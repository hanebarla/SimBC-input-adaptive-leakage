#!/usr/bin/env python3
"""Run the canonical three-way SARNN experiment for the Cloth task."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

CLOTH_WORLD_INDICES = tuple(range(21))


@dataclass(frozen=True)
class ExperimentJob:
    name: str
    command: tuple[str, ...]
    result_path: Path
    log_path: Path


def build_jobs(checkpoint: Path, output_dir: Path) -> list[ExperimentJob]:
    """Build commands for ANN, conventional SNN, and input-adaptive SNN."""

    repo_root = Path(__file__).resolve().parents[2]
    rollout_script = repo_root / "robo_manip_baselines" / "bin" / "Rollout.py"
    common = (
        sys.executable,
        str(rollout_script),
        "Sarnn",
        "MujocoUR5eCloth",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--world_idx_list",
        *(str(index) for index in CLOTH_WORLD_INDICES),
        "--use_test_offset",
        "--no_render",
        "--no_plot",
        "--auto_exit",
    )
    variants = {
        "qcfs_ann": ("--qcfs_T", "0"),
        "qcfs_snn_t3": (
            "--qcfs_T",
            "3",
            "--spike_dec_type",
            "default",
        ),
        "input_adaptive_t4": (
            "--qcfs_T",
            "4",
            "--no_reset",
            "--lif_cnt",
            "rule",
            "--la_rb_k",
            "32",
            "--la_rb_thresh",
            "0.03",
            "--spike_dec_type",
            "decay",
        ),
    }

    jobs = []
    for name, variant_args in variants.items():
        variant_dir = output_dir / "cloth" / name
        result_path = variant_dir / "result.yaml"
        log_path = variant_dir / "rollout.log"
        command = (*common, *variant_args, "--result_filename", str(result_path))
        jobs.append(ExperimentJob(name, command, result_path, log_path))
    return jobs


def _check_output_collisions(jobs: list[ExperimentJob], overwrite: bool) -> None:
    collisions = [
        path
        for job in jobs
        for path in (job.result_path, job.log_path)
        if path.exists()
    ]
    if collisions and not overwrite:
        paths = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(
            "Refusing to overwrite existing experiment outputs:\n"
            f"{paths}\nPass --overwrite to replace them."
        )


def _run_job(job: ExperimentJob, gpu: str) -> tuple[str, int]:
    job.result_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    if gpu.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    with job.log_path.open("w") as log_file:
        result = subprocess.run(
            job.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    return job.name, result.returncode


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0"],
        help="GPU identifiers; specify 'cpu' to disable CUDA",
    )
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=1,
        help="maximum simultaneous rollout processes per GPU",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    meta_path = checkpoint.parent / "model_meta_info.pkl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Model metadata does not exist: {meta_path}")
    if args.processes_per_gpu <= 0:
        raise ValueError("processes-per-gpu must be positive.")

    jobs = build_jobs(checkpoint, args.output_dir.resolve())
    _check_output_collisions(jobs, args.overwrite)
    gpu_slots = [gpu for _ in range(args.processes_per_gpu) for gpu in args.gpus]

    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpu_slots[index % len(gpu_slots)]
            print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(job.command)}")
        return 0

    failures = []
    executors = [ThreadPoolExecutor(max_workers=1) for _ in gpu_slots]
    try:
        futures = {
            executors[index % len(executors)].submit(
                _run_job, job, gpu_slots[index % len(gpu_slots)]
            ): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            name, returncode = future.result()
            print(f"[{name}] exit code: {returncode}", flush=True)
            if returncode != 0:
                failures.append(name)
    finally:
        for executor in executors:
            executor.shutdown()

    if failures:
        print(f"Failed variants: {', '.join(sorted(failures))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
