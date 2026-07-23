#!/usr/bin/env python3
"""GPU-aware batch runner shared by the three public MuJoCo tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from robo_manip_baselines.sarnn.lib.experiment_config import (
    DEFAULT_TASK,
    TASK_CONFIGS,
    WORLD_INDICES,
    VariantConfig,
    validate_world_index,
)


@dataclass(frozen=True)
class Job:
    task: str
    variant: VariantConfig
    world_idx: int


def build_jobs(task: str = DEFAULT_TASK, world_indices=WORLD_INDICES):
    """Build the Cartesian product of three variants and selected worlds."""

    config = TASK_CONFIGS[task]
    return [
        Job(task=task, variant=variant, world_idx=world_idx)
        for variant in config.variants
        for world_idx in world_indices
    ]


def parse_gpu_ids(value):
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--cuda-visible-devices requires at least one GPU ID")
    return gpu_ids


def parse_args(task=DEFAULT_TASK, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Run all public {task.title()} experiment variants."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--cropped-img-size", type=int, default=480)
    parser.add_argument("--skip", type=int, default=6)
    parser.add_argument(
        "--world-indices", type=int, nargs="+", default=list(WORLD_INDICES)
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--cuda-device-order", default="PCI_BUS_ID")
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--save-diagnostics", action="store_true")
    return parser.parse_args(argv)


def variant_arguments(variant):
    arguments = ["--qcfs_T", str(variant.qcfs_t)]
    if variant.no_reset:
        arguments.append("--no_reset")
    if variant.lif_controller != "none":
        arguments.extend(
            [
                "--lif_cnt",
                variant.lif_controller,
                "--LA_RB_k",
                str(variant.leakage_k),
                "--LA_RB_thresh",
                str(variant.leakage_threshold),
            ]
        )
    if variant.spike_decoder != "default":
        arguments.extend(["--spike_dec_type", variant.spike_decoder])
    return arguments


def build_command(args, config, job):
    command = [
        sys.executable,
        str(
            Path("robo_manip_baselines")
            / "sarnn"
            / "bin"
            / "rollout"
            / config.rollout_module
        ),
        "--checkpoint",
        str(args.checkpoint),
        "--output_dir",
        str(args.output_dir),
        "--cropped_img_size",
        str(args.cropped_img_size),
        "--skip",
        str(args.skip),
        "--world_idx",
        str(job.world_idx),
        *variant_arguments(job.variant),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.record_video:
        command.append("--record_video")
    if args.save_diagnostics:
        command.append("--save_diagnostics")
    return command


def format_command(command, gpu_id, cuda_device_order):
    environment = (
        f"CUDA_DEVICE_ORDER={shlex.quote(cuda_device_order)} "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)}"
    )
    return environment + " " + " ".join(shlex.quote(part) for part in command)


def aggregate_results(task, output_dir, jobs, failures):
    """Write task-level JSON and CSV summaries from per-world result files."""

    rows = []
    for job in jobs:
        result_path = (
            output_dir
            / task
            / job.variant.name
            / f"world_{job.world_idx}"
            / "result.json"
        )
        if not result_path.exists():
            continue
        with result_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        row = {
            "task": task,
            "variant": job.variant.name,
            "world_idx": job.world_idx,
            "checkpoint_sha256": result["checkpoint"]["sha256"],
            "inference_mean_seconds": result["inference"]["mean_seconds"],
            "inference_count": result["inference"]["count"],
        }
        row.update(result["task_metrics"])
        rows.append(row)

    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    failure_rows = [
        {
            "variant": job.variant.name,
            "world_idx": job.world_idx,
            "returncode": returncode,
            "log": str(log_path.relative_to(output_dir)),
        }
        for job, returncode, log_path in failures
    ]
    summary = {
        "schema_version": 1,
        "task": task,
        "expected_jobs": len(jobs),
        "completed_jobs": len(rows),
        "failed_jobs": failure_rows,
        "results": rows,
    }
    with (task_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    fieldnames = sorted({key for row in rows for key in row})
    with (task_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main(task=DEFAULT_TASK, argv=None):
    args = parse_args(task, argv)
    config = TASK_CONFIGS[task]
    if args.processes_per_gpu < 1:
        raise ValueError("--processes-per-gpu must be positive")
    world_indices = []
    for world_idx in args.world_indices:
        world_indices.append(validate_world_index(world_idx))
    if len(set(world_indices)) != len(world_indices):
        raise ValueError("--world-indices must not contain duplicates")
    gpu_ids = parse_gpu_ids(args.cuda_visible_devices)
    jobs = build_jobs(task, world_indices)

    if args.dry_run:
        for index, job in enumerate(jobs):
            command = build_command(args, config, job)
            print(
                format_command(
                    command,
                    gpu_ids[index % len(gpu_ids)],
                    args.cuda_device_order,
                )
            )
        return 0

    repository_root = Path(__file__).resolve().parents[3]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.output_dir.resolve() / task / "logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    failures = []
    stop_event = threading.Event()
    active_processes = set()
    process_lock = threading.Lock()

    def stop(_signum, _frame):
        stop_event.set()
        with process_lock:
            for process in tuple(active_processes):
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def worker(gpu_id, slot):
        while not stop_event.is_set():
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            command = build_command(args, config, job)
            log_path = log_dir / (
                f"{job.variant.name}_world-{job.world_idx}_gpu-{gpu_id}-{slot}.log"
            )
            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = args.cuda_device_order
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            with log_path.open("w", encoding="utf-8") as log:
                log.write(
                    format_command(command, gpu_id, args.cuda_device_order) + "\n"
                )
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=repository_root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with process_lock:
                    active_processes.add(process)
                returncode = process.wait()
                with process_lock:
                    active_processes.discard(process)
            if returncode:
                failures.append((job, returncode, log_path))
            job_queue.task_done()

    threads = []
    for gpu_id in gpu_ids:
        for slot in range(args.processes_per_gpu):
            thread = threading.Thread(target=worker, args=(gpu_id, slot))
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()

    aggregate_results(task, args.output_dir.resolve(), jobs, failures)
    if stop_event.is_set():
        return 130
    if failures:
        for job, returncode, log_path in failures:
            print(
                f"FAILED {job.variant.name} world={job.world_idx} "
                f"returncode={returncode} log={log_path}",
                file=sys.stderr,
            )
        return 1
    return 0
