#!/usr/bin/env python3
"""Validate legacy experiment checkpoints without copying or modifying them."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from robo_manip_baselines.sarnn.lib.LifController import RuleBaseController
from robo_manip_baselines.sarnn.lib.experiment_config import TASK_CONFIGS
from robo_manip_baselines.sarnn.lib.model_utils import (
    build_front_action_model,
    load_checkpoint,
    load_experiment_args,
    sha256_file,
)


def parse_mapping(value):
    try:
        task, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected TASK=PATH") from error
    if task not in TASK_CONFIGS:
        raise argparse.ArgumentTypeError(f"Unknown task: {task}")
    return task, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_mapping,
        required=True,
        metavar="TASK=PATH",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    for task, checkpoint_path in args.checkpoint:
        checkpoint_path = checkpoint_path.resolve()
        params = load_experiment_args(checkpoint_path.parent / "args.json")
        action_bounds = np.load(checkpoint_path.parent / "action_bounds.npy")
        joint_dim = action_bounds.shape[-1]
        image_size = params["im_size"]
        for variant in TASK_CONFIGS[task].variants:
            model = build_front_action_model(
                params,
                joint_dim,
                qcfs_t=variant.qcfs_t,
                reset=not variant.no_reset,
                spike_dec_type=variant.spike_decoder,
            )
            load_checkpoint(model, checkpoint_path)
            model.to(device).eval()
            if hasattr(model, "reset_state"):
                model.reset_state(reset_spike_count=True)
            image = torch.zeros(1, 3, image_size[0], image_size[1], device=device)
            action = torch.zeros(1, joint_dim, device=device)
            if variant.lif_controller == "rule":
                controller = RuleBaseController(
                    variant.leakage_k, variant.leakage_threshold
                )
                model.update_LIF_param(controller(image))
            with torch.inference_mode():
                image_output, action_output, *_ = model(image, action)
            if image_output.shape != image.shape:
                raise RuntimeError(
                    f"{task}/{variant.name}: image shape mismatch "
                    f"{image_output.shape} != {image.shape}"
                )
            if action_output.shape != action.shape:
                raise RuntimeError(
                    f"{task}/{variant.name}: action shape mismatch "
                    f"{action_output.shape} != {action.shape}"
                )
            print(f"ok task={task} variant={variant.name}")
        print(f"sha256 task={task} digest={sha256_file(checkpoint_path)}")


if __name__ == "__main__":
    main()
