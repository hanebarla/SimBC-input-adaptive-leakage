#!/usr/bin/env python3
"""Import and QCFS forward smoke test for the Apptainer environment."""

from __future__ import annotations

import argparse

import gymnasium
import mujoco
import torch
import torchvision

import robo_manip_baselines  # noqa: F401
from robo_manip_baselines.policy.sarnn import QCFSSarnnPolicy


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true")
    group.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requested but CUDA is unavailable")
    device = torch.device("cuda" if args.gpu else "cpu")
    model = QCFSSarnnPolicy(
        state_dim=7,
        num_images=1,
        image_size_list=[16, 16],
        num_attentions=3,
        lstm_hidden_dim=8,
        qcfs_L=8,
        qcfs_T=2,
    ).to(device)
    state = torch.zeros(1, 7, device=device)
    image = torch.zeros(1, 3, 16, 16, device=device)
    with torch.inference_mode():
        output = model(state, [image])
    assert output[0].shape == state.shape
    assert output[1][0].shape == image.shape
    print(
        "smoke-test ok "
        f"device={device} torch={torch.__version__} "
        f"torchvision={torchvision.__version__} "
        f"gymnasium={gymnasium.__version__} mujoco={mujoco.__version__}"
    )


if __name__ == "__main__":
    main()
