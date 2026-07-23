#!/usr/bin/env python3
"""Small import and forward smoke test for the official container."""

from __future__ import annotations

import argparse

import gymnasium
import mujoco
import torch
import torchvision
from eipl.model import SpikingSARNN

import robo_manip_baselines  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true")
    group.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requested but CUDA is unavailable")
    device = torch.device("cuda" if args.gpu else "cpu")
    model = SpikingSARNN(
        rec_dim=8,
        k_dim=3,
        joint_dim=7,
        im_size=[16, 16],
        qcfs_L=8,
        qcfs_T=2,
    ).to(device)
    image = torch.zeros(1, 3, 16, 16, device=device)
    action = torch.zeros(1, 7, device=device)
    with torch.inference_mode():
        output = model(image, action)
    assert output[0].shape == image.shape
    assert output[1].shape == action.shape
    print(
        "smoke-test ok "
        f"device={device} torch={torch.__version__} "
        f"torchvision={torchvision.__version__} "
        f"gymnasium={gymnasium.__version__} mujoco={mujoco.__version__}"
    )


if __name__ == "__main__":
    main()
