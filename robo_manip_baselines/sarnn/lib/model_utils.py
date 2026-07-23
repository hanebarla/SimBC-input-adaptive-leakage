"""Shared model construction and checkpoint compatibility helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


LEGACY_DEFAULTS: dict[str, Any] = {
    "compile": False,
    "heatmap_size": 0.1,
    "im_size": [128, 128],
    "k_dim": 20,
    "no_side_image": True,
    "no_wrench": True,
    "qcfs": False,
    "qcfs_L": 8,
    "qcfs_T": 0,
    "rec_dim": 50,
    "temperature": 1e-4,
    "vmax": 1.0,
    "vmin": 0.0,
    "wobn": True,
}


def load_experiment_args(path: str | Path) -> dict[str, Any]:
    """Load an experiment JSON and fill keys absent from older runs."""

    args_path = Path(path)
    with args_path.open(encoding="utf-8") as stream:
        stored = json.load(stream)
    if not isinstance(stored, dict):
        raise ValueError(f"Expected a JSON object in {args_path}")
    params = {**LEGACY_DEFAULTS, **stored}
    image_size = params["im_size"]
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    if len(image_size) != 2:
        raise ValueError(f"im_size must contain two values, got {image_size!r}")
    params["im_size"] = [int(image_size[0]), int(image_size[1])]
    return params


def build_front_action_model(
    params: dict[str, Any],
    joint_dim: int,
    *,
    qcfs_t: int | None = None,
    reset: bool = True,
    spike_dec_type: str = "default",
):
    """Build the front-image/action SARNN used by training, test, and rollout."""

    use_qcfs = bool(params.get("qcfs", False))
    if use_qcfs:
        from eipl.model import SpikingSARNN

        model = SpikingSARNN(
            rec_dim=int(params["rec_dim"]),
            joint_dim=joint_dim,
            k_dim=int(params["k_dim"]),
            heatmap_size=float(params["heatmap_size"]),
            temperature=float(params["temperature"]),
            im_size=params["im_size"],
            qcfs_L=int(params["qcfs_L"]),
            qcfs_T=int(params["qcfs_T"] if qcfs_t is None else qcfs_t),
            reset=reset,
            spike_dec_type=spike_dec_type,
        )
    else:
        if qcfs_t not in (None, 0):
            raise ValueError("qcfs_T can only be changed for a QCFS checkpoint")
        from eipl.model import SARNN

        model = SARNN(
            rec_dim=int(params["rec_dim"]),
            joint_dim=joint_dim,
            k_dim=int(params["k_dim"]),
            heatmap_size=float(params["heatmap_size"]),
            temperature=float(params["temperature"]),
            im_size=params["im_size"],
        )
    return model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, including weights saved from torch.compile."""

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    prefix = "_orig_mod."
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a checkpoint digest without loading the file in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
