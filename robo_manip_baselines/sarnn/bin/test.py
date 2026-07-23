"""Offline front-image/action evaluation through the shared model loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from eipl.utils import deprocess_img, normalization, tensor2numpy

from robo_manip_baselines.sarnn.lib.LifController import RuleBaseController
from robo_manip_baselines.sarnn.lib.model_utils import (
    build_front_action_model,
    load_checkpoint,
    load_experiment_args,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "--filename", dest="checkpoint", type=Path, required=True
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--qcfs_T", type=int)
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--lif_cnt", choices=("none", "rule"), default="none")
    parser.add_argument("--LA_RB_k", type=float, default=32.0)
    parser.add_argument("--LA_RB_thresh", type=float, default=0.03)
    parser.add_argument(
        "--spike_dec_type", choices=("default", "decay"), default="default"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no_side_image", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_wrench", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_experiment_args(args.checkpoint.resolve().parent / "args.json")
    if not params["no_side_image"] or not params["no_wrench"]:
        raise ValueError(
            "This public evaluator supports front-image/action models only"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    front_images = np.load(args.data_dir / "test" / "front_images.npy")[args.idx]
    actions = np.load(args.data_dir / "test" / "actions.npy")[args.idx]
    action_bounds = np.load(args.data_dir / "action_bounds.npy")
    runtime_t = params["qcfs_T"] if args.qcfs_T is None else args.qcfs_T
    model = build_front_action_model(
        params,
        actions.shape[-1],
        qcfs_t=runtime_t,
        reset=not args.no_reset,
        spike_dec_type=args.spike_dec_type,
    )
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    if hasattr(model, "reset_state"):
        model.reset_state(reset_spike_count=True)

    controller = None
    if args.lif_cnt == "rule":
        if runtime_t <= 0 or not args.no_reset or args.spike_dec_type != "decay":
            raise ValueError(
                "Rule control requires qcfs_T > 0, --no_reset, and decay decoding"
            )
        controller = RuleBaseController(args.LA_RB_k, args.LA_RB_thresh)

    value_range = [params["vmin"], params["vmax"]]
    state = None
    predicted_images = []
    predicted_actions = []
    encoder_points = []
    decoder_points = []
    leakage_alpha = []
    image_difference = []
    cumulative_spikes = []

    with torch.inference_mode():
        for image, action in zip(front_images, actions):
            image_input = normalization(image.transpose(2, 0, 1), (0, 255), value_range)
            image_input = torch.as_tensor(image_input[None], dtype=torch.float32)
            alpha = np.nan
            difference = np.nan
            if controller is not None:
                alpha = controller(image_input)
                difference = controller.latest_difference
                model.update_LIF_param(alpha)
            action_input = normalization(action, action_bounds, value_range)
            action_input = torch.as_tensor(
                action_input[None], dtype=torch.float32, device=device
            )
            image_output, action_output, enc_points, dec_points, state = model(
                image_input.to(device), action_input, state
            )
            prediction = deprocess_img(
                tensor2numpy(image_output[0]), params["vmin"], params["vmax"]
            ).transpose(1, 2, 0)
            predicted_images.append(prediction)
            predicted_actions.append(
                normalization(
                    tensor2numpy(action_output[0]), value_range, action_bounds
                )
            )
            encoder_points.append(tensor2numpy(enc_points[0]))
            decoder_points.append(tensor2numpy(dec_points[0]))
            leakage_alpha.append(alpha)
            image_difference.append(difference)
            if hasattr(model, "get_spike_cnt"):
                cumulative_spikes.append(
                    [float(count) for _, count in model.get_spike_cnt()]
                )

    action_error = np.asarray(predicted_actions) - actions
    print(f"frames={len(predicted_actions)} action_mse={np.mean(action_error**2):.8f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            predicted_images=np.asarray(predicted_images, dtype=np.float32),
            predicted_actions=np.asarray(predicted_actions, dtype=np.float32),
            encoder_points=np.asarray(encoder_points, dtype=np.float32),
            decoder_points=np.asarray(decoder_points, dtype=np.float32),
            leakage_alpha=np.asarray(leakage_alpha, dtype=np.float64),
            image_difference=np.asarray(image_difference, dtype=np.float64),
            cumulative_spikes=np.asarray(cumulative_spikes, dtype=np.float64),
        )


if __name__ == "__main__":
    main()
