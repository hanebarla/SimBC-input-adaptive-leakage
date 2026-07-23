"""SARNN policy loading and reproducible MuJoCo experiment runner."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from eipl.utils import deprocess_img, normalization, resize_img, tensor2numpy

from robo_manip_baselines.common import MotionStatus
from robo_manip_baselines.common.rollout import RolloutBase
from robo_manip_baselines.sarnn.lib.LifController import RuleBaseController
from robo_manip_baselines.sarnn.lib.experiment_config import validate_world_index
from robo_manip_baselines.sarnn.lib.model_utils import (
    build_front_action_model,
    load_checkpoint,
    load_experiment_args,
    sha256_file,
)


class RolloutSarnn(RolloutBase):
    """Shared SARNN rollout with SNN instrumentation kept out of generic rollouts."""

    TASK_NAME = None
    ENV_ID = None

    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument(
            "--output_dir", type=Path, required=self.TASK_NAME is not None
        )
        parser.add_argument("--cropped_img_size", default=480, type=int)
        parser.add_argument("--qcfs_T", type=int)
        parser.add_argument("--no_reset", action="store_true")
        parser.add_argument("--lif_cnt", choices=("none", "rule"), default="none")
        parser.add_argument("--LA_RB_k", type=float, default=32.0)
        parser.add_argument("--LA_RB_thresh", type=float, default=0.03)
        parser.add_argument(
            "--spike_dec_type", choices=("default", "decay"), default="default"
        )
        parser.add_argument(
            "--device", default="cuda", choices=("cpu", "cuda"), help="Inference device"
        )
        parser.add_argument("--record_video", action="store_true")
        parser.add_argument("--save_diagnostics", action="store_true")
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--max_policy_steps", type=int, default=420)

        super().setup_args(parser)

        if self.args.skip is None:
            self.args.skip = 6
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip
        validate_world_index(self.args.world_idx)
        if self.args.skip <= 0 or self.args.max_policy_steps <= 0:
            raise ValueError("skip and max_policy_steps must be positive")
        if self.args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    def setup_policy(self):
        checkpoint_dir = self.args.checkpoint.resolve().parent
        self.params = load_experiment_args(checkpoint_dir / "args.json")
        self.joint_bounds = np.load(checkpoint_dir / "action_bounds.npy")
        self.joint_dim = self.joint_bounds.shape[-1]
        self.joint_scales = [1.0] * (self.joint_dim - 1) + [0.01]
        self.im_size = int(self.params["im_size"][0])
        if self.params["im_size"][0] != self.params["im_size"][1]:
            raise ValueError("Rollout requires square input images")
        self.v_min_max = [self.params["vmin"], self.params["vmax"]]
        self.pred_action_list = np.empty((0, self.joint_dim))
        self.rnn_state = None
        self.device = torch.device(self.args.device)

        runtime_t = (
            self.params["qcfs_T"] if self.args.qcfs_T is None else self.args.qcfs_T
        )
        if runtime_t < 0:
            raise ValueError("qcfs_T must be non-negative")
        if self.args.lif_cnt == "rule":
            if runtime_t <= 0:
                raise ValueError("The rule controller requires qcfs_T > 0")
            if not self.args.no_reset or self.args.spike_dec_type != "decay":
                raise ValueError(
                    "The proposed method requires --no_reset and "
                    "--spike_dec_type decay"
                )
        if self.args.no_reset and runtime_t <= 0:
            raise ValueError("--no_reset is only meaningful when qcfs_T > 0")

        self.policy = build_front_action_model(
            self.params,
            self.joint_dim,
            qcfs_t=runtime_t,
            reset=not self.args.no_reset,
            spike_dec_type=self.args.spike_dec_type,
        )
        print(f"[RolloutSarnn] Load {self.args.checkpoint}")
        load_checkpoint(self.policy, self.args.checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        self.model_size_bytes = sum(
            tensor.nelement() * tensor.element_size()
            for tensor in self.policy.state_dict().values()
        )
        if hasattr(self.policy, "reset_state"):
            self.policy.reset_state(reset_spike_count=True)

        self.runtime_qcfs_t = int(runtime_t)
        self.controller = None
        if self.args.lif_cnt == "rule":
            self.controller = RuleBaseController(
                k=self.args.LA_RB_k, threshold=self.args.LA_RB_thresh
            )
        self.variant_name = self._variant_name()
        self.checkpoint_sha256 = sha256_file(self.args.checkpoint)

        self.trace_frame_indices = []
        self.trace_inference_seconds = []
        self.trace_alphas = []
        self.trace_differences = []
        self.trace_spikes = []
        self.trace_encoder_points = []
        self.trace_decoder_points = []
        self.diagnostic_inputs = []
        self.diagnostic_predictions = []
        self.video_frames = []

        if self.args.output_dir is not None:
            self.result_dir = (
                self.args.output_dir.resolve()
                / self.TASK_NAME
                / self.variant_name
                / f"world_{self.args.world_idx}"
            )
            existing = (
                list(self.result_dir.iterdir()) if self.result_dir.exists() else []
            )
            if existing and not self.args.overwrite:
                raise FileExistsError(
                    f"Results already exist in {self.result_dir}; pass --overwrite"
                )
            if self.args.overwrite:
                for filename in (
                    "result.json",
                    "trace.npz",
                    "diagnostics.npz",
                    "rollout.mp4",
                ):
                    artifact = self.result_dir / filename
                    if artifact.is_file():
                        artifact.unlink()
            self.result_dir.mkdir(parents=True, exist_ok=True)

    def _variant_name(self):
        if not self.params.get("qcfs", False):
            return "sarnn"
        if self.runtime_qcfs_t == 0:
            return "qcfs_ann"
        if self.args.lif_cnt == "rule":
            return f"input_adaptive_t{self.runtime_qcfs_t}"
        return f"qcfs_snn_t{self.runtime_qcfs_t}"

    def setup_env(self):
        if self.ENV_ID is None:
            super().setup_env()
            return
        import gymnasium as gym

        self.env = gym.make(self.ENV_ID, render_mode="rgb_array")

    def setup_plot(self):
        """Headless runs intentionally do not create OpenCV windows."""

    def infer_policy(self):
        if self.auto_time_idx % self.args.skip != 0:
            return False

        self.obs_front_image = self.info["rgb_images"]["front"]
        left, top = [
            (self.obs_front_image.shape[axis] - self.args.cropped_img_size) // 2
            for axis in (0, 1)
        ]
        if min(left, top) < 0:
            raise ValueError(
                f"cropped_img_size={self.args.cropped_img_size} exceeds image shape "
                f"{self.obs_front_image.shape[:2]}"
            )
        right, bottom = (
            left + self.args.cropped_img_size,
            top + self.args.cropped_img_size,
        )
        self.obs_front_image = self.obs_front_image[left:right, top:bottom, :]
        self.obs_front_image = resize_img(
            np.expand_dims(self.obs_front_image, 0), (self.im_size, self.im_size)
        )[0]
        front_input = self.obs_front_image.transpose(2, 0, 1)
        front_input = normalization(front_input, (0, 255), self.v_min_max)
        front_input = torch.as_tensor(
            np.expand_dims(front_input, 0), dtype=torch.float32
        )

        alpha = np.nan
        difference = np.nan
        if self.controller is not None:
            alpha = self.controller(front_input)
            difference = self.controller.latest_difference
            self.policy.update_LIF_param(alpha)

        action_input = self.motion_manager.get_action()
        action_input = normalization(action_input, self.joint_bounds, self.v_min_max)
        action_input = torch.as_tensor(
            np.expand_dims(action_input, 0), dtype=torch.float32, device=self.device
        )

        with torch.inference_mode():
            (
                front_output,
                action_output,
                enc_front_points,
                dec_front_points,
                self.rnn_state,
            ) = self.policy(front_input.to(self.device), action_input, self.rnn_state)

        self.pred_front_image = tensor2numpy(front_output[0])
        self.pred_front_image = deprocess_img(
            self.pred_front_image, self.params["vmin"], self.params["vmax"]
        ).transpose(1, 2, 0)
        self.pred_action = tensor2numpy(action_output[0])
        self.pred_action = normalization(
            self.pred_action, self.v_min_max, self.joint_bounds
        )
        self.pred_action_list = np.concatenate(
            [self.pred_action_list, np.expand_dims(self.pred_action, 0)]
        )
        self.enc_front_pts = (
            tensor2numpy(enc_front_points[0]).reshape(self.params["k_dim"], 2)
            * self.im_size
        )
        self.dec_front_pts = (
            tensor2numpy(dec_front_points[0]).reshape(self.params["k_dim"], 2)
            * self.im_size
        )

        self.trace_frame_indices.append(self.auto_time_idx)
        self.trace_alphas.append(alpha)
        self.trace_differences.append(difference)
        self.trace_encoder_points.append(self.enc_front_pts.copy())
        self.trace_decoder_points.append(self.dec_front_pts.copy())
        if hasattr(self.policy, "get_spike_cnt"):
            self.trace_spikes.append(
                [float(count) for _, count in self.policy.get_spike_cnt()]
            )
        else:
            self.trace_spikes.append([])
        if self.args.save_diagnostics:
            self.diagnostic_inputs.append(self.obs_front_image.copy())
            self.diagnostic_predictions.append(self.pred_front_image.copy())
        return True

    def draw_plot(self):
        """Default public runs are headless; video is captured from the environment."""

    def run(self):
        if self.TASK_NAME is None or self.args.output_dir is None:
            return super().run()

        self.obs, self.info = self.env.reset(seed=self.args.seed)
        self.auto_time_idx = 0
        phase_elapsed = 0.0
        if self.controller is not None:
            self.controller.reset()

        while True:
            if self.data_manager.status == MotionStatus.TELEOP:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                start = time.perf_counter()
                inference_called = self.infer_policy()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                duration = time.perf_counter() - start
                if inference_called:
                    self.trace_inference_seconds.append(duration)

            self.set_arm_command()
            self.set_gripper_command()
            action = self.motion_manager.get_action()
            self.obs, _, _, _, self.info = self.env.step(action)
            phase_elapsed += self.env.unwrapped.dt

            if self.args.record_video:
                self.video_frames.append(
                    np.asarray(self.info["rgb_images"]["front"]).copy()
                )

            if self.data_manager.status == MotionStatus.INITIAL and phase_elapsed > 1.0:
                self.data_manager.go_to_next_status()
                phase_elapsed = 0.0
            elif (
                self.data_manager.status == MotionStatus.PRE_REACH
                and phase_elapsed > 0.7
            ):
                self.data_manager.go_to_next_status()
                phase_elapsed = 0.0
            elif self.data_manager.status == MotionStatus.REACH and phase_elapsed > 0.3:
                self.data_manager.go_to_next_status()
                phase_elapsed = 0.0
            elif self.data_manager.status == MotionStatus.GRASP and phase_elapsed > 0.5:
                self.auto_time_idx = 0
                self.data_manager.go_to_next_status()
                phase_elapsed = 0.0
            elif self.data_manager.status == MotionStatus.TELEOP:
                self.auto_time_idx += 1
                if self.auto_time_idx >= self.args.max_policy_steps:
                    self.data_manager.go_to_next_status()
            elif self.data_manager.status == MotionStatus.END:
                break

        self._save_results()
        self.env.close()

    def _save_results(self):
        inference = np.asarray(self.trace_inference_seconds, dtype=np.float64)
        spikes = np.asarray(self.trace_spikes, dtype=np.float64)
        if spikes.size == 0:
            spikes = np.empty((len(self.trace_frame_indices), 0), dtype=np.float64)
        np.savez_compressed(
            self.result_dir / "trace.npz",
            frame_index=np.asarray(self.trace_frame_indices, dtype=np.int64),
            inference_seconds=inference,
            leakage_alpha=np.asarray(self.trace_alphas, dtype=np.float64),
            image_difference=np.asarray(self.trace_differences, dtype=np.float64),
            cumulative_spikes=spikes,
            encoder_points=np.asarray(self.trace_encoder_points, dtype=np.float32),
            decoder_points=np.asarray(self.trace_decoder_points, dtype=np.float32),
        )

        if self.args.save_diagnostics:
            np.savez_compressed(
                self.result_dir / "diagnostics.npz",
                input_images=np.asarray(self.diagnostic_inputs, dtype=np.uint8),
                predicted_images=np.asarray(
                    self.diagnostic_predictions, dtype=np.float32
                ),
            )
        if self.args.record_video and self.video_frames:
            import imageio.v3 as iio

            iio.imwrite(
                self.result_dir / "rollout.mp4",
                np.asarray(self.video_frames),
                fps=30,
            )

        spike_names = []
        if hasattr(self.policy, "get_spike_cnt"):
            raw_names = [name for name, _ in self.policy.get_spike_cnt()]
            midpoint = len(raw_names) // 2
            spike_names = [
                (
                    f"image_encoder.{name}"
                    if index < midpoint
                    else f"position_encoder.{name}"
                )
                for index, name in enumerate(raw_names)
            ]
        timing = {
            "count": int(inference.size),
            "mean_seconds": float(inference.mean()) if inference.size else None,
            "std_seconds": float(inference.std()) if inference.size else None,
            "min_seconds": float(inference.min()) if inference.size else None,
            "max_seconds": float(inference.max()) if inference.size else None,
        }
        model_info = {
            "class": self.policy.__class__.__name__,
            "parameter_bytes": int(self.model_size_bytes),
            "im_size": self.params["im_size"],
            "k_dim": int(self.params["k_dim"]),
            "rec_dim": int(self.params["rec_dim"]),
            "qcfs_L": int(self.params["qcfs_L"]),
            "qcfs_T": self.runtime_qcfs_t,
            "reset_between_inferences": not self.args.no_reset,
            "spike_decoder": self.args.spike_dec_type,
        }
        if hasattr(self.policy, "estimate_encoder_macs"):
            model_info["encoder_macs"] = self.policy.estimate_encoder_macs(
                self.params["im_size"]
            )
        result = {
            "schema_version": 1,
            "task": self.TASK_NAME,
            "variant": self.variant_name,
            "world_idx": self.args.world_idx,
            "seed": self.args.seed,
            "checkpoint": {
                "filename": self.args.checkpoint.name,
                "sha256": self.checkpoint_sha256,
            },
            "configuration": {
                "crop": self.args.cropped_img_size,
                "skip": self.args.skip,
                "max_policy_steps": self.args.max_policy_steps,
                "lif_controller": self.args.lif_cnt,
                "leakage_k": self.args.LA_RB_k if self.controller else None,
                "leakage_threshold": (
                    self.args.LA_RB_thresh if self.controller else None
                ),
            },
            "environment": {
                "id": self.ENV_ID,
                "num_worlds": int(self.env.unwrapped.num_worlds),
            },
            "model": model_info,
            "task_metrics": self.env.unwrapped.get_task_metrics(),
            "inference": timing,
            "spikes": {
                "layer_names": spike_names,
                "final_cumulative": (spikes[-1].tolist() if spikes.shape[0] else []),
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(self.device),
                "cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda"
                    else None
                ),
            },
        }
        with (self.result_dir / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
