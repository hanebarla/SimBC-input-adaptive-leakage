# Copyright (C) 2025-2026 Takehiro Habara
#
# This file is part of the input-adaptive leakage extension to EIPL and is
# released under the GNU Affero General Public License v3.0.

"""QCFS activation and temporal decoding layers for spiking SARNN.

The QCFS implementation is based in part on putshua/ANN_SNN_QCFS:
https://github.com/putshua/ANN_SNN_QCFS
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "ExpandTemporalDim",
    "GradFloor",
    "IF",
    "MergeTemporalDim",
    "ST2DECAY_FR",
    "ST2FR",
    "ZIF",
    "leaky_qcfs",
]


def _validate_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}.")
    return alpha


class MergeTemporalDim(nn.Module):
    """Merge a tensor shaped ``(T, B, ...)`` into ``(T * B, ...)``."""

    def __init__(self, T: int):
        super().__init__()
        self.T = _validate_nonnegative_int("T", T)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if self.T <= 0:
            raise ValueError("T must be positive when merging temporal data.")
        if x_seq.ndim < 2 or x_seq.shape[0] != self.T:
            raise ValueError(
                f"Expected leading temporal dimension {self.T}, "
                f"got shape {tuple(x_seq.shape)}."
            )
        return x_seq.flatten(0, 1).contiguous()


class ExpandTemporalDim(nn.Module):
    """Restore a tensor shaped ``(T * B, ...)`` to ``(T, B, ...)``."""

    def __init__(self, T: int):
        super().__init__()
        self.T = _validate_nonnegative_int("T", T)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if self.T <= 0:
            raise ValueError("T must be positive when expanding temporal data.")
        if x_seq.ndim == 0 or x_seq.shape[0] % self.T != 0:
            leading_dim = x_seq.shape[0] if x_seq.ndim else 0
            raise ValueError(
                f"Leading dimension {leading_dim} must be divisible by T={self.T}."
            )
        return x_seq.reshape(self.T, x_seq.shape[0] // self.T, *x_seq.shape[1:])


class ZIF(torch.autograd.Function):
    """Zero-one spiking function with a triangular surrogate gradient."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, gama: float) -> torch.Tensor:
        if not math.isfinite(float(gama)) or float(gama) <= 0:
            raise ValueError(f"gama must be finite and positive, got {gama!r}.")
        ctx.gama = float(gama)
        ctx.save_for_backward(input)
        return (input >= 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        gama = ctx.gama
        surrogate = (gama - input.abs()).clamp(min=0) / (gama * gama)
        return grad_output * surrogate, None


class GradFloor(torch.autograd.Function):
    """Floor during the forward pass and use a straight-through gradient."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


grad_floor = GradFloor.apply


def leaky_qcfs(
    membrane: torch.Tensor, threshold: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Leak membrane potential around half of the firing threshold."""

    alpha = _validate_alpha(alpha)
    return (membrane - 0.5 * threshold) * (1.0 - alpha) + 0.5 * threshold


class IF(nn.Module):
    """QCFS in ANN mode and integrate-and-fire activation in SNN mode."""

    def __init__(
        self,
        T: int = 0,
        L: int = 8,
        thresh: float = 8.0,
        tau: float = 1.0,
        gama: float = 1.0,
        reset: bool = True,
        lif_alpha: float = 0.0,
    ):
        super().__init__()
        self.T = _validate_nonnegative_int("T", T)
        if isinstance(L, bool) or not isinstance(L, int) or L <= 0:
            raise ValueError(f"L must be a positive integer, got {L!r}.")
        if not math.isfinite(float(thresh)) or float(thresh) <= 0:
            raise ValueError(f"thresh must be finite and positive, got {thresh!r}.")
        if not math.isfinite(float(gama)) or float(gama) <= 0:
            raise ValueError(f"gama must be finite and positive, got {gama!r}.")

        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([float(thresh)]))
        self.tau = float(tau)
        self.gama = float(gama)
        self.expand = ExpandTemporalDim(self.T)
        self.merge = MergeTemporalDim(self.T)
        self.L = L
        self.reset = bool(reset)
        self.lif_alpha = _validate_alpha(lif_alpha)
        self.register_buffer("mem", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.T == 0:
            quantized = torch.clamp(x / self.thresh, 0.0, 1.0)
            quantized = grad_floor(quantized * self.L + 0.5) / self.L
            return quantized * self.thresh

        threshold = self.thresh.detach()
        temporal = self.expand(x)
        if self.mem is None or self.reset:
            self.mem = 0.5 * threshold
        self.mem = leaky_qcfs(self.mem, threshold, self.lif_alpha)

        spikes = []
        for timestep in range(self.T):
            self.mem = self.mem + temporal[timestep]
            spike = self.act(self.mem - threshold, self.gama) * threshold
            self.mem = self.mem - spike
            spikes.append(spike)
        return self.merge(torch.stack(spikes, dim=0))

    def update_LIF_param(self, alpha: float) -> None:
        self.lif_alpha = _validate_alpha(alpha)

    def reset_state(self) -> None:
        self.mem = None

    def __repr__(self) -> str:
        return (
            f"IF(T={self.T}, L={self.L}, thresh={self.thresh}, "
            f"tau={self.tau}, gama={self.gama}, reset={self.reset})"
        )


class ST2FR(nn.Module):
    """Decode a temporal spike tensor using its mean firing rate."""

    def __init__(self, qcfs_T: int, dec_time: int = 0):
        super().__init__()
        self.qcfs_T = _validate_nonnegative_int("qcfs_T", qcfs_T)
        self.dec_time = _validate_nonnegative_int("dec_time", dec_time)
        if self.qcfs_T == 0:
            raise ValueError("qcfs_T must be positive for a spike decoder.")
        self.expand = ExpandTemporalDim(self.qcfs_T)
        self.register_buffer("window", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.expand(x)
        if self.dec_time > 0:
            temporal = self._update_window(temporal)
        return temporal.mean(dim=0)

    def _update_window(self, temporal: torch.Tensor) -> torch.Tensor:
        if self.qcfs_T > self.dec_time:
            self.window = temporal[-self.dec_time :]
        else:
            if self.window is None or self.window.shape[1:] != temporal.shape[1:]:
                self.window = temporal.new_zeros((self.dec_time, *temporal.shape[1:]))
            self.window = torch.cat((self.window, temporal), dim=0)[-self.dec_time :]
        return self.window

    def reset_state(self) -> None:
        self.window = None


class ST2DECAY_FR(ST2FR):
    """Mix the current firing rate with the preceding inference step."""

    def __init__(self, qcfs_T: int, alpha: float = 0.0, dec_time: int = 0):
        super().__init__(qcfs_T=qcfs_T, dec_time=dec_time)
        self.alpha = _validate_alpha(alpha)
        self.register_buffer("prev_fr", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.expand(x)
        if self.dec_time > 0:
            temporal = self._update_window(temporal)
        current_fr = temporal.mean(dim=0)
        if self.prev_fr is None:
            decoded = current_fr
        else:
            decay = 1.0 - self.alpha
            decoded = (decay * self.prev_fr + current_fr) / (decay + 1.0)
        # Preserve the experiment behavior: retain the current, unmixed rate.
        self.prev_fr = current_fr
        return decoded

    def update_LIF_param(self, alpha: float) -> None:
        self.alpha = _validate_alpha(alpha)

    def reset_state(self) -> None:
        super().reset_state()
        self.prev_fr = None
