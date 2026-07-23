"""Input-adaptive leakage controllers used by the published experiment."""

from __future__ import annotations

import math

import torch


class RuleBaseController:
    """Map normalized frame differences to a leakage coefficient.

    The first observation is compared with itself, so its difference is zero.
    Inputs are detached before being retained to avoid holding computation graphs.
    """

    def __init__(self, k: float, threshold: float):
        if not math.isfinite(float(k)) or float(k) <= 0:
            raise ValueError(f"k must be finite and positive, got {k!r}")
        if not math.isfinite(float(threshold)):
            raise ValueError(f"threshold must be finite, got {threshold!r}")
        self.k = float(k)
        self.threshold = float(threshold)
        self.previous_input: torch.Tensor | None = None
        self.differences: list[float] = []
        self.alphas: list[float] = []

    def __call__(self, image: torch.Tensor) -> float:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        current = image.detach()
        previous = current if self.previous_input is None else self.previous_input
        difference = torch.mean(torch.abs(current - previous))
        alpha = torch.sigmoid(self.k * (difference - self.threshold))
        difference_value = float(difference.item())
        alpha_value = float(alpha.item())
        self.previous_input = current.clone()
        self.differences.append(difference_value)
        self.alphas.append(alpha_value)
        return alpha_value

    @property
    def latest_difference(self) -> float:
        return self.differences[-1]

    def reset(self) -> None:
        self.previous_input = None
        self.differences.clear()
        self.alphas.clear()
