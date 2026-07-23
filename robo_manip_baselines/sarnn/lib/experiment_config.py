"""Canonical task and variant configuration for the SimBC experiments."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TASK = "ring"
WORLD_INDICES = tuple(range(21))


@dataclass(frozen=True)
class VariantConfig:
    """Runtime options for one checkpoint-conversion variant."""

    name: str
    qcfs_t: int
    no_reset: bool = False
    lif_controller: str = "none"
    leakage_k: float = 0.0
    leakage_threshold: float = 0.0
    spike_decoder: str = "default"


@dataclass(frozen=True)
class TaskConfig:
    """Properties that differ among the three public MuJoCo tasks."""

    name: str
    rollout_module: str
    standard_t: int
    adaptive_t: int
    adaptive_k: float
    adaptive_threshold: float

    @property
    def variants(self) -> tuple[VariantConfig, ...]:
        return (
            VariantConfig(name="qcfs_ann", qcfs_t=0),
            VariantConfig(name=f"qcfs_snn_t{self.standard_t}", qcfs_t=self.standard_t),
            VariantConfig(
                name=f"input_adaptive_t{self.adaptive_t}",
                qcfs_t=self.adaptive_t,
                no_reset=True,
                lif_controller="rule",
                leakage_k=self.adaptive_k,
                leakage_threshold=self.adaptive_threshold,
                spike_decoder="decay",
            ),
        )


TASK_CONFIGS = {
    "ring": TaskConfig(
        name="ring",
        rollout_module="RolloutSarnnMujocoUR5eRing.py",
        standard_t=8,
        adaptive_t=1,
        adaptive_k=32.0,
        adaptive_threshold=0.03,
    ),
    "particle": TaskConfig(
        name="particle",
        rollout_module="RolloutSarnnMujocoUR5eParticle.py",
        standard_t=4,
        adaptive_t=2,
        adaptive_k=32.0,
        adaptive_threshold=0.03,
    ),
    "cable": TaskConfig(
        name="cable",
        rollout_module="RolloutSarnnMujocoUR5eCable.py",
        standard_t=4,
        adaptive_t=4,
        adaptive_k=128.0,
        adaptive_threshold=0.0,
    ),
}


def validate_world_index(world_idx: int) -> int:
    """Return a valid world index, rejecting Python's negative indexing."""

    if isinstance(world_idx, bool) or not isinstance(world_idx, int):
        raise ValueError(f"world_idx must be an integer in [0, 20], got {world_idx!r}")
    if world_idx not in WORLD_INDICES:
        raise ValueError(f"world_idx must be in [0, 20], got {world_idx}")
    return world_idx
