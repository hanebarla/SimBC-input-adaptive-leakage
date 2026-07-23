import pytest

from robo_manip_baselines.sarnn.lib.experiment_config import (
    DEFAULT_TASK,
    TASK_CONFIGS,
    WORLD_INDICES,
    validate_world_index,
)


def test_default_task_is_ring():
    assert DEFAULT_TASK == "ring"
    assert next(iter(TASK_CONFIGS)) == DEFAULT_TASK


@pytest.mark.parametrize(
    ("task", "standard_t", "adaptive_t", "k", "threshold"),
    [
        ("ring", 8, 1, 32.0, 0.03),
        ("particle", 4, 2, 32.0, 0.03),
        ("cable", 4, 4, 128.0, 0.0),
    ],
)
def test_public_variant_defaults(task, standard_t, adaptive_t, k, threshold):
    variants = TASK_CONFIGS[task].variants
    assert [variant.qcfs_t for variant in variants] == [0, standard_t, adaptive_t]
    assert variants[2].leakage_k == k
    assert variants[2].leakage_threshold == threshold
    assert variants[2].no_reset
    assert variants[2].spike_decoder == "decay"


def test_world_indices_are_exactly_zero_through_twenty():
    assert WORLD_INDICES == tuple(range(21))
    assert validate_world_index(0) == 0
    assert validate_world_index(20) == 20
    for invalid in (-1, 21, True, 1.5):
        with pytest.raises(ValueError):
            validate_world_index(invalid)
