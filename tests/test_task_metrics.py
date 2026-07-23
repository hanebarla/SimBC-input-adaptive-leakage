import numpy as np
import pytest

from robo_manip_baselines.envs.mujoco.task_metrics import (
    point_in_angle_sector,
    point_in_polygon,
    resolve_world_index,
    segments_intersect,
)


def test_world_resolution_rejects_out_of_range_explicit_indices():
    assert resolve_world_index(0, None) == 0
    assert resolve_world_index(20, None) == 20
    assert resolve_world_index(None, 21) == 0
    for invalid in (-1, 21, True):
        with pytest.raises(ValueError):
            resolve_world_index(invalid, None)


def test_geometry_helpers():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert point_in_polygon((0.5, 0.5), square)
    assert not point_in_polygon((1.5, 0.5), square)
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
    assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
    assert point_in_angle_sector((1, 0), (0, 0), 0, 60)
