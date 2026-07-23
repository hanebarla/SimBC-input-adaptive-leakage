"""Shared validation and geometry helpers for public task metrics."""

from __future__ import annotations

import math

import numpy as np


NUM_EXPERIMENT_WORLDS = 21


def resolve_world_index(world_idx, cumulative_idx, num_worlds=NUM_EXPERIMENT_WORLDS):
    """Resolve explicit or cumulative indices and reject accidental wrapping."""

    if world_idx is None:
        if isinstance(cumulative_idx, bool) or not isinstance(cumulative_idx, int):
            raise ValueError(
                "cumulative_idx must be an integer when world_idx is omitted"
            )
        return cumulative_idx % num_worlds
    if isinstance(world_idx, bool) or not isinstance(world_idx, int):
        raise ValueError(f"world_idx must be an integer in [0, {num_worlds - 1}]")
    if not 0 <= world_idx < num_worlds:
        raise ValueError(f"world_idx must be in [0, {num_worlds - 1}], got {world_idx}")
    return world_idx


def point_in_polygon(point, polygon):
    """Return whether a 2-D point lies inside an ordered polygon."""

    x, y = point
    vertices = np.asarray(polygon, dtype=float)
    inside = False
    previous = len(vertices) - 1
    for current in range(len(vertices)):
        x_i, y_i = vertices[current]
        x_j, y_j = vertices[previous]
        crosses_y = (y_i > y) != (y_j > y)
        if crosses_y:
            crossing_x = (x_j - x_i) * (y - y_i) / (y_j - y_i) + x_i
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def segments_intersect(first_start, first_end, second_start, second_end):
    """Return whether two finite 2-D line segments have an intersection."""

    x1, y1 = np.asarray(first_start)[:2]
    x2, y2 = np.asarray(first_end)[:2]
    x3, y3 = np.asarray(second_start)[:2]
    x4, y4 = np.asarray(second_end)[:2]
    a1, b1, c1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
    a2, b2, c2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-12:
        return False
    x = (c1 * b2 - c2 * b1) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    def between(value, endpoint_a, endpoint_b):
        tolerance = 1e-9
        return (
            min(endpoint_a, endpoint_b) - tolerance
            <= value
            <= max(endpoint_a, endpoint_b) + tolerance
        )

    return all(
        (
            between(x, x1, x2),
            between(y, y1, y2),
            between(x, x3, x4),
            between(y, y3, y4),
        )
    )


def point_in_angle_sector(point, center, angle_min, angle_max):
    """Return whether a 2-D point angle is within an inclusive degree sector."""

    delta = np.asarray(point)[:2] - np.asarray(center)[:2]
    if np.allclose(delta, 0.0):
        return False
    angle = math.degrees(math.atan2(delta[1], delta[0])) % 360.0
    return angle_min <= angle <= angle_max
