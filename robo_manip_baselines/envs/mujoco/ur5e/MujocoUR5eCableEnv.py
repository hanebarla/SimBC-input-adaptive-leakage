"""MuJoCo UR5e cable task and its public evaluation metric."""

from os import path

import numpy as np

from robo_manip_baselines.envs.mujoco.task_metrics import (
    point_in_angle_sector,
    resolve_world_index,
    segments_intersect,
)

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eCableEnv(MujocoUR5eEnvBase):
    def __init__(self, **kwargs):
        super().__init__(
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_cable.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.25 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    0.0,
                ]
            ),
            **kwargs,
        )
        self.original_pole_pos = self.model.body("poles").pos.copy()
        self.pole_pos_offsets = np.column_stack(
            (
                np.arange(-0.03, 0.171, 0.01),
                np.zeros(21),
                np.zeros(21),
            )
        )

    @property
    def num_worlds(self):
        return len(self.pole_pos_offsets)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        world_idx = resolve_world_index(world_idx, cumulative_idx, self.num_worlds)
        self.model.body("poles").pos = (
            self.original_pole_pos + self.pole_pos_offsets[world_idx]
        )
        return world_idx

    def get_task_metrics(self):
        """Return the cable hook success predicate used in the experiments."""

        cable_points = np.asarray(
            [self.data.body(f"B{index}").xpos.copy() for index in range(25)]
        )
        pole_origin = self.data.body("poles").xpos.copy()
        red_pole = pole_origin + np.array([0.0, 0.0, 0.05])
        blue_pole = pole_origin + np.array([0.05, 0.0, 0.05])
        crosses_between_poles = any(
            segments_intersect(red_pole, blue_pole, start, end)
            for start, end in zip(cable_points[:-1], cable_points[1:])
        )
        cable_end = self.get_body_pose("cable_end")[:3]
        end_is_behind_red_pole = bool(cable_end[1] <= red_pole[1])
        surrounds_both_poles = any(
            point_in_angle_sector(point, red_pole, 210, 270)
            for point in cable_points[:-1]
        ) and any(
            point_in_angle_sector(point, blue_pole, 0, 60)
            for point in cable_points[:-1]
        )
        return {
            "cable_hooked": bool(
                crosses_between_poles
                and end_is_behind_red_pole
                and surrounds_both_poles
            )
        }
