"""MuJoCo UR5e ring task and its public evaluation metric."""

from os import path

import mujoco
import numpy as np

from robo_manip_baselines.envs.mujoco.task_metrics import (
    point_in_polygon,
    resolve_world_index,
)

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eRingEnv(MujocoUR5eEnvBase):
    def __init__(self, **kwargs):
        super().__init__(
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_ring.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.75 * np.pi,
                    -0.5 * np.pi,
                    0.0,
                    0.0,
                ]
            ),
            **kwargs,
        )
        self.original_pole_pos = self.model.body("pole").pos.copy()
        self.pole_pos_offsets = np.column_stack(
            (np.zeros(21), np.arange(0.0, 0.201, 0.01), np.zeros(21))
        )
        self.ring_geom_ids = []
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith("ring_"):
                self.ring_geom_ids.append(geom_id)

    @property
    def num_worlds(self):
        return len(self.pole_pos_offsets)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        world_idx = resolve_world_index(world_idx, cumulative_idx, self.num_worlds)
        self.model.body("pole").pos = (
            self.original_pole_pos + self.pole_pos_offsets[world_idx]
        )
        return world_idx

    def get_task_metrics(self):
        """Return whether the ordered ring polygon encloses the pole center."""

        ring_xy = np.asarray(
            [self.data.geom_xpos[geom_id][:2] for geom_id in self.ring_geom_ids]
        )
        if len(ring_xy) < 3:
            return {"ring_encloses_pole": False}
        pole_xy = self.data.body("pole").xpos[:2]
        lower = ring_xy.min(axis=0)
        upper = ring_xy.max(axis=0)
        in_bounds = bool(np.all(lower <= pole_xy) and np.all(pole_xy <= upper))
        return {
            "ring_encloses_pole": bool(in_bounds and point_in_polygon(pole_xy, ring_xy))
        }
