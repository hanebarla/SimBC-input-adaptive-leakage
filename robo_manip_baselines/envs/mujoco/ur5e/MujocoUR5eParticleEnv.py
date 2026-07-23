"""MuJoCo UR5e particle task and its public evaluation metric."""

import re
from os import path

import mujoco
import numpy as np

from robo_manip_baselines.envs.mujoco.task_metrics import resolve_world_index

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eParticleEnv(MujocoUR5eEnvBase):
    def __init__(self, **kwargs):
        super().__init__(
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_particle.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.25 * np.pi,
                    np.pi / 2,
                    np.pi,
                    0.0,
                ]
            ),
            **kwargs,
        )
        self.original_source_pos = self.model.body("source_case").pos.copy()
        self.original_particle_pos = self.model.body("particle").pos.copy()
        self.pos_offsets = np.column_stack(
            (np.zeros(21), np.arange(0.0, 0.201, 0.01), np.zeros(21))
        )
        self.particle_geom_ids = []
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and re.fullmatch(r"B\d+_\d+_\d+", name):
                self.particle_geom_ids.append(geom_id)

    @property
    def num_worlds(self):
        return len(self.pos_offsets)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        world_idx = resolve_world_index(world_idx, cumulative_idx, self.num_worlds)
        offset = self.pos_offsets[world_idx]
        self.model.body("source_case").pos = self.original_source_pos + offset
        self.model.body("particle").pos = self.original_particle_pos + offset
        return world_idx

    def _is_in_goal(self, position):
        positive_x = self.data.geom("goal_case_px").xpos[0]
        negative_x = self.data.geom("goal_case_nx").xpos[0]
        positive_y = self.data.geom("goal_case_py").xpos[1]
        negative_y = self.data.geom("goal_case_ny").xpos[1]
        return bool(
            negative_x < position[0] < positive_x
            and negative_y < position[1] < positive_y
        )

    def get_task_metrics(self):
        """Return the number of particle geoms whose centers are in the goal."""

        count = sum(
            self._is_in_goal(self.data.geom_xpos[geom_id])
            for geom_id in self.particle_geom_ids
        )
        return {"particle_count": int(count)}
