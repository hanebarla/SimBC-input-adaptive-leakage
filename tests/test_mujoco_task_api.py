import gymnasium as gym
import pytest

import robo_manip_baselines  # noqa: F401


@pytest.mark.parametrize(
    ("env_id", "metric_name"),
    [
        ("robo_manip_baselines/MujocoUR5eCableEnv-v0", "cable_hooked"),
        ("robo_manip_baselines/MujocoUR5eParticleEnv-v0", "particle_count"),
        ("robo_manip_baselines/MujocoUR5eRingEnv-v0", "ring_encloses_pole"),
    ],
)
def test_world_bounds_and_task_metric_schema(env_id, metric_name):
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        assert env.unwrapped.num_worlds == 21
        assert env.unwrapped.modify_world(world_idx=0) == 0
        assert env.unwrapped.modify_world(world_idx=20) == 20
        for invalid in (-1, 21):
            with pytest.raises(ValueError):
                env.unwrapped.modify_world(world_idx=invalid)
        env.reset(seed=0)
        metrics = env.unwrapped.get_task_metrics()
        assert metric_name in metrics
    finally:
        env.close()
