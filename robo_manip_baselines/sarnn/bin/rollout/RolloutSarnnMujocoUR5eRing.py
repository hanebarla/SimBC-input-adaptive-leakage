from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing


class RolloutSarnnMujocoUR5eRing(RolloutSarnn, RolloutMujocoUR5eRing):
    TASK_NAME = "ring"
    ENV_ID = "robo_manip_baselines/MujocoUR5eRingEnv-v0"


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eRing()
    rollout.run()
