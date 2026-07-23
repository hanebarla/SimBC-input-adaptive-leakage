from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCable


class RolloutSarnnMujocoUR5eCable(RolloutSarnn, RolloutMujocoUR5eCable):
    TASK_NAME = "cable"
    ENV_ID = "robo_manip_baselines/MujocoUR5eCableEnv-v0"


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eCable()
    rollout.run()
