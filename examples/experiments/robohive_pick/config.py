import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    JoystickIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    ControllerType
)
from robohive_hil_sim.robohive_wrapper import RobohiveWrapper

from experiments.pick_cube_sim.wrapper import GripperPenaltyWrapper

from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig


class TrainConfig(DefaultTrainingConfig):
    controller_type = ControllerType.XBOX
    image_keys = ["franka0_front_cam", "franka0_wrist_cam"]
    classifier_keys = ["franka0_front_cam", "franka0_wrist_cam"]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]
    buffer_period = 2000
    replay_buffer_capacity = 50000
    batch_size = 256
    random_steps = 0
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    fake_env = False
    classifier = False
    pretraining_steps: int = 2000
    training_starts: int = 0

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env_name = 'FrankaPickPlaceOctoSingleFixedColorRobotiqRL-v0'
        image_obs = True
        env = RobohiveWrapper(env_name, image_obs=image_obs)
        if not fake_env:
            env = JoystickIntervention(env=env, controller_type=self.controller_type, robohive_env=True)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs))[0] > 0.95)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env