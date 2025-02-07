import time

import mujoco
import mujoco.viewer
import numpy as np

from robohive_hil_sim.robohive_wrapper import RobohiveWrapper

env_name = 'FrankaPickPlaceOctoSingleFixedColorRobotiqRL-v0'
image_obs = True
env = RobohiveWrapper(env_name, image_obs=image_obs)
action_spec = env.action_space


def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


obs, info = env.reset()
frames = []

for i in range(200):
    a = sample()
    obs, rew, done, truncated, info = env.step(a)
    images = obs["images"]
    img = np.concatenate([images[k] for k in env.env_image_keys], axis=1)
    frames.append(img)

    if done:
        obs, info = env.reset()

import imageio

imageio.mimsave("robohive_render_test.mp4", frames, fps=20)
