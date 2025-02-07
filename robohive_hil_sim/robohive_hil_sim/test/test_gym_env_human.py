import time

import mujoco
import mujoco.viewer
import numpy as np
from robohive_hil_sim.robohive_wrapper import RobohiveWrapper

from robohive_hil_sim.viewer_utils import OpenCVViewer

env_name = 'FrankaPickPlaceOctoSingleFixedColorRobotiqRL-v0'
image_obs = True
env = RobohiveWrapper(env_name, image_obs=image_obs)
action_spec = env.action_space

def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


key_reset = False
key_esc = False
KEY_ESC = 27
KEY_SPACE = 32

def key_callback(keycode):
    if keycode == KEY_SPACE:
        global key_reset
        key_reset = True
    elif keycode == KEY_ESC:
        global key_esc
        key_esc = True


env.reset()

# Create the dual viewer
dual_viewer = OpenCVViewer(env.env.unwrapped)

with dual_viewer as viewer:
    start = time.time()
    while viewer.is_running():
        if key_reset:
            env.reset()
            key_reset = False
        elif key_esc:
            break
        else:
            step_start = time.time()
            env.step(sample())
            viewer.sync()
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

env.close()