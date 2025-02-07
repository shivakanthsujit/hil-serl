import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np

# from franka_sim import envs
# import gymnasium as gym

# # import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention
from franka_env.spacemouse.spacemouse_expert import ControllerType

from robohive_hil_sim.robohive_wrapper import RobohiveWrapper

from robohive_hil_sim.viewer_utils import OpenCVViewer
from robohive.utils.quat_math import euler2quat, mat2quat, mat2euler
from termcolor import cprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="xbox", help="Controller type. xbox|ps5")

    args = parser.parse_args()
    controller_type = ControllerType[args.controller.upper()]

env_name = 'FrankaPickPlaceOctoSingleFixedColorRobotiqRL-v0'
image_obs = True
env = RobohiveWrapper(env_name, image_obs=image_obs)
env = JoystickIntervention(env, controller_type=controller_type, robohive_env=True)

env: RobohiveWrapper
env.reset()
done = False

# Create the dual viewer
dual_viewer = OpenCVViewer(env.base_env)

data = env.base_env.sim.data.ptr
model = env.base_env.sim.model.ptr
site_id = model.site("end_effector").id
current_tcp = data.site(site_id).xpos.copy()
current_xmat = data.site(site_id).xmat.copy()
current_xmat = np.array(current_xmat).reshape((3, 3))
euler = mat2euler(current_xmat)
max_val = 0.3
# max_val = 0.2
step = np.linspace(0, max_val, 20)
step = np.concatenate([step, step[::-1]], axis=0)
step = np.concatenate([step, -step], axis=0)
solved_threshold = [0.264, 0.1]
# intervene on position control
with dual_viewer as viewer:
    for i in range(100000):
        if done:
            print("Resetting")
            print(f"{terminated=}, {truncated=}")
            env.reset()
        step_start = time.time()
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        done = terminated or truncated
        in_bin = env.base_env.check_block_in_bin("red")
        if in_bin:
            cprint(f"Reached bin {in_bin}", "green")
        viewer.sync()
        time_until_next_step = env.control_dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
