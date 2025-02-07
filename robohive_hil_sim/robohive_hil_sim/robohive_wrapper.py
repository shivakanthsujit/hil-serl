from typing import Literal
import mujoco
import numpy as np
import robohive_multi
import gym as oldgym
import gymnasium as gym
from gymnasium import spaces
from robohive_multi.envs.single_arms.pick_place_octo_robotiq import PickPlaceV0
from robohive_multi.utils.data_utils import visualkey_to_info
from robohive_multi.utils.diffkn_controller import compute_control
from robohive.utils.quat_math import euler2quat, mat2quat, mat2euler

GRIPPER_BOUNDS = np.array([
                            [-0.50, 0.25, 0.50], 
                            [ 0.50, 0.75, 1.40]
                            ])

class RobohiveWrapper():
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
            self, 
            env_name,
            action_scale: np.ndarray = np.asarray([0.1, 1]),
            render_mode: Literal["rgb_array", "human"] = "rgb_array",
            image_obs: bool = False,
            reward_type: str = "sparse",
            seed=0):
        
        assert reward_type == "sparse"
        rotation_rep = "euler"
        self.env = oldgym.make(env_name)
        self.env.seed(seed)
        self.env = EEPoseControlWrapper(self.env, rotation_rep=rotation_rep, site_name="end_effector")
        self.env = DeltaEEPoseControlWrapper(self.env, action_scale=action_scale, site_name="end_effector", rotation_rep=rotation_rep)
        self.control_dt = self.env.unwrapped.dt
        
        self._action_scale = action_scale
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.env.dt)),
        }

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float32
                        ),
                        "tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(6,), dtype=np.float32
                        ),
                        "gripper_pose": spaces.Box(
                            -1, 1, shape=(1,), dtype=np.float32
                        ),
                    }
                )
            }
        )

        self.rgb_keys = [k for k in self.env.visual_keys if "rgb" in k]
        self.key_infos = {k: visualkey_to_info(k) for k in self.rgb_keys}
        self.image_key_map = {k: self.key_infos[k].camera for k in self.rgb_keys}
        self.env_image_keys = list(self.image_key_map.values())

        if self.image_obs:
            image_space_dict = {
                v.camera:  spaces.Box(
                                low=0,
                                high=255,
                                shape=(v.height, v.width, 3),
                                dtype=np.uint8,
                            )
                for k, v in self.key_infos.items()
            }
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {
                            "tcp_pose": spaces.Box(
                                -np.inf, np.inf, shape=(7,), dtype=np.float32
                            ),
                            "tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
                            ),
                            "gripper_pose": spaces.Box(
                                -1, 1, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": spaces.Dict(image_space_dict),
                }
            )

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.site_name = "end_effector"
        self.model = self.base_env.sim.model.ptr
        self.data = self.base_env.sim.data.ptr
        self.site_id = self.model.site(self.site_name).id
        self.gripper_ctrl_id = 7

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, **kwargs):
        _ = self.env.reset()
        obs = self._compute_observation()
        return obs, {}
    
    def step(self, action):
        _, reward, done, info = self.env.step(action)
        
        obs = self._compute_observation()
        success = self.base_env.check_block_in_bin("red")
        out_of_bounds = self.base_env.check_block_outofbounds("red")
        truncated = info.get("TimeLimit.truncated", False)
        done = success or out_of_bounds
        reward = int(success)
        info = {}
        info["succeed"] = success
        return obs, reward, done, truncated, info
    
    def get_visuals(self):
        return self.env.unwrapped.get_visuals()
    
    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self.data.site(self.site_id).xpos.copy()
        tcp_mat = self.data.site(self.site_id).xmat.copy()
        tcp_mat = np.array(tcp_mat).reshape((3, 3))
        tcp_quat = mat2quat(tcp_mat)
        obs["state"]["tcp_pose"] = np.concatenate([tcp_pos, tcp_quat]).astype(np.float32)

        # Jacobian.
        njoints = 7
        jac = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)
        jac = jac[:, :njoints]
        dq = self.data.qvel[:njoints]
        dx = jac @ dq
        obs["state"]["tcp_vel"] = dx.astype(np.float32)

        gripper_pose = np.array(self.data.qpos[self.gripper_ctrl_id])
        obs["state"]["gripper_pose"] = gripper_pose[None]

        if self.image_obs:
            image_dict = self.get_visuals()
            obs["images"] = {self.image_key_map[k]: image_dict[k] for k in self.rgb_keys}
        else:
            block_pos = self.base_env.get_box_pos()
            assert len(block_pos) == 1
            obs["state"]["block_pos"] = block_pos[0]
        return obs

    def close(self):
        self.env.close()
    
    @property
    def base_env(self) -> PickPlaceV0:
        return self.env.unwrapped
    
class EEPoseControlWrapper(gym.ActionWrapper):
    def __init__(self, env, rotation_rep="quat", site_name="end_effector"):
        super().__init__(env)
        self.rotation_rep = rotation_rep
        assert self.rotation_rep in ["quat", "euler"]
        self.gripper_config = 'robotiq_gripper' if 'robotiq' in env.robot.robot_config else 'franka_hand'
        act_dims = 4 # TCP + gripper
        if rotation_rep == "quat":
            act_dims += 4
        elif rotation_rep == "euler":
            act_dims += 3
        
        self.action_space = spaces.Box(
            low=np.ones(act_dims) * -1.0,
            high=np.ones(act_dims) * 1.0,
            dtype=np.float32,
        )
        self.act_dims = act_dims

        self.site_name = site_name
        self.model = self.env.sim.model.ptr
        self.data = self.env.sim.data.ptr
        self.site_id = self.model.site(site_name).id
        self.bounds = np.array([
                            [-0.5,  0.25,  0.5], 
                            [0.5, 0.7, 1.4]
                            ])

    def step(self, action):
        assert len(action) == self.act_dims
        # assert self.action_space.contains(action), f"{action} not in {self.action_space}"
        tcp = action[:3]

        # clip the tcp position to the limits
        tcp = np.clip(tcp, *self.bounds)

        rotation = action[3:-1]
        gripper_width = action[-1]
        rotation = np.array(rotation)
        if self.rotation_rep == "quat":
            quat = rotation.copy()
            assert quat.shape == (4,)
        elif self.rotation_rep == "euler":
            assert rotation.shape == (3,)
            quat = euler2quat(rotation)
            
        joint_pos = compute_control(self.env.sim.model.ptr, self.env.sim.data.ptr, tcp, quat, self.site_name)
                        
        if 'robotiq' in self.gripper_config.lower():
            act = np.concatenate([joint_pos, [gripper_width]])
        else:
            act = np.concatenate([joint_pos, [gripper_width, gripper_width]])

        obs, reward, done, info = self.env.step(act)
        return obs, reward, done, info

class DeltaEEPoseControlWrapper(gym.ActionWrapper):
    def __init__(self, env, action_scale=np.asarray([0.05, 1]), site_name="end_effector", rotation_rep="quat"):
        super().__init__(env)
        self.action_scale = action_scale
        self.last_action = None
        self.rotation_rep = rotation_rep
        assert self.rotation_rep in ["quat", "euler"]
        
        self.site_name = site_name
        self.model = self.env.sim.model.ptr
        self.data = self.env.sim.data.ptr
        self.site_id = self.model.site(site_name).id

        self.gripper_ctrl_id = 7 # Robotiq gripper
        self.bounds = np.array([
                            [-0.5,  0.25,  0.5], 
                            [0.5, 0.7, 1.4]
                            ])

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.start_tcp = self.data.site(self.site_id).xpos.copy()
        start_xmat = self.data.site(self.site_id).xmat.copy()
        self.start_xmat = np.array(start_xmat).reshape((3, 3))
        self.last_tcp = self.start_tcp.copy()
        self.last_gripper = self.data.qpos[self.gripper_ctrl_id]
        return obs
    
    def action(self, action):
        # assert self.action_space.contains(action), f"{action} not in {self.action_space}"
        delta_action = action.copy()
        delta_tcp = action[:3]
        delta_rotation = action[3:-1]
        delta_gripper_width = action[-1]
        
        current_tcp = self.last_tcp.copy()
        dtcp = delta_tcp * self.action_scale[0]
        new_tcp = current_tcp + dtcp
        new_tcp = np.clip(new_tcp, *self.bounds)
        self.last_tcp = new_tcp.copy()

        # target_xmat = self.data.site(self.site_id).xmat.copy()
        target_xmat = self.start_xmat.copy()
        target_xmat = np.array(target_xmat).reshape((3, 3))
        if self.rotation_rep == "quat":
            new_rotation = mat2quat(target_xmat)
        elif self.rotation_rep == "euler":
            new_rotation = mat2euler(target_xmat)

        current_gripper_width = self.last_gripper.copy()
        dgripper_width = delta_gripper_width * self.action_scale[1]
        new_gripper_width = current_gripper_width + dgripper_width
        new_gripper_width = np.clip(new_gripper_width, 0.0, 1.0)
        self.last_gripper = new_gripper_width.copy()
        new_gripper_width = new_gripper_width[None]
        action = np.concatenate([new_tcp, new_rotation, new_gripper_width])
        return action
        
def get_shapes(obj):
    if isinstance(obj, dict):
        return {k: get_shapes(v) for k, v in obj.items()}
    else:
        return np.array(obj).shape