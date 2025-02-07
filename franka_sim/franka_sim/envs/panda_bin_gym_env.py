

from pathlib import Path
from typing import Literal
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
import numpy as np

from franka_sim.mujoco_gym_env import GymRenderingSpec

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_bin.xml"
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.4, 0.15]])

class PandaBinGymEnv(PandaPickCubeGymEnv):
    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.05, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        xml_path: Path = _XML_PATH,
        sampling_bounds: np.ndarray = _SAMPLING_BOUNDS,
    ):  
        assert reward_type == "sparse"
        super().__init__(
            action_scale=action_scale,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            reward_type=reward_type,
            xml_path=xml_path,
            sampling_bounds=sampling_bounds,
        )
    
    def check_if_in_bin(self):
        block_pos = self._data.sensor("block_pos").data
        bin_pos = self._data.sensor("bin_pos").data
        target_error = bin_pos - block_pos
        abs_error = np.abs(target_error)
        bounds = np.array([0.264, 0.21, 0.05])
        # ! BIN rotated so change bounds
        x_bound = bounds[0]
        y_bound = bounds[1]
        bounds[0] = y_bound
        bounds[1] = x_bound
        in_bin = np.all(abs_error < bounds)
        return in_bin
    
    def _compute_reward(self) -> float:
        return float(self.check_if_in_bin())

    def _is_success(self) -> bool:
        return self.check_if_in_bin()
