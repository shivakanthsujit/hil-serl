from typing import Optional, Tuple, Union
import mujoco
import numpy as np
from dm_robotics.transformations import transformations as tr

def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def serl_control_action(action, model, data, pinch_site_id, base_pos, panda_dof_ids, panda_ctrl_ids, gripper_ctrl_id, action_scale, bounds, n_substeps):
    x, y, z, rx, ry, rz, grasp = action

    # Set gripper grasp.
    g = data.ctrl[gripper_ctrl_id] / 255
    dg = grasp * action_scale[1]
    ng = np.clip(g + dg, 0.0, 1.0)
    data.ctrl[gripper_ctrl_id] = ng * 255

    dpos = np.asarray([x, y, z]) * action_scale[0]

    currpos = data.site_xpos[pinch_site_id]
    # currorient = data.site_xmat[pinch_site_id].reshape((3, 3))
    # current_quat = tr.mat_to_quat(currorient)

    current_quat = data.mocap_quat[0].copy()

    position_d_ = currpos.copy() # Current (interpolated) target for the opspace controller (operating at 500Hz)
    position_d_target_ = np.clip(currpos + dpos, *bounds) # Current target given by the RL policy (operating at 10Hz)
    
    # # Set the mocap orientation.
    # delta_quat = axis_angle_to_quaternion(np.array([rx, ry, rz]) * action_scale[0])
    # new_quat = quaternion_multiply(current_quat, delta_quat)
    new_quat = current_quat.copy()

    orientation_d_ = current_quat.copy()
    orientation_d_target_ = new_quat.copy()

    orientation_d_ = new_quat.copy()

    q_d_nullspace_ = base_pos.copy()

    error_i = np.zeros(6)
    filter_params_ = 0.4

    pos_gains = np.zeros(3)
    ori_gains = np.zeros(3)
    nullspace_stiffness = 0
    Ki_ = np.zeros(3)

    pos_gain_val = 2000
    ori_gain_val = 150
    pos_gains_target = np.array([pos_gain_val, pos_gain_val, pos_gain_val])
    ori_gains_target = np.array([ori_gain_val, ori_gain_val, ori_gain_val])
    nullspace_stiffness_target = 0.5
    # Ki_target_ = np.array([0.1, 0.1, 0.1])
    Ki_target_ = np.zeros(3)

    translational_clip_min = np.array([-0.002, -0.005, -0.005])
    translational_clip_max = np.array([0.002, 0.005, 0.005])
    # translational_clip_min = None
    # translational_clip_max = None

    for _ in range(n_substeps):
        tau_J_d = data.ctrl[panda_ctrl_ids].copy()
        tau, error_i = opspace(
            model=model,
            data=data,
            site_id=pinch_site_id,
            dof_ids=panda_dof_ids,
            position_d_=position_d_,
            orientation_d_=orientation_d_,
            q_d_nullspace_=q_d_nullspace_,
            pos_gains=pos_gains,
            pos_int_gains=Ki_.copy(),
            ori_gains=ori_gains,
            ori_int_gains=Ki_.copy(),
            nullspace_stiffness=nullspace_stiffness,
            gravity_comp=True,
            tau_J_d=tau_J_d,
            translational_clip_min = translational_clip_min,
            translational_clip_max = translational_clip_max,
            error_i=error_i,
            delta_tau_max = 1,
        )
        data.ctrl[panda_ctrl_ids] = tau
        mujoco.mj_step(model, data)

        pos_gains = filter_params_ * pos_gains_target + (1.0 - filter_params_) * pos_gains
        ori_gains = filter_params_ * ori_gains_target + (1.0 - filter_params_) * ori_gains
        nullspace_stiffness = filter_params_ * nullspace_stiffness_target + (1.0 - filter_params_) * nullspace_stiffness
        position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_
        # orientation_d_ = slerp(orientation_d_, orientation_d_target_, filter_params_)
        Ki_ = filter_params_ * Ki_target_ + (1.0 - filter_params_) * Ki_


def opspace(
    model,
    data,
    site_id,
    dof_ids: np.ndarray,
    position_d_: Optional[np.ndarray] = None,
    orientation_d_: Optional[np.ndarray] = None,
    q_d_nullspace_: Optional[np.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    pos_int_gains: Union[Tuple[float, float, float], np.ndarray] = (0, 0, 0),
    ori_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    ori_int_gains: Union[Tuple[float, float, float], np.ndarray] = (0, 0, 0),
    damping_ratio: float = 1.0,
    nullspace_stiffness: float = 0.5,
    gravity_comp: bool = True,
    tau_J_d: np.ndarray = np.zeros(7),
    translational_clip_min = None,
    translational_clip_max = None,
    error_i = None,
    delta_tau_max = None,
) -> np.ndarray:
    
    # if pos is None:
    #     x_des = data.site_xpos[site_id]
    # else:
    #     x_des = np.asarray(pos)
    # if ori is None:
    #     xmat = data.site_xmat[site_id].reshape((3, 3))
    #     quat_des = tr.mat_to_quat(xmat.reshape((3, 3)))
    # else:
    #     ori = np.asarray(ori)
    #     if ori.shape == (3, 3):
    #         quat_des = tr.mat_to_quat(ori)
    #     else:
    #         quat_des = ori
    # if joint is None:
    #     q_des = data.qpos[dof_ids]
    # else:
    #     q_des = np.asarray(joint)

    kp_pos = np.asarray(pos_gains)
    ki_pos = np.asarray(pos_int_gains)
    kd_pos = damping_ratio * 2 * np.sqrt(kp_pos)

    kp_ori = np.asarray(ori_gains)
    ki_ori = np.asarray(ori_int_gains)
    kd_ori = damping_ratio * 2 * np.sqrt(kp_ori)

    # Compute Jacobian of the eef site in world frame.
    J_v = np.zeros((3, model.nv), dtype=np.float64)
    J_w = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        model,
        data,
        J_v,
        J_w,
        site_id,
    )

    J_v = J_v[:, dof_ids]
    J_w = J_w[:, dof_ids]
    J = np.concatenate([J_v, J_w], axis=0)

    # Compute inertia matrix in joint space.
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)
    M = M[dof_ids, :][:, dof_ids]

    # Compute inertia matrix in task space.
    M_inv = np.linalg.inv(M)
    Mx_inv = J @ M_inv @ J.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    # Get current state.
    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]

    error = np.zeros(6)

    position = data.site_xpos[site_id]
    error[:3] = position - position_d_
    if translational_clip_min is not None and translational_clip_max is not None:
        error[:3] = np.clip(error[:3], translational_clip_min, translational_clip_max)

    # Keeping orientation error as zero for now
    orientation = data.site_xmat[site_id].reshape((3, 3))
    orientation = tr.mat_to_quat(orientation)
    if orientation @ orientation_d_ < 0.0:
        orientation *= -1.0
    quat_err = tr.quat_diff_active(source_quat=orientation_d_, target_quat=orientation)
    error[3:] = tr.quat_to_axisangle(quat_err)

    translational_integral_min, translational_integral_max = -0.1, 0.1
    rotational_integral_min, rotational_integral_max = -0.3, 0.3

    error_i[:3] = np.clip(error_i[:3] + error[:3], translational_integral_min, translational_integral_max)
    error_i[3:] = np.clip(error_i[3:] + error[3:], rotational_integral_min, rotational_integral_max)

    # Compute position PD control.
    ddx = -kp_pos * error[:3] - kd_pos * (J_v @ dq) - ki_pos * error_i[:3]

    # Compute orientation PD control.
    dw = -kp_ori * error[3:] - kd_ori * (J_w @ dq) - ki_ori * error_i[3:]

    # Compute generalized forces.
    ddx_dw = np.concatenate([ddx, dw], axis=0)
    tau_task = J.T @ Mx @ ddx_dw

    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = damping_ratio * 2 * np.sqrt(kp_joint)

    qe = q - q_d_nullspace_
    qe[0] *= 100
    dqe = dq.copy()
    dqe[0] *= 2 * np.sqrt(100)

    # Add joint task in nullspace.
    ddq = -kp_joint * qe - kd_joint * dqe
    Jnull = M_inv @ J.T @ Mx
    tau_nullspace = (np.eye(len(q)) - J.T @ Jnull.T) @ ddq

    tau_d = tau_task + tau_nullspace

    if gravity_comp:
        tau_d += data.qfrc_bias[dof_ids]

    tau_d = saturate_torques(tau_d, tau_J_d, delta_tau_max)
    return tau_d, error_i

def saturate_torques(tau, tau_J_d, delta_tau_max_):
    difference  = tau - tau_J_d
    return tau_J_d + np.clip(difference, -delta_tau_max_, delta_tau_max_)