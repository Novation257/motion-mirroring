# ik_open_manipulator.py
# Adapted from DeepMind dm_control inverse_kinematics.py
# Works with robotis_mujoco_menagerie scene.xml

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------- helpers ----------

def rpy_to_quat(roll, pitch, yaw):
    """Roll-pitch-yaw (XYZ extrinsic) → quaternion [w, x, y, z] (MuJoCo convention)."""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    x, y, z, w = r.as_quat()   # scipy gives [x,y,z,w]
    return np.array([w, x, y, z])

def quat_error(target_quat, current_quat):
    """
    Orientation error as a 3-vector (axis-angle) in world frame.
    Both quats are [w, x, y, z] (MuJoCo convention).
    """
    # Conjugate of current: [w, -x, -y, -z]
    cq = current_quat.copy()
    cq[1:] *= -1
    # Error quaternion = target * conj(current)
    tw, tx, ty, tz = target_quat
    cw, cx, cy, cz = cq
    ew = tw*cw - tx*cx - ty*cy - tz*cz
    ex = tw*cx + tx*cw + ty*cz - tz*cy
    ey = tw*cy - tx*cz + ty*cw + tz*cx
    ez = tw*cz + tx*cy - ty*cx + tz*cw
    # Convert to axis-angle (rotvec)
    err_quat = np.array([ew, ex, ey, ez])
    # mju_quat2Vel equivalent: 2 * ln(q) ≈ 2 * [x,y,z] when w ≈ 1
    if err_quat[0] < 0:
        err_quat = -err_quat  # ensure shortest path
    return 2.0 * err_quat[1:]  # [ex, ey, ez]

# ---------- main IK solver ----------

def ik(model, data, site_name, target_pos, target_rpy,
       max_steps=200, tol=1e-4, step_size=0.4, damping=5e-4,
       pos_weight=1.0, rot_weight=0.5):
    """
    Damped least-squares Jacobian IK for MuJoCo.

    Args:
        model, data : MuJoCo model and data objects
        site_name   : name of the end-effector site in the MJCF
        target_pos  : [x, y, z] target position (meters)
        target_rpy  : [roll, pitch, yaw] target orientation (radians, XYZ extrinsic)
        max_steps   : maximum Jacobian iterations
        tol         : convergence tolerance (norm of 6D error)
        step_size   : how aggressively to move joints per iteration
        damping     : damping factor λ for (JJᵀ + λI)⁻¹
        pos_weight  : weight on position error (reduce if diverging)
        rot_weight  : weight on rotation error (often < 1 for 4-DOF arms)

    Returns:
        np.array of joint angles (qpos), or None if failed
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        # Fall back to body if site not found
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site_name)
        use_body = True
    else:
        use_body = False

    target_quat = rpy_to_quat(*target_rpy)
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for i in range(max_steps):
        mujoco.mj_forward(model, data)

        # --- get current EE state ---
        if use_body:
            curr_pos  = data.body(site_name).xpos.copy()
            curr_quat = data.body(site_name).xquat.copy()  # [w,x,y,z]
            mujoco.mj_jacBody(model, data, jacp, jacr,
                              mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site_name))
        else:
            curr_pos  = data.site(site_name).xpos.copy()
            # site xmat is a 3x3 rotation matrix → convert to quat
            xmat = data.site(site_name).xmat.reshape(3, 3)
            curr_quat = np.zeros(4)
            mujoco.mju_mat2Quat(curr_quat, xmat.flatten())
            mujoco.mj_jacSite(model, data, jacp, jacr,
                              mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))

        # --- compute 6D error ---
        pos_err = (target_pos - curr_pos) * pos_weight
        rot_err = quat_error(target_quat, curr_quat) * rot_weight
        err6    = np.concatenate([pos_err, rot_err])

        if np.linalg.norm(err6) < tol:
            print(f"  IK converged in {i+1} steps")
            return data.qpos.copy()

        # --- damped least squares update ---
        J   = np.vstack([jacp, jacr])                    # 6 × nv
        JJT = J @ J.T                                    # 6 × 6
        dq  = J.T @ np.linalg.solve(JJT + damping * np.eye(6), err6)
        data.qpos[:nv] += step_size * dq

        # clamp to joint limits
        for j in range(model.njnt):
            jnt_type = model.jnt_type[j]
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE or jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                qpos_adr = model.jnt_qposadr[j]
                lo = model.jnt_range[j, 0]
                hi = model.jnt_range[j, 1]
                if lo < hi:  # limits enabled
                    data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr], lo, hi)

    print(f"  IK did not converge (final err={np.linalg.norm(err6):.5f})")
    return data.qpos.copy()  # return best attempt


# ---------- example usage ----------

if __name__ == "__main__":
    import os

    xml_path = "robotis_open_manipulator_x/scene.xml"  # adjust path
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Print all site and body names so you can find the right EE name
    print("Sites:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                     for i in range(model.nsite)])
    print("Bodies:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                      for i in range(model.nbody)])

    # --- solve IK ---
    target_pos = np.array([0.25, 0.0, 0.15])
    target_rpy = np.array([0.0, -0.3, 0.0])   # roll, pitch, yaw from BNO085

    joint_angles = ik(model, data, "end_effector",   # ← replace with actual site/body name
                      target_pos, target_rpy)

    print("Joint angles (rad):", joint_angles[:4])
    print("Joint angles (deg):", np.degrees(joint_angles[:4]))
