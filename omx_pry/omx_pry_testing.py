"""
omx_pry_testing.py
------------------
MuJoCo safety test for the motion-mirroring project.

Mirrors scorpio.py's receiver exactly — same HOST, PORT, and message
format ("x,y,z,pitch,yaw,roll,flex") — then pipes the live data through
the IK solver and runs the arm in MuJoCo simulation with safety checks.

Run on your PC (needs a display for the viewer):
    python omx_pry_testing.py

Libra (libra-c.py) sends:
    "x,y,z,pitch,yaw,roll,flex\n"
    x/y/z in metres, pitch/yaw/roll in degrees, flex 0.0-1.0
"""

import socket
import threading
import math
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────

# Adjust this path to wherever you cloned robotis_mujoco_menagerie
XML_PATH = "../robotis_mujoco_menagerie/robotis_open_manipulator_x/scene.xml"

# Must match scorpio.py exactly
HOST = "0.0.0.0"
PORT = 5000

# Scale glove position (metres) down to arm workspace (metres)
# Glove double-integration drifts a lot — keep this small
POSITION_SCALE = 0.3

# Workspace clamp — keep arm in a safe reachable box (metres)
WS_MIN = np.array([0.05, -0.20, 0.05])
WS_MAX = np.array([0.30,  0.20, 0.35])

# IK settings (tuned for 4-DOF arm)
IK_MAX_STEPS  = 150
IK_TOL        = 1e-4
IK_STEP_SIZE  = 0.4
IK_DAMPING    = 5e-4
IK_ROT_WEIGHT = 0.4   # less than 1.0 — arm can't hit all 6 DOF

# Max joint movement per IK cycle (radians) — prevents snap from bad IMU readings
MAX_JOINT_DELTA = 0.05   # ~2.9 degrees per step

# IK runs at this rate to match Libra's 10 Hz send rate
IK_HZ = 10.0


# ─────────────────────────────────────────────────────
#  SHARED STATE  (network thread writes, sim thread reads)
# ─────────────────────────────────────────────────────

state = {
    "x":         0.0,
    "y":         0.0,
    "z":         0.0,
    "pitch":     0.0,   # degrees
    "yaw":       0.0,   # degrees
    "roll":      0.0,   # degrees
    "flex":      0.0,   # 0.0 open .. 1.0 closed
    "active":    False, # False = frozen (no connection yet)
}
state_lock = threading.Lock()


# ─────────────────────────────────────────────────────
#  NETWORK THREAD  — mirrors scorpio.py receiver logic
# ─────────────────────────────────────────────────────

def network_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    print(f"[NET] Waiting for Libra on port {PORT} ...")

    while True:
        try:
            conn, addr = server.accept()
            print(f"[NET] Libra connected: {addr}")

            buffer = b""
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break

                buffer += chunk

                # Process all complete newline-terminated messages in buffer
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    message = line.decode().strip()
                    if not message:
                        continue

                    # Same split as scorpio.py
                    try:
                        x, y, z, pitch, yaw, roll, flex = message.split(",")
                        with state_lock:
                            state["x"]      = float(x)
                            state["y"]      = float(y)
                            state["z"]      = float(z)
                            state["pitch"]  = float(pitch)
                            state["yaw"]    = float(yaw)
                            state["roll"]   = float(roll)
                            state["flex"]   = float(flex)
                            state["active"] = True
                    except ValueError:
                        print(f"[NET] Bad message: {message!r}")

        except Exception as e:
            print(f"[NET] Error: {e}")
        finally:
            with state_lock:
                state["active"] = False
            print("[NET] Libra disconnected — arm frozen. Waiting for reconnect...")


# ─────────────────────────────────────────────────────
#  IK SOLVER
# ─────────────────────────────────────────────────────

def rpy_to_quat_mujoco(roll_r, pitch_r, yaw_r):
    """XYZ extrinsic RPY radians -> MuJoCo quaternion [w, x, y, z]."""
    x, y, z, w = R.from_euler('xyz', [roll_r, pitch_r, yaw_r]).as_quat()
    return np.array([w, x, y, z])


def quat_error_vec(tq, cq):
    """Orientation error axis-angle vector. Both quats [w,x,y,z]."""
    cq = cq.copy(); cq[1:] *= -1
    tw,tx,ty,tz = tq;  cw,cx,cy,cz = cq
    e = np.array([
        tw*cw - tx*cx - ty*cy - tz*cz,
        tw*cx + tx*cw + ty*cz - tz*cy,
        tw*cy - tx*cz + ty*cw + tz*cx,
        tw*cz + tx*cy - ty*cx + tz*cw,
    ])
    if e[0] < 0: e = -e
    return 2.0 * e[1:]


def solve_ik(model, data, target_pos, target_rpy_rad):
    """Damped least-squares Jacobian IK on body 'link5'. Returns qpos."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    tq      = rpy_to_quat_mujoco(*target_rpy_rad)
    nv      = model.nv
    jacp    = np.zeros((3, nv))
    jacr    = np.zeros((3, nv))

    for _ in range(IK_MAX_STEPS):
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.body("link5").xpos
        rot_err = quat_error_vec(tq, data.body("link5").xquat) * IK_ROT_WEIGHT
        err6    = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(err6) < IK_TOL:
            break
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J   = np.vstack([jacp, jacr])
        dq  = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), err6)
        data.qpos[:nv] += IK_STEP_SIZE * dq
        for j in range(model.njnt):
            a = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]
            if lo < hi:
                data.qpos[a] = np.clip(data.qpos[a], lo, hi)

    return data.qpos.copy()


# ─────────────────────────────────────────────────────
#  SAFETY CHECKS
# ─────────────────────────────────────────────────────

def joints_within_limits(model, qpos):
    for j in range(model.njnt):
        a = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        if lo < hi and not (lo - 1e-5 <= qpos[a] <= hi + 1e-5):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"[SAFETY] Limit violation: {name} = "
                  f"{math.degrees(qpos[a]):.1f}deg  "
                  f"(range [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}])")
            return False
    return True


def clamp_velocity(prev, nxt, model):
    """Limit each joint to MAX_JOINT_DELTA radians per cycle."""
    out = nxt.copy()
    for j in range(4):
        a     = model.jnt_qposadr[j]
        delta = out[a] - prev[a]
        if abs(delta) > MAX_JOINT_DELTA:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"[SAFETY] Velocity clamped: {name} "
                  f"{math.degrees(delta):.1f}deg -> "
                  f"+/-{math.degrees(MAX_JOINT_DELTA):.1f}deg")
            out[a] = prev[a] + np.sign(delta) * MAX_JOINT_DELTA
    return out


def no_nan(data):
    return not (np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)))


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────

def main():
    print(f"[SIM] Loading: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print("[SIM] Joint limits:")
    for j in range(4):
        name   = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[j]
        print(f"      {name}: [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg")
    print()

    threading.Thread(target=network_thread, daemon=True).start()

    prev_qpos    = data.qpos.copy()
    last_ik_time = 0.0
    ik_interval  = 1.0 / IK_HZ

    print("[SIM] Viewer opening — arm frozen until Libra connects.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            now = time.time()

            with state_lock:
                active    = state["active"]
                pitch_deg = state["pitch"]
                yaw_deg   = state["yaw"]
                roll_deg  = state["roll"]
                flex      = state["flex"]
                gx        = state["x"]
                gy        = state["y"]
                gz        = state["z"]

            # ── Frozen ─────────────────────────────────────────────
            if not active:
                mujoco.mj_step(model, data)
                if not no_nan(data):
                    print("[SAFETY] NaN — resetting to home")
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                    prev_qpos = data.qpos.copy()
                viewer.sync()
                time.sleep(0.01)
                continue

            # ── IK at 10 Hz ────────────────────────────────────────
            if (now - last_ik_time) >= ik_interval:
                last_ik_time = now

                # Convert degrees to radians (Libra sends degrees)
                rpy_rad = np.array([
                    math.radians(roll_deg),
                    math.radians(pitch_deg),
                    math.radians(yaw_deg),
                ])

                # Scale and clamp position to safe workspace
                target_pos = np.clip(
                    np.array([gx, gy, gz]) * POSITION_SCALE,
                    WS_MIN, WS_MAX
                )

                qpos_before   = data.qpos.copy()
                qpos_solution = solve_ik(model, data, target_pos, rpy_rad)

                # Safety gate 1 — joint limits
                if not joints_within_limits(model, qpos_solution):
                    print("[SAFETY] Pose rejected — holding position")
                    data.qpos[:] = qpos_before
                    qpos_solution = qpos_before

                # Safety gate 2 — velocity clamp
                qpos_solution = clamp_velocity(prev_qpos, qpos_solution, model)

                # Apply to sim
                data.qpos[:]  = qpos_solution
                data.ctrl[:4] = qpos_solution[:4]

                # Gripper: flex 0.0=open, 1.0=closed
                # Gripper slide range is [-0.01, 0.019] m
                data.ctrl[4] = np.interp(flex, [0.0, 1.0], [0.019, -0.01])

                prev_qpos = qpos_solution.copy()

                # Status
                ee  = data.body("link5").xpos
                jd  = [math.degrees(data.qpos[model.jnt_qposadr[j]]) for j in range(4)]
                print(f"[SIM] rpy=({roll_deg:.1f},{pitch_deg:.1f},{yaw_deg:.1f})deg  "
                      f"flex={flex:.2f}  "
                      f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  "
                      f"J=[{jd[0]:.1f},{jd[1]:.1f},{jd[2]:.1f},{jd[3]:.1f}]deg")

            # ── Physics step + render ──────────────────────────────
            mujoco.mj_step(model, data)
            if not no_nan(data):
                print("[SAFETY] NaN in physics — resetting to home")
                mujoco.mj_resetDataKeyframe(model, data, 0)
                prev_qpos = data.qpos.copy()
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()
