import socket
import math
from dynamixel_sdk import PortHandler, PacketHandler
 
# ─────────────────────────────────────────────────────
#  DYNAMIXEL SETTINGS
#  OpenManipulator-X uses XM430-W350 (Protocol 2.0)
# ─────────────────────────────────────────────────────
 
DEVICENAME = "/dev/ttyUSB0"   # change to /dev/ttyACM0 if needed
BAUDRATE   = 1000000
PROTOCOL   = 2.0
 
# Control table addresses (XM430)
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
 
TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0
 
# Dynamixel IDs on the OpenManipulator-X
JOINT_IDS  = [11, 12, 13, 14]
GRIPPER_ID = 15
 
# Dynamixel position: 0-4095 maps to 0-300 degrees
# Centre (home) = 2048 = 150 degrees
DXL_HOME = 2048
 
# Joint home positions (ticks) — safe resting pose
HOME_POSITIONS = {
  11: 2048,   # Joint1 base:     0 deg
  12: 2048,   # Joint2 shoulder: 0 deg
  13: 2048,   # Joint3 elbow:    0 deg
  14: 2048,   # Joint4 wrist:    0 deg
  15: 1900,   # Gripper open
}
 
# Joint limits in degrees relative to home (0 deg)
# Based on OpenManipulator-X mechanical limits
JOINT_LIMITS_DEG = {
  1: (-162.0,  162.0),  # base      (-2.827 to +2.827 rad)
  2: (-102.6,   90.0),  # shoulder  (-1.791 to +1.571 rad)
  3: ( -54.0,   79.2),  # elbow     (-0.942 to +1.382 rad)
  4: (-102.6,  117.0),  # wrist     (-1.791 to +2.042 rad)
}
 
# Gripper tick range
GRIPPER_OPEN   = 3300
GRIPPER_CLOSED = 1700

# Max ticks allowed to change per control cycle (~3.3 deg per step at 10 Hz = 33 deg/s)
MAX_TICK_DELTA = 40
 
 
# ─────────────────────────────────────────────────────
#  DYNAMIXEL HELPERS
# ─────────────────────────────────────────────────────
 
def degrees_to_ticks(degrees):
  """Convert degrees offset from home to Dynamixel ticks. XM430: 360deg = 4096 ticks."""
  return clamp(int(DXL_HOME + (degrees / 360.0) * 4096), 0, 4095)
 
 
def clamp(value, low, high):
  return max(low, min(high, value))
 
 
def set_torque(dxl_id, enable):
  val = TORQUE_ENABLE if enable else TORQUE_DISABLE
  result, _ = packet_handler.write1ByteTxRx(
    port_handler, dxl_id, ADDR_TORQUE_ENABLE, val)
  if result != 0:
    print(f"[WARN] Torque set failed on ID {dxl_id}: "
          f"{packet_handler.getTxRxResult(result)}")
 
 
def set_goal_position(dxl_id, ticks):
  result, _ = packet_handler.write4ByteTxRx(
    port_handler, dxl_id, ADDR_GOAL_POSITION, ticks)
  if result != 0:
    print(f"[WARN] Goal position failed on ID {dxl_id}: "
          f"{packet_handler.getTxRxResult(result)}")
 
 
def move_to_home():
  print("[ARM] Moving to home position...")
  for dxl_id, ticks in HOME_POSITIONS.items():
    set_goal_position(dxl_id, ticks)
  print("[ARM] Home reached.")


def rate_limit(target, last, max_delta=MAX_TICK_DELTA):
  delta = target - last
  if abs(delta) > max_delta:
    return last + int(math.copysign(max_delta, delta))
  return target
 
 
# ─────────────────────────────────────────────────────
#  MAP GLOVE DATA TO JOINT ANGLES
# ─────────────────────────────────────────────────────
 
def glove_to_joints(x, y, z, pitch, yaw, roll):
  """
  Maps BNO085 orientation (degrees) to joint angles (degrees).
 
  Joint1 (base rotation)  <- yaw   of glove
  Joint2 (shoulder pitch) <- roll of glove, scaled down
  Joint3 (elbow)          <- roll of glove, opposing shoulder
  Joint4 (wrist pitch)    <- pitch  of glove
 
  All outputs clamped to each joint's physical limit.
  """
  j1 = clamp(yaw,           *JOINT_LIMITS_DEG[1])
  j2 = clamp(z *  10,   *JOINT_LIMITS_DEG[2])
  j3 = clamp(y *  10,   *JOINT_LIMITS_DEG[3])
  j4 = clamp(-pitch * 0.75, *JOINT_LIMITS_DEG[4])
  return j1, j2, j3, j4
 
 
def flex_to_gripper(flex):
  """flex 0.0 = open, flex 1.0 = closed."""
  flex = clamp(flex, 0.0, 1.0)
  return int(GRIPPER_OPEN - flex * abs(GRIPPER_CLOSED - GRIPPER_OPEN))
 
 
# ─────────────────────────────────────────────────────
#  SETUP DYNAMIXEL
# ─────────────────────────────────────────────────────
 
port_handler   = PortHandler(DEVICENAME)
packet_handler = PacketHandler(PROTOCOL)
 
if not port_handler.openPort():
  raise RuntimeError(f"Failed to open port {DEVICENAME}")
print(f"[ARM] Opened port {DEVICENAME}")
 
if not port_handler.setBaudRate(BAUDRATE):
  raise RuntimeError(f"Failed to set baud rate {BAUDRATE}")
print(f"[ARM] Baud rate set to {BAUDRATE}")
 
# Enable torque on all joints + gripper
for dxl_id in JOINT_IDS + [GRIPPER_ID]:
  set_torque(dxl_id, True)
  print(f"[ARM] Torque enabled: ID {dxl_id}")
 
move_to_home()

last_ticks = dict(HOME_POSITIONS)


# ─────────────────────────────────────────────────────
#  NETWORK — receives from omx_libra.py
# ─────────────────────────────────────────────────────
 
HOST = "0.0.0.0"
PORT = 5000
 
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen()
 
print("Waiting for connection...")
 
conn, addr = server.accept()
print("Connected to:", addr)
 
buffer = b""
 
try:
  while True:
    data = conn.recv(1024)
 
    if not data:
      break
 
    buffer += data
 
    # Process all complete newline-terminated messages
    while b"\n" in buffer:
      line, buffer = buffer.split(b"\n", 1)
      message = line.decode().strip()
      if not message:
        continue
 
      try:
        x, y, z, roll, yaw, pitch, flex = message.split(",")
        x     = float(x)
        y     = float(y)
        z     = float(z)
        pitch = float(pitch)
        yaw   = float(yaw)
        roll  = float(roll)
        flex  = float(flex)
      except ValueError:
        print("Bad message:", message)
        continue

      # Reject NaN — float("nan") parses without error but corrupts joint commands
      if any(math.isnan(v) for v in (pitch, yaw, roll, flex)):
        print("[SAFETY] NaN in message — skipping")
        continue

      # Map glove orientation to joint angles
      j1, j2, j3, j4 = glove_to_joints(x, y, z, pitch, yaw, roll)

      # Convert to Dynamixel ticks
      t1 = degrees_to_ticks(j1)
      t2 = degrees_to_ticks(j2)
      t3 = degrees_to_ticks(j3)
      t4 = degrees_to_ticks(j4)
      tg = flex_to_gripper(flex)

      # Rate limit — prevent the arm snapping from drift spikes or bad IMU readings
      t1 = rate_limit(t1, last_ticks[11])
      t2 = rate_limit(t2, last_ticks[12])
      t3 = rate_limit(t3, last_ticks[13])
      t4 = rate_limit(t4, last_ticks[14])
      tg = rate_limit(tg, last_ticks[15], MAX_TICK_DELTA * 2)

      last_ticks[11] = t1
      last_ticks[12] = t2
      last_ticks[13] = t3
      last_ticks[14] = t4
      last_ticks[15] = tg

      # Send to arm
      set_goal_position(11, t1)
      set_goal_position(12, t2)
      set_goal_position(13, t3)
      set_goal_position(14, t4)
      set_goal_position(15, tg)

      print(f"Pitch:{pitch:.1f} Yaw:{yaw:.1f} Roll:{roll:.1f} Flex:{flex:.2f}")
      print(f"  J=[{j1:.1f},{j2:.1f},{j3:.1f},{j4:.1f}] deg  Gripper:{tg}")
      print("----------------")
 
finally:
  # Safe shutdown
  print("[ARM] Connection lost — returning to home and disabling torque")
  move_to_home()
  for dxl_id in JOINT_IDS + [5]:
    set_torque(dxl_id, False)
  port_handler.closePort()
 
