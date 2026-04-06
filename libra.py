import socket
import time
import board
import busio
import math

from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, BNO_REPORT_LINEAR_ACCELERATION
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ---------- NETWORK SETTINGS ----------
# SERVER_IP = input("Enter Scorpio IP: ")   # Replace with Pi 5 IP
# PORT = 5000

# ---------- CONNECT TO PI 5 ----------
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect((SERVER_IP, PORT))
# print("Connected to Pi 5")

# ---------- I2C SETUP ----------
i2c = busio.I2C(board.SCL, board.SDA)

# ---------- IMU SETUP ----------
bno = BNO08X_I2C(i2c)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
print("IMU initialized")

# ---------- FLEX SENSOR SETUP ----------
ads = ADS.ADS1115(i2c)
flex_channel = AnalogIn(ads, 0)
print("ADC initialized")

# ------ Sensor Processing Variables ------
# State
px = py = pz = 0.0
vx = vy = vz = 0.0

global last_time
last_time = time.time()

# Tunable parameters
ACCEL_THRESHOLD = 0.08     # noise cutoff
STATIONARY_THRESHOLD = 0.05
DAMPING = 0.9             # velocity decay (0.9–0.99)
ZUPT_COUNT_REQUIRED = 5   # frames of stillness

stationary_counter = 0

# Quaternion rotate vector into world frame
def rotate_vector(qx, qy, qz, qw, x, y, z):
  # Quaternion rotation: v' = q * v * q_conj
  # Optimized form
  ix =  qw * x + qy * z - qz * y
  iy =  qw * y + qz * x - qx * z
  iz =  qw * z + qx * y - qy * x
  iw = -qx * x - qy * y - qz * z

  rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
  ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
  rz = iz * qw + iw * -qz + ix * -qy - iy * -qx

  return rx, ry, rz

def compute_position():
  now = time.time()
  dt = now - last_time
  last_time = now

  quat = bno.quaternion
  accel = bno.linear_acceleration

  if quat is None or accel is None:
    return px, py, pz

  qx, qy, qz, qw = quat
  ax, ay, az = accel

  # Rotate accel into world frame
  ax, ay, az = rotate_vector(qx, qy, qz, qw, ax, ay, az)

  # Noise filter
  if abs(ax) < ACCEL_THRESHOLD:
    ax = 0
  if abs(ay) < ACCEL_THRESHOLD:
    ay = 0
  if abs(az) < ACCEL_THRESHOLD:
    az = 0

  # Detect stationary
  if abs(ax) < STATIONARY_THRESHOLD and \
    abs(ay) < STATIONARY_THRESHOLD and \
    abs(az) < STATIONARY_THRESHOLD:
    stationary_counter += 1
  else:
    stationary_counter = 0

  # ZUPT (Zero Velocity Update)
  if stationary_counter > ZUPT_COUNT_REQUIRED:
    vx = vy = vz = 0.0
  else:
    vx += ax * dt
    vy += ay * dt
    vz += az * dt

  # Velocity damping (prevents runaway drift)
  vx *= DAMPING
  vy *= DAMPING
  vz *= DAMPING

  # Position
  px += vx * dt
  py += vy * dt
  pz += vz * dt

  return px, py, pz

def compute_rotation():
  quat_i, quat_j, quat_k, quat_real = bno.quaternion

  ysqr = quat_j * quat_j

  t0 = 2.0 * (quat_real * quat_i + quat_j * quat_k)
  t1 = 1.0 - 2.0 * (quat_i * quat_i + ysqr)
  roll = math.degrees(math.atan2(t0, t1))

  t2 = 2.0 * (quat_real * quat_j - quat_k * quat_i)
  t2 = max(min(t2, 1.0), -1.0)
  pitch = math.degrees(math.asin(t2))

  t3 = 2.0 * (quat_real * quat_k + quat_i * quat_j)
  t4 = 1.0 - 2.0 * (ysqr + quat_k * quat_k)
  yaw = math.degrees(math.atan2(t3, t4))

  return pitch, yaw, roll


while True:
  # ----- READ IMU -----
  x, y, z = compute_position()
  p, y, r = compute_rotation()

  # ----- READ FLEX SENSOR -----
  flex_value = flex_channel.value

  # ----- CREATE MESSAGE -----
  message = f"{x:.2f},{y:.2f},{z:.2f},{p:.2f},{y:.2f},{r:.2f},{flex_value}\n"

  # ----- SEND DATA -----
  # client.send(message.encode())

  print("Sent:", message)

  # time.sleep(0.1)