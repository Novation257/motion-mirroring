# Program to be ran on the Raspberry Pi Zero

import socket
import time
import board
import busio
import math
import numpy as np

from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, \
                            BNO_REPORT_LINEAR_ACCELERATION, \
                            BNO_REPORT_GEOMAGNETIC_ROTATION_VECTOR
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

class Kalman1D:
  def __init__(self):
    # State: [position, velocity]
    self.x = np.array([[0.0], [0.0]])

    # Covariance
    self.P = np.eye(2) * 1.0

    # Process noise
    self.Q = np.array([[0.05, 0],
                        [0, 0.5]])

    # Measurement noise (velocity correction via ZUPT)
    self.R = np.array([[0.05]])

    # Measurement matrix (we observe velocity during ZUPT)
    self.H = np.array([[0, 1]])

  def predict(self, a, dt):
    # State transition
    F = np.array([[1, dt],
                  [0, 1]])

    # Control input (acceleration)
    B = np.array([[0.5 * dt * dt],
                  [dt]])

    # Predict state
    self.x = F @ self.x + B * a

    # Predict covariance
    self.P = F @ self.P @ F.T + self.Q

  def update_velocity(self, measured_v=0):
    z = np.array([[measured_v]])

    y = z - (self.H @ self.x)
    S = self.H @ self.P @ self.H.T + self.R
    K = self.P @ self.H.T @ np.linalg.inv(S)

    self.x = self.x + K @ y
    self.P = (np.eye(2) - K @ self.H) @ self.P

  def get_state(self):
    return self.x[0, 0], self.x[1, 0]

# MotionTracker class for IMU position and rotation tracking
class MotionTracker():
  def __init__(self, i2c, debug=False):
    self.debug = debug

    # Kalman filters (x, y, z)
    self.kx = Kalman1D()
    self.ky = Kalman1D()
    self.kz = Kalman1D()

    # Filtered acceleration
    self.ax_f = self.ay_f = self.az_f = 0

    # Bias
    self.bias_ax = self.bias_ay = self.bias_az = 0

    # Timing
    self.tick = time.perf_counter()

    # Sensor setup
    self.bno = BNO08X_I2C(i2c_bus=i2c)
    self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)

    # Calibration
    time.sleep(2)
    print("Calibrating...")
    self.calibrate_bias()

  def calibrate_bias(self, samples=200):
    sx = sy = sz = 0
    for _ in range(samples):
      ax, ay, az = self.bno.linear_acceleration
      sx += ax
      sy += ay
      sz += az
      time.sleep(0.005)

    self.bias_ax = sx / samples
    self.bias_ay = sy / samples
    self.bias_az = sz / samples

    print("Bias calibrated:", self.bias_ax, self.bias_ay, self.bias_az)

  # Returns the euler angle representation of a quaternion
  def quaternion_to_euler(self, quaternion):
    x, y, z, w = quaternion

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    p_deg = math.degrees(pitch)
    y_deg = math.degrees(yaw)
    r_deg = math.degrees(roll)

    return p_deg, y_deg, r_deg

  # Time since last delta call
  def delta(self):
    now = time.perf_counter()
    dt = now - self.tick
    self.tick = now
    return dt

  # Quaternion rotate vector into world frame
  def rotate_vector(self, qx, qy, qz, qw, x, y, z):
    ix =  qw * x + qy * z - qz * y
    iy =  qw * y + qz * x - qx * z
    iz =  qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z

    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx

    return rx, ry, rz

  # Iterative calculations to track position and rotation
  def update(self):
    # Get sensor and time data
    dt = self.delta()
    qx, qy, qz, qw = self.bno.quaternion
    ax, ay, az = self.bno.linear_acceleration

    # Remove bias
    ax -= self.bias_ax
    ay -= self.bias_ay
    az -= self.bias_az

    # Rotate accel vector into world frame
    ax, ay, az = self.rotate_vector(qx, qy, qz, qw, ax, ay, az)

    # Get euler angle representation of rotation
    w, r, t = self.quaternion_to_euler((qx, qy, qz, qw))

    # Low-pass filter
    alpha = 0.2
    self.ax_f = alpha * ax + (1 - alpha) * self.ax_f
    self.ay_f = alpha * ay + (1 - alpha) * self.ay_f
    self.az_f = alpha * az + (1 - alpha) * self.az_f

    # Kalman predict
    self.kx.predict(self.ax_f, dt)
    self.ky.predict(self.ay_f, dt)
    self.kz.predict(self.az_f, dt)

    # Zero-velocity detection (ZUPT)
    if abs(self.ax_f) < 0.05 and abs(self.ay_f) < 0.05 and abs(self.az_f) < 0.05:
      self.kx.update_velocity(0)
      self.ky.update_velocity(0)
      self.kz.update_velocity(0)

    # Read out Kalman prediction values
    px, vx = self.kx.get_state()
    py, vy = self.ky.get_state()
    pz, vz = self.kz.get_state()

    # Print vars if in debug mode
    if self.debug:
      print(f"dt: {dt:.4f}")
      print(f"pos: {px:.3f}, {py:.3f}, {pz:.3f}")
      print(f"vel: {vx:.3f}, {vy:.3f}, {vz:.3f}")

    return px, py, pz, w, r, t

# Processes the raw flex sensor readings into a percent
def process_flex(value, raw = False):
  # output raw value if raw is true
  if raw: return value

  nominal = 5000 # Sensor value when unflexed
  max = 3000 # Sensor value when flexed 90deg forward
  deadzone = 250

  # Deadzone - return 0% flex when sensor is close to nominal value
  if value >= (nominal - deadzone): return 0
  
  # Forward flex - return percentage of flex
  if value < (nominal - deadzone): return min((nominal-value) / (nominal-max), 1)

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
mt = MotionTracker(i2c, debug=True)

# ---------- FLEX SENSOR SETUP ----------
ads = ADS.ADS1115(i2c)
flex_channel = AnalogIn(ads, 0)
print("ADC initialized")

# ------ Sensor Processing Variables ------
# State
px = py = pz = 0.0
vx = vy = vz = 0.0
last_time = time.time()

while True:
  # ----- READ IMU -----
  x, y, z, w, r, t = mt.update()

  # ----- READ FLEX SENSOR -----
  flex_value = process_flex(flex_channel.value, raw = False)

  # ----- CREATE MESSAGE -----
  message = f"{x:.6f}\n{y:.6f}\n{z:.6f}\n\n{w:.6f}\n{r:.6f}\n{t:.6f}\n\n{flex_value}\n"

  # ----- SEND DATA -----
  # client.send(message.encode())

  # print("")
  # print("Sent:")
  # print("--Position--")
  # print("X: ", x)
  # print("Y: ", y)
  # print("Z: ", z)
  # print("--Rotation--")
  # print("P: ", w)
  # print("Y: ", r)
  # print("R: ", t)
  # print("--Flex--")
  # print("F: ", flex_value)