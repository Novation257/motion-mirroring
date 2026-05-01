# Program to be ran on the Raspberry Pi Zero

import socket
import time
import board
import busio
import math
import numpy as np
import RPi.GPIO as GPIO
import sys

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

    # Covariance matrix
    self.P = np.eye(2) * 1.0

    # Process noise (dynamically adjusted per dt)
    self.Q = np.array([[0.05, 0],
                        [0, 0.5]])

    # Measurement noise (velocity correction via ZUPT)
    self.R = np.array([[0.01]])  # Start with small trust in velocity measurements

    # Measurement matrix (we observe velocity during ZUPT)
    self.H = np.array([[0, 1]])

    # Low-pass filter coefficient (for raw accel smoothing)
    self.alpha = 0.2
    self.velocity = 0  # Smoothing of velocity

  def predict(self, a, dt):
    # Dynamically adjusting process noise based on time step (dt)
    q = 0.1  # Process noise coefficient (you can tune this)
    self.Q = np.array([
        [0.25 * dt**4, 0.5 * dt**3],
        [0.5 * dt**3, dt**2]
    ]) * q

    # State transition matrix (position, velocity)
    F = np.array([[1, dt],
                  [0, 1]])

    # Control matrix (acceleration)
    B = np.array([[0.5 * dt**2],
                  [dt]])

    # Prediction step: Predict state
    self.x = F @ self.x + B * a

    # Predict covariance
    self.P = F @ self.P @ F.T + self.Q

  def update_velocity(self, measured_v=0):
    # Measurement update: Update velocity (via ZUPT)
    z = np.array([[measured_v]])

    y = z - (self.H @ self.x)  # Innovation or residual
    S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
    K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

    # Update state estimate
    self.x = self.x + K @ y
    self.P = (np.eye(2) - K @ self.H) @ self.P

  def get_state(self):
    # Return the current position and velocity
    return self.x[0, 0], self.x[1, 0]

  def apply_velocity_smoothing(self, vx):
    # Simple velocity smoothing (exponential filter)
    self.velocity = 0.9 * self.velocity + 0.1 * vx
    return self.velocity

  def zero_velocity_update(self, ax, ay, az, threshold=0.1):
    # # Zero-velocity detection using the magnitude of the acceleration
    # acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    # if acc_mag < threshold:
    #     return True  # ZUPT: we assume zero velocity here
    return False

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

    # Last position and rotation
    self.last_position = (0, 0, 0)
    self.last_rotation = (0, 0, 0)

    # Sensor setup
    self.bno = BNO08X_I2C(i2c_bus=i2c)
    self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)

    # Calibration
    time.sleep(2)
    print("Calibrating...")
    # self.calibrate_bias()

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

    # Calculate deltas for position and rotation
    position_delta = (px - self.last_position[0], py - self.last_position[1], pz - self.last_position[2])
    rotation_delta = (w - self.last_rotation[0], r - self.last_rotation[1], t - self.last_rotation[2])

    # Update last position and rotation
    self.last_position = (px, py, pz)
    self.last_rotation = (w, r, t)

    # Print vars if in debug mode
    if self.debug:
      print(f"""dt: {dt:.4f} delta_pos: {position_delta[0]:.3f}, {position_delta[1]:.3f}, {position_delta[2]:.3f}
               delta_rot: {rotation_delta[0]:.3f}, {rotation_delta[1]:.3f}, {rotation_delta[2]:.3f}""")

    # Return position and rotation deltas
    return position_delta + rotation_delta

# Processes the raw flex sensor readings into a percent
def process_flex(value, raw = False):
  # output raw value if raw is true
  if raw: return value

  nominal = 1400 # Sensor value when unflexed
  flexed = 700 # Diff between nominal and max value

  return max(0.0, min((value - nominal) / flexed, 1.0))

# ---------- NETWORK SETTINGS ----------
# networking = False
networking = bool(input("Connect to Scorpio? (y/n) ") == 'y')

if networking:
  SERVER_IP = input("Enter Scorpio IP: ")   # Replace with Pi 5 IP
  PORT = 5000

# ---------- CONNECT TO PI 5 ----------
if networking:
  client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client.connect((SERVER_IP, PORT))
  print("Connected to Pi 5")

# --------- GPIO (DEAD MAN SWITCH) ---------
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ---------- I2C SETUP ----------
i2c = busio.I2C(board.SCL, board.SDA)

# ---------- IMU SETUP ----------
mt = MotionTracker(i2c, debug=False)

# ---------- FLEX SENSOR SETUP ----------
ads = ADS.ADS1115(i2c)
flex_channel = AnalogIn(ads, 0)
print("ADC initialized")

# ------ Sensor Processing Variables ------
# State
point_rotation = (0, 0, 0, 0, 0, 0)
flex_out = 1.0
last_DMT_state = False
last_time = time.time()

while True:
  # ----- READ IMU -----
  deltas = mt.update()

  # ----- READ FLEX SENSOR -----
  flex_value = process_flex(flex_channel.value, raw = False)

  # ----- READ DEAD MAN'S TRIGGER -----
  button_pressed = (GPIO.input(BUTTON_PIN) == GPIO.LOW)
  # button_just_pressed = True if (button_pressed == True and last_DMT_state == False) else False
  # button_just_released = True if (button_pressed == False and last_DMT_state == True) else False
  # last_DMT_state = button_pressed

  # Only allow movement when DMT is pressed... clamp angles to (-180, 180)
  if button_pressed: 
    point_rotation = np.add(point_rotation, deltas)
    flex_out = flex_value
  x, y, z, w, r, t = point_rotation
  for angle in (w, r, t): ((angle + 180) % 360) - 180

  # ----- CREATE MESSAGE -----


  # message = f"{x:.2f}\n{y:.2f}\n{z:.2f}\n\n{w:.2f}\n{r:.2f}\n{t:.2f}\n\n{flex_value}\n"
  message = f"{x:.6f},{y:.6f},{z:.6f},{w:.6f},{r:.6f},{t:.6f},{flex_out}\n"

  printable = f"{x:.2f}, {y:.2f}, {z:.2f}   {w:.2f}, {r:.2f}, {t:.2f}   flex={flex_value} --"
  # print(f"\r{printable}", end='', flush=True)
  print(printable)

  # ----- SEND DATA -----
  if networking: client.send(message.encode())