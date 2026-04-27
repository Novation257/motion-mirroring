import socket
import time
import board
import busio
import math
import numpy as np
import RPi.GPIO as GPIO

from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, \
                            BNO_REPORT_LINEAR_ACCELERATION
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class Kalman1D:
  def __init__(self):
    self.x = np.array([[0.0], [0.0]])
    self.P = np.eye(2) * 1.0
    self.Q = np.array([[0.05, 0], [0, 0.5]])
    self.R = np.array([[0.01]])
    self.H = np.array([[0, 1]])

  def predict(self, a, dt):
    q = 0.1
    self.Q = np.array([
        [0.25 * dt**4, 0.5 * dt**3],
        [0.5 * dt**3,  dt**2]
    ]) * q
    F = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5 * dt**2], [dt]])
    self.x = F @ self.x + B * a
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


def _wrap_delta(delta):
  """Normalize an angle delta to [-180, 180] to handle IMU ±180° boundary crossings."""
  return ((delta + 180) % 360) - 180


class MotionTracker():
  def __init__(self, i2c, debug=False):
    self.debug = debug

    self.kx = Kalman1D()
    self.ky = Kalman1D()
    self.kz = Kalman1D()

    self.ax_f = self.ay_f = self.az_f = 0
    self.bias_ax = self.bias_ay = self.bias_az = 0
    self.tick = time.perf_counter()
    self.last_position = (0, 0, 0)
    self.last_rotation = (0, 0, 0)

    self.bno = BNO08X_I2C(i2c_bus=i2c)
    self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)

    time.sleep(2)
    print("Calibrating...")

  def quaternion_to_euler(self, quaternion):
    x, y, z, w = quaternion
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

  def delta(self):
    now = time.perf_counter()
    dt = now - self.tick
    self.tick = now
    return dt

  def rotate_vector(self, qx, qy, qz, qw, x, y, z):
    ix =  qw * x + qy * z - qz * y
    iy =  qw * y + qz * x - qx * z
    iz =  qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return rx, ry, rz

  def update(self):
    dt = self.delta()
    qx, qy, qz, qw = self.bno.quaternion
    ax, ay, az = self.bno.linear_acceleration

    ax -= self.bias_ax
    ay -= self.bias_ay
    az -= self.bias_az

    ax, ay, az = self.rotate_vector(qx, qy, qz, qw, ax, ay, az)
    w, r, t = self.quaternion_to_euler((qx, qy, qz, qw))

    alpha = 0.2
    self.ax_f = alpha * ax + (1 - alpha) * self.ax_f
    self.ay_f = alpha * ay + (1 - alpha) * self.ay_f
    self.az_f = alpha * az + (1 - alpha) * self.az_f

    self.kx.predict(self.ax_f, dt)
    self.ky.predict(self.ay_f, dt)
    self.kz.predict(self.az_f, dt)

    if abs(self.ax_f) < 0.05: self.kx.update_velocity(0)
    if abs(self.ay_f) < 0.05: self.ky.update_velocity(0)
    if abs(self.az_f) < 0.05: self.kz.update_velocity(0)

    px, _ = self.kx.get_state()
    py, _ = self.ky.get_state()
    pz, _ = self.kz.get_state()

    position_delta = (px - self.last_position[0], py - self.last_position[1], pz - self.last_position[2])
    rotation_delta = (
      _wrap_delta(w - self.last_rotation[0]),
      _wrap_delta(r - self.last_rotation[1]),
      _wrap_delta(t - self.last_rotation[2]),
    )

    self.last_position = (px, py, pz)
    self.last_rotation = (w, r, t)

    if self.debug:
      print(f"\rdt:{dt:.4f}  dpos:({position_delta[0]:.3f},{position_delta[1]:.3f},{position_delta[2]:.3f})"
            f"  drot:({rotation_delta[0]:.3f},{rotation_delta[1]:.3f},{rotation_delta[2]:.3f})", end='')

    return position_delta + rotation_delta


def process_flex(value, raw=False):
  if raw: return value
  nominal  = 5000
  max      = 3000
  deadzone = 250
  if value >= (nominal - deadzone): return 0
  if value  < (nominal - deadzone): return min((nominal - value) / (nominal - max), 1)
  return 0


# ---------- NETWORK SETTINGS ----------
# networking = False
networking = bool(input("Connect to Scorpio? (y/n) ") == 'y')

if networking:
  SERVER_IP = input("Enter Scorpio IP: ")
  PORT = 5000
  client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client.connect((SERVER_IP, PORT))
  print("Connected to Scorpio")

# --------- GPIO (DEAD MAN'S TRIGGER) ---------
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

# ------ Accumulated pose (position + rotation) ------
point_rotation = np.zeros(6)

while True:
  # ----- READ IMU -----
  deltas = mt.update()

  # ----- READ FLEX SENSOR -----
  flex_value = process_flex(flex_channel.value, raw=False)

  # ----- READ DEAD MAN'S TRIGGER -----
  button_pressed = (GPIO.input(BUTTON_PIN) == GPIO.LOW)

  # Only accumulate movement while button is held
  if button_pressed:
    point_rotation = np.add(point_rotation, deltas)

  x, y, z, w, r, t = point_rotation

  # ----- CREATE MESSAGE -----
  # "x,y,z,pitch,yaw,roll,flex\n" — matches omx_scorpio.py and omx_pry_testing.py
  message = f"{x:.6f},{y:.6f},{z:.6f},{w:.6f},{r:.6f},{t:.6f},{flex_value}\n"

  # ----- SEND DATA -----
  if networking:
    client.send(message.encode())

  print(f"\rX:{x:.3f} Y:{y:.3f} Z:{z:.3f}  P:{w:.1f} Yw:{r:.1f} R:{t:.1f}  Flex:{flex_value:.2f}  DMT:{'ON' if button_pressed else 'off'}",
        end='', flush=True)

  time.sleep(0.1)
