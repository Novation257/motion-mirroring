# Program to be ran on the Raspberry Pi Zero

import socket
import time
import board
import busio
import math

from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, \
                            BNO_REPORT_LINEAR_ACCELERATION, \
                            BNO_REPORT_GEOMAGNETIC_ROTATION_VECTOR
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# MotionTracker class for IMU position and rotation tracking
class MotionTracker():
  def __init__(self, i2c, accel_deadzone = 0.15, vel_deadzone = 0.2, debug = False):
    self.px = self.py = self.pz = 0 # Position (m)
    self.vx = self.vy = self.vz = 0 # Velocity (m/s)
    self.tick = self.last_tick = self.dt = 0 # Time vars

    self.accel_deadzone = accel_deadzone # Deadzone for acceleration readings around zero (m/s^2)
    self.vel_deadzone = vel_deadzone # Deadzone for velocity readings around zero (m/s)
    self.debug = debug # Print values on every iteration

    # Sensor setup
    self.bno = BNO08X_I2C(i2c_bus=i2c)
    self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)

    # This is stupid but it clears the first few bad accel readings
    time.sleep(2)
    for _ in range(5): self.get_accel()

    # Init time vars
    self.delta()
    self.delta()

    # print("IMU initialized")
    pass

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

  # Quaternion rotate vector into world frame
  def rotate_vector(self, qx, qy, qz, qw, x, y, z):
    # Quaternion rotation: v' = q * v * q_conj
    ix =  qw * x + qy * z - qz * y
    iy =  qw * y + qz * x - qx * z
    iz =  qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z

    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx

    return rx, ry, rz

  # Gets acceleration tuple from sensor (m/s^2)
  def get_accel(self): return self.bno.linear_acceleration 

  # Gets rotation tuple from sensor 
  # (Quaternion, with x+ aligned with North and z- aligned with gravity vector)
  def get_rot_quat(self): return self.bno.quaternion

  # Time since last delta call, updates all time vars
  def delta(self):
    curr_time = time.time()
    self.last_tick = self.tick
    self.tick = curr_time
    self.dt = self.tick - self.last_tick
    return self.dt
  
  # Iterative calculations to track position and rotation
  def update(self):
    # Get sensor and time data
    qx, qy, qz, qw = self.get_rot_quat()
    ax, ay, az = self.get_accel()
    dt = self.delta()

    # Rotate accel vector into world frame
    ax, ay, az = self.rotate_vector(qx, qy, qz, qw, ax, ay, az)

    # Get euler angle representation of rotation
    w, r, t = self.quaternion_to_euler((qx, qy, qz, qw))

    # Integrate acceleration to get velocity
    if(abs(ax) > self.accel_deadzone): self.vx += ax * dt
    if(abs(ay) > self.accel_deadzone): self.vy += ay * dt
    if(abs(az) > self.accel_deadzone): self.vz += az * dt

    # Velocity decay
    vx *= 0.95
    vy *= 0.95
    vz *= 0.95

    # Integrate velocity to get position
    if(abs(self.vx) > self.vel_deadzone): self.px += self.vx * dt
    if(abs(self.vy) > self.vel_deadzone): self.py += self.vy * dt
    if(abs(self.vz) > self.vel_deadzone): self.pz += self.vz * dt

    # Print vars if in debug mode
    if self.debug:
      print("")
      print(f"dt:{dt:.4f}")
      print("")
      print(f"ax:{ax:.4f}")
      print(f"ay:{ay:.4f}")
      print(f"az:{az:.4f}")
      print("")
      print(f"vx:{self.vx:.4f}")
      print(f"vy:{self.vy:.4f}")
      print(f"vz:{self.vz:.4f}")
      print("----")
      print(f"px:{self.px:.4f}")
      print(f"py:{self.py:.4f}")
      print(f"pz:{self.pz:.4f}")
      print("")
      print(f"w:{w:.4f}")
      print(f"r:{r:.4f}")
      print(f"t:{t:.4f}")
    
    # Return position and rotation
    return(self.px, self.py, self.pz, w, r, t)

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
networking = bool(input("Connect to Scorpio? (y/n) ") == 'y')

if networking:
  SERVER_IP = input("Enter Scorpio IP: ")   # Replace with Pi 5 IP
  PORT = 5000

# ---------- CONNECT TO PI 5 ----------
if networking:
  client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client.connect((SERVER_IP, PORT))
  print("Connected to Pi 5")

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
  if networking: client.send(message.encode())

  print("")
  print("Sent:")
  print("--Position--")
  print("X: ", x)
  print("Y: ", y)
  print("Z: ", z)
  print("--Rotation--")
  print("P: ", w)
  print("Y: ", r)
  print("R: ", t)
  print("--Flex--")
  print("F: ", flex_value)