import socket
import time
import board
import busio
import math

from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ---------- NETWORK SETTINGS ----------
SERVER_IP = "192.168.1.100"   # Replace with Pi 5 IP
PORT = 5000

# ---------- CONNECT TO PI 5 ----------
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER_IP, PORT))

print("Connected to Pi 5")

# ---------- I2C SETUP ----------
i2c = busio.I2C(board.SCL, board.SDA)

# ---------- IMU SETUP ----------
bno = BNO08X_I2C(i2c)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

# ---------- FLEX SENSOR SETUP ----------
ads = ADS.ADS1115(i2c)
flex_channel = AnalogIn(ads, ADS.P0)

while True:

    # ----- READ IMU -----
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

    # ----- READ FLEX SENSOR -----
    flex_value = flex_channel.value

    # ----- CREATE MESSAGE -----
    message = f"{yaw:.2f},{pitch:.2f},{roll:.2f},{flex_value}\n"

    # ----- SEND DATA -----
    client.send(message.encode())

    print("Sent:", message)

    time.sleep(0.1)