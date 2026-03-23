import time
import math
import socket
import board
import busio

from adafruit_bno08x import (
    BNO_REPORT_ROTATION_VECTOR,
    BNO_REPORT_LINEAR_ACCELERATION,
)
from adafruit_bno08x.i2c import BNO08X_I2C

# UDP setup
# UDP_IP = "192.168.1.147"
# UDP_PORT = 4210
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# I2C + IMU
i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO08X_I2C(i2c)

bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)

print("IMU initialized")

# State
px = py = pz = 0.0
vx = vy = vz = 0.0

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


while True:
    now = time.time()
    dt = now - last_time
    last_time = now

    quat = bno.quaternion
    accel = bno.linear_acceleration

    if quat is None or accel is None:
        continue

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

    # Output
    output = f"{qx:.5f},{qy:.5f},{qz:.5f},{qw:.5f}," \
                f"{px:.2f},{py:.2f},{pz:.2f}"

    # sock.sendto(output.encode(), (UDP_IP, UDP_PORT))
    print(output)

    time.sleep(0.01)