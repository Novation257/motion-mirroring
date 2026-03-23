import time
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize sensor
bno = BNO08X_I2C(i2c)

# Enable rotation vector report
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

print("BNO085 Test Running...")

while True:
    quat = bno.quaternion
    print("Quaternion:", quat)
    time.sleep(0.5)