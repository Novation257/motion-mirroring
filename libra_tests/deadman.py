import RPi.GPIO as GPIO
import time

# ---------- SETUP ----------

BUTTON_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print(GPIO.getmode())
print("Dead Man Switch Test Started")
print("Press button = ACTIVE | Release = STOP\n")

last_state = None

try:
    while True:
        state = GPIO.input(BUTTON_PIN)

    if state == GPIO.LOW:
        print("ACTIVE (PRESSED)")
    else:
        print("STOP (RELEASED)")

    time.sleep(0.05)  # debounce

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    GPIO.cleanup()
