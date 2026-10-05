#!/usr/bin/env python3
"""Motor test script to run to specific angle. 
Rubie - angle 130 is the platform being flat for motor encoders"""
import sys
import time

import pigpio

SERVO_GPIO = 18
MIN_US, MAX_US = 600, 2400   # conservative MG995 range; widen to 500-2500 if needed


def move(pi, angle):
    angle = max(0, min(180, angle))
    pi.set_servo_pulsewidth(SERVO_GPIO, MIN_US + (MAX_US - MIN_US) * angle / 180)

def move_to(pi, current, target, speed_dps=60, rate_hz=50):
    """Move smoothly from current to target angle at speed_dps degrees/second.
    Returns the final angle (store it for the next call)."""
    if current == target:
        return target
    step = speed_dps / rate_hz               # degrees per update
    direction = 1 if target > current else -1
    angle = current
    while (target - angle) * direction > 0:
        angle += direction * min(step, abs(target - angle))
        pi.set_servo_pulsewidth(SERVO_GPIO, MIN_US + (MAX_US - MIN_US) * angle / 180)
        time.sleep(1 / rate_hz)
    return target


def sweep(pi, start, end, step=2, delay=0.02):
    direction = 1 if end >= start else -1
    for a in range(start, end + direction, step * direction):
        move(pi, a)
        time.sleep(delay)


pi = pigpio.pi()
if not pi.connected:
    sys.exit("Can't reach pigpiod. Start it with: sudo systemctl start pigpiod")

try:
    print("Center (90)")
    move_to(pi, 90, 135)
    time.sleep(1)
  
except KeyboardInterrupt:
    print("\nStopping")
finally:
    pi.set_servo_pulsewidth(SERVO_GPIO, 0)   # stop driving the servo
    pi.stop()