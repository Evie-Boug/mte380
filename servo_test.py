import sys
import time
import pigpio

servo_gpio = 18
pi = pigpio.pi()
pigpio.pi().set_servo_pulsewidth(servo_gpio, 1000)
