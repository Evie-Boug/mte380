from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
kit.servo[0].angle = 95
kit.servo[1].angle = 80
kit.servo[2].angle = 80