# Hardware side of the ball balancer: sensor calibration file, ball position from the
# light sensors, Kalman filter, and the servo.
#
# Conventions:
#   x     ball position in mm, 0 = middle of the beam, + towards the linkage end
#   theta beam angle in deg, + = linkage end up (so +theta rolls the ball towards -x)
import json
import os

import numpy as np

import kinematics

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "balancer_config.json")

# Servo settings, found with test.py
SERVO_GPIO = 18
MIN_US, MAX_US = 600, 2400   # pulse width at 0 and 180 deg
LEVEL_DEG = 130.0            # servo angle where the beam is flat
SERVO_DIR = 1                # increasing servo angle raises the linkage end
MAX_BEAM_DEG = 11.0          # about the most the linkage can do with +/-40 deg on the motor

DETECT_THRESHOLD = 0.3       # how covered the darkest sensor has to be to count as the ball


def load_config(path=CONFIG_FILE):
    cfg = {
        # sensor positions along the beam (25 mm apart), calibrate.py flips them if needed
        "sensor_x_mm": [25.0 * (i - 3.5) for i in range(8)],
        "baseline": None,   # readings with no ball
        "covered": None,    # readings with the ball right on top
    }
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg, path=CONFIG_FILE):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved {path}")


class BallEstimator:
    def __init__(self, cfg):
        if cfg["baseline"] is None or cfg["covered"] is None:
            raise RuntimeError("Sensors not calibrated, run: python calibrate.py")
        self.x = np.array(cfg["sensor_x_mm"], dtype=float)
        self.base = np.array(cfg["baseline"], dtype=float)
        span = np.array(cfg["covered"], dtype=float) - self.base
        self.span = np.where(np.abs(span) < 1, np.inf, span)   # dead sensor just reads 0

    def coverage(self, values):
        # 0 = no ball over the sensor, 1 = ball right on top
        d = np.asarray(values, dtype=float) - self.base
        # Room lighting shifts every sensor at once. The ball only covers 2-3 of the 8,
        # so the median is the lighting shift and we subtract it.
        d -= np.median(d)
        return np.clip(d / self.span, 0.0, 1.0)

    def position(self, values):
        # Returns (x in mm or None if no ball, coverage of each sensor)
        c = self.coverage(values)
        k = int(np.argmax(c))
        if c[k] < DETECT_THRESHOLD:
            return None, c
        # weighted average of the darkest sensor and its neighbours, so we get
        # positions in between sensors too
        lo, hi = max(k - 1, 0), min(k + 2, len(c))
        w = c[lo:hi]
        return float(np.dot(w, self.x[lo:hi]) / w.sum()), c


class BallFilter:
    # Kalman filter on [position, velocity, accel offset].
    # Uses the beam angle we command to predict the ball's acceleration, so the velocity
    # comes out smooth without the lag of a big low-pass filter. Taking the derivative of
    # the raw position directly was way too noisy and made the servo jerk.
    G = 4000.0          # mm/s^2 per sin(theta), fitted from a run (theory for a hollow ball is 3/5 g)
    MEAS_STD = 6.0      # position noise (mm)
    ACCEL_STD = 600.0   # how wrong the accel model can be (mm/s^2)
    BIAS_STD = 50.0     # how fast the offset can drift, e.g. beam not perfectly level
    GATE = 4.0          # ignore readings more than 4 std devs away from the prediction
    MAX_REJECTS = 3     # unless it happens 3 times in a row, then the ball really moved

    def __init__(self):
        self.s = None
        self.rejects = 0

    def reset(self):
        self.s = None

    def predict(self, dt, theta_deg):
        if self.s is None:
            return
        F = np.array([[1, dt, dt * dt / 2],
                      [0, 1, dt],
                      [0, 0, 1]])
        g = np.array([dt * dt / 2, dt, 0])
        a = -self.G * np.sin(np.radians(theta_deg))
        self.s = F @ self.s + g * a
        Q = np.outer(g, g) * self.ACCEL_STD**2
        Q[2, 2] = self.BIAS_STD**2 * dt
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        # returns False if the reading got rejected
        if self.s is None:
            self.s = np.array([z, 0.0, 0.0])
            self.P = np.diag([self.MEAS_STD**2, 300.0**2, 300.0**2])
            return True
        err = z - self.s[0]
        S = self.P[0, 0] + self.MEAS_STD**2
        if err * err > self.GATE**2 * S:
            self.rejects += 1
            if self.rejects >= self.MAX_REJECTS:
                self.s, self.rejects = None, 0
                return self.update(z)
            return False
        self.rejects = 0
        K = self.P[:, 0] / S
        self.s = self.s + K * err
        self.P = self.P - np.outer(K, self.P[0, :])
        return True

    @property
    def x(self):
        return None if self.s is None else float(self.s[0])

    @property
    def v(self):
        return 0.0 if self.s is None else float(self.s[1])


class BeamServo:
    def __init__(self):
        import pigpio
        self.max_beam = min(MAX_BEAM_DEG, -kinematics.BEAM_MIN_DEG, kinematics.BEAM_MAX_DEG)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Can't reach pigpiod, start it with: sudo systemctl start pigpiod")

    def set_servo_deg(self, angle):
        angle = max(0.0, min(180.0, angle))
        self.pi.set_servo_pulsewidth(SERVO_GPIO, MIN_US + (MAX_US - MIN_US) * angle / 180)

    def set_beam(self, theta_deg):
        # returns the angle actually sent after clamping
        theta = max(-self.max_beam, min(self.max_beam, theta_deg))
        self.set_servo_deg(LEVEL_DEG + SERVO_DIR * kinematics.motor_angle(theta))
        return theta

    def release(self):
        self.pi.set_servo_pulsewidth(SERVO_GPIO, 0)
        self.pi.stop()
