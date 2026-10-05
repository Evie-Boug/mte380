# Linkage kinematics for the beam, using the dimensions from our hand calcs.
# Beam pivot is at (0, 0), motor shaft is at (W, -H). Angles in degrees.
# theta   = beam angle, positive when the linkage end of the beam goes up
# theta11 = motor (crank) angle from horizontal, 0 when the beam is level
import numpy as np

L = 100.0     # pivot to rod attachment on the beam (mm)
W = 70.0      # horizontal distance pivot to motor
H = 100.0     # vertical distance pivot to motor
L11 = 30.0    # crank
L12 = 100.0   # connecting rod

MOTOR_LIMIT_DEG = 40.0


def beam_angle(theta11_deg):
    # Forward kinematics. The beam end has to be L from the pivot and L12 from the crank tip,
    # so it's where those two circles intersect.
    t11 = np.radians(theta11_deg)
    cx = W + L11 * np.cos(t11)
    cy = -H + L11 * np.sin(t11)
    d = np.hypot(cx, cy)
    a = (L**2 - L12**2 + d**2) / (2 * d)
    h = np.sqrt(np.maximum(L**2 - a**2, 0.0))
    px, py = a * cx / d, a * cy / d
    x1, y1 = px + h * cy / d, py - h * cx / d
    x2, y2 = px - h * cy / d, py + h * cx / d
    # the real solution is the one on the motor side
    x = np.where(x1 > x2, x1, x2)
    y = np.where(x1 > x2, y1, y2)
    return np.degrees(np.arctan2(y, x))


# beam angle vs motor angle is monotonic over +/-40 deg, so the inverse is just a lookup table
_motor_table = np.linspace(-MOTOR_LIMIT_DEG, MOTOR_LIMIT_DEG, 801)
_beam_table = beam_angle(_motor_table)
BEAM_MIN_DEG = float(_beam_table[0])
BEAM_MAX_DEG = float(_beam_table[-1])


def motor_angle(theta_deg):
    # Inverse kinematics: beam angle -> motor angle
    return float(np.interp(theta_deg, _beam_table, _motor_table))


if __name__ == "__main__":
    print(f"motor +/-{MOTOR_LIMIT_DEG:.0f} deg gives beam {BEAM_MIN_DEG:.2f} to {BEAM_MAX_DEG:.2f} deg")
    for t in (-10, -5, 0, 5, 10):
        print(f"beam {t:+5.1f} -> motor {motor_angle(t):+6.2f}")
