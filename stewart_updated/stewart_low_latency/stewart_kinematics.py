"""Three-actuator triangular platform inverse kinematics and slow PCA9685 control.

Geometry convention follows the user's supplied kinematics:
B_i = Rb*e_i
E_i = B_i + L1*(cos(q_i)*e_i + sin(q_i)*z)
P_i = [0,0,H] + R*(Rp*e_i)
||P_i - E_i|| = L2

q is horn elevation above horizontal. The configured IK branch is -1
(elbow-out branch). Servo commands are calibrated around the measured
level positions: [95, 79, 77] degrees. The observed servo direction is
negative for all channels: increasing the ServoKit angle lowers that side.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
STATE_PATH = HERE / "last_pose.json"
HOME_POSE = {"heave_mm": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0}
MAX_SERVO_SPEED_DPS = 20.0
ACCELERATION_DPS2 = 15.0
UPDATE_HZ = 50.0

def load_config():
    cfg = json.loads(CONFIG_PATH.read_text())
    g = cfg["geometry"]
    if len(g["radial_angles_deg"]) != 3:
        raise ValueError("Need exactly three radial angles")
    if float(g["ik_branch"]) not in (-1.0, 1.0):
        raise ValueError("ik_branch must be -1 or +1")
    if len(cfg["servos"]) != 3:
        raise ValueError("Need exactly three servo definitions")
    for s in cfg["servos"]:
        if s["servo_direction"] not in (-1, 1):
            raise ValueError("servo_direction must be -1 or +1")
    return cfg

def rotation_matrix(roll, pitch):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], float)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], float)
    return Ry @ Rx

def geometry(cfg):
    g = cfg["geometry"]
    beta = np.deg2rad(np.asarray(g["radial_angles_deg"], float))
    e = np.column_stack((np.cos(beta), np.sin(beta), np.zeros(3)))
    B = float(g["base_radius_mm"]) * e
    p = float(g["platform_radius_mm"]) * e
    return e, B, p

def pose_points(pose, cfg):
    g = cfg["geometry"]
    _, _, p = geometry(cfg)
    R = rotation_matrix(math.radians(pose.get("roll_deg", 0.0)),
                        math.radians(pose.get("pitch_deg", 0.0)))
    H = float(g["nominal_heave_mm"]) + float(pose.get("heave_mm", 0.0))
    return (R @ p.T).T + np.array([0.0, 0.0, H])

def inverse_kinematics(pose, cfg):
    e, B, _ = geometry(cfg)
    g = cfg["geometry"]
    L1 = float(g["horn_length_mm"])
    L2 = float(g["upper_link_length_mm"])
    branch = float(g["ik_branch"])
    margin = float(g.get("ik_margin", 0.98))
    P = pose_points(pose, cfg)
    q = np.zeros(3)
    for i in range(3):
        d = P[i] - B[i]
        A = float(d @ e[i])
        C = float(d[2])
        K = (float(d @ d) + L1**2 - L2**2) / (2*L1)
        rho = math.hypot(A, C)
        if rho < 1e-9 or abs(K) > margin*rho:
            raise ValueError(f"Actuator {i}: pose unreachable or near singularity")
        phi = math.atan2(C, A)
        delta = math.acos(K/rho)
        qi = math.atan2(math.sin(phi + branch*delta),
                        math.cos(phi + branch*delta))
        if math.cos(qi) <= 0:
            raise ValueError(f"Actuator {i}: horn points inward; check IK branch")
        q[i] = math.degrees(qi)
        s = cfg["servos"][i]
        if not float(s["motor_min_deg"]) <= q[i] <= float(s["motor_max_deg"]):
            raise ValueError(f"Actuator {i}: motor angle {q[i]:.2f} outside configured limits")
    return q

def neutral_motor_angles(cfg):
    return inverse_kinematics(HOME_POSE, cfg)

def servo_commands(pose, cfg):
    q = inverse_kinematics(pose, cfg)
    q0 = neutral_motor_angles(cfg)
    commands = []
    for i, s in enumerate(cfg["servos"]):
        a = float(s["servo_at_neutral_deg"]) + int(s["servo_direction"])*(q[i]-q0[i])
        if not float(s["servo_min_deg"]) <= a <= float(s["servo_max_deg"]):
            raise ValueError(f"Servo channel {s['channel']} command {a:.2f} outside "
                             f"[{s['servo_min_deg']}, {s['servo_max_deg']}]")
        commands.append(a)
    return q, np.asarray(commands)

def make_kit(cfg):
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16, address=int(cfg["pca9685_address"], 16),
                   frequency=int(cfg.get("pwm_frequency_hz", 50)))
    for s in cfg["servos"]:
        servo = kit.servo[int(s["channel"])]
        servo.set_pulse_width_range(int(s["pulse_min_us"]), int(s["pulse_max_us"]))
        servo.actuation_range = float(s.get("actuation_range_deg", 180))
    return kit

def load_last_pose():
    try:
        return json.loads(STATE_PATH.read_text())["pose"]
    except (OSError, ValueError, KeyError):
        return None

def save_pose(pose):
    STATE_PATH.write_text(json.dumps({"pose": pose}, indent=2))

def move_pose(target, cfg, kit, start=None, speed_override=None):
    """Move with adaptive subdivision; persist each completed intermediate pose."""
    if start is None:
        start = load_last_pose()
    if start is None:
        start = HOME_POSE.copy()
        print("No saved pose found.")
        print("Assuming the mechanism is physically at calibrated level [95, 79, 77].")
        print("If it is not, stop and manually re-home before moving.")

    # Normalize pose keys so interpolation is deterministic.
    keys = ("heave_mm", "roll_deg", "pitch_deg")
    start = {k: float(start.get(k, 0.0)) for k in keys}
    target = {k: float(target.get(k, 0.0)) for k in keys}

    _, a0 = servo_commands(start, cfg)
    _, a1 = servo_commands(target, cfg)

    motion = cfg.get("motion", {})
    speed = float(speed_override if speed_override is not None
                  else MAX_SERVO_SPEED_DPS)
    hz = 40.0
    if speed <= 0 or hz <= 0:
        raise ValueError("Speed and update rate must be positive")
    max_step = speed / hz

    # Pose-space interpolation is nonlinear in servo space. Increase the
   # Smooth trapezoidal/triangular velocity profile
    distance = float(np.max(np.abs(a1 - a0)))

    if distance < 1e-6:
        return target

    # Determine total movement time based on acceleration and speed
    accel_time = MAX_SERVO_SPEED_DPS / ACCELERATION_DPS2
    accel_distance = 0.5 * ACCELERATION_DPS2 * accel_time**2

    if distance < 2 * accel_distance:
        # Triangular profile: never reaches maximum speed
        accel_time = math.sqrt(distance / ACCELERATION_DPS2)
        cruise_time = 0.0
    else:
        cruise_time = (distance - 2 * accel_distance) / MAX_SERVO_SPEED_DPS

    total_time = 2 * accel_time + cruise_time
    n = max(1, math.ceil(total_time * UPDATE_HZ))

    print(f"Moving over {total_time:.2f} seconds in {n} steps.")

    completed_pose = start.copy()

    try:
        for k in range(1, n + 1):
            t = min(k / UPDATE_HZ, total_time)

            # Acceleration / constant velocity / deceleration
            if t < accel_time:
                distance_moved = 0.5 * ACCELERATION_DPS2 * t**2

            elif t < accel_time + cruise_time:
                distance_moved = (
                    accel_distance
                    + MAX_SERVO_SPEED_DPS * (t - accel_time)
                )

            else:
                remaining = total_time - t
                distance_moved = distance - (
                    0.5 * ACCELERATION_DPS2 * remaining**2
                )

            f = min(1.0, distance_moved / distance)

            pose = {
                key: start[key] + (target[key] - start[key]) * f
                for key in keys
            }

            _, cmd = servo_commands(pose, cfg)

            for i, s in enumerate(cfg["servos"]):
                kit.servo[int(s["channel"])].angle = float(cmd[i])

            completed_pose = pose
            save_pose(completed_pose)

            time.sleep(1.0 / UPDATE_HZ)

    except BaseException:
        save_pose(completed_pose)
        print("Move interrupted. Saved last commanded pose:")
        print(completed_pose)
        raise

    print("Move complete.")
    return target
    print(f"Start pose: {start}")
    print(f"Target pose: {target}")
    print("Target servo commands (deg):", np.round(commands[-1], 3).tolist())
    print(f"Moving in {n} steps; max command increment "
          f"{largest_step:.4f} deg at {hz:.0f} Hz.")

    completed_pose = start.copy()
    try:
        for k in range(1, n + 1):
            pose = poses[k]
            cmd = commands[k]
            # Send all three channels for each coordinated waypoint.
            for i, s in enumerate(cfg["servos"]):
                kit.servo[int(s["channel"])].angle = float(cmd[i])
            completed_pose = pose
            # Save after EACH completed coordinated waypoint. If a later
            # command is interrupted, the next run starts from this estimate.
            save_pose(completed_pose)
            time.sleep(1.0 / hz)
    except BaseException:
        save_pose(completed_pose)
        print("\\nMove interrupted/failed. Saved last completed commanded pose:")
        print(completed_pose)
        print("This is a command estimate, not physical feedback.")
        raise

    print("Move complete. Last commanded pose saved.")
    return target

def print_dry_run(pose, cfg):
    q, cmds = servo_commands(pose, cfg)
    print("Pose:", pose)
    print("Motor horn angles (deg):", np.round(q, 3).tolist())
    print("ServoKit commands (deg):", np.round(cmds, 3).tolist())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--heave", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--speed", type=float, default=None, help="max servo speed in deg/s")
    a = ap.parse_args()
    cfg = load_config()
    pose = {"heave_mm": a.heave, "roll_deg": a.roll, "pitch_deg": a.pitch}
    if a.dry_run:
        print_dry_run(pose, cfg)
    else:
        kit = make_kit(cfg)
        move_pose(pose, cfg, kit, speed_override=a.speed)
