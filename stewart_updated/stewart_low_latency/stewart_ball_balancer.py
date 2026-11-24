#!/usr/bin/env python3
"""2-axis camera ball balancer for the three-servo Stewart-style platform.

Run: python stewart_ball_balancer.py              (GUI, dry-run by default)
     python stewart_ball_balancer.py --hardware   (enable PCA9685)
     python stewart_ball_balancer.py --headless   (camera-only dry-run)

First calibrate camera-to-platform XY with the GUI's Calibrate 3 points button.
Keep the camera fixed. Verify movement sign with LOW gains before balancing.

Architecture (built for a Raspberry Pi 4)
  * TWO processes. The control process owns the camera, ball detection,
    Kalman filter, trajectory planner, PID and servos, pinned to cores 2-3.
    The Tk GUI runs in the main process on cores 0-1. Python's GIL is per
    process, so nothing the GUI does (redraws, button handling, preview
    rendering) can delay a control step.
  * They share state through shared memory (written every control step) and
    a command queue. Preview frames are a plain memory copy a few times per
    second; undistortion and drawing happen in the GUI process.
  * Capture thread only grab()s; a frame is decoded only when control is ready.
  * Kalman filter (position, velocity, plate bias) with latency prediction.
  * No CSV logging.

Control law (per axis, output in physical degrees)
  u = Kp*(p - p_ref) + Kd*(v - v_ref) + Ki*integral + (b - Kff*a_ref)/G
where G = ball acceleration per degree of tilt and b is the plate-bias
acceleration estimated by the Kalman filter.  Tracking velocity error
(instead of damping velocity to zero) is what lets the ball follow a shape.
"""
import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

import cv2
import numpy as np

# Avoid OpenCV spawning extra worker pools on CPU-limited Raspberry Pi systems.
cv2.setNumThreads(1)

import stewart_kinematics as kin
from stewart_ball_detection import BallDetector
from trajectory_planner import TrajectoryPlanner, SHAPES

HERE = Path(__file__).resolve().parent

# ============== TEMPORARY KINEMATICS CORRECTION ==============
# Measured physical tilt is ~1/2 of kinematics requested tilt.
# Multiply desired physical roll/pitch by 2 before calling IK.
KINEMATICS_ANGLE_FACTOR = 2.0
# =============================================================

# ---------------- camera ----------------
CAMERA_ID = 0
CAMERA_FPS = 60
# Auto-exposure often drops a webcam to 15 FPS indoors and blurs the ball.
# On V4L2: CAMERA_AUTO_EXPOSURE = 1 means manual, 3 means auto. Then set
# CAMERA_EXPOSURE (driver units, often 50-300; lower = shorter). None = leave.
CAMERA_AUTO_EXPOSURE = None
CAMERA_EXPOSURE = None
FRAME_WAIT_S = 0.10
# Preview is the most expensive thing on a Pi (remap + Tk image upload), and
# the control loop never needs it. 8 FPS at half size is ~10x cheaper than
# 20 FPS full size. Full size is used automatically while calibrating.
PREVIEW_HZ = 8.0
PREVIEW_SCALE = 0.5
GUI_HZ = 10.0
# Python hands the GIL between threads every 5 ms by default; a shorter
# interval stops GUI work from delaying the control thread.
GIL_SWITCH_INTERVAL_S = 0.001
# Pi 4 has 4 cores: control process on cores 2-3, GUI process on cores 0-1.
PIN_CPUS = True
CONTROL_CPUS = (2, 3)
GUI_CPUS = (0, 1)

# ---------------- control ----------------
KP, KI, KD = 0.025, 0.0, 0.015   # deg/mm, deg/(mm*s), deg*s/mm
KFF = 1.0                        # acceleration feed-forward gain (1 = physics model)
MAX_PHYSICAL_TILT_DEG = 4.0
MAX_IK_TILT_DEG = 10.0
MAX_SERVO_RATE_DPS = 240.0       # was 30: that alone added ~0.5 s lag
SERVO_DEADBAND_DEG = 0.1
LOST_TIMEOUT_S = 0.30
MAX_INTEGRAL_MM_S = 200.0
MAX_POSITION_JUMP_MM = 50.0
# Positive physical Rx makes +Y ball roll toward -Y; positive physical Ry
# makes +X ball roll toward +X. Therefore to pull +X ball back use -pitch,
# and to pull +Y ball back use +roll. Verify experimentally at low angles.
ROLL_SIGN = +1.0
PITCH_SIGN = -1.0

# Ball rolling on a tilted plane: a = g*sin(theta) / (1 + I/(m r^2)).
# Ping-pong ball (thin shell) I/(m r^2) = 2/3; solid ball = 2/5.
BALL_INERTIA_RATIO = 2.0 / 3.0
BALL_ACCEL_PER_DEG = 9810.0 / (1.0 + BALL_INERTIA_RATIO) * math.pi / 180.0  # mm/s^2 per deg

# ---------------- estimation / latency ----------------
ESTIMATOR_ACCEL_SIGMA = 600.0    # mm/s^2 unmodelled acceleration (higher = less lag, more noise)
ESTIMATOR_MEAS_SIGMA_MM = 0.8
MODEL_INPUT_GAIN = 1.0           # 0 disables the tilt->acceleration model in the filter
DEFAULT_LATENCY_COMP_MS = 60.0   # camera exposure/USB + servo response not seen by timestamps
BIAS_RANDOM_WALK = 25.0          # mm/s^2 per sqrt(s): how fast the plate-bias estimate adapts
MAX_BIAS_ACCEL = 1.5 * BALL_ACCEL_PER_DEG   # cap bias compensation at 1.5 deg
BIAS_COMPENSATION = True

# ---------------- trajectories ----------------
BALL_EDGE_MARGIN_MM = 20.0
DEFAULT_TRAJECTORY_SPEED_MM_S = 30.0
DEFAULT_TRAJECTORY_ACCEL_MM_S2 = 120.0


class BallEstimator:
    """Kalman filter per axis with state [position, velocity, bias acceleration].

    Ball acceleration model: a = -G*u + b, where u is the commanded axis output
    and b is a slowly varying bias (plate not level, trim error, gain error).
    The controller cancels b with feed-forward, which removes steady offsets
    while tracking a moving shape far better than an integral term can.

    Both axes share dt and noise, so they share ONE covariance. Everything is
    plain-float arithmetic: on a Pi, numpy calls on 3x3 matrices cost far more
    in per-call overhead than the maths itself.
    """

    def __init__(self, accel_sigma=ESTIMATOR_ACCEL_SIGMA, meas_sigma=ESTIMATOR_MEAS_SIGMA_MM,
                 bias_sigma=BIAS_RANDOM_WALK):
        self.q = accel_sigma ** 2
        self.r = meas_sigma ** 2
        self.qb = bias_sigma ** 2
        self._b = [0.0, 0.0]
        self.reset()

    # numpy views for callers (cheap; not used inside the filter)
    p = property(lambda self: np.array(self._p))
    v = property(lambda self: np.array(self._v))
    b = property(lambda self: np.array(self._b))

    def reset(self, z=None, keep_bias=True):
        """Reset position/velocity. The bias is a plate property, so it is kept."""
        self.valid = z is not None
        self._p = [0.0, 0.0] if z is None else [float(z[0]), float(z[1])]
        self._v = [0.0, 0.0]
        if not keep_bias:
            self._b = [0.0, 0.0]
        b_var = 15.0 ** 2 if keep_bias and any(self._b) else 40.0 ** 2
        # symmetric covariance P00 P01 P02 P11 P12 P22
        self._P = [self.r, 0.0, 0.0, 200.0 ** 2, 0.0, b_var]

    def predict(self, dt, accel):
        if not self.valid:
            return
        h, h2 = dt, 0.5 * dt * dt
        for k in (0, 1):
            a = float(accel[k]) + self._b[k]
            self._p[k] += self._v[k] * h + a * h2
            self._v[k] += a * h
        P00, P01, P02, P11, P12, P22 = self._P
        r00 = P00 + h * P01 + h2 * P02
        r01 = P01 + h * P11 + h2 * P12
        r02 = P02 + h * P12 + h2 * P22
        r11 = P11 + h * P12
        r12 = P12 + h * P22
        q = self.q
        self._P = [r00 + h * r01 + h2 * r02 + q * h2 * h2,
                   r01 + h * r02 + q * h2 * h,
                   r02,
                   r11 + h * r12 + q * h * h,
                   r12,
                   P22 + self.qb * dt]

    def correct(self, z):
        P00, P01, P02, P11, P12, P22 = self._P
        S = P00 + self.r
        K0, K1, K2 = P00 / S, P01 / S, P02 / S
        for k in (0, 1):
            y = float(z[k]) - self._p[k]
            self._p[k] += K0 * y
            self._v[k] += K1 * y
            self._b[k] = min(MAX_BIAS_ACCEL, max(-MAX_BIAS_ACCEL, self._b[k] + K2 * y))
        self._P = [P00 - K0 * P00, P01 - K0 * P01, P02 - K0 * P02,
                   P11 - K1 * P01, P12 - K1 * P02, P22 - K2 * P02]

    def distance_to(self, z):
        return math.hypot(float(z[0]) - self._p[0], float(z[1]) - self._p[1])

    def predicted(self, horizon, accel):
        h, h2 = horizon, 0.5 * horizon * horizon
        pp, pv = [], []
        for k in (0, 1):
            a = float(accel[k]) + self._b[k]
            pp.append(self._p[k] + self._v[k] * h + a * h2)
            pv.append(self._v[k] + a * h)
        return pp, pv


class AxisPID:
    """PD + I + acceleration feed-forward tracking a full reference state."""

    def __init__(self, kp=KP, ki=KI, kd=KD, kff=KFF):
        self.kp, self.ki, self.kd, self.kff = kp, ki, kd, kff
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, position, velocity, ref_p, ref_v, ref_a, dt, bias_accel=0.0):
        e = position - ref_p
        ev = velocity - ref_v
        # Feed-forward: produce the reference acceleration and cancel plate bias.
        u_ff = (-self.kff * ref_a + bias_accel) / BALL_ACCEL_PER_DEG
        i_try = min(MAX_INTEGRAL_MM_S, max(-MAX_INTEGRAL_MM_S, self.integral + e * dt))
        u = self.kp * e + self.kd * ev + self.ki * i_try + u_ff
        if abs(u) < MAX_PHYSICAL_TILT_DEG:          # conditional anti-windup
            self.integral = i_try
        return min(MAX_PHYSICAL_TILT_DEG, max(-MAX_PHYSICAL_TILT_DEG, u))


class PlatformDriver:
    """Rate-limited servo output with a scalar copy of stewart_kinematics' IK.

    The IK maths is identical to kin.inverse_kinematics (roll/pitch only,
    heave 0) but uses plain floats: ~10x cheaper per call on a Pi.
    """

    def __init__(self, cfg, hardware=False):
        self.cfg = cfg
        self.kit = kin.make_kit(cfg) if hardware else None
        self.hardware = hardware
        g = cfg['geometry']
        servos = cfg['servos']
        self._L1 = float(g['horn_length_mm'])
        self._L2 = float(g['upper_link_length_mm'])
        self._branch = float(g['ik_branch'])
        self._margin = float(g.get('ik_margin', 0.98))
        self._H = float(g['nominal_heave_mm'])
        rb, rp = float(g['base_radius_mm']), float(g['platform_radius_mm'])
        self._act = []
        for i, beta in enumerate(g['radial_angles_deg']):
            ex, ey = math.cos(math.radians(beta)), math.sin(math.radians(beta))
            s = servos[i]
            self._act.append((ex, ey, rb * ex, rb * ey, rp * ex, rp * ey,
                              float(s['motor_min_deg']), float(s['motor_max_deg']),
                              float(s['servo_at_neutral_deg']), float(s['servo_direction']),
                              float(s['servo_min_deg']), float(s['servo_max_deg']),
                              int(s['channel'])))
        self.channels = [a[12] for a in self._act]
        self.q0 = self._ik(0.0, 0.0)
        self.neutral = [a[8] for a in self._act]
        self.angles = list(self.neutral)        # target angle after rate limiter
        self.sent_angles = list(self.neutral)   # last angle actually sent over I2C
        self.last_pose = dict(kin.HOME_POSE)
        self.last_error = ''
        # No automatic motion on initialization: only command on tick.

    def _ik(self, roll_deg, pitch_deg):
        """Horn angles (deg). Same equations and checks as kin.inverse_kinematics."""
        r, p = math.radians(roll_deg), math.radians(pitch_deg)
        cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
        L1, L2, H = self._L1, self._L2, self._H
        q = []
        for i, (ex, ey, bx, by, px, py, qmin, qmax, *_rest) in enumerate(self._act):
            # P = Ry(p) Rx(r) [px, py, 0] + [0, 0, H]
            dx = cp * px + sp * sr * py - bx
            dy = cr * py - by
            dz = -sp * px + cp * sr * py + H
            A = dx * ex + dy * ey
            K = (dx * dx + dy * dy + dz * dz + L1 * L1 - L2 * L2) / (2 * L1)
            rho = math.hypot(A, dz)
            if rho < 1e-9 or abs(K) > self._margin * rho:
                raise ValueError(f'Actuator {i}: pose unreachable or near singularity')
            ang = math.atan2(dz, A) + self._branch * math.acos(K / rho)
            qi = math.atan2(math.sin(ang), math.cos(ang))
            if math.cos(qi) <= 0:
                raise ValueError(f'Actuator {i}: horn points inward; check IK branch')
            qi = math.degrees(qi)
            if not qmin <= qi <= qmax:
                raise ValueError(f'Actuator {i}: motor angle {qi:.2f} outside configured limits')
            q.append(qi)
        return q

    def _servo_goal(self, roll_deg, pitch_deg):
        q = self._ik(roll_deg, pitch_deg)
        goal = []
        for i, a in enumerate(self._act):
            cmd = a[8] + a[9] * (q[i] - self.q0[i])
            if not a[10] <= cmd <= a[11]:
                raise ValueError(f'Servo channel {a[12]} command {cmd:.2f} outside [{a[10]}, {a[11]}]')
            goal.append(cmd)
        return goal

    def command(self, roll_deg, pitch_deg, dt):
        ik_roll = float(roll_deg) * KINEMATICS_ANGLE_FACTOR
        ik_pitch = float(pitch_deg) * KINEMATICS_ANGLE_FACTOR
        if max(abs(ik_roll), abs(ik_pitch)) > MAX_IK_TILT_DEG:
            self.last_error = 'Requested angle exceeds IK safety bound'
            return False
        try:
            goal = self._servo_goal(ik_roll, ik_pitch)
        except (ValueError, ArithmeticError) as exc:
            self.last_error = 'IK rejected pose: ' + str(exc)
            return False
        rate = MAX_SERVO_RATE_DPS * max(0.001, min(dt, 0.2))
        for i in range(3):
            step = min(rate, max(-rate, goal[i] - self.angles[i]))
            nxt = self.angles[i] + step
            if self.kit and abs(nxt - self.sent_angles[i]) >= SERVO_DEADBAND_DEG:
                self.kit.servo[self.channels[i]].angle = nxt
                self.sent_angles[i] = nxt
            self.angles[i] = nxt
        self.last_pose = {'heave_mm': 0.0, 'roll_deg': ik_roll, 'pitch_deg': ik_pitch}
        self.last_error = ''
        return True

    def close(self):
        # Don't disable PWM: that can allow platform to drop under gravity.
        if self.hardware:
            kin.save_pose(self.last_pose)


# ======================================================================
# Shared state between the control process and the GUI process
# ======================================================================
FIELDS = ('alive', 'valid', 'x', 'y', 'vx', 'vy', 'roll', 'pitch', 'ref_x', 'ref_y',
          'goal_x', 'goal_y', 'active', 'pace', 'lap', 'laps', 'running', 'stopping', 'mode',
          'fps', 'capture_fps', 'decode_ms', 'detect_ms', 'control_ms', 'loop_ms_peak',
          'age_ms', 'bias_x', 'bias_y', 'calibrated',
          # preview frame block, protected by the frame_seq seqlock
          'frame_seq', 'f_found', 'f_px', 'f_py', 'f_rad', 'f_has_mm', 'f_mmx', 'f_mmy',
          'f_ref_x', 'f_ref_y', 'f_active')


class S:
    """Indices into the shared state array, e.g. state[S.X]."""


for _i, _name in enumerate(FIELDS):
    setattr(S, _name.upper(), _i)

MODES = ('Hold', 'Point') + tuple(SHAPES)


class Controller:
    """Runs INSIDE the control process: camera, detection, estimation, control, servos.

    Nothing here touches Tk, so GUI work can never delay a control step.
    The GUI talks to it only through a command queue (in), an event queue
    (out, rare messages) and shared memory (state every frame, preview frames).
    """

    def __init__(self, state, frame_buf, events, hardware=False, color='orange'):
        sys.setswitchinterval(GIL_SWITCH_INTERVAL_S)
        self.st = state
        self.frame_buf = frame_buf
        self.events = events
        self.cfg = kin.load_config()
        self.detector = BallDetector(self.cfg, color_mode=color)
        self.driver = PlatformDriver(self.cfg, hardware)
        self.pid_x, self.pid_y = AxisPID(), AxisPID()
        self.estimator = BallEstimator()
        center, radius = self.detector.calibration_circle_mm()
        if np.linalg.norm(center) > 1e-3:
            raise ValueError('Calibration circle is off-centre from platform XY origin; '
                             'trajectory planner expects concentric radial attachment points')
        self.planner = TrajectoryPlanner(radius - BALL_EDGE_MARGIN_MM,
                                         DEFAULT_TRAJECTORY_SPEED_MM_S,
                                         DEFAULT_TRAJECTORY_ACCEL_MM_S2)
        self.trim = [0.0, 0.0]
        self.latency_comp_s = DEFAULT_LATENCY_COMP_MS / 1000.0
        self.active = False
        self.position = None
        self.roll = self.pitch = 0.0
        self.sp = [0.0, 0.0]
        self._u = (0.0, 0.0)
        self.last_seen = -1e9
        self.last_step = None
        self.fps = self.capture_fps = 0.0
        self.decode_ms = self.detect_ms = self.control_ms = 0.0
        self._peak = self._peak_shown = 0.0
        self._peak_t0 = time.monotonic()
        self.preview_enabled = True
        self._last_preview = 0.0
        self._sent_path = None
        self._sent_driver_error = ''
        self._running = False
        self._frame_cond = threading.Condition()
        self._want_frame = threading.Event()
        self._frame = None
        self._frame_stamp = 0.0
        self._frame_seq = 0
        self._last_capture_stamp = None
        self.cap, self.camera_info = self._open_camera()

    # ------------------------------------------------------------ camera
    def _open_camera(self):
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()          # some Pi camera stacks do not expose V4L2
            cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            raise RuntimeError('Could not open camera')
        # Order matters on V4L2: format, then size, then rate.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.detector.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.detector.size[1])
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if CAMERA_AUTO_EXPOSURE is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAMERA_AUTO_EXPOSURE)
        if CAMERA_EXPOSURE is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = ''.join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        info = (f'Camera {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
                f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):g} FPS, '
                f'format {fourcc!r}, exposure {cap.get(cv2.CAP_PROP_EXPOSURE):g} '
                f'(auto {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g})')
        print(info, flush=True)
        return cap, info

    def _capture_loop(self):
        """grab() every frame to keep the driver queue empty; decode only on demand."""
        while self._running:
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            stamp = time.monotonic()
            if self._last_capture_stamp is not None:
                d = max(stamp - self._last_capture_stamp, 1e-3)
                self.capture_fps = 0.9 * self.capture_fps + 0.1 / d
            self._last_capture_stamp = stamp
            if not self._want_frame.is_set():
                continue                     # control busy: skip decode entirely
            ok, frame = self.cap.retrieve()
            if not ok:
                continue
            self.decode_ms = 0.9 * self.decode_ms + 0.1 * (time.monotonic() - stamp) * 1000.0
            self._want_frame.clear()
            with self._frame_cond:
                self._frame = frame
                self._frame_stamp = stamp
                self._frame_seq += 1
                self._frame_cond.notify()

    # ------------------------------------------------------------ commands
    def _start_xy(self):
        return self.position.copy() if self.position is not None else np.array(self.sp)

    def set_active(self, active):
        if active and self.detector.affine is None:
            raise RuntimeError('Calibrate platform X/Y points before enabling PID')
        self.active = bool(active)
        self.reset_pid()
        if not active:
            self.planner.stop(immediate=True)

    def reset_pid(self):
        self.pid_x.reset()
        self.pid_y.reset()

    def handle(self, cmd, *args):
        if cmd == 'param':
            key, val = args
            if key in ('kp', 'ki', 'kd', 'kff'):
                setattr(self.pid_x, key, float(val))
                setattr(self.pid_y, key, float(val))
            elif key == 'latency_ms':
                self.latency_comp_s = float(val) / 1000.0
            elif key == 'trim_roll':
                self.trim[0] = float(val)
            elif key == 'trim_pitch':
                self.trim[1] = float(val)
        elif cmd == 'active':
            self.set_active(args[0])
        elif cmd == 'point':
            xy, speed, accel = args
            self.planner.point(xy, self._start_xy(), speed=speed, accel=accel)
        elif cmd == 'shape':
            name, center, radius, speed, laps, accel = args
            self.planner.shape(name, center, radius, self._start_xy(),
                               speed=speed, laps=laps, accel=accel)
        elif cmd == 'stop':
            self.planner.stop()
        elif cmd == 'reset_pid':
            self.reset_pid()
        elif cmd == 'reset_mapping':
            self.set_active(False)
            self.detector.reset_mapping()
            self.estimator.reset()
        elif cmd == 'reload_mapping':
            self.detector.load_mapping()
            self.estimator.reset()
            self.position = None
            self.reset_pid()
        elif cmd == 'preview':
            self.preview_enabled = bool(args[0])
        elif cmd == 'quit':
            self._running = False
        else:
            raise ValueError(f'Unknown command {cmd!r}')

    def _drain_commands(self, cmd_q):
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                return
            try:
                self.handle(*cmd)
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                self.events.put(('error', cmd[0], str(exc)))

    # ------------------------------------------------------------ control
    def _go_level(self, dt):
        self.reset_pid()
        self._u = (0.0, 0.0)
        self.roll, self.pitch = float(self.trim[0]), float(self.trim[1])
        self.driver.command(self.roll, self.pitch, dt)

    def step(self, frame, capture_stamp):
        start = time.monotonic()
        now = capture_stamp
        dt = 1 / 30 if self.last_step is None else min(max(now - self.last_step, 0.001), 0.2)
        self.last_step = now
        self.fps = 0.92 * self.fps + 0.08 / dt

        result = self.detector.detect(frame, raw_frame=True)
        t_detect = time.monotonic()
        measurement = result['xy_mm']

        est = self.estimator
        g = -BALL_ACCEL_PER_DEG * MODEL_INPUT_GAIN
        accel_in = (g * self._u[0], g * self._u[1])
        est.predict(dt, accel_in)
        accepted = False
        if measurement is not None:
            if not est.valid:
                est.reset(measurement)
                accepted = True
            elif est.distance_to(measurement) <= MAX_POSITION_JUMP_MM:
                est.correct(measurement)
                accepted = True
            if accepted:
                self.last_seen = now
        lost = (not est.valid) or (now - self.last_seen > LOST_TIMEOUT_S)
        if lost:
            est.reset()
            self.position = None
        else:
            self.position = est.p

        planner = self.planner
        if self.active and not lost and accepted:
            # Predict the ball to the time this command will actually act.
            horizon = self.latency_comp_s + (time.monotonic() - capture_stamp)
            pred_p, pred_v = est.predicted(horizon, accel_in)
            ref_p, ref_v, ref_a = planner.update(now, pred_p, horizon)
            self.sp = [float(planner.reference[0]), float(planner.reference[1])]
            b = est._b if BIAS_COMPENSATION else (0.0, 0.0)
            ux = self.pid_x.update(pred_p[0], pred_v[0], ref_p[0], ref_v[0], ref_a[0], dt, b[0])
            uy = self.pid_y.update(pred_p[1], pred_v[1], ref_p[1], ref_v[1], ref_a[1], dt, b[1])
            self._u = (ux, uy)
            # x error drives pitch, y error drives roll.
            roll = ROLL_SIGN * uy + self.trim[0]
            pitch = PITCH_SIGN * ux + self.trim[1]
            self.roll = min(MAX_PHYSICAL_TILT_DEG, max(-MAX_PHYSICAL_TILT_DEG, roll))
            self.pitch = min(MAX_PHYSICAL_TILT_DEG, max(-MAX_PHYSICAL_TILT_DEG, pitch))
            self.driver.command(self.roll, self.pitch, dt)
        elif self.active and not lost:
            # One missed/rejected frame: keep the last command, don't snap level.
            self.driver.command(self.roll, self.pitch, dt)
        else:
            planner.update(now, None)          # freeze the path clock
            self.sp = [float(planner.reference[0]), float(planner.reference[1])]
            self._go_level(dt)

        end = time.monotonic()
        self.detect_ms = 0.9 * self.detect_ms + 0.1 * (t_detect - start) * 1000.0
        self.control_ms = 0.9 * self.control_ms + 0.1 * (end - t_detect) * 1000.0
        total = (end - start) * 1000.0
        self._peak = max(self._peak, total)
        if end - self._peak_t0 >= 2.0:
            self._peak_shown, self._peak, self._peak_t0 = self._peak, 0.0, end
        self._publish(result, capture_stamp, start)

    def _publish(self, result, capture_stamp, start):
        st = self.st
        p = self.position
        st[S.VALID] = 0.0 if p is None else 1.0
        if p is not None:
            st[S.X], st[S.Y] = float(p[0]), float(p[1])
            st[S.VX], st[S.VY] = self.estimator._v
        st[S.ROLL], st[S.PITCH] = self.roll, self.pitch
        st[S.REF_X], st[S.REF_Y] = self.sp
        pl = self.planner
        st[S.GOAL_X], st[S.GOAL_Y] = float(pl.target[0]), float(pl.target[1])
        st[S.ACTIVE] = 1.0 if self.active else 0.0
        st[S.PACE] = pl.pace
        st[S.LAP], st[S.LAPS] = pl.lap, pl.laps
        st[S.RUNNING] = 1.0 if pl.running else 0.0
        st[S.STOPPING] = 1.0 if pl.stopping else 0.0
        st[S.MODE] = MODES.index(pl.mode) if pl.mode in MODES else 0
        st[S.FPS], st[S.CAPTURE_FPS] = self.fps, self.capture_fps
        st[S.DECODE_MS], st[S.DETECT_MS], st[S.CONTROL_MS] = self.decode_ms, self.detect_ms, self.control_ms
        st[S.LOOP_MS_PEAK] = self._peak_shown
        st[S.AGE_MS] = (time.monotonic() - capture_stamp) * 1000.0
        st[S.BIAS_X], st[S.BIAS_Y] = self.estimator._b
        st[S.CALIBRATED] = 0.0 if self.detector.affine is None else 1.0
        # Rare messages only when something actually changed.
        if pl.display_path is not self._sent_path:
            self._sent_path = pl.display_path
            self.events.put(('path', np.array(pl.display_path)))
        if self.driver.last_error != self._sent_driver_error:
            self._sent_driver_error = self.driver.last_error
            self.events.put(('driver_error', self.driver.last_error))
        # Preview: a ~1 MB memcpy a few times a second; all drawing is done by the GUI.
        if (self.preview_enabled and self.frame_buf is not None
                and start - self._last_preview >= 1.0 / PREVIEW_HZ):
            self._last_preview = start
            seq = st[S.FRAME_SEQ]
            st[S.FRAME_SEQ] = seq + 1                    # odd: writing
            np.copyto(self.frame_buf, self._frame_for_preview)
            found = result['found']
            st[S.F_FOUND] = 1.0 if found else 0.0
            if found:
                st[S.F_PX], st[S.F_PY] = result['pixel']
                st[S.F_RAD] = result['radius_px']
                mm = result['xy_mm']
                st[S.F_HAS_MM] = 0.0 if mm is None else 1.0
                if mm is not None:
                    st[S.F_MMX], st[S.F_MMY] = mm
            st[S.F_REF_X], st[S.F_REF_Y] = self.sp
            st[S.F_ACTIVE] = st[S.ACTIVE]
            st[S.FRAME_SEQ] = seq + 2                    # even: complete

    # ------------------------------------------------------------ main loop
    def run(self, cmd_q):
        self._running = True
        cap_thread = threading.Thread(target=self._capture_loop, daemon=True, name='camera')
        cap_thread.start()
        self.st[S.ALIVE] = 1.0
        parent = mp.parent_process()
        last_seq = 0
        last_parent_check = time.monotonic()
        try:
            while self._running:
                self._drain_commands(cmd_q)
                self._want_frame.set()
                with self._frame_cond:
                    self._frame_cond.wait_for(lambda: self._frame_seq != last_seq,
                                              timeout=FRAME_WAIT_S)
                    seq, frame, stamp = self._frame_seq, self._frame, self._frame_stamp
                now = time.monotonic()
                if now - last_parent_check > 1.0:
                    last_parent_check = now
                    if parent is not None and not parent.is_alive():
                        break                       # GUI died: level and exit
                if seq == last_seq or frame is None:
                    # Camera stalled. Hold briefly, then level once the ball is "lost".
                    if now - self.last_seen > LOST_TIMEOUT_S:
                        self.position = None
                        self.estimator.reset()
                        self._go_level(FRAME_WAIT_S)
                        self.st[S.VALID] = 0.0
                    continue
                last_seq = seq
                self._frame_for_preview = frame
                self.step(frame, stamp)
        finally:
            self._running = False
            cap_thread.join(timeout=1.0)

    def close(self):
        self._running = False
        self.active = False
        try:
            self.driver.command(float(self.trim[0]), float(self.trim[1]), 0.2)
            self.driver.close()
        finally:
            self.cap.release()


def _pin_to_cpus(cpus):
    if PIN_CPUS:
        try:
            available = os.sched_getaffinity(0)
            chosen = set(cpus) & available
            if chosen and len(available) >= 4:
                os.sched_setaffinity(0, chosen)
        except (AttributeError, OSError):
            pass


def control_process_main(state, frame_raw, shape, cmd_q, events, hardware, color):
    """Entry point of the control process."""
    cv2.setNumThreads(1)
    _pin_to_cpus(CONTROL_CPUS)
    frame_buf = np.frombuffer(frame_raw, dtype=np.uint8).reshape(shape)
    ctrl = None
    try:
        ctrl = Controller(state, frame_buf, events, hardware, color)
        events.put(('ready', ctrl.camera_info))
        ctrl.run(cmd_q)
    except Exception as exc:
        traceback.print_exc()
        events.put(('fatal', f'{type(exc).__name__}: {exc}'))
    finally:
        try:
            if ctrl is not None:
                ctrl.close()
        finally:
            state[S.ALIVE] = 0.0


class BalancerClient:
    """GUI-side handle to the control process."""

    def __init__(self, hardware=False, color='orange'):
        ctx = mp.get_context('spawn')
        self.cfg = kin.load_config()
        self.hardware = hardware
        # A detector copy in THIS process for preview drawing and calibration clicks.
        self.detector = BallDetector(self.cfg, color_mode=color)
        center, radius = self.detector.calibration_circle_mm()
        self.surface_center_mm, self.surface_radius_mm = center, radius
        self.limit = radius - BALL_EDGE_MARGIN_MM
        w, h = self.detector.size
        self.frame_shape = (h, w, 3)
        self.state = ctx.RawArray('d', len(FIELDS))
        self._frame_raw = ctx.RawArray('B', h * w * 3)
        self._frame = np.frombuffer(self._frame_raw, dtype=np.uint8).reshape(self.frame_shape)
        self.cmd_q = ctx.Queue()
        self.events = ctx.Queue()
        self.proc = ctx.Process(target=control_process_main, name='balancer-control',
                                args=(self.state, self._frame_raw, self.frame_shape,
                                      self.cmd_q, self.events, hardware, color))
        self.proc.start()
        _pin_to_cpus(GUI_CPUS)
        self._last_frame_seq = 0.0

    def send(self, *cmd):
        self.cmd_q.put(cmd)

    def __getitem__(self, idx):
        return self.state[idx]

    def poll_events(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def alive(self):
        return self.proc.is_alive()

    def read_frame(self):
        """Newest preview frame (copy) + its detection result, or None if nothing new."""
        st = self.state
        seq = st[S.FRAME_SEQ]
        if seq == self._last_frame_seq or int(seq) % 2:
            return None
        frame = self._frame.copy()
        found = st[S.F_FOUND] > 0.5
        result = {'found': found,
                  'pixel': (st[S.F_PX], st[S.F_PY]) if found else None,
                  'radius_px': st[S.F_RAD] if found else None,
                  'xy_mm': (st[S.F_MMX], st[S.F_MMY]) if found and st[S.F_HAS_MM] > 0.5 else None}
        ref = (st[S.F_REF_X], st[S.F_REF_Y])
        active = st[S.F_ACTIVE] > 0.5
        if st[S.FRAME_SEQ] != seq:
            return None                      # overwritten while copying; skip
        self._last_frame_seq = seq
        return frame, result, ref, active

    def close(self, timeout=5.0):
        if self.proc.is_alive():
            self.send('quit')
            self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(1.0)


def render_preview(detector, frame, result, ref, active, path, scale):
    """Undistort + draw overlays for display (runs in the GUI process)."""
    img = detector.undistort(frame, scale)
    detector.draw(img, result, ref, path_mm=path, scale=scale)
    k = max(scale, 0.6)
    thick = 2 if scale >= 0.75 else 1
    cv2.putText(img, 'PID ON' if active else 'PAUSED / LEVEL', (8, int(26 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 220, 0) if active else (0, 180, 255), thick)
    if detector.affine is None:
        cv2.putText(img, 'CALIBRATE 3 PLATFORM POINTS', (8, int(54 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 255), thick)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true', help='Enable real PCA9685 servo output')
    parser.add_argument('--headless', action='store_true', help='OpenCV monitor without Tk GUI; PID disabled')
    parser.add_argument('--color', choices=['orange', 'white'], default='orange')
    args = parser.parse_args()
    client = BalancerClient(hardware=args.hardware, color=args.color)
    try:
        if args.headless:
            print('Headless monitoring only; PID remains disabled. Press q to exit.')
            path = None
            while True:
                for ev in client.poll_events():
                    if ev[0] == 'fatal':
                        raise RuntimeError(ev[1])
                    if ev[0] == 'path':
                        path = ev[1]
                got = client.read_frame()
                if got is not None:
                    cv2.imshow('Stewart Ball Camera',
                               render_preview(client.detector, *got, path, 1.0))
                if cv2.waitKey(30) & 0xff == ord('q') or not client.alive():
                    break
        else:
            from stewart_balancer_gui import BalancerGUI
            BalancerGUI(client).run()
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()