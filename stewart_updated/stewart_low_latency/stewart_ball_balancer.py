#!/usr/bin/env python3
"""2-axis camera ball balancer for the three-servo Stewart-style platform.

Run: python stewart_ball_balancer.py              (GUI, dry-run by default)
     python stewart_ball_balancer.py --hardware   (enable PCA9685)
     python stewart_ball_balancer.py --headless   (camera-only dry-run)

First calibrate camera-to-platform XY with the GUI's Calibrate 3 points button.
Keep the camera fixed. Verify movement sign with LOW gains before balancing.

Latency pipeline (what changed)
  * Capture thread only grab()s (no JPEG decode); a frame is decoded only when
    the control thread is ready, so control always gets the newest frame.
  * A Kalman filter estimates ball position AND velocity (much less lag than
    the old EMA velocity), and the state is predicted forward by the measured
    pipeline age + a tunable latency-compensation time before control.
  * Servo slew limit raised from 30 to 240 deg/s; the old limit alone added
    ~0.5 s of actuator lag.
  * A missed frame holds the last command instead of snapping the plate level.
  * Preview is rendered event-driven at up to 20 FPS instead of being up to
    two 200 ms timer periods old.

Control law (per axis, output in physical degrees)
  u = Kp*(p - p_ref) + Kd*(v - v_ref) + Ki*integral + (b - Kff*a_ref)/G
where G = ball acceleration per degree of tilt and b is the plate-bias
acceleration estimated by the Kalman filter.  Tracking velocity error
(instead of damping velocity to zero) is what lets the ball follow a shape.
"""
import argparse
from collections import deque
import csv
import math
from pathlib import Path
import time
import threading

import cv2
import numpy as np

# Avoid OpenCV spawning extra worker pools on CPU-limited Raspberry Pi systems.
cv2.setNumThreads(1)

import stewart_kinematics as kin
from stewart_ball_detection import BallDetector
from trajectory_planner import TrajectoryPlanner

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
PREVIEW_HZ = 20.0
GUI_HZ = 25.0

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
    Both axes share dt and noise, so one 3x3 covariance serves both.
    """

    def __init__(self, accel_sigma=ESTIMATOR_ACCEL_SIGMA, meas_sigma=ESTIMATOR_MEAS_SIGMA_MM,
                 bias_sigma=BIAS_RANDOM_WALK):
        self.q = accel_sigma ** 2
        self.r = meas_sigma ** 2
        self.qb = bias_sigma ** 2
        self.b = np.zeros(2)
        self.reset()

    def reset(self, z=None, keep_bias=True):
        """Reset position/velocity. The bias is a plate property, so it is kept."""
        self.valid = z is not None
        self.p = np.zeros(2) if z is None else np.asarray(z, float).copy()
        self.v = np.zeros(2)
        if not keep_bias:
            self.b = np.zeros(2)
        b_var = 40.0 ** 2 if not keep_bias or not np.any(self.b) else 15.0 ** 2
        self.P = np.diag([self.r, 200.0 ** 2, b_var])

    def predict(self, dt, accel):
        if not self.valid:
            return
        a = accel + self.b
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        F = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        g = np.array([0.5 * dt * dt, dt, 0.0])
        Q = self.q * np.outer(g, g)
        Q[2, 2] += self.qb * dt
        self.P = F @ self.P @ F.T + Q

    def correct(self, z):
        S = self.P[0, 0] + self.r
        K = self.P[:, 0] / S
        y = np.asarray(z, float) - self.p
        self.p = self.p + K[0] * y
        self.v = self.v + K[1] * y
        self.b = np.clip(self.b + K[2] * y, -MAX_BIAS_ACCEL, MAX_BIAS_ACCEL)
        self.P = self.P - np.outer(K, self.P[0, :])

    def predicted(self, horizon, accel):
        h = horizon
        a = accel + self.b
        return self.p + self.v * h + 0.5 * a * h * h, self.v + a * h


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
        i_try = float(np.clip(self.integral + e * dt, -MAX_INTEGRAL_MM_S, MAX_INTEGRAL_MM_S))
        u = self.kp * e + self.kd * ev + self.ki * i_try + u_ff
        if abs(u) < MAX_PHYSICAL_TILT_DEG:          # conditional anti-windup
            self.integral = i_try
        return float(np.clip(u, -MAX_PHYSICAL_TILT_DEG, MAX_PHYSICAL_TILT_DEG))


class PlatformDriver:
    def __init__(self, cfg, hardware=False):
        self.cfg = cfg
        self.kit = kin.make_kit(cfg) if hardware else None
        self.hardware = hardware
        servos = cfg['servos']
        # Neutral IK is cached: the old path re-solved it on every command.
        self.q0 = kin.inverse_kinematics(kin.HOME_POSE, cfg)
        self.neutral = np.array([float(s['servo_at_neutral_deg']) for s in servos])
        self.direction = np.array([int(s['servo_direction']) for s in servos], float)
        self.lo = np.array([float(s['servo_min_deg']) for s in servos])
        self.hi = np.array([float(s['servo_max_deg']) for s in servos])
        self.channels = [int(s['channel']) for s in servos]
        self.angles = self.neutral.copy()        # target angle after rate limiter
        self.sent_angles = self.neutral.copy()   # last angle actually sent over I2C
        self.last_pose = dict(kin.HOME_POSE)
        self.last_error = ''
        # No automatic motion on initialization: only command on tick.

    def _servo_goal(self, pose):
        q = kin.inverse_kinematics(pose, self.cfg)
        goal = self.neutral + self.direction * (q - self.q0)
        bad = (goal < self.lo) | (goal > self.hi)
        if np.any(bad):
            i = int(np.argmax(bad))
            raise ValueError(f'Servo channel {self.channels[i]} command {goal[i]:.2f} '
                             f'outside [{self.lo[i]}, {self.hi[i]}]')
        return goal

    def command(self, roll_deg, pitch_deg, dt):
        pose = {'heave_mm': 0.0,
                'roll_deg': float(roll_deg) * KINEMATICS_ANGLE_FACTOR,
                'pitch_deg': float(pitch_deg) * KINEMATICS_ANGLE_FACTOR}
        if max(abs(pose['roll_deg']), abs(pose['pitch_deg'])) > MAX_IK_TILT_DEG:
            self.last_error = 'Requested angle exceeds IK safety bound'
            return False
        try:
            goal = self._servo_goal(pose)
        except (ValueError, ArithmeticError) as exc:
            self.last_error = 'IK rejected pose: ' + str(exc)
            return False
        rate = MAX_SERVO_RATE_DPS * max(0.001, min(dt, 0.2))
        next_angles = self.angles + np.clip(goal - self.angles, -rate, rate)
        if self.kit:
            for i, ch in enumerate(self.channels):
                if abs(next_angles[i] - self.sent_angles[i]) >= SERVO_DEADBAND_DEG:
                    self.kit.servo[ch].angle = float(next_angles[i])
                    self.sent_angles[i] = next_angles[i]
        self.angles = next_angles
        self.last_pose = pose
        self.last_error = ''
        return True

    def close(self):
        # Don't disable PWM: that can allow platform to drop under gravity.
        if self.hardware:
            kin.save_pose(self.last_pose)


class StewartBalancer:
    def __init__(self, hardware=False, color='orange'):
        self.cfg = kin.load_config()
        self.detector = BallDetector(self.cfg, color_mode=color)
        self.driver = PlatformDriver(self.cfg, hardware)
        self.pid_x, self.pid_y = AxisPID(), AxisPID()
        self.estimator = BallEstimator()
        self.sp = np.zeros(2, float)          # displayed reference
        self.trajectory_lock = threading.RLock()
        self.surface_center_mm, self.surface_radius_mm = self.detector.calibration_circle_mm()
        if np.linalg.norm(self.surface_center_mm) > 1e-3:
            raise ValueError('Calibration circle is off-centre from platform XY origin; '
                             'trajectory planner expects concentric radial attachment points')
        self.planner = TrajectoryPlanner(self.surface_radius_mm - BALL_EDGE_MARGIN_MM,
                                         DEFAULT_TRAJECTORY_SPEED_MM_S,
                                         DEFAULT_TRAJECTORY_ACCEL_MM_S2)
        self.trim = np.zeros(2, float)        # physical roll, pitch degrees
        self.latency_comp_s = DEFAULT_LATENCY_COMP_MS / 1000.0
        self.active = False                   # disarmed by default
        self.position = None                  # filtered (not predicted) position
        self.velocity = np.zeros(2)
        self.raw = None
        self.roll = 0.0
        self.pitch = 0.0
        self._u = np.zeros(2)                 # last axis outputs (ux, uy), deg
        self.last_seen = -1e9
        self.last_step = None
        self.fps = 0.0
        self.capture_fps = 0.0
        self.processing_ms = 0.0
        self.frame_age_ms = 0.0
        self.latest_overlay = None
        self.latest_frame_id = 0
        self.worker_error = ''
        self._frame_cond = threading.Condition()
        self._want_frame = threading.Event()
        self._frame = None
        self._frame_stamp = 0.0
        self._frame_seq = 0
        self._overlay_cond = threading.Condition()
        self._overlay_src = None
        self._last_overlay_time = 0.0
        self._running = False
        self._threads = []
        self._last_capture_stamp = None
        self.history = deque(maxlen=1500)
        self.t0 = time.monotonic()
        self.cap = self._open_camera()
        logs = HERE / 'logs'
        logs.mkdir(exist_ok=True)
        self.log_path = logs / time.strftime('stewart_%Y%m%d_%H%M%S.csv')
        # Large buffer: logging every frame costs microseconds, not disk waits.
        self.csv_file = self.log_path.open('w', newline='', buffering=1 << 20)
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['time_s', 'raw_x_mm', 'raw_y_mm', 'x_mm', 'y_mm',
                              'vx_mm_s', 'vy_mm_s', 'ref_x_mm', 'ref_y_mm',
                              'ref_vx_mm_s', 'ref_vy_mm_s', 'path_pace',
                              'physical_roll_deg', 'physical_pitch_deg', 'enabled',
                              'frame_age_ms', 'servo0_deg', 'servo1_deg', 'servo2_deg'])

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
        print('Camera mode:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x',
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), '@', cap.get(cv2.CAP_PROP_FPS),
              'FPS, format', repr(fourcc), '| exposure', cap.get(cv2.CAP_PROP_EXPOSURE),
              'auto', cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        return cap

    # ------------------------------------------------------------ trajectory API
    def trajectory_snapshot(self):
        with self.trajectory_lock:
            p = self.planner
            return (p.display_path, p.reference.copy(), p.target.copy(),
                    p.mode, p.running, p.status_text())

    def _start_xy(self):
        return self.position.copy() if self.position is not None else self.sp.copy()

    def go_to_point(self, xy, speed=DEFAULT_TRAJECTORY_SPEED_MM_S,
                    accel=DEFAULT_TRAJECTORY_ACCEL_MM_S2):
        with self.trajectory_lock:
            self.planner.point(xy, self._start_xy(), speed=speed, accel=accel)

    def trace_shape(self, name, center, radius, speed=DEFAULT_TRAJECTORY_SPEED_MM_S,
                    laps=1, accel=DEFAULT_TRAJECTORY_ACCEL_MM_S2):
        with self.trajectory_lock:
            self.planner.shape(name, center, radius, self._start_xy(),
                               speed=speed, laps=laps, accel=accel)

    def stop_trajectory(self, immediate=False):
        with self.trajectory_lock:
            self.planner.stop(immediate=immediate)

    def reset_pid(self):
        self.pid_x.reset()
        self.pid_y.reset()

    def set_active(self, active):
        if active and self.detector.affine is None:
            raise RuntimeError('Calibrate platform X/Y points before enabling PID')
        self.active = bool(active)
        self.reset_pid()
        if not active:
            self.stop_trajectory(immediate=True)

    # ------------------------------------------------------------ control
    def _go_level(self, dt):
        self.reset_pid()
        self._u[:] = 0.0
        self.roll, self.pitch = float(self.trim[0]), float(self.trim[1])
        self.driver.command(self.roll, self.pitch, dt)

    def step(self, frame, capture_stamp):
        """One control iteration, called ONLY by the control worker."""
        start = time.monotonic()
        now = capture_stamp
        dt = 1 / 30 if self.last_step is None else min(max(now - self.last_step, 0.001), 0.2)
        self.last_step = now
        self.fps = 0.92 * self.fps + 0.08 / dt

        result = self.detector.detect(frame, raw_frame=True)
        measurement = result['xy_mm']
        self.raw = measurement

        est = self.estimator
        accel_in = -BALL_ACCEL_PER_DEG * MODEL_INPUT_GAIN * self._u
        est.predict(dt, accel_in)
        accepted = False
        if measurement is not None:
            z = np.asarray(measurement, float)
            if not est.valid:
                est.reset(z)
                accepted = True
            elif np.linalg.norm(z - est.p) <= MAX_POSITION_JUMP_MM:
                est.correct(z)
                accepted = True
            if accepted:
                self.last_seen = now
        lost = (not est.valid) or (now - self.last_seen > LOST_TIMEOUT_S)
        if lost:
            est.reset()
            self.position = None
            self.velocity = np.zeros(2)
        else:
            self.position = est.p.copy()
            self.velocity = est.v.copy()

        ref_v = np.zeros(2)
        if self.active and not lost and accepted:
            # Predict the ball to the time this command will actually act.
            horizon = self.latency_comp_s + (time.monotonic() - capture_stamp)
            pred_p, pred_v = est.predicted(horizon, accel_in)
            with self.trajectory_lock:
                ref_p, ref_v, ref_a = self.planner.update(now, pred_p, horizon)
                self.sp[:] = self.planner.reference
            bias = est.b if BIAS_COMPENSATION else np.zeros(2)
            ux = self.pid_x.update(pred_p[0], pred_v[0], ref_p[0], ref_v[0], ref_a[0], dt, bias[0])
            uy = self.pid_y.update(pred_p[1], pred_v[1], ref_p[1], ref_v[1], ref_a[1], dt, bias[1])
            self._u[:] = (ux, uy)
            # x error drives pitch, y error drives roll.
            roll = ROLL_SIGN * uy + self.trim[0]
            pitch = PITCH_SIGN * ux + self.trim[1]
            self.roll = float(np.clip(roll, -MAX_PHYSICAL_TILT_DEG, MAX_PHYSICAL_TILT_DEG))
            self.pitch = float(np.clip(pitch, -MAX_PHYSICAL_TILT_DEG, MAX_PHYSICAL_TILT_DEG))
            self.driver.command(self.roll, self.pitch, dt)
        elif self.active and not lost:
            # One missed/rejected frame: keep the last command, don't snap level.
            self.driver.command(self.roll, self.pitch, dt)
        else:
            with self.trajectory_lock:
                self.planner.update(now, None)       # freeze the path clock
                self.sp[:] = self.planner.reference
            self._go_level(dt)

        t = now - self.t0
        pos = self.position
        self.history.append((t, pos[0] if pos is not None else None,
                             pos[1] if pos is not None else None,
                             self.roll, self.pitch, float(self.sp[0]), float(self.sp[1])))
        while self.history and t - self.history[0][0] > 10:
            self.history.popleft()
        age_ms = (time.monotonic() - capture_stamp) * 1000.0
        self.writer.writerow([
            round(t, 4), *(np.round(measurement, 3) if measurement is not None else ('', '')),
            *(np.round(pos, 3) if pos is not None else ('', '')),
            round(float(self.velocity[0]), 2), round(float(self.velocity[1]), 2),
            round(float(self.sp[0]), 3), round(float(self.sp[1]), 3),
            round(float(ref_v[0]), 2), round(float(ref_v[1]), 2), round(self.planner.pace, 3),
            round(self.roll, 3), round(self.pitch, 3), int(self.active), round(age_ms, 1),
            *np.round(self.driver.angles, 2)])

        if start - self._last_overlay_time >= 1.0 / PREVIEW_HZ:
            result.pop('mask', None)
            with self._overlay_cond:
                self._overlay_src = (frame, result, tuple(self.sp), self.planner.display_path)
                self._overlay_cond.notify()
            self._last_overlay_time = start

        self.processing_ms = (time.monotonic() - start) * 1000.0
        self.frame_age_ms = (time.monotonic() - capture_stamp) * 1000.0

    # ------------------------------------------------------------ workers
    def _overlay_loop(self):
        """Event-driven: renders as soon as the control loop hands over a frame."""
        while self._running:
            with self._overlay_cond:
                self._overlay_cond.wait_for(
                    lambda: self._overlay_src is not None or not self._running, timeout=0.5)
                src, self._overlay_src = self._overlay_src, None
            if src is None:
                continue
            frame, result, sp, path = src
            try:
                corrected = self.detector.undistort(frame)
                overlay = self.detector.draw(corrected, result, sp, path_mm=path)
                mode = 'PID ON' if self.active else 'PAUSED / LEVEL'
                cv2.putText(overlay, mode, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7,
                            (0, 220, 0) if self.active else (0, 180, 255), 2)
                if self.detector.affine is None:
                    cv2.putText(overlay, 'CALIBRATE 3 PLATFORM POINTS', (12, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
                self.latest_overlay = overlay
                self.latest_frame_id += 1
            except Exception as exc:  # preview must never stop control
                print('Preview error:', exc)

    def start(self):
        """Capture and control are independent of Tkinter redraws."""
        if self._running:
            return
        self._running = True
        self._threads = [threading.Thread(target=fn, daemon=True, name=name)
                         for fn, name in ((self._capture_loop, 'camera'),
                                          (self._control_loop, 'control'),
                                          (self._overlay_loop, 'overlay'))]
        for th in self._threads:
            th.start()

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
            self._want_frame.clear()
            with self._frame_cond:
                self._frame = frame
                self._frame_stamp = stamp
                self._frame_seq += 1
                self._frame_cond.notify()

    def _control_loop(self):
        last_seq = 0
        while self._running:
            self._want_frame.set()
            with self._frame_cond:
                self._frame_cond.wait_for(
                    lambda: not self._running or self._frame_seq != last_seq,
                    timeout=FRAME_WAIT_S)
                if not self._running:
                    break
                seq, frame, stamp = self._frame_seq, self._frame, self._frame_stamp
            try:
                if seq == last_seq or frame is None:
                    # Camera stalled. Hold briefly, then level once the ball is "lost".
                    if time.monotonic() - self.last_seen > LOST_TIMEOUT_S:
                        self.position = None
                        self.estimator.reset()
                        self._go_level(FRAME_WAIT_S)
                    continue
                last_seq = seq
                self.step(frame, stamp)
            except Exception as exc:
                self.active = False
                self.worker_error = f'{type(exc).__name__}: {exc}'
                self.position = None
                self.reset_pid()
                try:
                    self.driver.command(float(self.trim[0]), float(self.trim[1]), 0.05)
                except Exception:
                    pass
                self._running = False
                break

    def stop(self):
        self._running = False
        with self._frame_cond:
            self._frame_cond.notify_all()
        with self._overlay_cond:
            self._overlay_cond.notify_all()
        for th in self._threads:
            th.join(timeout=1.0)

    def history_snapshot(self):
        try:
            return list(self.history)
        except RuntimeError:
            return []

    def close(self):
        self.stop()
        self.active = False
        try:
            self.driver.command(float(self.trim[0]), float(self.trim[1]), 0.2)
            self.driver.close()
        finally:
            self.cap.release()
            self.csv_file.close()
            print('CSV log:', self.log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true', help='Enable real PCA9685 servo output')
    parser.add_argument('--headless', action='store_true', help='OpenCV monitor without Tk GUI; PID disabled')
    parser.add_argument('--color', choices=['orange', 'white'], default='orange')
    args = parser.parse_args()
    ctrl = StewartBalancer(hardware=args.hardware, color=args.color)
    try:
        if args.headless:
            print('Headless monitoring only; PID remains disabled. Press q to exit.')
            ctrl.start()
            last_id = -1
            while True:
                if ctrl.latest_frame_id != last_id and ctrl.latest_overlay is not None:
                    last_id = ctrl.latest_frame_id
                    cv2.imshow('Stewart Ball Camera', ctrl.latest_overlay)
                if ctrl.worker_error:
                    raise RuntimeError(ctrl.worker_error)
                if cv2.waitKey(10) & 0xff == ord('q'):
                    break
        else:
            from stewart_balancer_gui import BalancerGUI
            BalancerGUI(ctrl).run()
    finally:
        ctrl.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()