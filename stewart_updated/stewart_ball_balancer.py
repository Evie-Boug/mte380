#!/usr/bin/env python3
"""2-axis camera PID ball balancer for the three-servo Stewart-style platform.

Run: python stewart_ball_balancer.py              (GUI, dry-run by default)
     python stewart_ball_balancer.py --hardware   (enable PCA9685)
     python stewart_ball_balancer.py --headless   (camera-only dry-run)

First calibrate camera-to-platform XY with GUI's Calibrate 3 points button.
Keep the camera fixed. Verify movement sign with LOW gains before balancing.
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

import stewart_kinematics as kin
from stewart_ball_detection import BallDetector

HERE = Path(__file__).resolve().parent

# ============== TEMPORARY KINEMATICS CORRECTION ==============
# Measured physical tilt is ~1/2 of kinematics requested tilt.
# Multiply desired physical roll/pitch by 2 before calling IK.
KINEMATICS_ANGLE_FACTOR = 2.0
# =============================================================

CAMERA_ID = 0
KP, KI, KD = 0.025, 0.0, 0.015  # per-axis: deg/mm, deg/(mm*s), deg*s/mm
MAX_PHYSICAL_TILT_DEG = 4.0
MAX_IK_TILT_DEG = 10.0
MAX_SERVO_RATE_DPS = 30.0
CONTROL_HZ = 50.0  # upper target; real rate bounded by camera and CPU
GUI_HZ = 10.0
MAX_FRAME_AGE_S = 0.15  # drop delayed frames rather than commanding from stale positions
LOST_TIMEOUT_S = 0.30
MAX_INTEGRAL_MM_S = 200.0
VELOCITY_ALPHA = 0.25
# Positive physical Rx makes +Y ball roll toward -Y; positive physical Ry
# makes +X ball roll toward +X. Therefore to pull +X ball back use -pitch,
# and to pull +Y ball back use +roll. Verify experimentally at low angles.
ROLL_SIGN = +1.0
PITCH_SIGN = -1.0
# Safety check for independent axis orientation: flip sign only if required.
MAX_POSITION_JUMP_MM = 50.0


class AxisPID:
    def __init__(self, kp=KP, ki=KI, kd=KD):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.velocity = 0.0
        self.last_position = None

    def reset(self):
        self.integral = 0.0
        self.velocity = 0.0
        self.last_position = None

    def update(self, position, target, dt):
        e = position - target
        if self.last_position is not None:
            measured_velocity = (position - self.last_position) / dt
            self.velocity = (VELOCITY_ALPHA * measured_velocity
                             + (1 - VELOCITY_ALPHA) * self.velocity)
        self.last_position = position
        i_try = float(np.clip(self.integral + e * dt,
                              -MAX_INTEGRAL_MM_S, MAX_INTEGRAL_MM_S))
        u = self.kp * e + self.kd * self.velocity + self.ki * i_try
        if abs(u) < MAX_PHYSICAL_TILT_DEG:
            self.integral = i_try
        return float(np.clip(u, -MAX_PHYSICAL_TILT_DEG, MAX_PHYSICAL_TILT_DEG))


class PlatformDriver:
    def __init__(self, cfg, hardware=False):
        self.cfg = cfg
        self.kit = kin.make_kit(cfg) if hardware else None
        self.hardware = hardware
        self.neutral = np.asarray(kin.servo_commands(kin.HOME_POSE, cfg)[1], float)
        self.angles = self.neutral.copy()  # last commanded servo angles
        self.last_pose = dict(kin.HOME_POSE)
        self.last_error = ''
        # No automatic motion on initialization: only command on tick.

    def command(self, roll_deg, pitch_deg, dt):
        # Generate a physically desired tilt, then compensate before IK.
        pose = {'heave_mm': 0.0,
                'roll_deg': float(roll_deg) * KINEMATICS_ANGLE_FACTOR,
                'pitch_deg': float(pitch_deg) * KINEMATICS_ANGLE_FACTOR}
        if max(abs(pose['roll_deg']), abs(pose['pitch_deg'])) > MAX_IK_TILT_DEG:
            self.last_error = 'Requested angle exceeds IK safety bound'
            return False
        try:
            _, goal = kin.servo_commands(pose, self.cfg)
        except (ValueError, ArithmeticError) as exc:
            self.last_error = 'IK rejected pose: ' + str(exc)
            return False
        rate = MAX_SERVO_RATE_DPS * max(0.001, min(dt, 0.2))
        next_angles = self.angles + np.clip(goal - self.angles, -rate, rate)
        # All target servo bounds are checked by IK; intermediate angles between
        # current valid command and a valid target remain in servo bounds.
        if self.kit:
            for i, s in enumerate(self.cfg['servos']):
                self.kit.servo[int(s['channel'])].angle = float(next_angles[i])
        self.angles = next_angles
        self.last_pose = pose
        self.last_error = ''
        return True

    def close(self):
        # Don't disable PWM: that can allow platform to drop under gravity.
        # This stores the final REQUESTED pose, not verified encoder feedback.
        if self.hardware:
            kin.save_pose(self.last_pose)


class StewartBalancer:
    def __init__(self, hardware=False, color='orange'):
        self.cfg = kin.load_config()
        self.detector = BallDetector(self.cfg, color_mode=color)
        self.driver = PlatformDriver(self.cfg, hardware)
        self.pid_x, self.pid_y = AxisPID(), AxisPID()
        self.sp = np.zeros(2, float)
        self.trim = np.zeros(2, float)  # physical roll, pitch degrees
        self.active = False  # disarmed by default
        self.position = None
        self.raw = None
        self.roll = 0.0
        self.pitch = 0.0
        self.last_seen = 0.0
        self.last_step = None
        self.fps = 0.0
        self.capture_fps = 0.0
        self.processing_ms = 0.0
        self.frame_age_ms = 0.0
        self.latest_overlay = None
        self.latest_frame_id = 0
        self._frame_cond = threading.Condition()
        self._frame = None
        self._frame_stamp = 0.0
        self._frame_seq = 0
        self._running = False
        self._camera_thread = None
        self._control_thread = None
        self._last_capture_stamp = None
        self.worker_error = ''
        self.history = deque(maxlen=1500)
        self.t0 = time.monotonic()
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.detector.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.detector.size[1])
        # Supported by some V4L2 drivers; others ignore this property.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError('Could not open camera')
        logs = HERE / 'logs'
        logs.mkdir(exist_ok=True)
        self.log_path = logs / time.strftime('stewart_%Y%m%d_%H%M%S.csv')
        self.csv_file = self.log_path.open('w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['time_s','raw_x_mm','raw_y_mm','x_mm','y_mm',
                              'vx_mm_s','vy_mm_s','target_x_mm','target_y_mm',
                              'physical_roll_deg','physical_pitch_deg',
                              'ik_roll_deg','ik_pitch_deg','enabled',
                              'servo0_deg','servo1_deg','servo2_deg'])

    def reset_pid(self):
        self.pid_x.reset()
        self.pid_y.reset()

    def set_active(self, active):
        if active and self.detector.affine is None:
            raise RuntimeError('Calibrate platform X/Y points before enabling PID')
        self.active = bool(active)
        self.reset_pid()

    def step(self, frame, capture_stamp):
        """One control iteration, called ONLY by the control worker."""
        start = time.monotonic()
        now = capture_stamp
        dt = 1 / CONTROL_HZ if self.last_step is None else min(max(now-self.last_step, 0.001), 0.2)
        self.last_step = now
        self.fps = 0.92 * self.fps + 0.08 / dt
        corrected = self.detector.undistort(frame)
        result = self.detector.detect(corrected)
        measurement = result['xy_mm']
        self.raw = measurement
        if measurement is not None:
            if self.position is None or np.linalg.norm(np.asarray(measurement) - self.position) <= MAX_POSITION_JUMP_MM:
                self.position = np.asarray(measurement, float)
                self.last_seen = now
            else:
                measurement = None  # jump/outlier, reject
        if now - self.last_seen > LOST_TIMEOUT_S:
            if self.position is not None:
                self.reset_pid()
            self.position = None
        if self.active and self.position is not None and measurement is not None:
            ux = self.pid_x.update(self.position[0], self.sp[0], dt)
            uy = self.pid_y.update(self.position[1], self.sp[1], dt)
            # x error drives pitch, y error drives roll.
            roll = ROLL_SIGN * uy + self.trim[0]
            pitch = PITCH_SIGN * ux + self.trim[1]
        else:
            # Loss of ball or pause returns platform to trimmed level.
            if not self.active or self.position is None:
                self.reset_pid()
            roll, pitch = self.trim
        self.roll = float(np.clip(roll, -MAX_PHYSICAL_TILT_DEG, MAX_PHYSICAL_TILT_DEG))
        self.pitch = float(np.clip(pitch, -MAX_PHYSICAL_TILT_DEG, MAX_PHYSICAL_TILT_DEG))
        self.driver.command(self.roll, self.pitch, dt)
        t = now-self.t0
        self.history.append((t, *(self.position if self.position is not None else (None, None)), self.roll, self.pitch))
        while self.history and t-self.history[0][0] > 10:
            self.history.popleft()
        self.writer.writerow([round(t,4), *(measurement if measurement else ('','')),
                              *(self.position if self.position is not None else ('','')),
                              self.pid_x.velocity, self.pid_y.velocity, *self.sp,
                              self.roll, self.pitch, self.roll*KINEMATICS_ANGLE_FACTOR,
                              self.pitch*KINEMATICS_ANGLE_FACTOR, int(self.active),
                              *self.driver.angles])
        # Camera overlays only need GUI rate, not every control iteration.
        draw_overlay = (self.latest_overlay is None or
                        (time.monotonic() - getattr(self, '_last_overlay_time', 0.0)) >= 1.0 / GUI_HZ)
        if draw_overlay:
            overlay = self.detector.draw(corrected, result, tuple(self.sp))
            mode = 'PID ON' if self.active else 'PAUSED / LEVEL'
            cv2.putText(overlay, mode, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 220, 0) if self.active else (0, 180, 255), 2)
            if self.detector.affine is None:
                cv2.putText(overlay, 'CALIBRATE 3 PLATFORM POINTS', (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
            self.latest_overlay = overlay
            self.latest_frame_id += 1
            self._last_overlay_time = time.monotonic()
        self.processing_ms = (time.monotonic() - start) * 1000.0
        self.frame_age_ms = (time.monotonic() - capture_stamp) * 1000.0


    def start(self):
        """Capture and control are independent of Tkinter redraws."""
        if self._running:
            return
        self._running = True
        self._camera_thread = threading.Thread(target=self._capture_loop, daemon=True, name='camera')
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True, name='control')
        self._camera_thread.start()
        self._control_thread.start()

    def _capture_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            stamp = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            if self._last_capture_stamp is not None:
                dt = max(stamp - self._last_capture_stamp, 1e-3)
                self.capture_fps = 0.9 * self.capture_fps + 0.1 / dt
            self._last_capture_stamp = stamp
            # Replace unread frames. No growing queue of old observations.
            with self._frame_cond:
                self._frame = frame
                self._frame_stamp = stamp
                self._frame_seq += 1
                self._frame_cond.notify()

    def _control_loop(self):
        last_seq = 0
        while self._running:
            with self._frame_cond:
                self._frame_cond.wait_for(lambda: not self._running or self._frame_seq != last_seq,
                                          timeout=LOST_TIMEOUT_S / 2)
                if not self._running:
                    break
                seq = self._frame_seq
                frame = self._frame
                stamp = self._frame_stamp
            is_new = seq != last_seq
            last_seq = seq
            if not is_new or frame is None or time.monotonic()-stamp > MAX_FRAME_AGE_S:
                # No fresh camera feedback. Bring platform toward trim and disarm PID state.
                self.position = None
                self.reset_pid()
                self.roll, self.pitch = float(self.trim[0]), float(self.trim[1])
                self.driver.command(self.roll, self.pitch, 1.0 / CONTROL_HZ)
                continue
            try:
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
            # A modest upper rate limit prevents busy looping with fast cameras.
            remaining = 1.0 / CONTROL_HZ - (time.monotonic() - stamp)
            if remaining > 0:
                time.sleep(remaining)

    def stop(self):
        self._running = False
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._camera_thread:
            self._camera_thread.join(timeout=1.0)
        if self._control_thread:
            self._control_thread.join(timeout=1.0)

    def history_snapshot(self):
        # Snapshot when graph refreshes (not on control path).
        try:
            return list(self.history)
        except RuntimeError:
            return []

    def close(self):
        self.stop()
        self.active = False
        try:
            # Level at shutdown through the command rate limit. This may not
            # physically reach level if app exits immediately, so explicitly
            # request a final level pose after stopping PID.
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
    parser.add_argument('--color', choices=['orange','white'], default='orange')
    args = parser.parse_args()
    ctrl = StewartBalancer(hardware=args.hardware, color=args.color)
    try:
        if args.headless:
            print('Headless monitoring only; PID remains disabled. Press q to exit.')
            ctrl.start()
            while True:
                frame = ctrl.latest_overlay
                if frame is not None:
                    cv2.imshow('Stewart Ball Camera', frame)
                if ctrl.worker_error:
                    raise RuntimeError(ctrl.worker_error)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
                time.sleep(0.01)
        else:
            from stewart_balancer_gui import BalancerGUI
            BalancerGUI(ctrl).run()
    finally:
        ctrl.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
