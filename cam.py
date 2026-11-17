import argparse
import collections
import csv
import math
import os
import threading
import time

import cv2
import numpy as np

import kinematics
from ball_detection import BallDetector


# ============================================================
# CONTROL PARAMETERS
# ============================================================

KP = 0.05
KI = 0.00       # Start with zero!
KD = 0.04

MAX_BEAM_DEG = 11.0
MAX_RATE = 60.0             # deg/s
LOST_TIMEOUT = 0.30         # seconds

# Ball dynamics:
# x_ddot ~= -G * sin(theta)
G = 4000.0                  # mm/s^2

# Vision conversion
BEAM_LENGTH_MM = 220.0

# Camera
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Velocity filtering
VELOCITY_ALPHA = 0.25

# Integral limit
MAX_INTEGRAL = 300.0


# ============================================================
# SERVO
# ============================================================

class BeamServo:

    SERVO_GPIO = 18
    MIN_US = 600
    MAX_US = 2400

    LEVEL_DEG = 130.0
    SERVO_DIR = 1

    def __init__(self):

        import pigpio

        self.max_beam = min(
            MAX_BEAM_DEG,
            -kinematics.BEAM_MIN_DEG,
            kinematics.BEAM_MAX_DEG
        )

        self.pi = pigpio.pi()

        if not self.pi.connected:
            raise RuntimeError(
                "Can't reach pigpiod. "
                "Start it with: sudo systemctl start pigpiod"
            )

    def set_servo_deg(self, angle):

        angle = max(0.0, min(180.0, angle))

        pulse = (
            self.MIN_US
            + (self.MAX_US - self.MIN_US)
            * angle / 180.0
        )

        self.pi.set_servo_pulsewidth(
            self.SERVO_GPIO,
            pulse
        )

    def set_beam(self, theta_deg):

        theta = max(
            -self.max_beam,
            min(self.max_beam, theta_deg)
        )

        servo_angle = (
            self.LEVEL_DEG
            + self.SERVO_DIR
            * kinematics.motor_angle(theta)
        )

        self.set_servo_deg(servo_angle)

        return theta

    def release(self):

        self.pi.set_servo_pulsewidth(
            self.SERVO_GPIO,
            0
        )

        self.pi.stop()


# ============================================================
# RED CENTER MARKER
# ============================================================

class CenterMarkerDetector:

    def __init__(self):

        # Red wraps around HSV hue=0, so two ranges are needed.
        self.lower1 = np.array(
            [0, 120, 100],
            dtype=np.uint8
        )

        self.upper1 = np.array(
            [10, 255, 255],
            dtype=np.uint8
        )

        self.lower2 = np.array(
            [170, 120, 100],
            dtype=np.uint8
        )

        self.upper2 = np.array(
            [179, 255, 255],
            dtype=np.uint8
        )

    def detect(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        mask1 = cv2.inRange(
            hsv,
            self.lower1,
            self.upper1
        )

        mask2 = cv2.inRange(
            hsv,
            self.lower2,
            self.upper2
        )

        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # The center marker is expected to be approximately
        # circular and small.
        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 10:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            circularity = (
                4.0 * math.pi * area
                / (perimeter * perimeter)
            )

            (x, y), radius = cv2.minEnclosingCircle(
                contour
            )

            if circularity > 0.55:
                candidates.append(
                    (area, x, y, radius)
                )

        if not candidates:
            return None

        # Largest plausible red circular object
        _, x, y, radius = max(
            candidates,
            key=lambda q: q[0]
        )

        return float(x), float(y), float(radius)


# ============================================================
# VISION
# ============================================================

class VisionSystem:

    def __init__(self):

        self.ball = BallDetector()
        self.center_marker = CenterMarkerDetector()

        # pixels/mm.
        #
        # IMPORTANT:
        # This should be calibrated from the actual camera.
        #
        # Initial approximation assumes the entire 220 mm beam
        # occupies 500 pixels horizontally.
        self.pixels_per_mm = -1.9596

        self.last_center = None

    def detect(self, frame):

        # -------------------------
        # Ball
        # -------------------------

        found, ball_center, radius, _ = \
            self.ball.detect_ball(frame)

        # -------------------------
        # Red center
        # -------------------------

        marker = self.center_marker.detect(frame)

        if marker is None:
            return {
                "found": False,
                "ball_px": None,
                "center_px": None,
                "x_mm": None,
                "marker": None
            }

        center_x, center_y, center_radius = marker

        # -------------------------
        # Ball position
        # -------------------------

        if not found:

            return {
                "found": False,
                "ball_px": None,
                "center_px": (center_x, center_y),
                "x_mm": None,
                "marker": marker
            }

        ball_x, ball_y = ball_center

        # Positive x is toward the right side
        # of the image.
        #
        # IMPORTANT:
        # Flip this sign if your physical +x direction
        # is opposite to image-right.
        dx_pixels = ball_x - center_x

        x_mm = dx_pixels / self.pixels_per_mm

        return {
            "found": True,
            "ball_px": (ball_x, ball_y),
            "center_px": (center_x, center_y),
            "x_mm": x_mm,
            "marker": marker
        }

    def draw(self, frame, result, setpoint):

        output = frame.copy()

        h, w = output.shape[:2]

        # Center marker
        if result["center_px"] is not None:

            cx, cy = result["center_px"]

            cv2.circle(
                output,
                (int(cx), int(cy)),
                6,
                (0, 0, 255),
                2
            )

            cv2.putText(
                output,
                "CENTER",
                (int(cx) + 10, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        # Ball
        if result["found"]:

            bx, by = result["ball_px"]

            cv2.circle(
                output,
                (int(bx), int(by)),
                10,
                (0, 255, 0),
                2
            )

            cv2.circle(
                output,
                (int(bx), int(by)),
                3,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                output,
                f"x = {result['x_mm']:+.1f} mm",
                (int(bx) - 40, int(by) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

        return output


# ============================================================
# PID CONTROLLER
# ============================================================

class PIDController:

    def __init__(self, kp, ki, kd):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0

        self.velocity = 0.0

        self.last_x = None

    def reset(self):

        self.integral = 0.0
        self.velocity = 0.0
        self.last_x = None

    def update(self, x, setpoint, dt):

        error = x - setpoint

        # -----------------------------------------
        # Estimate velocity from measured position
        # -----------------------------------------

        if self.last_x is None:

            raw_velocity = 0.0

        else:

            raw_velocity = (
                x - self.last_x
            ) / dt

        self.last_x = x

        # Low-pass velocity.
        #
        # This is much better than directly differentiating
        # noisy camera measurements.
        self.velocity = (
            VELOCITY_ALPHA * raw_velocity
            + (1.0 - VELOCITY_ALPHA)
            * self.velocity
        )

        # -----------------------------------------
        # Integral
        # -----------------------------------------

        new_integral = (
            self.integral
            + error * dt
        )

        new_integral = max(
            -MAX_INTEGRAL,
            min(MAX_INTEGRAL, new_integral)
        )

        # -----------------------------------------
        # PID
        # -----------------------------------------

        theta = (
            self.kp * error
            + self.kd * self.velocity
            + self.ki * new_integral
        )

        # Anti-windup
        if abs(theta) < MAX_BEAM_DEG:

            self.integral = new_integral

        return theta


# ============================================================
# MAIN BALANCER
# ============================================================

class BallBalancer:

    def __init__(self, kp, ki, kd):

        self.vision = VisionSystem()

        self.servo = BeamServo()

        self.pid = PIDController(
            kp,
            ki,
            kd
        )

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.setpoint = 0.0
        self.trim = 0.0

        self.enabled = True

        self.x = None
        self.x_meas = None
        self.v = 0.0

        self.theta = 0.0

        self.last_seen = 0.0

        self.running = False

        self.rate_hz = 0.0

        self.history = collections.deque()

        self.log_dir = "logs"
        os.makedirs(
            self.log_dir,
            exist_ok=True
        )

        self.log_path = os.path.join(
            self.log_dir,
            time.strftime(
                "vision_run_%Y%m%d_%H%M%S.csv"
            )
        )

        self.log_file = open(
            self.log_path,
            "w",
            newline=""
        )

        self.log = csv.writer(
            self.log_file
        )

        self.log.writerow([
            "t",
            "x_meas_mm",
            "x_mm",
            "v_mm_s",
            "setpoint_mm",
            "theta_deg",
            "kp",
            "ki",
            "kd"
        ])

    # --------------------------------------------------------
    # Integral reset
    # --------------------------------------------------------

    def reset_integral(self):

        self.pid.integral = 0.0

    # --------------------------------------------------------
    # Servo command
    # --------------------------------------------------------

    def command(self, requested_theta, dt):

        # Rate limit
        step = MAX_RATE * dt

        requested_theta = max(
            self.theta - step,
            min(
                self.theta + step,
                requested_theta
            )
        )

        self.theta = self.servo.set_beam(
            requested_theta
        )

    # --------------------------------------------------------
    # One control iteration
    # --------------------------------------------------------

    def step(self, frame, dt, now):

        result = self.vision.detect(frame)

        self.x_meas = result["x_mm"]

        # -----------------------------------------
        # No ball
        # -----------------------------------------

        if self.x_meas is None:

            if now - self.last_seen > LOST_TIMEOUT:

                self.x = None
                self.v = 0.0

                self.pid.reset()

                self.command(
                    self.trim,
                    dt
                )

                return result

            return result

        # -----------------------------------------
        # Ball detected
        # -----------------------------------------

        self.last_seen = now

        self.x = self.x_meas

        # -----------------------------------------
        # Level / paused
        # -----------------------------------------

        if not self.enabled:

            self.command(
                self.trim,
                dt
            )

            return result

        # -----------------------------------------
        # PID
        # -----------------------------------------

        theta = self.pid.update(
            self.x,
            self.setpoint,
            dt
        )

        self.v = self.pid.velocity

        theta += self.trim

        self.command(
            theta,
            dt
        )

        return result

    # --------------------------------------------------------
    # Camera loop
    # --------------------------------------------------------

    def run(self):

        cap = cv2.VideoCapture(
            CAMERA_ID
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )


        if not cap.isOpened():

            raise RuntimeError(
                "Could not open camera"
            )

        self.running = True

        t0 = time.monotonic()
        last_time = time.monotonic()

        self.theta = self.servo.set_beam(
            self.trim
        )

        try:

            while self.running:

                ret, frame = cap.read()

                if not ret:

                    print(
                        "Camera frame failed"
                    )

                    continue

                now = time.monotonic()

                dt = now - last_time

                last_time = now

                dt = max(
                    0.001,
                    min(dt, 0.2)
                )

                self.rate_hz += (
                    0.05
                    * (
                        1.0 / dt
                        - self.rate_hz
                    )
                )

                result = self.step(
                    frame,
                    dt,
                    now
                )

                # -----------------------------
                # Visualization
                # -----------------------------

                vis = self.vision.draw(
                    frame,
                    result,
                    self.setpoint
                )

                cv2.imshow(
                    "Ball Balancer Camera",
                    vis
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):

                    break

                # -----------------------------
                # Logging
                # -----------------------------

                t = now - t0

                self.history.append(
                    (
                        t,
                        self.x,
                        self.theta
                    )
                )

                while (
                    self.history
                    and
                    t - self.history[0][0] > 10
                ):

                    self.history.popleft()

                self.log.writerow([
                    f"{t:.4f}",
                    "" if self.x_meas is None
                    else f"{self.x_meas:.3f}",

                    "" if self.x is None
                    else f"{self.x:.3f}",

                    f"{self.v:.2f}",

                    f"{self.setpoint:.3f}",

                    f"{self.theta:.3f}",

                    self.kp,
                    self.ki,
                    self.kd
                ])

        finally:

            cap.release()

            cv2.destroyAllWindows()

    def shutdown(self):

        self.running = False

        try:

            self.servo.set_beam(
                0.0
            )

            time.sleep(0.3)

        finally:

            self.servo.release()

            self.log_file.close()

            print(
                f"Log saved to {self.log_path}"
            )


# ============================================================
# GUI
# ============================================================

class GUI:

    W = 600

    def __init__(self, controller):

        import tkinter as tk
        from tkinter import ttk

        self.c = controller

        self.root = tk.Tk()

        self.root.title(
            "Ball Balancer"
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.root.quit
        )

        # -----------------------------
        # Beam
        # -----------------------------

        self.beam = tk.Canvas(
            self.root,
            width=self.W,
            height=150,
            bg="white"
        )

        self.beam.pack(
            padx=10,
            pady=(10, 0)
        )

        # -----------------------------
        # Chart
        # -----------------------------

        self.chart = tk.Canvas(
            self.root,
            width=self.W,
            height=220,
            bg="white"
        )

        self.chart.pack(
            padx=10,
            pady=5
        )

        # -----------------------------
        # Status
        # -----------------------------

        self.status = ttk.Label(
            self.root,
            font=("TkFixedFont", 11)
        )

        self.status.pack()

        # -----------------------------
        # Sliders
        # -----------------------------

        sliders = ttk.Frame(
            self.root
        )

        sliders.pack(
            padx=10,
            pady=5,
            fill="x"
        )

        self.vars = {}
        self.labels = {}

        rows = [
            (
                "kp",
                "Kp [deg/mm]",
                0,
                0.3,
                controller.kp
            ),
            (
                "ki",
                "Ki [deg/(mm*s)]",
                0,
                0.1,
                controller.ki
            ),
            (
                "kd",
                "Kd [deg*s/mm]",
                0,
                0.15,
                controller.kd
            ),
            (
                "setpoint",
                "Setpoint [mm]",
                -80,
                80,
                controller.setpoint
            ),
            (
                "trim",
                "Level trim [deg]",
                -3,
                3,
                controller.trim
            ),
        ]

        for row, (
            name,
            label,
            lo,
            hi,
            value
        ) in enumerate(rows):

            self.vars[name] = \
                tk.DoubleVar(
                    value=value
                )

            ttk.Label(
                sliders,
                text=label,
                width=16
            ).grid(
                row=row,
                column=0,
                sticky="w"
            )

            ttk.Scale(
                sliders,
                from_=lo,
                to=hi,
                variable=self.vars[name],
                length=380
            ).grid(
                row=row,
                column=1
            )

            self.labels[name] = ttk.Label(
                sliders,
                width=8
            )

            self.labels[name].grid(
                row=row,
                column=2
            )

        # -----------------------------
        # Buttons
        # -----------------------------

        buttons = ttk.Frame(
            self.root
        )

        buttons.pack(
            pady=(0, 10)
        )

        self.pause_btn = ttk.Button(
            buttons,
            text="Pause (level)",
            command=self.toggle
        )

        self.pause_btn.pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Reset integral",
            command=controller.reset_integral
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Quit",
            command=self.root.quit
        ).pack(
            side="left",
            padx=5
        )

        self.update()

    def toggle(self):

        self.c.enabled = not self.c.enabled

        self.pause_btn.config(
            text=(
                "Pause (level)"
                if self.c.enabled
                else "Resume control"
            )
        )

    def px(self, x_mm):

        return (
            self.W / 2
            + x_mm
            * (self.W - 40)
            / BEAM_LENGTH_MM
        )

    def draw_beam(self):

        c = self.c
        cv = self.beam

        cv.delete("all")

        y = 115

        # Beam
        cv.create_line(
            self.px(-110),
            y + 2,
            self.px(110),
            y + 2,
            width=4,
            fill="#555"
        )

        # Motor direction
        cv.create_text(
            self.px(110),
            y + 30,
            text="motor end ->",
            anchor="e",
            fill="#888"
        )

        # Center / setpoint
        sp = self.px(
            c.setpoint
        )

        cv.create_line(
            sp,
            5,
            sp,
            y,
            fill="#2a9d8f",
            dash=(3, 2),
            width=2
        )

        # Ball
        if c.x is not None:

            r = 20 * (
                self.W - 40
            ) / BEAM_LENGTH_MM

            cv.create_oval(
                self.px(c.x) - r,
                y - 2 * r,
                self.px(c.x) + r,
                y,
                fill="#f4f1de",
                outline="#333"
            )

    def draw_chart(self):

        cv = self.chart

        cv.delete("all")

        hist = list(
            self.c.history
        )

        # Position
        lanes = [
            (
                "x [mm]",
                1,
                110,
                "#264653",
                0
            ),
            (
                "theta [deg]",
                2,
                self.c.servo.max_beam,
                "#e76f51",
                110
            )
        ]

        for (
            label,
            idx,
            scale,
            color,
            top
        ) in lanes:

            mid = top + 55

            cv.create_line(
                0,
                mid,
                self.W,
                mid,
                fill="#ddd"
            )

            cv.create_text(
                5,
                top + 5,
                text=label,
                anchor="nw",
                fill=color
            )

            if not hist:
                continue

            t_end = hist[-1][0]

            pts = []

            for h in hist:

                if h[idx] is None:

                    if len(pts) >= 4:

                        cv.create_line(
                            *pts,
                            fill=color,
                            width=2
                        )

                    pts = []

                    continue

                pts += [
                    self.W
                    * (
                        1
                        - (
                            t_end - h[0]
                        ) / 10
                    ),

                    mid
                    - 50
                    * h[idx]
                    / scale
                ]

            if len(pts) >= 4:

                cv.create_line(
                    *pts,
                    fill=color,
                    width=2
                )

        # Setpoint
        sp_y = (
            55
            - 50
            * self.c.setpoint
            / 110
        )

        cv.create_line(
            0,
            sp_y,
            self.W,
            sp_y,
            fill="#2a9d8f",
            dash=(3, 2)
        )

    def update(self):

        c = self.c

        c.kp = self.vars["kp"].get()
        c.ki = self.vars["ki"].get()
        c.kd = self.vars["kd"].get()

        c.pid.kp = c.kp
        c.pid.ki = c.ki
        c.pid.kd = c.kd

        c.setpoint = \
            self.vars["setpoint"].get()

        c.trim = \
            self.vars["trim"].get()

        for name, var in self.vars.items():

            self.labels[name].config(
                text=f"{var.get():.3g}"
            )

        self.draw_beam()
        self.draw_chart()

        x = (
            "  --  "
            if c.x is None
            else f"{c.x:+6.1f}"
        )

        self.status.config(
            text=(
                f"x {x} mm   "
                f"v {c.v:+6.0f} mm/s   "
                f"theta {c.theta:+5.2f} deg   "
                f"I {c.ki * c.pid.integral:+5.2f} deg   "
                f"{c.rate_hz:4.1f} Hz"
            )
        )

        self.root.after(
            50,
            self.update
        )

    def run(self):

        self.root.mainloop()

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kp",
        type=float,
        default=KP
    )

    parser.add_argument(
        "--ki",
        type=float,
        default=KI
    )

    parser.add_argument(
        "--kd",
        type=float,
        default=KD
    )

    args = parser.parse_args()

    controller = BallBalancer(
        args.kp,
        args.ki,
        args.kd
    )

    worker = threading.Thread(
        target=controller.run,
        daemon=True
    )

    worker.start()

    try:

        GUI(controller).run()

    except KeyboardInterrupt:

        pass

    finally:

        controller.running = False

        worker.join(
            timeout=1.0
        )

        controller.shutdown()


if __name__ == "__main__":
    main()
