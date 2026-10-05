#!/usr/bin/env python3
# Ball balancer main program.
# light sensors -> ball position -> Kalman filter -> PID -> beam angle -> inverse kinematics -> servo
#
#   python ball_balancer.py             GUI with sliders for the gains
#   python ball_balancer.py --headless  no GUI, set gains with --kp --ki --kd
#
# Run calibrate.py first. Units are mm and degrees, so Kp is deg/mm, Ki is deg/(mm*s), Kd is deg/(mm/s).
# Every run gets logged to logs/.
import argparse
import collections
import csv
import os
import threading
import time

from balancer_hw import BallEstimator, BallFilter, BeamServo, load_config
from read_sensors import SensorReader

KP, KI, KD = 0.05, 0.01, 0.04
LOST_TIMEOUT = 0.3     # s, level the beam if we haven't seen the ball for this long
MAX_RATE = 60.0        # deg/s, max beam speed so the servo doesn't jerk


class Controller:
    def __init__(self, cfg, kp, ki, kd):
        self.est = BallEstimator(cfg)
        self.reader = SensorReader()
        self.servo = BeamServo()
        self.max_beam = self.servo.max_beam
        self.sensor_x = cfg["sensor_x_mm"]

        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = 0.0
        self.trim = 0.0         # manual level correction (deg)
        self.enabled = True     # False = just hold the beam level

        self.x = None           # filtered position
        self.x_meas = None      # raw position from the sensors
        self.v = 0.0
        self.theta = 0.0
        self.coverage = [0.0] * 8
        self.rate_hz = 0.0
        self.history = collections.deque()   # last 10 s for the plot

        self.filter = BallFilter()
        self.integral = 0.0
        self.last_seen = 0.0
        self.running = False

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, time.strftime("run_%Y%m%d_%H%M%S.csv"))
        self.log_file = open(self.log_path, "w", newline="")
        self.log = csv.writer(self.log_file)
        self.log.writerow(["t", "x_meas_mm", "x_mm", "v_mm_s", "setpoint_mm", "theta_deg", "kp", "ki", "kd"]
                          + [f"s{i}" for i in range(8)])

    def reset_integral(self):
        self.integral = 0.0

    def command(self, theta, dt):
        # limit how far the beam can move in one step
        step = MAX_RATE * dt
        theta = min(max(theta, self.theta - step), self.theta + step)
        self.theta = self.servo.set_beam(theta)

    def step(self, values, dt, now):
        self.filter.predict(dt, self.theta)
        self.x_meas, self.coverage = self.est.position(values)
        if self.x_meas is not None and self.filter.update(self.x_meas):
            self.last_seen = now

        if now - self.last_seen > LOST_TIMEOUT:
            # ball is gone, go back to level and start over
            self.filter.reset()
            self.x, self.v, self.integral = None, 0.0, 0.0
            self.command(self.trim, dt)
            return
        # if we only lost it for a couple samples, keep going on the filter's prediction
        self.x, self.v = self.filter.x, self.filter.v
        if self.x is None:
            return

        if not self.enabled:
            self.command(self.trim, dt)
            return

        # ball on the + side needs a + angle to roll it back
        error = self.x - self.setpoint
        new_integral = self.integral + error * dt
        u = self.kp * error + self.kd * self.v + self.ki * new_integral + self.trim
        if abs(u) < self.max_beam:   # anti windup
            self.integral = new_integral
        self.command(u, dt)

    def run(self):
        self.running = True
        count, last_stamp, t0 = self.reader.count, None, time.monotonic()
        self.theta = self.servo.set_beam(self.trim)
        while self.running:
            got = self.reader.wait_new(count, timeout=0.5)
            if got is None:
                print("No sensor data, check the serial connection")
                self.theta = self.servo.set_beam(self.trim)
                continue
            values, stamp, count = got
            dt = 0.05 if last_stamp is None else min(max(stamp - last_stamp, 1e-3), 0.2)
            last_stamp = stamp
            self.rate_hz += 0.05 * (1 / dt - self.rate_hz)

            self.step(values, dt, stamp)

            t = stamp - t0
            self.history.append((t, self.x, self.theta))
            while self.history and t - self.history[0][0] > 10:
                self.history.popleft()
            self.log.writerow([f"{t:.4f}", "" if self.x_meas is None else f"{self.x_meas:.2f}",
                               "" if self.x is None else f"{self.x:.2f}", f"{self.v:.1f}",
                               self.setpoint, f"{self.theta:.3f}", self.kp, self.ki, self.kd] + values)

    def shutdown(self):
        self.running = False
        try:
            self.servo.set_beam(0.0)
            time.sleep(0.3)
        finally:
            self.servo.release()
            self.reader.close()
            self.log_file.close()
            print(f"Log saved to {self.log_path}")


class GUI:
    W = 600

    def __init__(self, ctrl):
        import tkinter as tk
        from tkinter import ttk
        self.c = ctrl
        self.root = tk.Tk()
        self.root.title("Ball Balancer")
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)

        self.beam = tk.Canvas(self.root, width=self.W, height=150, bg="white")
        self.beam.pack(padx=10, pady=(10, 0))
        self.chart = tk.Canvas(self.root, width=self.W, height=220, bg="white")
        self.chart.pack(padx=10, pady=5)
        self.status = ttk.Label(self.root, font=("TkFixedFont", 11))
        self.status.pack()

        sliders = ttk.Frame(self.root)
        sliders.pack(padx=10, pady=5, fill="x")
        self.vars, self.labels = {}, {}
        rows = [
            ("kp", "Kp [deg/mm]", 0, 0.3, ctrl.kp),
            ("ki", "Ki [deg/(mm*s)]", 0, 0.1, ctrl.ki),
            ("kd", "Kd [deg*s/mm]", 0, 0.15, ctrl.kd),
            ("setpoint", "Setpoint [mm]", -80, 80, ctrl.setpoint),
            ("trim", "Level trim [deg]", -3, 3, ctrl.trim),
        ]
        for row, (name, label, lo, hi, val) in enumerate(rows):
            self.vars[name] = tk.DoubleVar(value=val)
            ttk.Label(sliders, text=label, width=16).grid(row=row, column=0, sticky="w")
            ttk.Scale(sliders, from_=lo, to=hi, variable=self.vars[name], length=380).grid(row=row, column=1)
            self.labels[name] = ttk.Label(sliders, width=8)
            self.labels[name].grid(row=row, column=2)

        buttons = ttk.Frame(self.root)
        buttons.pack(pady=(0, 10))
        self.pause_btn = ttk.Button(buttons, text="Pause (level)", command=self.toggle)
        self.pause_btn.pack(side="left", padx=5)
        ttk.Button(buttons, text="Reset integral", command=ctrl.reset_integral).pack(side="left", padx=5)
        ttk.Button(buttons, text="Quit", command=self.root.quit).pack(side="left", padx=5)
        self.update()

    def toggle(self):
        self.c.enabled = not self.c.enabled
        self.pause_btn.config(text="Pause (level)" if self.c.enabled else "Resume control")

    def px(self, x_mm):
        # mm on the beam to canvas pixels, 220 mm beam
        return self.W / 2 + x_mm * (self.W - 40) / 220

    def draw_beam(self):
        c, cv = self.c, self.beam
        cv.delete("all")
        y = 115
        cv.create_line(self.px(-110), y + 2, self.px(110), y + 2, width=4, fill="#555")
        cv.create_text(self.px(110), y + 30, text="motor end ->", anchor="e", fill="#888")
        # orange bars = how covered each sensor is
        for xs, cov in zip(c.sensor_x, c.coverage):
            cv.create_rectangle(self.px(xs) - 6, y + 7, self.px(xs) + 6, y + 7 + 15 * cov, fill="#e8a33d", outline="")
            cv.create_rectangle(self.px(xs) - 6, y + 7, self.px(xs) + 6, y + 22, outline="#aaa")
        sp = self.px(c.setpoint)
        cv.create_line(sp, 5, sp, y, fill="#2a9d8f", dash=(3, 2), width=2)
        if c.x is not None:
            r = 20 * (self.W - 40) / 220
            cv.create_oval(self.px(c.x) - r, y - 2 * r, self.px(c.x) + r, y, fill="#f4f1de", outline="#333")

    def draw_chart(self):
        cv, hist = self.chart, list(self.c.history)
        cv.delete("all")
        # top half is position, bottom half is beam angle
        lanes = [("x [mm]", 1, 110, "#264653", 0), ("theta [deg]", 2, self.c.max_beam, "#e76f51", 110)]
        for label, idx, scale, color, top in lanes:
            mid = top + 55
            cv.create_line(0, mid, self.W, mid, fill="#ddd")
            cv.create_text(5, top + 5, text=label, anchor="nw", fill=color)
            if not hist:
                continue
            t_end = hist[-1][0]
            pts = []
            for h in hist:
                if h[idx] is None:
                    if len(pts) >= 4:
                        cv.create_line(*pts, fill=color, width=2)
                    pts = []
                    continue
                pts += [self.W * (1 - (t_end - h[0]) / 10), mid - 50 * h[idx] / scale]
            if len(pts) >= 4:
                cv.create_line(*pts, fill=color, width=2)
        sp_y = 55 - 50 * self.c.setpoint / 110
        cv.create_line(0, sp_y, self.W, sp_y, fill="#2a9d8f", dash=(3, 2))

    def update(self):
        c = self.c
        c.kp, c.ki, c.kd = (self.vars[k].get() for k in ("kp", "ki", "kd"))
        c.setpoint, c.trim = self.vars["setpoint"].get(), self.vars["trim"].get()
        for name, var in self.vars.items():
            self.labels[name].config(text=f"{var.get():.3g}")
        self.draw_beam()
        self.draw_chart()
        x = "  --  " if c.x is None else f"{c.x:+6.1f}"
        self.status.config(text=f"x {x} mm   v {c.v:+6.0f} mm/s   theta {c.theta:+5.2f} deg   "
                                f"I {c.ki * c.integral:+5.2f} deg   {c.rate_hz:4.1f} Hz")
        self.root.after(50, self.update)

    def run(self):
        self.root.mainloop()
        self.root.destroy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kp", type=float, default=KP)
    p.add_argument("--ki", type=float, default=KI)
    p.add_argument("--kd", type=float, default=KD)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    ctrl = Controller(load_config(), args.kp, args.ki, args.kd)
    worker = threading.Thread(target=ctrl.run, daemon=True)
    worker.start()
    try:
        if args.headless:
            while True:
                time.sleep(0.2)
                x = "  --  " if ctrl.x is None else f"{ctrl.x:+6.1f}"
                print(f"x {x} mm  v {ctrl.v:+6.0f} mm/s  theta {ctrl.theta:+5.2f}  {ctrl.rate_hz:4.1f} Hz")
        else:
            GUI(ctrl).run()
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.running = False
        worker.join(timeout=1.0)
        ctrl.shutdown()


if __name__ == "__main__":
    main()
