"""Tk GUI: camera view, top-down X/Y platform, history plots, and PID sliders."""
import math
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

from stewart_ball_balancer import (GUI_HZ, MAX_PHYSICAL_TILT_DEG,
                                   KINEMATICS_ANGLE_FACTOR)


class BalancerGUI:
    def __init__(self, controller):
        self.c = controller
        self.root = tk.Tk()
        self.root.title('Stewart Platform Ball Balancer')
        self.root.protocol('WM_DELETE_WINDOW', self.root.quit)
        self.calibrating = False
        self.photo = None
        self.last_frame = None
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, padx=12, pady=8, sticky='n')
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, padx=12, pady=8, sticky='n')

        ttk.Label(left, text='Camera (intrinsics corrected)').pack(anchor='w')
        self.camera = tk.Canvas(left, width=640, height=480, bg='black', highlightthickness=0)
        self.camera.pack()
        self.camera.bind('<Button-1>', self.click_camera)
        self.status = ttk.Label(left, text='', width=85)
        self.status.pack(anchor='w', pady=5)
        self.mapping_label = ttk.Label(left, text='', wraplength=640)
        self.mapping_label.pack(anchor='w')
        ttk.Button(left, text='Calibrate 3 platform points', command=self.start_calibration).pack(anchor='w', pady=4)

        ttk.Label(right, text='Top view (X/Y in mm)').pack(anchor='w')
        self.platform = tk.Canvas(right, width=430, height=295, bg='white')
        self.platform.pack()
        ttk.Label(right, text='Last 10 seconds: X, Y and physical roll/pitch').pack(anchor='w')
        self.chart = tk.Canvas(right, width=430, height=170, bg='white')
        self.chart.pack()

        self.vars = {}
        sliders = ttk.LabelFrame(right, text='PID and setpoints (both axes share gains)')
        sliders.pack(fill='x', pady=5)
        params = [
            ('kp', 'Kp [deg/mm]', 0, .20, self.c.pid_x.kp),
            ('ki', 'Ki [deg/(mm s)]', 0, .08, self.c.pid_x.ki),
            ('kd', 'Kd [deg s/mm]', 0, .12, self.c.pid_x.kd),
            ('target_x', 'Target X [mm]', -100, 100, 0),
            ('target_y', 'Target Y [mm]', -100, 100, 0),
            ('trim_roll', 'Roll trim [deg]', -2, 2, 0),
            ('trim_pitch', 'Pitch trim [deg]', -2, 2, 0)
        ]
        self.numbers = {}
        for i, (key, label, low, high, initial) in enumerate(params):
            self.vars[key] = tk.DoubleVar(value=initial)
            ttk.Label(sliders, text=label, width=23).grid(row=i, column=0, sticky='w')
            ttk.Scale(sliders, from_=low, to=high, variable=self.vars[key], length=240).grid(row=i, column=1)
            lbl = ttk.Label(sliders, text='', width=8)
            lbl.grid(row=i, column=2)
            self.numbers[key] = lbl

        actions = ttk.Frame(right)
        actions.pack(pady=8)
        self.toggle_button = ttk.Button(actions, text='Enable PID', command=self.toggle)
        self.toggle_button.pack(side='left', padx=5)
        ttk.Button(actions, text='Reset integral', command=self.c.reset_pid).pack(side='left', padx=5)
        ttk.Button(actions, text='Quit', command=self.root.quit).pack(side='left', padx=5)
        self.hardware_label = ttk.Label(right, text='HARDWARE ENABLED' if self.c.driver.hardware
                                        else 'DRY RUN: no motor commands sent')
        self.hardware_label.pack(anchor='w')
        ttk.Label(right, text=f'Temporary IK angle factor = {KINEMATICS_ANGLE_FACTOR:g}x | '
                 f'max physical angle = {MAX_PHYSICAL_TILT_DEG:g} deg').pack(anchor='w')
        self.update_period_ms = max(1, round(1000 / GUI_HZ))
        self.last_display_id = -1

    def start_calibration(self):
        if self.c.active:
            self.c.set_active(False)
            self.toggle_button.configure(text='Enable PID')
        self.c.detector.reset_mapping()
        self.calibrating = True
        self.mapping_label.configure(text='Click platform attachment 1, 2, 3 IN THE ORDER of geometry.radial_angles_deg '
                               '(default: top, bottom-right, bottom-left). Use the level pose.')

    def click_camera(self, event):
        if not self.calibrating:
            return
        try:
            n = self.c.detector.add_reference_click(event.x, event.y)
        except ValueError as e:
            messagebox.showerror('Calibration', str(e))
            return
        if n == 3:
            self.calibrating = False
            self.c.position = None
            self.c.reset_pid()
            self.mapping_label.configure(text='Mapping saved. Confirm X/Y axes and measurement before enabling PID.')
        else:
            self.mapping_label.configure(text=f'Click platform attachment point {n+1} of 3.')

    def toggle(self):
        try:
            self.c.set_active(not self.c.active)
            self.toggle_button.configure(text='Pause (level)' if self.c.active else 'Enable PID')
        except RuntimeError as e:
            messagebox.showwarning('Cannot enable PID', str(e))

    def coordinate(self, x, y):
        scale = 1.03
        return (215 + x * scale, 147 - y * scale)

    def draw_platform(self):
        cv = self.platform
        cv.delete('all')
        c = self.c
        pts = c.detector.target_points_mm()
        poly = []
        for x, y in pts:
            poly.extend(self.coordinate(x, y))
        cv.create_polygon(*poly, outline='#415a77', fill='#e7eef3', width=2)
        x0, y0 = self.coordinate(0, 0)
        cv.create_line(15, y0, 410, y0, fill='#aaa', dash=(3,3))
        cv.create_line(x0, 8, x0, 282, fill='#aaa', dash=(3,3))
        cv.create_text(412, y0-8, text='+X', anchor='e')
        cv.create_text(x0+6, 12, text='+Y', anchor='w')
        sx, sy = self.coordinate(*c.sp)
        cv.create_line(sx-8, sy, sx+8, sy, fill='#279d8f', width=2)
        cv.create_line(sx, sy-8, sx, sy+8, fill='#279d8f', width=2)
        if c.position is not None:
            bx, by = self.coordinate(*c.position)
            cv.create_oval(bx-10, by-10, bx+10, by+10, fill='#f0c95b', outline='#333', width=2)
        cv.create_text(10, 270, anchor='w', text='Yellow = ball    Green = target', fill='#333')

    def draw_chart(self):
        cv = self.chart
        cv.delete('all')
        hist = self.c.history_snapshot()
        lanes = [('X mm', 1, 130, '#285d81', 0),
                 ('Y mm', 2, 130, '#d37a2c', 0),
                 ('roll deg', 3, MAX_PHYSICAL_TILT_DEG, '#2a9d8f', 85),
                 ('pitch deg', 4, MAX_PHYSICAL_TILT_DEG, '#ae4b75', 85)]
        for top in [0,85]:
            cv.create_line(0, top+43, 430, top+43, fill='#ddd')
        for label, idx, scale, color, top in lanes:
            cv.create_text(6 if idx in [1,3] else 100, top+3, text=label, anchor='nw', fill=color)
            if not hist:
                continue
            end = hist[-1][0]
            segments = []
            for h in hist:
                v = h[idx]
                if v is None:
                    if len(segments) >= 4:
                        cv.create_line(*segments, fill=color, width=2)
                    segments = []
                    continue
                segments += [430 * (1 - (end-h[0])/10), top+43 - 36*v/scale]
            if len(segments) >= 4:
                cv.create_line(*segments, fill=color, width=2)

    def tick(self):
        try:
            self.c.pid_x.kp = self.c.pid_y.kp = self.vars['kp'].get()
            self.c.pid_x.ki = self.c.pid_y.ki = self.vars['ki'].get()
            self.c.pid_x.kd = self.c.pid_y.kd = self.vars['kd'].get()
            self.c.sp[:] = [self.vars['target_x'].get(), self.vars['target_y'].get()]
            self.c.trim[:] = [self.vars['trim_roll'].get(), self.vars['trim_pitch'].get()]
            for key, var in self.vars.items():
                self.numbers[key].configure(text=f'{var.get():+.3f}')
            if self.c.worker_error:
                raise RuntimeError(self.c.worker_error)
            frame = self.c.latest_overlay
            if frame is not None and self.c.latest_frame_id != self.last_display_id:
                self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.camera.delete('frame')
                self.camera.create_image(0, 0, image=self.photo, anchor='nw', tags='frame')
                self.last_display_id = self.c.latest_frame_id
            self.draw_platform()
            self.draw_chart()
            p = self.c.position
            pos = 'NO BALL' if p is None else f'X={p[0]:+.1f}, Y={p[1]:+.1f} mm'
            self.status.configure(text=(f'{pos}  |  vx={self.c.pid_x.velocity:+.1f} vy={self.c.pid_y.velocity:+.1f} mm/s  '
                                        f'| roll={self.c.roll:+.2f} pitch={self.c.pitch:+.2f} deg  | control {self.c.fps:.1f} / camera {self.c.capture_fps:.1f} fps | processing {self.c.processing_ms:.0f}ms'))
            if self.c.driver.last_error:
                self.mapping_label.configure(text=self.c.driver.last_error)
        except Exception as exc:
            self.c.active = False
            messagebox.showerror('Control stopped', f'{type(exc).__name__}: {exc}')
            self.root.quit()
            return
        self.root.after(self.update_period_ms, self.tick)

    def run(self):
        self.c.start()
        self.tick()
        self.root.mainloop()
        self.root.destroy()
