"""Tk GUI: camera view, top-down X/Y platform, history plots, gains and trajectory planning.

Latency notes: the GUI polls every 40 ms and only does cheap work per tick.
The camera image is pasted into ONE persistent PhotoImage (no new Tk image per
frame), platform markers are moved with coords() instead of delete/re-create,
and the history chart redraws at 4 Hz.
"""
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk
import numpy as np

from stewart_ball_balancer import (MAX_PHYSICAL_TILT_DEG, KINEMATICS_ANGLE_FACTOR,
                                   DEFAULT_TRAJECTORY_SPEED_MM_S,
                                   DEFAULT_TRAJECTORY_ACCEL_MM_S2,
                                   DEFAULT_LATENCY_COMP_MS, KFF)
from trajectory_planner import SHAPES

GUI_PERIOD_MS = 40
CHART_PERIOD_S = 0.25
PW, PH = 430, 295        # platform canvas
CW, CH = 430, 170        # chart canvas


class BalancerGUI:
    def __init__(self, controller):
        self.c = controller
        self.root = tk.Tk()
        self.root.title('Stewart Platform Ball Balancer')
        self.root.protocol('WM_DELETE_WINDOW', self.root.quit)
        self.calibrating = False
        self.last_display_id = -1
        self._last_chart_draw = 0.0
        self._last_path_signature = None
        self._label_cache = {}

        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, padx=12, pady=8, sticky='n')
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, padx=12, pady=8, sticky='n')

        # ---------------- left: camera + gains ----------------
        w, h = self.c.detector.size
        ttk.Label(left, text='Camera (intrinsics corrected)').pack(anchor='w')
        self.camera = tk.Canvas(left, width=w, height=h, bg='black', highlightthickness=0)
        self.camera.pack()
        self.photo = ImageTk.PhotoImage('RGB', (w, h))
        self.camera.create_image(0, 0, image=self.photo, anchor='nw')
        self.camera.bind('<Button-1>', self.click_camera)
        self.status = ttk.Label(left, text='', width=95)
        self.status.pack(anchor='w', pady=4)
        self.mapping_label = ttk.Label(left, text='', wraplength=w)
        self.mapping_label.pack(anchor='w')
        ttk.Button(left, text='Calibrate 3 platform points',
                   command=self.start_calibration).pack(anchor='w', pady=4)

        self.vars, self.numbers = {}, {}
        sliders = ttk.LabelFrame(left, text='Controller (both axes share gains)')
        sliders.pack(fill='x', pady=4)
        params = [
            ('kp', 'Kp [deg/mm]', 0, .20, self.c.pid_x.kp),
            ('ki', 'Ki [deg/(mm s)]', 0, .08, self.c.pid_x.ki),
            ('kd', 'Kd [deg s/mm]', 0, .12, self.c.pid_x.kd),
            ('kff', 'Accel feed-forward', 0, 2.0, KFF),
            ('latency', 'Latency comp [ms]', 0, 200, DEFAULT_LATENCY_COMP_MS),
            ('trim_roll', 'Roll trim [deg]', -2, 2, 0),
            ('trim_pitch', 'Pitch trim [deg]', -2, 2, 0),
        ]
        for i, (key, label, low, high, initial) in enumerate(params):
            self._slider(sliders, i, key, label, low, high, initial, 300)

        # ---------------- right: views + planning ----------------
        ttk.Label(right, text='Top view (X/Y in mm)').pack(anchor='w')
        self.platform = tk.Canvas(right, width=PW, height=PH, bg='white')
        self.platform.pack()
        self.platform.bind('<Button-1>', self.click_platform)
        ttk.Label(right, text='Last 10 s: X, Y (dashed = reference) and physical roll/pitch').pack(anchor='w')
        self.chart = tk.Canvas(right, width=CW, height=CH, bg='white')
        self.chart.pack()

        plan = ttk.LabelFrame(right, text='Trajectory planning')
        plan.pack(fill='x', pady=4)
        ttk.Label(plan, text='Shape').grid(row=0, column=0, padx=5, sticky='w')
        self.shape_var = tk.StringVar(value='Circle')
        ttk.Combobox(plan, values=list(SHAPES), textvariable=self.shape_var,
                     state='readonly', width=15).grid(row=0, column=1, sticky='w')
        ttk.Label(plan, text='Laps').grid(row=0, column=2, sticky='e')
        self.laps_var = tk.IntVar(value=3)
        ttk.Spinbox(plan, from_=1, to=99, textvariable=self.laps_var, width=4).grid(
            row=0, column=3, sticky='w', padx=4)
        self._slider(plan, 1, 'radius', 'Shape radius [mm]', 10, 70, 35.0, 190, fmt='{:.0f}')
        self._slider(plan, 2, 'speed', 'Max path speed [mm/s]', 5, 100,
                     DEFAULT_TRAJECTORY_SPEED_MM_S, 190, fmt='{:.0f}')
        self._slider(plan, 3, 'accel', 'Max path accel [mm/s²]', 20, 300,
                     DEFAULT_TRAJECTORY_ACCEL_MM_S2, 190, fmt='{:.0f}')
        self._slider(plan, 4, 'target_x', 'Goal X [mm]', -95, 95, 0, 190)
        self._slider(plan, 5, 'target_y', 'Goal Y [mm]', -95, 95, 0, 190)
        buttons = ttk.Frame(plan)
        buttons.grid(row=6, column=0, columnspan=4, pady=5)
        ttk.Button(buttons, text='Go to point', command=self.go_to_point).pack(side='left', padx=3)
        ttk.Button(buttons, text='Trace shape', command=self.trace_shape).pack(side='left', padx=3)
        ttk.Button(buttons, text='Stop path', command=self.c.stop_trajectory).pack(side='left', padx=3)
        self.plan_status = ttk.Label(plan, text='Hold')
        self.plan_status.grid(row=7, column=0, columnspan=4, sticky='w')
        ttk.Label(plan, text='Click top view to choose a goal; shape centre = (0,0).').grid(
            row=8, column=0, columnspan=4, sticky='w')

        actions = ttk.Frame(right)
        actions.pack(pady=6)
        self.toggle_button = ttk.Button(actions, text='Enable PID', command=self.toggle)
        self.toggle_button.pack(side='left', padx=5)
        ttk.Button(actions, text='Reset integral', command=self.c.reset_pid).pack(side='left', padx=5)
        ttk.Button(actions, text='Quit', command=self.root.quit).pack(side='left', padx=5)
        ttk.Label(right, text='HARDWARE ENABLED' if self.c.driver.hardware
                  else 'DRY RUN: no motor commands sent').pack(anchor='w')
        ttk.Label(right, text=f'Temporary IK angle factor = {KINEMATICS_ANGLE_FACTOR:g}x | '
                  f'max physical angle = {MAX_PHYSICAL_TILT_DEG:g} deg').pack(anchor='w')

        self._build_platform_items()
        self._build_chart_grid()

    def _slider(self, parent, row, key, label, low, high, initial, length, fmt='{:+.3f}'):
        self.vars[key] = tk.DoubleVar(value=initial)
        ttk.Label(parent, text=label, width=23).grid(row=row, column=0, sticky='w', padx=5)
        ttk.Scale(parent, from_=low, to=high, variable=self.vars[key], length=length).grid(
            row=row, column=1, columnspan=2, sticky='w')
        lbl = ttk.Label(parent, text='', width=8)
        lbl.grid(row=row, column=3)
        self.numbers[key] = (lbl, fmt)

    # ---------------------------------------------------------------- actions
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
            self.c.estimator.reset()
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

    def _scale(self):
        return min(190, 128) / self.c.surface_radius_mm

    def coordinate(self, x, y):
        s = self._scale()
        return (215 + x * s, 147 - y * s)

    def click_platform(self, event):
        s = self._scale()
        xy = np.array([(event.x - 215) / s, (147 - event.y) / s])
        limit = self.c.planner.limit
        length = np.linalg.norm(xy)
        if length > limit:
            xy *= limit / length
        self.vars['target_x'].set(float(xy[0]))
        self.vars['target_y'].set(float(xy[1]))
        self.go_to_point()

    def go_to_point(self):
        try:
            self.c.go_to_point((self.vars['target_x'].get(), self.vars['target_y'].get()),
                               self.vars['speed'].get(), self.vars['accel'].get())
        except ValueError as exc:
            messagebox.showwarning('Invalid trajectory', str(exc))

    def trace_shape(self):
        try:
            laps = int(self.laps_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning('Invalid laps', 'Laps must be a whole number 1-99')
            return
        try:
            self.c.trace_shape(self.shape_var.get(), (0.0, 0.0), self.vars['radius'].get(),
                               self.vars['speed'].get(), laps, self.vars['accel'].get())
        except ValueError as exc:
            messagebox.showwarning('Invalid trajectory', str(exc))

    # ---------------------------------------------------------------- drawing
    def _build_platform_items(self):
        cv, c = self.platform, self.c
        cx, cy = self.coordinate(*c.surface_center_mm)
        r = self.coordinate(c.surface_radius_mm, 0)[0] - cx
        cv.create_oval(cx-r, cy-r, cx+r, cy+r, outline='#415a77', fill='#e7eef3', width=2)
        safe_r = self.coordinate(c.planner.limit, 0)[0] - cx
        cv.create_oval(cx-safe_r, cy-safe_r, cx+safe_r, cy+safe_r, outline='#a1b2bc', dash=(4, 4))
        for x, y in c.detector.target_points_mm():
            ax, ay = self.coordinate(x, y)
            cv.create_oval(ax-3, ay-3, ax+3, ay+3, fill='#415a77', outline='')
        x0, y0 = self.coordinate(0, 0)
        cv.create_line(15, y0, 410, y0, fill='#aaa', dash=(3, 3))
        cv.create_line(x0, 8, x0, 282, fill='#aaa', dash=(3, 3))
        cv.create_text(412, y0-8, text='+X', anchor='e')
        cv.create_text(x0+6, 12, text='+Y', anchor='w')
        self._goal = cv.create_oval(0, 0, 0, 0, outline='#355c85', width=2, tags='dyn')
        self._ref_h = cv.create_line(0, 0, 0, 0, fill='#279d8f', width=2, tags='dyn')
        self._ref_v = cv.create_line(0, 0, 0, 0, fill='#279d8f', width=2, tags='dyn')
        self._ball = cv.create_oval(0, 0, 0, 0, fill='#f0c95b', outline='#333', width=2,
                                    state='hidden', tags='dyn')
        self._legend = cv.create_text(8, 282, anchor='w', fill='#333', tags='dyn')

    def draw_platform(self, snapshot):
        cv = self.platform
        path, reference, goal, mode, running, _ = snapshot
        signature = (id(path), mode)
        if signature != self._last_path_signature:
            cv.delete('path')
            if len(path) > 1:
                coords = [v for pt in path for v in self.coordinate(*pt)]
                cv.create_line(*coords, fill='#6b69ba', width=2, tags='path')
                cv.tag_raise('dyn')
            self._last_path_signature = signature
        gx, gy = self.coordinate(*goal)
        cv.coords(self._goal, gx-5, gy-5, gx+5, gy+5)
        sx, sy = self.coordinate(*reference)
        cv.coords(self._ref_h, sx-8, sy, sx+8, sy)
        cv.coords(self._ref_v, sx, sy-8, sx, sy+8)
        p = self.c.position
        if p is None:
            cv.itemconfigure(self._ball, state='hidden')
        else:
            bx, by = self.coordinate(*p)
            cv.coords(self._ball, bx-10, by-10, bx+10, by+10)
            cv.itemconfigure(self._ball, state='normal')
        self._set_text(cv, self._legend, f'Yellow: ball  Green: reference  Purple: path  [{mode}]')

    def _set_text(self, widget, item, text):
        key = (str(widget), item)
        if self._label_cache.get(key) != text:
            self._label_cache[key] = text
            if item is None:
                widget.configure(text=text)
            else:
                widget.itemconfigure(item, text=text)

    def _build_chart_grid(self):
        cv = self.chart
        for top in (0, 85):
            cv.create_line(0, top + 43, CW, top + 43, fill='#ddd')
        for text, x, y, color in (('X mm', 6, 3, '#285d81'), ('Y mm', 60, 3, '#d37a2c'),
                                  ('roll deg', 6, 88, '#2a9d8f'), ('pitch deg', 80, 88, '#ae4b75')):
            cv.create_text(x, y, text=text, anchor='nw', fill=color)

    def draw_chart(self):
        cv = self.chart
        cv.delete('data')
        hist = self.c.history_snapshot()
        if not hist:
            return
        if len(hist) > 160:
            hist = hist[::max(1, len(hist) // 160)] + [hist[-1]]
        end = hist[-1][0]
        xy_scale = self.c.surface_radius_mm
        lanes = [(5, xy_scale, '#285d81', 0, (4, 3), 1),
                 (6, xy_scale, '#d37a2c', 0, (4, 3), 1),
                 (1, xy_scale, '#285d81', 0, None, 2),
                 (2, xy_scale, '#d37a2c', 0, None, 2),
                 (3, MAX_PHYSICAL_TILT_DEG, '#2a9d8f', 85, None, 2),
                 (4, MAX_PHYSICAL_TILT_DEG, '#ae4b75', 85, None, 2)]
        for idx, scale, color, top, dash, width in lanes:
            seg = []
            for h in hist:
                v = h[idx]
                if v is None:
                    if len(seg) >= 4:
                        cv.create_line(*seg, fill=color, width=width, dash=dash, tags='data')
                    seg = []
                    continue
                seg += [CW * (1 - (end - h[0]) / 10), top + 43 - 36 * v / scale]
            if len(seg) >= 4:
                cv.create_line(*seg, fill=color, width=width, dash=dash, tags='data')

    def _update_camera(self):
        frame = self.c.latest_overlay
        fid = self.c.latest_frame_id
        if frame is None or fid == self.last_display_id:
            return
        self.photo.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.last_display_id = fid

    # ---------------------------------------------------------------- loop
    def tick(self):
        try:
            c, v = self.c, self.vars
            c.pid_x.kp = c.pid_y.kp = v['kp'].get()
            c.pid_x.ki = c.pid_y.ki = v['ki'].get()
            c.pid_x.kd = c.pid_y.kd = v['kd'].get()
            c.pid_x.kff = c.pid_y.kff = v['kff'].get()
            c.latency_comp_s = v['latency'].get() / 1000.0
            c.trim[:] = [v['trim_roll'].get(), v['trim_pitch'].get()]
            for key, (lbl, fmt) in self.numbers.items():
                self._set_text(lbl, None, fmt.format(v[key].get()))
            if c.worker_error:
                raise RuntimeError(c.worker_error)
            self._update_camera()
            snapshot = c.trajectory_snapshot()
            self.draw_platform(snapshot)
            self._set_text(self.plan_status, None, snapshot[5])
            now = time.monotonic()
            if now - self._last_chart_draw >= CHART_PERIOD_S:
                self.draw_chart()
                self._last_chart_draw = now
            p = c.position
            pos = 'NO BALL' if p is None else f'X={p[0]:+.1f}, Y={p[1]:+.1f} mm'
            self._set_text(self.status, None,
                           f'{pos} | vx={c.velocity[0]:+.0f} vy={c.velocity[1]:+.0f} mm/s | '
                           f'roll={c.roll:+.2f} pitch={c.pitch:+.2f} deg | control {c.fps:.0f} / '
                           f'camera {c.capture_fps:.0f} fps | proc {c.processing_ms:.0f} ms | '
                           f'age {c.frame_age_ms:.0f} ms')
            if c.driver.last_error:
                self._set_text(self.mapping_label, None, c.driver.last_error)
        except Exception as exc:
            self.c.active = False
            messagebox.showerror('Control stopped', f'{type(exc).__name__}: {exc}')
            self.root.quit()
            return
        self.root.after(GUI_PERIOD_MS, self.tick)

    def run(self):
        self.c.start()
        self.tick()
        self.root.mainloop()
        self.root.destroy()