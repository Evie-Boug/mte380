"""Tk GUI for the ball balancer. Runs in the MAIN process; control runs in its own process.

Nothing here can slow the control loop: the GUI only reads shared memory and
sends commands. Its own CPU use is kept small anyway so the buttons stay
responsive on a Pi 4:
  * 10 Hz tick; sliders push values only when moved;
  * camera preview rendered in a background thread at half size
    (full size while calibrating) and can be switched off;
  * graphs are OFF by default; when on, they sample at the GUI rate and
    redraw twice a second.
"""
from collections import deque
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk
import numpy as np

from stewart_ball_balancer import (S, MODES, MAX_PHYSICAL_TILT_DEG, KINEMATICS_ANGLE_FACTOR,
                                   DEFAULT_TRAJECTORY_SPEED_MM_S, DEFAULT_TRAJECTORY_ACCEL_MM_S2,
                                   DEFAULT_LATENCY_COMP_MS, KP, KI, KD, KFF,
                                   PREVIEW_HZ, PREVIEW_SCALE, render_preview)
from trajectory_planner import SHAPES

GUI_PERIOD_MS = 100
CHART_PERIOD_S = 0.5
CHART_SECONDS = 10.0
PW, PH = 430, 295        # platform canvas
CW, CH = 430, 170        # chart canvas


class BalancerGUI:
    def __init__(self, client):
        self.c = client
        self.det = client.detector
        self.root = tk.Tk()
        self.root.title('Stewart Platform Ball Balancer')
        self.root.protocol('WM_DELETE_WINDOW', self.root.quit)
        self.calibrating = False
        self._path = np.zeros((1, 2))
        self._path_drawn = None
        self._label_cache = {}
        self._history = deque()
        self._last_chart_draw = 0.0
        self._rendered = None
        self._rendered_id = 0
        self._shown_id = 0
        self._render_running = True
        self.camera_info = 'Starting camera...'

        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, padx=12, pady=8, sticky='n')
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, padx=12, pady=8, sticky='n')

        # ---------------- left: camera + gains ----------------
        top = ttk.Frame(left)
        top.pack(fill='x')
        ttk.Label(top, text='Camera (intrinsics corrected)').pack(side='left')
        self.show_camera = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='Camera preview', variable=self.show_camera,
                        command=self._camera_toggled).pack(side='left', padx=12)
        self.show_graphs = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='Graphs', variable=self.show_graphs,
                        command=self._graphs_toggled).pack(side='left')
        self.camera = tk.Canvas(left, bg='black', highlightthickness=0)
        self.camera.pack(anchor='w')
        self.photo = None
        self._photo_size = None
        self._image_item = self.camera.create_image(0, 0, anchor='nw')
        self._camera_off_text = self.camera.create_text(10, 10, anchor='nw', fill='#aaa',
                                                        text='', font=('TkDefaultFont', 11))
        self._set_preview_scale(PREVIEW_SCALE)
        self.camera.bind('<Button-1>', self.click_camera)
        self.status = ttk.Label(left, text='', width=95)
        self.status.pack(anchor='w', pady=(4, 0))
        self.timing = ttk.Label(left, text='', width=95)
        self.timing.pack(anchor='w')
        self.mapping_label = ttk.Label(left, text='', wraplength=640)
        self.mapping_label.pack(anchor='w')
        ttk.Button(left, text='Calibrate 3 platform points',
                   command=self.start_calibration).pack(anchor='w', pady=4)

        self.vars, self.numbers = {}, {}
        sliders = ttk.LabelFrame(left, text='Controller (both axes share gains)')
        sliders.pack(fill='x', pady=4)
        for i, (key, label, low, high, initial) in enumerate([
                ('kp', 'Kp [deg/mm]', 0, .20, KP),
                ('ki', 'Ki [deg/(mm s)]', 0, .08, KI),
                ('kd', 'Kd [deg s/mm]', 0, .12, KD),
                ('kff', 'Accel feed-forward', 0, 2.0, KFF),
                ('latency_ms', 'Latency comp [ms]', 0, 200, DEFAULT_LATENCY_COMP_MS),
                ('trim_roll', 'Roll trim [deg]', -2, 2, 0),
                ('trim_pitch', 'Pitch trim [deg]', -2, 2, 0)]):
            self._slider(sliders, i, key, label, low, high, initial, 300, send=True)

        # ---------------- right: views + planning ----------------
        ttk.Label(right, text='Top view (X/Y in mm)').pack(anchor='w')
        self.platform = tk.Canvas(right, width=PW, height=PH, bg='white')
        self.platform.pack()
        self.platform.bind('<Button-1>', self.click_platform)
        self.chart_title = ttk.Label(right, text='Last 10 s: X, Y (dashed = reference) and roll/pitch')
        self.chart = tk.Canvas(right, width=CW, height=CH, bg='white')

        plan = ttk.LabelFrame(right, text='Trajectory planning')
        self.plan_frame = plan
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
        ttk.Button(buttons, text='Stop path', command=lambda: self.c.send('stop')).pack(side='left', padx=3)
        self.plan_status = ttk.Label(plan, text='Hold')
        self.plan_status.grid(row=7, column=0, columnspan=4, sticky='w')
        ttk.Label(plan, text='Click top view to choose a goal; shape centre = (0,0).').grid(
            row=8, column=0, columnspan=4, sticky='w')

        actions = ttk.Frame(right)
        actions.pack(pady=6)
        self.toggle_button = ttk.Button(actions, text='Enable PID', command=self.toggle)
        self.toggle_button.pack(side='left', padx=5)
        ttk.Button(actions, text='Reset integral',
                   command=lambda: self.c.send('reset_pid')).pack(side='left', padx=5)
        ttk.Button(actions, text='Quit', command=self.root.quit).pack(side='left', padx=5)
        ttk.Label(right, text='HARDWARE ENABLED' if self.c.hardware
                  else 'DRY RUN: no motor commands sent').pack(anchor='w')
        ttk.Label(right, text=f'Temporary IK angle factor = {KINEMATICS_ANGLE_FACTOR:g}x | '
                  f'max physical angle = {MAX_PHYSICAL_TILT_DEG:g} deg').pack(anchor='w')

        self._build_platform_items()
        self._build_chart_grid()
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True,
                                               name='preview-render')

    # ---------------------------------------------------------------- widgets
    def _slider(self, parent, row, key, label, low, high, initial, length,
                fmt='{:+.3f}', send=False):
        var = self.vars[key] = tk.DoubleVar(value=initial)
        ttk.Label(parent, text=label, width=23).grid(row=row, column=0, sticky='w', padx=5)
        ttk.Scale(parent, from_=low, to=high, variable=var, length=length).grid(
            row=row, column=1, columnspan=2, sticky='w')
        lbl = ttk.Label(parent, text=fmt.format(initial), width=8)
        lbl.grid(row=row, column=3)
        self.numbers[key] = (lbl, fmt)
        var.trace_add('write', lambda *_: self._on_slider(key, send))
        if send:
            self.c.send('param', key, float(initial))

    def _on_slider(self, key, send):
        try:
            val = self.vars[key].get()
        except tk.TclError:
            return
        lbl, fmt = self.numbers[key]
        lbl.configure(text=fmt.format(val))
        if send:
            self.c.send('param', key, val)

    def _set_text(self, widget, text, item=None):
        key = (str(widget), item)
        if self._label_cache.get(key) != text:
            self._label_cache[key] = text
            if item is None:
                widget.configure(text=text)
            else:
                widget.itemconfigure(item, text=text)

    # ---------------------------------------------------------------- toggles
    def _set_preview_scale(self, scale):
        self.preview_scale = scale
        w, h = self.det.size
        self.camera.configure(width=int(round(w * scale)), height=int(round(h * scale)))

    def _camera_toggled(self):
        on = self.show_camera.get() or self.calibrating
        self.c.send('preview', on)
        self.camera.itemconfigure(self._image_item, state='normal' if on else 'hidden')
        self.camera.itemconfigure(self._camera_off_text,
                                  text='' if on else 'Camera preview off (saves CPU)')

    def _graphs_toggled(self):
        if self.show_graphs.get():
            self._history.clear()
            self.chart_title.pack(anchor='w', before=self.plan_frame)
            self.chart.pack(before=self.plan_frame)
        else:
            self.chart.delete('data')
            self.chart.pack_forget()
            self.chart_title.pack_forget()
            self._history.clear()

    # ---------------------------------------------------------------- actions
    def start_calibration(self):
        self.c.send('reset_mapping')            # also disables PID in the control process
        self.det.reset_mapping()
        self.calibrating = True
        self._set_preview_scale(1.0)            # full resolution for accurate clicks
        self._camera_toggled()
        self.mapping_label.configure(text='Click platform attachment 1, 2, 3 IN THE ORDER of '
                                     'geometry.radial_angles_deg (default: top, bottom-right, '
                                     'bottom-left). Use the level pose.')

    def click_camera(self, event):
        if not self.calibrating:
            return
        s = self.preview_scale
        try:
            n = self.det.add_reference_click(event.x / s, event.y / s)
        except ValueError as e:
            messagebox.showerror('Calibration', str(e))
            return
        if n == 3:
            self.calibrating = False
            self._set_preview_scale(PREVIEW_SCALE)
            self._camera_toggled()
            self.c.send('reload_mapping')       # control process reads the saved mapping
            self.mapping_label.configure(text='Mapping saved. Confirm X/Y axes and measurement '
                                         'before enabling PID.')
        else:
            self.mapping_label.configure(text=f'Click platform attachment point {n+1} of 3.')

    def toggle(self):
        self.c.send('active', self.c[S.ACTIVE] < 0.5)

    def _scale(self):
        return 128.0 / self.c.surface_radius_mm

    def coordinate(self, x, y):
        s = self._scale()
        return (215 + x * s, 147 - y * s)

    def click_platform(self, event):
        s = self._scale()
        xy = np.array([(event.x - 215) / s, (147 - event.y) / s])
        length = np.linalg.norm(xy)
        if length > self.c.limit:
            xy *= self.c.limit / length
        self.vars['target_x'].set(float(xy[0]))
        self.vars['target_y'].set(float(xy[1]))
        self.go_to_point()

    def go_to_point(self):
        v = self.vars
        self.c.send('point', (v['target_x'].get(), v['target_y'].get()),
                    v['speed'].get(), v['accel'].get())

    def trace_shape(self):
        try:
            laps = int(self.laps_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning('Invalid laps', 'Laps must be a whole number 1-99')
            return
        v = self.vars
        self.c.send('shape', self.shape_var.get(), (0.0, 0.0), v['radius'].get(),
                    v['speed'].get(), laps, v['accel'].get())

    # ---------------------------------------------------------------- drawing
    def _build_platform_items(self):
        cv, c = self.platform, self.c
        cx, cy = self.coordinate(*c.surface_center_mm)
        r = self.coordinate(c.surface_radius_mm, 0)[0] - cx
        cv.create_oval(cx-r, cy-r, cx+r, cy+r, outline='#415a77', fill='#e7eef3', width=2)
        safe_r = self.coordinate(c.limit, 0)[0] - cx
        cv.create_oval(cx-safe_r, cy-safe_r, cx+safe_r, cy+safe_r, outline='#a1b2bc', dash=(4, 4))
        for x, y in self.det.target_points_mm():
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

    def draw_platform(self, st, mode):
        cv = self.platform
        if self._path is not self._path_drawn:
            cv.delete('path')
            if len(self._path) > 1:
                coords = [v for pt in self._path for v in self.coordinate(*pt)]
                cv.create_line(*coords, fill='#6b69ba', width=2, tags='path')
                cv.tag_raise('dyn')
            self._path_drawn = self._path
        gx, gy = self.coordinate(st[S.GOAL_X], st[S.GOAL_Y])
        cv.coords(self._goal, gx-5, gy-5, gx+5, gy+5)
        sx, sy = self.coordinate(st[S.REF_X], st[S.REF_Y])
        cv.coords(self._ref_h, sx-8, sy, sx+8, sy)
        cv.coords(self._ref_v, sx, sy-8, sx, sy+8)
        if st[S.VALID] > 0.5:
            bx, by = self.coordinate(st[S.X], st[S.Y])
            cv.coords(self._ball, bx-10, by-10, bx+10, by+10)
            cv.itemconfigure(self._ball, state='normal')
        else:
            cv.itemconfigure(self._ball, state='hidden')
        self._set_text(cv, f'Yellow: ball  Green: reference  Purple: path  [{mode}]', self._legend)

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
        hist = self._history
        if len(hist) < 2:
            return
        end = hist[-1][0]
        r = self.c.surface_radius_mm
        lanes = [(5, r, '#285d81', 0, (4, 3), 1), (6, r, '#d37a2c', 0, (4, 3), 1),
                 (1, r, '#285d81', 0, None, 2), (2, r, '#d37a2c', 0, None, 2),
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
                seg += [CW * (1 - (end - h[0]) / CHART_SECONDS), top + 43 - 36 * v / scale]
            if len(seg) >= 4:
                cv.create_line(*seg, fill=color, width=width, dash=dash, tags='data')

    # ---------------------------------------------------------------- preview
    def _render_loop(self):
        """Background thread: copy + undistort + draw (OpenCV releases the GIL)."""
        while self._render_running:
            time.sleep(0.5 / PREVIEW_HZ)
            if not (self.show_camera.get() or self.calibrating):
                continue
            got = self.c.read_frame()
            if got is None:
                continue
            try:
                img = render_preview(self.det, *got, self._path, self.preview_scale)
                self._rendered = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._rendered_id += 1
            except Exception:
                pass                        # e.g. mapping being reset mid-draw

    def _update_camera(self):
        rgb = self._rendered
        if rgb is None or self._rendered_id == self._shown_id:
            return
        self._shown_id = self._rendered_id
        size = (rgb.shape[1], rgb.shape[0])
        if size != self._photo_size:
            self.photo = ImageTk.PhotoImage('RGB', size)
            self.camera.itemconfigure(self._image_item, image=self.photo)
            self._photo_size = size
        self.photo.paste(Image.fromarray(rgb))

    # ---------------------------------------------------------------- loop
    def _handle_events(self):
        for ev in self.c.poll_events():
            kind = ev[0]
            if kind == 'path':
                self._path = ev[1]
            elif kind == 'ready':
                self.camera_info = ev[1]
            elif kind == 'error':
                title = {'active': 'Cannot enable PID'}.get(ev[1], 'Invalid trajectory'
                                                           if ev[1] in ('point', 'shape') else ev[1])
                messagebox.showwarning(title, ev[2])
            elif kind == 'driver_error':
                if ev[1]:
                    self.mapping_label.configure(text=ev[1])
            elif kind == 'fatal':
                raise RuntimeError(ev[1])

    def tick(self):
        try:
            self._handle_events()
            if not self.c.alive():
                raise RuntimeError('Control process exited')
            st = self.c.state
            self._update_camera()
            mode = MODES[int(st[S.MODE])] if 0 <= int(st[S.MODE]) < len(MODES) else 'Hold'
            self.draw_platform(st, mode)
            active = st[S.ACTIVE] > 0.5
            self._set_text(self.toggle_button, 'Pause (level)' if active else 'Enable PID')
            if st[S.RUNNING] > 0.5:
                pace = f'pace {100 * st[S.PACE]:.0f}%'
                if st[S.STOPPING] > 0.5:
                    plan = f'{mode}: stopping ({pace})'
                elif st[S.LAPS] > 0:
                    lap = 'lead-in' if st[S.LAP] < 1 else f'lap {int(st[S.LAP])}/{int(st[S.LAPS])}'
                    plan = f'{mode}: {lap}, {pace}'
                else:
                    plan = f'{mode}: {pace}'
            else:
                plan = 'Hold'
            self._set_text(self.plan_status, plan)
            valid = st[S.VALID] > 0.5
            pos = f'X={st[S.X]:+.1f}, Y={st[S.Y]:+.1f} mm' if valid else 'NO BALL'
            self._set_text(self.status,
                           f'{pos} | v=({st[S.VX]:+.0f},{st[S.VY]:+.0f}) mm/s | roll={st[S.ROLL]:+.2f} '
                           f'pitch={st[S.PITCH]:+.2f} deg | plate bias=({st[S.BIAS_X]:+.0f},'
                           f'{st[S.BIAS_Y]:+.0f}) mm/s²')
            self._set_text(self.timing,
                           f'control {st[S.FPS]:.0f} / camera {st[S.CAPTURE_FPS]:.0f} fps | decode '
                           f'{st[S.DECODE_MS]:.1f} | detect {st[S.DETECT_MS]:.1f} | control '
                           f'{st[S.CONTROL_MS]:.1f} | peak {st[S.LOOP_MS_PEAK]:.0f} | age '
                           f'{st[S.AGE_MS]:.0f} ms     {self.camera_info}')
            if self.show_graphs.get():
                t = time.monotonic()
                self._history.append((t, st[S.X] if valid else None, st[S.Y] if valid else None,
                                      st[S.ROLL], st[S.PITCH], st[S.REF_X], st[S.REF_Y]))
                while self._history and t - self._history[0][0] > CHART_SECONDS:
                    self._history.popleft()
                if t - self._last_chart_draw >= CHART_PERIOD_S:
                    self.draw_chart()
                    self._last_chart_draw = t
        except Exception as exc:
            messagebox.showerror('Control stopped', f'{type(exc).__name__}: {exc}')
            self.root.quit()
            return
        self.root.after(GUI_PERIOD_MS, self.tick)

    def run(self):
        self._render_thread.start()
        self.tick()
        try:
            self.root.mainloop()
        finally:
            self._render_running = False
            self.root.destroy()