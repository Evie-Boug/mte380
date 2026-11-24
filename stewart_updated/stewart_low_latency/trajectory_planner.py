"""Feasible, time-parameterised ball reference trajectories (mm, s).

Why the old planner oscillated
------------------------------
It handed the PID a *position-only* carrot a few mm ahead of the ball, and the
PID's D term damped the ball's velocity toward ZERO.  So every step the
controller tried to stop the ball on the carrot, progress stalled because it
was tied to the ball's projection, and the ball hopped from setpoint to
setpoint.

New algorithm
-------------
1. Geometry: optional straight lead-in from the ball to the nearest point on
   the shape, then N identical closed laps.  Square/triangle corners are
   filleted so the ball can keep rolling through them.
2. Resample at uniform arc length and compute signed curvature kappa(s).
3. Speed profile v(s): v <= v_max, v <= sqrt(a_lat / |kappa|) (centripetal
   limit), and forward/backward passes enforce |dv/dt| <= a_tan.  v = 0 at the
   start, the end, and the lead-in corner.
4. Integrate to t(s).  The reference is a full state:
       p(t),  v(t) = T*v,  a(t) = T*dv/dt + N*kappa*v^2
   The controller tracks position AND velocity and adds acceleration
   feed-forward, so the ball keeps moving along the curve.
5. Time-scaling governor: the path clock tau advances at rate f in [0, 1].
   f falls smoothly when the ball falls behind (tracking error grows) and
   rises back when it catches up.  v and a are scaled by f and f^2, so the
   reference never runs away from the ball, yet never stop-and-goes.
"""
import math
import numpy as np

PATH_DS_MM = 0.5                 # arc-length resolution of the planned path
LOOP_SAMPLES = 1440              # dense samples per analytic lap before resampling
CURVATURE_HALF_WINDOW = 4        # samples each side for curvature smoothing
CORNER_TURN_DEG = 15.0           # turn per sample treated as a sharp corner
FILLET_FRACTION = 0.25           # square/triangle corner fillet = this * radius
MIN_FILLET_MM = 5.0
TANGENTIAL_ACCEL_FRACTION = 0.6  # a_tan = this * a_lat (leaves margin for both)
MIN_INTERIOR_SPEED = 2.0         # mm/s floor (avoids zero-length time steps)
GOVERNOR_FULL_SPEED_ERR_MM = 8.0  # tracking error below this: full pace
GOVERNOR_STOP_ERR_MM = 25.0       # tracking error above this: path clock frozen
GOVERNOR_RATE_PER_S = 1.5         # max change of the pace factor per second
STOP_PACE = 0.02
MAX_PATH_SAMPLES = 400_000
SHAPES = ('Circle', 'Ellipse', 'Square', 'Triangle', 'Figure eight')


def clip_to_disc(xy, limit):
    xy = np.asarray(xy, dtype=float)
    if not np.all(np.isfinite(xy)):
        raise ValueError('Waypoint coordinates must be finite')
    length = float(np.linalg.norm(xy))
    return xy * (min(1.0, limit / length) if length else 1.0)


def _dedupe(vertices):
    v = np.asarray(vertices, float)
    if len(v) < 2:
        return v
    keep = np.r_[True, np.linalg.norm(np.diff(v, axis=0), axis=1) > 1e-9]
    return v[keep]


def resample_polyline(vertices, spacing=PATH_DS_MM):
    """Resample a polyline at (nearly) uniform arc-length spacing."""
    v = _dedupe(vertices)
    if len(v) < 2:
        return v[:1].copy()
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(v, axis=0), axis=1))]
    n = max(2, int(math.ceil(d[-1] / spacing)) + 1)
    s = np.linspace(0.0, d[-1], n)
    return np.column_stack((np.interp(s, d, v[:, 0]), np.interp(s, d, v[:, 1])))


def _outward_normal_angle(a, b):
    d = b - a                       # CCW polygon edge; outward normal = (dy, -dx)
    return math.atan2(-d[0], d[1])


def _rounded_polygon(sides, radius, first_angle):
    """Regular CCW polygon whose outer extent is `radius`, with filleted corners."""
    fillet = min(max(MIN_FILLET_MM, FILLET_FRACTION * radius), 0.9 * radius)
    angles = first_angle + 2 * np.pi * np.arange(sides) / sides
    corners = (radius - fillet) * np.column_stack((np.cos(angles), np.sin(angles)))
    arc_n = max(12, LOOP_SAMPLES // (2 * sides))
    pts = []
    for i in range(sides):
        prev_c, c, next_c = corners[i - 1], corners[i], corners[(i + 1) % sides]
        a0 = _outward_normal_angle(prev_c, c)
        a1 = _outward_normal_angle(c, next_c)
        while a1 < a0:
            a1 += 2 * np.pi
        th = np.linspace(a0, a1, arc_n)
        pts.append(c + fillet * np.column_stack((np.cos(th), np.sin(th))))
    pts = np.vstack(pts)
    return np.vstack((pts, pts[:1]))


def shape_loop(name, center, radius):
    """One closed CCW lap (last point == first point)."""
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError('Shape radius must be positive and finite')
    t = np.linspace(0, 2 * math.pi, LOOP_SAMPLES + 1)
    if name == 'Circle':
        raw = np.column_stack((radius * np.cos(t), radius * np.sin(t)))
    elif name == 'Ellipse':
        raw = np.column_stack((radius * np.cos(t), 0.6 * radius * np.sin(t)))
    elif name == 'Figure eight':
        raw = np.column_stack((radius * np.sin(t), 0.5 * radius * np.sin(2 * t)))
    elif name == 'Square':
        raw = _rounded_polygon(4, radius, math.pi / 4)
    elif name == 'Triangle':
        raw = _rounded_polygon(3, radius, math.pi / 2)
    else:
        raise ValueError('Unknown trajectory shape: ' + str(name))
    raw[-1] = raw[0]
    return raw + np.asarray(center, float)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def plan_speed_profile(P, v_max, a_lat, a_tan, stop_indices=()):
    """Curvature- and acceleration-limited speed profile along polyline P.

    Returns (s, t, v, V, A, kappa): arc length, time, speed, velocity vectors,
    acceleration vectors and signed curvature at every sample.
    """
    P = np.asarray(P, float)
    n = len(P)
    seg = np.diff(P, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.zeros(n)
    turn[1:-1] = _wrap(np.diff(heading))
    # Signed curvature: summed turning over a small window / window length.
    csum = np.r_[0.0, np.cumsum(turn)]
    idx = np.arange(n)
    lo = np.clip(idx - CURVATURE_HALF_WINDOW, 0, n)
    hi = np.clip(idx + CURVATURE_HALF_WINDOW + 1, 0, n)
    kappa = (csum[hi] - csum[lo]) / ((hi - lo) * float(np.mean(ds)))

    vlim = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-9)))
    sharp = np.abs(turn) > math.radians(CORNER_TURN_DEG)
    vlim[sharp] = 0.0
    hard_stops = [0, n - 1] + [int(i) for i in stop_indices]
    vlim[hard_stops] = 0.0

    v = vlim.copy()
    two_a_ds = 2.0 * a_tan * ds
    for i in range(1, n):                       # acceleration limit
        cap = math.sqrt(v[i - 1] * v[i - 1] + two_a_ds[i - 1])
        if v[i] > cap:
            v[i] = cap
    for i in range(n - 2, -1, -1):              # deceleration limit
        cap = math.sqrt(v[i + 1] * v[i + 1] + two_a_ds[i])
        if v[i] > cap:
            v[i] = cap
    interior = np.ones(n, bool)
    interior[hard_stops] = False
    v[interior] = np.maximum(v[interior], MIN_INTERIOR_SPEED)

    dt = 2.0 * ds / np.maximum(v[:-1] + v[1:], 1e-9)
    t = np.r_[0.0, np.cumsum(dt)]

    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    N = np.column_stack((-T[:, 1], T[:, 0]))
    a_t = np.gradient(0.5 * v * v, s)
    V = T * v[:, None]
    A = T * a_t[:, None] + N * (kappa * v * v)[:, None]
    return s, t, v, V, A, kappa


class TrajectoryPlanner:
    """Plans a full-state reference (position, velocity, acceleration)."""

    def __init__(self, usable_radius_mm, speed_mm_s=30.0, accel_mm_s2=120.0):
        self.limit = float(usable_radius_mm)
        self.speed = float(speed_mm_s)
        self.accel = float(accel_mm_s2)
        self._last_update = None
        self._set_hold(np.zeros(2))

    # ------------------------------------------------------------------ setup
    def _set_hold(self, xy):
        xy = np.asarray(xy, float).copy()
        self.mode = 'Hold'
        self.running = False
        self.stopping = False
        self.reference = xy.copy()
        self.target = xy.copy()
        self.path = xy.reshape(1, 2).copy()
        self.display_path = self.path
        self._cols = None
        self.tau = 0.0
        self.duration = 0.0
        self.pace = 0.0
        self.laps = 0
        self.lap = 0
        self._lead_len = 0.0
        self._lap_len = 0.0

    def _begin(self, P, mode, speed, accel, stop_indices=(), lead_len=0.0,
               lap_len=0.0, laps=0, display_end=None):
        speed = self.speed if speed is None else float(speed)
        accel = self.accel if accel is None else float(accel)
        if not (math.isfinite(speed) and speed > 0):
            raise ValueError('Speed must be positive')
        if not (math.isfinite(accel) and accel > 0):
            raise ValueError('Acceleration must be positive')
        P = np.asarray(P, float)
        if np.max(np.linalg.norm(P, axis=1)) > self.limit + 1e-6:
            raise ValueError('Trajectory exceeds the safe plate radius')
        if len(P) < 3:
            self._set_hold(P[-1])
            return
        self.speed, self.accel = speed, accel
        s, t, v, V, A, _ = plan_speed_profile(
            P, speed, accel, TANGENTIAL_ACCEL_FRACTION * accel, stop_indices)
        # One table row per sample: t, s, px, py, vx, vy, ax, ay
        self._cols = np.column_stack((t, s, P, V, A))
        self._t = t
        self.path = P
        end = len(P) if display_end is None else int(display_end)
        step = max(1, end // 400)
        disp = P[:end:step]
        self.display_path = np.vstack((disp, P[end - 1:end]))
        self.duration = float(t[-1])
        self.tau = 0.0
        self.pace = 1.0
        self.mode = mode
        self.running = True
        self.stopping = False
        self.target = P[-1].copy()
        self.reference = P[0].copy()
        self.laps, self.lap = int(laps), 0
        self._lead_len, self._lap_len = float(lead_len), float(lap_len)
        self._last_update = None

    def point(self, goal, current, now=None, speed=None, accel=None):
        end = clip_to_disc(goal, self.limit)
        start = clip_to_disc(current, self.limit)
        self._begin(resample_polyline([start, end]), 'Point', speed, accel)

    def shape(self, name, center, radius, current, now=None, speed=None,
              laps=1, accel=None):
        laps = int(laps)
        if laps < 1:
            raise ValueError('Laps must be at least 1')
        center = clip_to_disc(center, self.limit)
        if radius + np.linalg.norm(center) > self.limit + 1e-6:
            raise ValueError(f'Shape does not fit: radius {radius:.0f} mm + centre offset '
                             f'must be <= {self.limit:.0f} mm')
        loop = shape_loop(name, center, radius)[:-1]
        start = clip_to_disc(current, self.limit)
        j = int(np.argmin(np.linalg.norm(loop - start, axis=1)))
        loop = np.roll(loop, -j, axis=0)            # begin the lap nearest the ball
        lap = resample_polyline(np.vstack((loop, loop[:1])))
        lap_len = float(np.sum(np.linalg.norm(np.diff(lap, axis=0), axis=1)))
        if laps * len(lap) > MAX_PATH_SAMPLES:
            raise ValueError('Too many laps for this shape size; reduce laps')
        laps_pts = np.vstack([lap] + [lap[1:]] * (laps - 1))
        lead_len = float(np.linalg.norm(lap[0] - start))
        if lead_len >= 2 * PATH_DS_MM:
            lead = resample_polyline([start, lap[0]])
            P = np.vstack((lead[:-1], laps_pts))
            stops = (len(lead) - 1,)                # stop at the lead-in corner
            lead_n = len(lead) - 1
        else:
            P, stops, lead_n, lead_len = laps_pts, (), 0, 0.0
        self._begin(P, name, speed, accel, stops, lead_len, lap_len, laps,
                    display_end=lead_n + len(lap))

    def stop(self, immediate=False):
        """Soft stop decelerates along the path, then holds; immediate holds now."""
        if immediate or not self.running:
            self._set_hold(self.reference)
        else:
            self.stopping = True

    # --------------------------------------------------------------- runtime
    def _row(self, tau):
        """Linearly interpolated table row at path time tau (one lookup)."""
        tab, t = self._cols, self._t
        tau = min(max(tau, 0.0), self.duration)
        i = int(np.searchsorted(t, tau))
        if i <= 0:
            return tab[0]
        t0, t1 = t[i - 1], t[i]
        w = (tau - t0) / (t1 - t0) if t1 > t0 else 0.0
        return tab[i - 1] + w * (tab[i] - tab[i - 1])

    def _sample(self, tau):
        r = self._row(tau)
        return r[2:4], r[4:6], r[6:8]

    def update(self, now, position=None, horizon=0.0):
        """Advance the path clock and return (p_ref, v_ref, a_ref).

        `position` should be the latency-compensated ball estimate; the
        reference is sampled `horizon` seconds ahead (scaled by the pace) so
        it lines up with that prediction.  position=None freezes the clock.
        """
        zero = np.zeros(2)
        if not self.running:
            self._last_update = now
            return self.reference.copy(), zero, zero.copy()
        if position is None:
            self._last_update = now
            self.pace = 0.0
            return self.reference.copy(), zero, zero.copy()
        dt = 0.0 if self._last_update is None else min(max(now - self._last_update, 0.0), 0.1)
        self._last_update = now
        horizon = max(0.0, float(horizon))

        f = self.pace
        look = self._row(self.tau + f * horizon)
        err = math.hypot(float(position[0]) - look[2], float(position[1]) - look[3])
        if self.stopping:
            goal = 0.0
        else:
            goal = (GOVERNOR_STOP_ERR_MM - err) / (GOVERNOR_STOP_ERR_MM - GOVERNOR_FULL_SPEED_ERR_MM)
            goal = min(1.0, max(0.0, goal))
        step = GOVERNOR_RATE_PER_S * dt
        f_new = f + min(step, max(-step, goal - f))
        f_dot = (f_new - f) / dt if dt > 0 else 0.0
        self.pace = f = f_new
        self.tau += f * dt

        if self.tau >= self.duration:
            self._set_hold(self.target)
            return self.reference.copy(), zero, zero.copy()
        now_row = self._row(self.tau)
        if self.stopping and f <= STOP_PACE:
            self._set_hold(now_row[2:4])
            return self.reference.copy(), zero, zero.copy()

        self.reference = now_row[2:4].copy()
        if self.laps and self._lap_len > 0:
            s_now = float(now_row[1])
            if s_now < self._lead_len:
                self.lap = 0
            else:
                self.lap = min(self.laps, int((s_now - self._lead_len) // self._lap_len) + 1)
        r = self._row(self.tau + f * horizon)
        V = r[4:6]
        return r[2:4], f * V, f * f * r[6:8] + f_dot * V

    def status_text(self):
        if not self.running:
            return 'Hold'
        pace = f'pace {100 * self.pace:3.0f}%'
        if self.stopping:
            return f'{self.mode}: stopping ({pace})'
        if self.laps:
            lap = 'lead-in' if self.lap == 0 else f'lap {self.lap}/{self.laps}'
            return f'{self.mode}: {lap}, {pace}'
        return f'{self.mode}: {pace}'