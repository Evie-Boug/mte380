"""Distortion-corrected, planar X/Y ping-pong ball tracking.

Calibrate by clicking THREE platform attachment points in the camera view,
IN THE ORDER of config.json geometry.radial_angles_deg. Assumes these points
lie at platform_radius_mm in the platform plane. Keep camera fixed afterward.
The affine mapping is approximate when the platform tilts or the ball is above
its surface; it is not a full camera-to-moving-plane pose solution.
"""
from pathlib import Path
import json
import math
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIBRATION_PATH = HERE / 'camera_intrinsics.npz'
MAPPING_PATH = HERE / 'platform_camera_mapping.json'

# Tune these for YOUR ball. Orange is the default, as in the old script.
# For a white ball, use COLOR_MODE = 'white' with a dark/non-white plate.
COLOR_MODE = 'orange'  # 'orange' or 'white'
ORANGE_LOW = (5, 110, 90)
ORANGE_HIGH = (25, 255, 255)
WHITE_LOW = (0, 0, 145)
WHITE_HIGH = (179, 80, 255)
MIN_BALL_RADIUS_PX = 4
MAX_BALL_RADIUS_PX = 65
MIN_CIRCULARITY = 0.48
ROI_HALF_PX = 70

MAX_POSITION_RADIUS_MM = 160.0


class BallDetector:
    def __init__(self, config, calibration_path=CALIBRATION_PATH,
                 mapping_path=MAPPING_PATH, color_mode=COLOR_MODE):
        self.cfg = config
        self.color_mode = color_mode
        self.mapping_path = Path(mapping_path)
        with np.load(calibration_path) as npz:
            self.K = npz['camera_matrix']
            self.dist = npz['dist_coeffs']
            self.size = tuple(int(v) for v in npz['image_size'])
        # Precompute maps once rather than calling cv2.undistort on every frame.
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, self.size, cv2.CV_16SC2)
        self.affine = None
        self.image_clicks = []
        self._overlay_geometry = None
        self._inverse_affine = None
        self.load_mapping()
        self._last_raw_px = None

    def target_points_mm(self):
        g = self.cfg['geometry']
        r = float(g['platform_radius_mm'])
        angles = np.deg2rad(g['radial_angles_deg'])
        return np.column_stack((r * np.cos(angles), r * np.sin(angles))).astype(np.float32)

    def calibration_circle_mm(self):
        """Circumcircle through the three camera calibration attachment points.

        The image-to-platform affine mapping takes this circle through the
        corresponding three calibration clicks in the undistorted image.
        """
        pts = np.asarray(self.target_points_mm(), dtype=float)
        a, b, c = pts
        system = 2.0 * np.vstack((b - a, c - a))
        rhs = np.array([b @ b - a @ a, c @ c - a @ a])
        if abs(np.linalg.det(system)) < 1e-8:
            raise ValueError('Calibration points cannot define a circle (collinear)')
        center = np.linalg.solve(system, rhs)
        return center, float(np.linalg.norm(a - center))

    def load_mapping(self):
        if not self.mapping_path.exists():
            return
        d = json.loads(self.mapping_path.read_text())
        if tuple(d['image_size']) != self.size:
            raise ValueError('Saved platform mapping resolution differs from camera intrinsics')
        if not np.allclose(d['platform_points_mm'], self.target_points_mm(), atol=0.01):
            raise ValueError('Platform geometry changed since mapping. Recalibrate mapping.')
        self.affine = np.array(d['image_to_platform_affine'], dtype=float)
        self.image_clicks = list(d['image_points_px'])
        self._invalidate_overlay_geometry()

    def _invalidate_overlay_geometry(self):
        self._overlay_geometry = None
        self._inverse_affine = None

    def reset_mapping(self):
        self.affine = None
        self.image_clicks = []
        self._invalidate_overlay_geometry()

    def add_reference_click(self, x, y):
        self.image_clicks.append([float(x), float(y)])
        if len(self.image_clicks) == 3:
            src = np.asarray(self.image_clicks, np.float32)
            dst = self.target_points_mm()
            d1, d2 = src[1] - src[0], src[2] - src[0]
            area = abs(float(d1[0] * d2[1] - d1[1] * d2[0])) / 2  # 2-D cross (np.cross 2-D is removed in NumPy 2)
            if area < 150:
                self.reset_mapping()
                raise ValueError('Reference points nearly collinear. Please re-click.')
            self.affine = cv2.getAffineTransform(src, dst).astype(float)
            self._invalidate_overlay_geometry()
            self.mapping_path.write_text(json.dumps({
                'image_size': self.size,
                'image_points_px': self.image_clicks,
                'platform_points_mm': dst.tolist(),
                'image_to_platform_affine': self.affine.tolist()
            }, indent=2))
        return len(self.image_clicks)

    def undistort(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f'Camera frame resolution must be {self.size}, got '
                             f'{(frame.shape[1], frame.shape[0])}')
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def pixel_to_mm(self, xy):
        if self.affine is None:
            return None
        p = self.affine @ np.array([xy[0], xy[1], 1.0])
        return float(p[0]), float(p[1])

    def mm_to_pixel(self, xy):
        if self.affine is None:
            return None
        if self._inverse_affine is None:
            A = np.vstack((self.affine, [0, 0, 1]))
            self._inverse_affine = np.linalg.inv(A)
        p = self._inverse_affine @ np.array([xy[0], xy[1], 1.0])
        return tuple(int(round(v)) for v in p[:2])

    def detect(self, frame, raw_frame=False):
        x0 = y0 = 0
        sub = frame
        if self._last_raw_px is not None:
            h, w = frame.shape[:2]
            cx, cy = self._last_raw_px
            x0, x1 = max(0, int(cx) - ROI_HALF_PX), min(w, int(cx) + ROI_HALF_PX)
            y0, y1 = max(0, int(cy) - ROI_HALF_PX), min(h, int(cy) + ROI_HALF_PX)
            sub = frame[y0:y1, x0:x1]
        result = self._detect_in(sub, x0, y0, raw_frame)
        if not result['found'] and sub is not frame:
            result = self._detect_in(frame, 0, 0, raw_frame)  # lost: full search
        return result

    def _detect_in(self, img, x0, y0, raw_frame):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.color_mode == 'orange':
            mask = cv2.inRange(hsv, np.array(ORANGE_LOW, np.uint8), np.array(ORANGE_HIGH, np.uint8))
        elif self.color_mode == 'white':
            mask = cv2.inRange(hsv, np.array(WHITE_LOW, np.uint8), np.array(WHITE_HIGH, np.uint8))
        else:
            raise ValueError('color_mode must be orange or white')
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area <= 0 or perimeter <= 0 or 4 * math.pi * area / perimeter**2 < MIN_CIRCULARITY:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            if not MIN_BALL_RADIUS_PX <= radius <= MAX_BALL_RADIUS_PX:
                continue
            rx, ry = float(x) + x0, float(y) + y0          # full-frame raw pixel
            if raw_frame:
                corrected = cv2.undistortPoints(
                    np.array([[[rx, ry]]], dtype=np.float32), self.K, self.dist, P=self.K)[0, 0]
                xy = (float(corrected[0]), float(corrected[1]))
            else:
                xy = (rx, ry)
            xy_mm = self.pixel_to_mm(xy)
            if xy_mm is not None and math.hypot(*xy_mm) > MAX_POSITION_RADIUS_MM:
                continue
            candidates.append((area, xy, float(radius), xy_mm, (rx, ry)))
        if not candidates:
            self._last_raw_px = None
            return {'found': False, 'pixel': None, 'radius_px': None, 'xy_mm': None, 'mask': mask}
        _, pixel, radius, mm, raw_px = max(candidates, key=lambda cand: cand[0])
        self._last_raw_px = raw_px
        return {'found': True, 'pixel': pixel, 'radius_px': radius, 'xy_mm': mm, 'mask': mask}

    def draw(self, frame, result, setpoint=(0, 0), path_mm=None):
        out = frame.copy()
        if self.affine is not None:
            # Circle through all three calibrated attachment points.
            # Affine projection may look elliptical in camera perspective.
            # Static geometry is projected ONCE per calibration, not each GUI frame.
            if self._overlay_geometry is None:
                center, plate_r = self.calibration_circle_mm()
                angles = np.linspace(0, 2 * np.pi, 121, endpoint=False)
                xy = center + plate_r * np.column_stack((np.cos(angles), np.sin(angles)))
                inv = self._inverse_affine
                if inv is None:
                    inv = np.linalg.inv(np.vstack((self.affine, [0, 0, 1])))
                    self._inverse_affine = inv
                uv = np.column_stack((xy, np.ones(len(xy)))) @ inv.T
                outline = np.rint(uv[:, :2]).astype(np.int32).reshape(-1, 1, 2)
                anchors = [self.mm_to_pixel(p) for p in self.target_points_mm()]
                self._overlay_geometry = (outline, anchors, self.mm_to_pixel((0, 0)),
                                          self.mm_to_pixel((50, 0)), self.mm_to_pixel((0, 50)))
            outline, anchors, origin, axis_x, axis_y = self._overlay_geometry
            cv2.polylines(out, [outline], True, (255, 190, 0), 2)
            if path_mm is not None and len(path_mm) > 1:
                pts = np.asarray(path_mm, float)
                uv = np.column_stack((pts, np.ones(len(pts)))) @ self._inverse_affine.T
                cv2.polylines(out, [np.rint(uv[:, :2]).astype(np.int32).reshape(-1, 1, 2)],
                              False, (186, 105, 107), 1, cv2.LINE_AA)
            for pt in anchors:
                cv2.circle(out, pt, 4, (255, 190, 0), 2)
            cv2.circle(out, origin, 5, (0, 0, 255), 2)
            goal = self.mm_to_pixel(setpoint)
            cv2.drawMarker(out, goal, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.arrowedLine(out, origin, axis_x, (255, 0, 0), 2)
            cv2.arrowedLine(out, origin, axis_y, (0, 255, 0), 2)
        for i, p in enumerate(self.image_clicks):
            cv2.circle(out, tuple(map(int, p)), 6, (255, 0, 255), 2)
            cv2.putText(out, str(i+1), (int(p[0])+8, int(p[1])-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if result['found']:
            x, y = map(int, result['pixel'])
            cv2.circle(out, (x, y), int(result['radius_px']), (0, 255, 0), 2)
            cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
            if result['xy_mm'] is not None:
                cv2.putText(out, 'x={:+.1f} y={:+.1f} mm'.format(*result['xy_mm']),
                            (max(5, x-80), max(22, y-22)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)
        return out