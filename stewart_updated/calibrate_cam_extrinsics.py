"""Track a checkerboard rigidly mounted on a Stewart platform.

Uses intrinsics produced by calibrate_camera.py. Camera must stay FIXED.
Press h at the physically level/home platform pose to establish the reference.

IMPORTANT: Relative XYZ and Euler angles are expressed in the checkerboard's
HOME coordinate frame. They equal platform heave/roll/pitch only when axes are
aligned and the checkerboard-to-platform-center offset is correctly specified.
"""
from pathlib import Path
import math

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIBRATION_PATH = HERE / "camera_intrinsics.npz"
CAMERA_ID = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
# Coordinates of the PLATFORM CENTER relative to the checkerboard's first
# INNER corner, expressed along checkerboard x/y/z axes, in millimetres.
# Replace after measuring your board's placement on the platform.
BOARD_ORIGIN_TO_PLATFORM_CENTER_MM = np.array([0.0, 0.0, 0.0])


def rotation_to_angles_ry_rx(R):
    """Angles consistent with R = Ry(pitch) @ Rx(roll), when yaw ~ zero."""
    roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    pitch = math.degrees(math.atan2(-R[2, 0], R[0, 0]))
    # For perfectly Ry @ Rx motion, R[1,0] == 0. Nonzero signals yaw/misalignment.
    return roll, pitch


def put(frame, line, y, color=(255, 255, 255)):
    cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def main():
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"Calibration file missing: {CALIBRATION_PATH}")
    with np.load(CALIBRATION_PATH) as cal:
        K = cal["camera_matrix"]
        dist = cal["dist_coeffs"]
        expected_size = tuple(map(int, cal["image_size"]))
        pattern = tuple(map(int, cal["checkerboard_inner_corners"]))
        square_mm = float(cal["square_size_mm"])

    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= square_mm
    axis_mm = square_mm * 2.0
    axes = np.float32([[0, 0, 0], [axis_mm, 0, 0],
                       [0, axis_mm, 0], [0, 0, -axis_mm]])

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    home_R = None
    home_t = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    print("Keep the overhead camera fixed and mount the checkerboard to the moving plate.")
    print("At physical home pose, press h to set reference. Press r to reset; q to quit.")
    print("Axes and relative translations use the checkerboard's HOME frame.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            if (width, height) != expected_size:
                raise RuntimeError(
                    f"Camera resolution {(width, height)} differs from calibrated "
                    f"resolution {expected_size}; reconfigure camera or recalibrate."
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            current = None
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                success, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.reshape(3)
                    current = (R, t)
                    cv2.drawChessboardCorners(frame, pattern, corners, True)
                    projected, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
                    pts = np.round(projected.reshape(-1, 2)).astype(int)
                    origin = tuple(pts[0])
                    for endpoint, color in zip(pts[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        cv2.line(frame, origin, tuple(endpoint), color, 3)
                    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                    rms = float(np.sqrt(np.mean(np.sum(
                        (proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
                    put(frame, f"Board detected | PnP reprojection RMS: {rms:.2f} px", 28, (0, 255, 0))
                    put(frame, f"Board in camera frame (mm): X={t[0]:+.1f} Y={t[1]:+.1f} Z={t[2]:+.1f}", 55)
                    if home_R is not None:
                        relative_R = home_R.T @ R
                        # Track platform center instead of board origin to account for
                        # displacement caused by tilt around the platform center.
                        cur_center = t + R @ BOARD_ORIGIN_TO_PLATFORM_CENTER_MM
                        home_center = home_t + home_R @ BOARD_ORIGIN_TO_PLATFORM_CENTER_MM
                        dp = home_R.T @ (cur_center - home_center)
                        roll, pitch = rotation_to_angles_ry_rx(relative_R)
                        put(frame, f"Relative center (home-board axes, mm): {dp[0]:+.1f}, {dp[1]:+.1f}, {dp[2]:+.1f}", 84)
                        put(frame, f"Relative roll={roll:+.2f} deg  pitch={pitch:+.2f} deg", 112)
                        put(frame, "Not platform heave/roll/pitch until board alignment is verified", 140, (0, 220, 255))
                    else:
                        put(frame, "Move to physical home, then press h", 84, (0, 220, 255))
                else:
                    put(frame, "Checkerboard found; solvePnP failed", 28, (0, 0, 255))
            else:
                put(frame, "Checkerboard not found; pose unavailable", 28, (0, 0, 255))
            put(frame, "h: set home  |  r: reset home  |  q: quit", height - 14)
            cv2.imshow("Stewart Platform Pose Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                home_R = home_t = None
                print("Home reference cleared")
            if key == ord("h"):
                if current is None:
                    print("Cannot set home: checkerboard pose not found")
                else:
                    home_R, home_t = (current[0].copy(), current[1].copy())
                    print("Home reference recorded (for this session only)")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
