"""Interactive checkerboard intrinsic calibration for a Stewart-platform camera.

Set CHECKERBOARD_INNER_CORNERS to the number of *inner intersections*,
not the number of black/white squares. Keep camera focus, zoom and resolution
fixed between calibration and use.

OUTPUT: intrinsic cam matrix and distortion coefficients
"""
import time
from pathlib import Path

import cv2
import numpy as np

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CHECKERBOARD_INNER_CORNERS = (10,7)  # columns, rows of internal corners; CHANGE to match yours
SQUARE_SIZE_MM = 23.5  # CHANGE to measured square side length
MIN_SAMPLES = 15
TARGET_SAMPLES = 25
MIN_SAMPLE_INTERVAL_S = 0.7
MIN_CORNER_SHIFT_PX = 22.0
OUTPUT_PATH = Path(__file__).with_name("camera_intrinsics.npz")


def object_point_grid():
    columns, rows = CHECKERBOARD_INNER_CORNERS
    grid = np.zeros((rows * columns, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    grid *= SQUARE_SIZE_MM
    return grid


def calibrate(object_points, image_points, image_size):
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view_errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        err = np.linalg.norm(img.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        per_view_errors.append(float(np.sqrt(np.mean(err ** 2))))
    np.savez(
        OUTPUT_PATH,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        checkerboard_inner_corners=np.array(CHECKERBOARD_INNER_CORNERS),
        square_size_mm=SQUARE_SIZE_MM,
        rms_reprojection_error=rms,
        per_view_rms_errors=np.array(per_view_errors),
    )
    print("\nCalibration saved to:", OUTPUT_PATH)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coeffs.ravel())
    print(f"Overall RMS reprojection error: {rms:.3f} pixels")
    print(f"Worst sample RMS reprojection error: {max(per_view_errors):.3f} pixels")
    print("Lower reprojection error is better, but also check coverage across the image.")


def main():
    if SQUARE_SIZE_MM <= 0:
        raise ValueError("SQUARE_SIZE_MM must be positive")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    image_size = None
    obj_template = object_point_grid()
    object_points = []
    image_points = []
    last_sample_corners = None
    last_sample_time = 0.0

    print("Show the entire checkerboard and move/tilt it across the field of view.")
    print("Press c to calibrate after at least", MIN_SAMPLES, "samples; q to quit.")
    print("Automatically calibrates at", TARGET_SAMPLES, "samples.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if image_size is None:
                image_size = (w, h)
                print("Actual camera resolution:", image_size)
            elif image_size != (w, h):
                raise RuntimeError("Camera resolution changed during calibration")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKERBOARD_INNER_CORNERS,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(frame, CHECKERBOARD_INNER_CORNERS, corners, found)
                now = time.monotonic()
                shifted = (last_sample_corners is None or
                           float(np.mean(np.linalg.norm(
                               corners.reshape(-1, 2) - last_sample_corners.reshape(-1, 2),
                               axis=1))) >= MIN_CORNER_SHIFT_PX)
                if shifted and now - last_sample_time >= MIN_SAMPLE_INTERVAL_S:
                    object_points.append(obj_template.copy())
                    image_points.append(corners.copy())
                    last_sample_corners = corners.copy()
                    last_sample_time = now
                    print(f"Accepted calibration sample {len(image_points)}/{TARGET_SAMPLES}")

            cv2.putText(frame, f"Checkerboard: {'FOUND' if found else 'not found'}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 220, 0) if found else (0, 0, 255), 2)
            cv2.putText(frame, f"Samples: {len(image_points)}/{TARGET_SAMPLES} | c: calibrate | q: quit",
                        (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Stewart Platform Camera Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopped without saving calibration")
                break
            if key == ord('c') and len(image_points) >= MIN_SAMPLES:
                calibrate(object_points, image_points, image_size)
                break
            if key == ord('c'):
                print(f"Need at least {MIN_SAMPLES} samples, got {len(image_points)}")
            if len(image_points) >= TARGET_SAMPLES:
                calibrate(object_points, image_points, image_size)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
