"""
Pixel -> mm calibration for the ball balancer.

Coordinates: x = 0 mm at the motor end, x = 220 mm at the far end.
Keep the beam level (servo at LEVEL_DEG) and the camera fixed.

For each target position, place the ball CENTER at that mark, press SPACE
to capture. Press 'q' to finish early. A line fit of pixel_x vs mm gives
pixels_per_mm, and tells you which image direction the motor end is.
"""
import cv2
import numpy as np

from ball_detection import BallDetector

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Positions (mm from motor end) to place the ball at
POSITIONS_MM = [0, 40, 80, 110, 140, 180, 220]
SAMPLES = 30  # frames averaged per capture


def capture(cap, det):
    xs = []
    while len(xs) < SAMPLES:
        ok, frame = cap.read()
        if not ok:
            continue
        found, center, _, _ = det.detect_ball(frame)
        if found:
            xs.append(center[0])
    return float(np.median(xs))


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    det = BallDetector()
    mm_pts, px_pts = [], []

    try:
        for target in POSITIONS_MM:
            print(f"\nPlace ball center at x = {target} mm, then press SPACE "
                  f"(q to finish)")
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                found, center, radius, _ = det.detect_ball(frame)
                vis = frame.copy()
                if found:
                    cv2.circle(vis, (int(center[0]), int(center[1])),
                               int(radius), (0, 255, 0), 2)
                    cv2.putText(vis, f"px_x = {center[0]:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis, f"Place ball at {target} mm  [SPACE=capture]",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.imshow("Calibration", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
                if key == ord(" ") and found:
                    px = capture(cap, det)
                    mm_pts.append(target)
                    px_pts.append(px)
                    print(f"  captured: {target} mm -> {px:.1f} px")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(mm_pts) < 2:
        print("Need at least 2 points.")
        return

    mm = np.array(mm_pts, float)
    px = np.array(px_pts, float)
    slope, intercept = np.polyfit(mm, px, 1)  # px = slope*mm + intercept
    resid = px - (slope * mm + intercept)

    print("\n================ RESULT ================")
    print(f"pixels_per_mm   = {abs(slope):.4f}")
    print(f"RMS fit error   = {np.sqrt(np.mean(resid**2)):.2f} px "
          f"({np.sqrt(np.mean(resid**2)) / abs(slope):.2f} mm)")
    print(f"Pixel x at 0 mm (motor end) = {intercept:.1f}")
    print(f"Pixel x at 110 mm (center)  = {slope * 110 + intercept:.1f}")
    if slope > 0:
        print("Motor end is on the LEFT of the image (px increases away "
              "from motor).")
    else:
        print("Motor end is on the RIGHT of the image (px decreases away "
              "from motor).")
    print("\nIn your balancer script set:")
    print(f"  self.pixels_per_mm = {abs(slope):.4f}")
    print("Sign of x_mm = (ball_x - center_x) / pixels_per_mm is image-right "
          "positive;")
    print("flip it if your physical +x direction (toward the motor?) is "
          "opposite.")


if __name__ == "__main__":
    main()