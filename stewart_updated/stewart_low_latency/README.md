# Lower-latency Stewart ball balancer

## Replace the old files

Place all five `.py` files in your project directory, alongside your **existing** `config.json`, `camera_intrinsics.npz`, and `platform_camera_mapping.json` (if previously calibrated). These three calibration/config files were not uploaded and are not included here. Keep backups of the originals.

Run dry-run: `python stewart_ball_balancer.py`

Run servo output: `python stewart_ball_balancer.py --hardware`

Run camera-only diagnostic: `python stewart_ball_balancer.py --headless`

## Changes

* Detect in raw camera frames. Undistort only ball center coordinates at control frequency. Full-frame rectification for the Tk preview runs at 5 Hz, preserving calibration coordinates.
* Camera requests MJPEG 60 FPS with 1-frame buffering when driver supports it; prints actual reported camera mode.
* Servo PWM writes skipped when under 0.15 degree from *last transmitted command*; target angle still rate-limited.
* CSV logging downsampled to 10 Hz; graph and preview rates remain low.
* Path follows a forward lookahead on the continuous geometric contour, with bounded forward progress from actual ball position, instead of a clock-only advancing setpoint.
* Both go-to-point and all five shapes remain, including GUI sliders for PID gains, speed and size, trim, and camera calibration.

## Start conservatively

1. **Begin dry-run** and verify tracking direction, axis signs and camera calibration before enabling servo movement.
2. Test the **point** target before shapes. Start at a slow path speed of 5-10 mm/s and shape radius 15-25 mm. Once stable, increase gradually.
3. Observe `camera FPS`, `control FPS`, `proc ms`, and `age ms` in the GUI. Camera frame timestamp is attached **after** OpenCV `read()`; the age display does **not** measure sensor exposure-to-delivery latency. To measure true end-to-end lag, use an LED/clock-in-view test.
4. The camera may ignore MJPEG/FPS/buffer settings. Use `v4l2-ctl --list-formats-ext -d /dev/video0` to see supported modes (where V4L2 is available). If frame delay persists, investigate USB bandwidth, camera driver internal buffering, exposure time and operating mode.
5. If desktop mouse remains unresponsive, test GUI vs `--headless` to isolate display load. Disable unnecessary Pi desktop processes or use Pi on-camera-control headless with remote display. Check `top`, `vcgencmd get_throttled` and CPU temperatures for throttling.
6. The existing 30 deg/s servo slew cap can also cause *physical* lag. Do not increase it before confirming geometry and motion safety. The supplied temporary 2x kinematic correction remains unchanged.

## Limitations

This was statically checked and trajectory logic was tested in isolation, but was **not run on a Raspberry Pi**. OpenCV driver support and servo timing require testing. The ball-to-platform map remains an approximate level-plane affine mapping, not tilt-aware 3D calibration. A faster vision loop cannot compensate for an inaccurate mapping, wrong motor sign, PID gains, or slow actuators. Track a suitable colored ball under even lighting.
