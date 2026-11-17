# Low-latency Stewart platform balancer (experimental update)

This package replaces the previous synchronous GUI + PID loop with:

- Dedicated camera capture thread, replacing stale/unread frames instead of queuing them.
- Independent vision/PID/servo thread. GUI redraws cannot block control.
- GUI refresh at 10 Hz while the controller targets up to 50 Hz (actual rate is camera/CPU dependent).
- Cached OpenCV undistortion remap, computed once at startup.
- Reduced camera-overlay rendering rate.
- Runtime display of processing ms and camera/control FPS.
- Fails safe toward level on a stale or missing camera frame (within limits of servo slew rate).

Run using the same commands as before:

```bash
python stewart_ball_balancer.py                 # GUI, no servo output
python stewart_ball_balancer.py --hardware     # real servo output
python stewart_ball_balancer.py --hardware --color white
python stewart_ball_balancer.py --headless     # camera monitor, no PID
```

## Important

- KEEP your existing `platform_camera_mapping.json` beside the scripts if you have already completed the three-point calibration. Otherwise recalibrate in the GUI. You can copy the JSON from the original program directory.
- In `stewart_ball_balancer.py`, `KINEMATICS_ANGLE_FACTOR = 2.0` remains unchanged, compensating for the observed ~2x IK-to-physical angle discrepancy. It is not a verified calibration.
- The hard maximum `MAX_SERVO_RATE_DPS = 30` still caps command changes, which may be too conservative for good balancing. Do not increase it without first testing low-angle motion and observing actual mechanical speed, backlash, and power constraints.
- `CONTROL_HZ = 50` is an upper target, not a guaranteed real-time deadline. 30 fps cameras cannot generate 50 fresh independent observations/sec.
- Keep the camera at its calibrated resolution. Do not resize images after calibration or your X/Y mapping will be wrong.
- Camera-to-platform mapping is approximate when the platform tilts, and PID direction signs must be experimentally verified.
- **Not tested on actual Raspberry Pi hardware or camera.** Try with PID disabled and hardware off first, check measured camera/control FPS and processing ms, then test low-angle motor commands while supervised and with a physical emergency power cutoff.
- GUI calculations access a few shared Python variables. They are adequate for an experimental prototype but are not a hardened real-time safety system.
- Servo command safety is based on requested angles, not encoder feedback; servo mechanical lag and camera exposure delay still affect stability.

## Performance diagnosis

1. Observe `camera FPS` vs `control FPS` vs `processing ms` in GUI.
2. If camera FPS is low or ball position visibly lags, check exposure/auto-exposure, USB bandwidth, camera driver buffering, and camera FPS mode. The OS/backend may ignore `CAP_PROP_BUFFERSIZE=1`.
3. If processing takes >20-30 ms, investigate ROI cropping, optimized OpenCV build, and resolution. Do not change calibrated resolution without recalibration.
4. If servo motion follows commands slowly despite prompt ball detection, measure physical servo travel, power supply droop, and output slew-rate limit. Increasing PID gains does not cure mechanical latency.

# Stewart platform ball balancer

Three replacement scripts:
- `stewart_ball_detection.py`: HSV ball detector, intrinsic undistortion, pixel-to-platform XY mapping.
- `stewart_ball_balancer.py`: two-axis PID controller and PCA9685 IK bridge.
- `stewart_balancer_gui.py`: camera, 2D platform, 10-second plots, live sliders.

The original `stewart_kinematics.py`, `config.json`, and `camera_intrinsics.npz` are included unchanged.

## Install and run

On the computer with the camera, Python, and PCA9685:

```bash
pip install numpy opencv-python pillow adafruit-circuitpython-servokit
python stewart_ball_balancer.py             # dry run, no servo output
python stewart_ball_balancer.py --hardware  # real servos
python stewart_ball_balancer.py --hardware --color white
```

The GUI uses Tkinter, which may need `sudo apt install python3-tk` on a Raspberry Pi.

## Camera-to-platform calibration

Intrinsic parameters only remove lens distortion. They **do not** calibrate platform center, axes, or mm/pixel.

1. Keep platform physically level and camera stationary. Ensure all three attachment point centers are visible.
2. Click **Calibrate 3 platform points** in GUI.
3. Click the centers of the *moving platform's* three upper link attachment joints, in exact `config.json -> geometry.radial_angles_deg` order. For the included config: 90 degrees (top), -30 degrees (bottom right), -150 degrees (bottom left), when viewed in platform coordinates. These are *platform* directions, NOT necessarily camera directions.
4. Blue outline should match the platform attachment triangle. Blue X and green Y arrows should point along actual physical +X and +Y. Save happens automatically to `platform_camera_mapping.json`.
5. Verify moving the ball along known distances yields corresponding mm readings; remeasure real radial attachment points if necessary.

This is an **affine, level-plane approximation**, not extrinsic tracking of the moving plate. Ball height, camera perspective, and tilt introduce errors, especially off-center. For better accuracy use a plate homography and/or continuously track fiducials attached to the moving plate. If camera views strong perspective, don't expect mm-level accuracy from three-point affine calibration.

## Tuning and safety

- `KINEMATICS_ANGLE_FACTOR = 2.0` multiplies *physical* requested roll/pitch before passing to existing IK, as requested. It does **not** change motor/servo angle, and is only a temporary correction for the stated factor-of-two mismatch.
- X and Y control are independent PIDs. With the included `R = Ry(pitch) @ Rx(roll)`, X error drives pitch and Y error drives roll. Sign choices are `PITCH_SIGN=-1`, `ROLL_SIGN=+1`; **test physical sign** with tiny gains and a ball placed off center. Incorrect axes/signs cause positive feedback.
- Starts with PID disabled and hardware disabled by default. `--hardware` must be passed to drive motors. First check the camera mapping and direction with no ball and tiny gains.
- Start with **Kp around 0.01 to 0.03, Ki = 0, Kd around 0.01**; adjust after measuring servo responsiveness. Settings are intentionally conservative starting guesses, not tuned gains.
- Camera loss longer than 0.30 s returns platform to trimmed level and resets PID. IK failures reject targets. Servo output is slew-limited.
- The last requested IK pose is recorded on shutdown; it is NOT encoder feedback. The mechanism is NOT automatically moved to neutral on startup. Ensure it is manually leveled before enabling hardware. PWM remains active on shutdown to avoid a free-fall.
- The controller's shutdown sends one rate-limited level command, so the platform might not finish physically leveling when you quit. Always supervise operation and provide a physical emergency power cutoff.
- Uses `heave_mm=0` throughout; the platform's existing IK configuration determines level elevation.
- Writes CSV logs into `logs/`.
