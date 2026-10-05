# Sensor calibration. Redo this whenever the lighting changes.
# Saves the no-ball and ball-on-top readings for each sensor to balancer_config.json.
import time

import numpy as np

from balancer_hw import BallEstimator, load_config, save_config
from read_sensors import SensorReader


def record(reader, seconds):
    samples, count, end = [], reader.count, time.monotonic() + seconds
    while time.monotonic() < end:
        got = reader.wait_new(count)
        if got:
            vals, _, count = got
            samples.append(vals)
    return np.array(samples, dtype=float)


def main():
    cfg = load_config()
    reader = SensorReader()
    try:
        input("\n1) Take the ball off the beam and press Enter")
        empty = record(reader, 2.0)
        base, noise = empty.mean(axis=0), empty.std(axis=0)
        print(f"  {len(empty)} samples ({len(empty) / 2:.0f} Hz)")
        print("  no ball:", np.round(base).astype(int).tolist())
        print("  noise:  ", np.round(noise, 1).tolist())

        input("\n2) Press Enter, then slowly roll the ball over every sensor a few times (15 s)")
        roll = record(reader, 15.0)
        # for each sensor, take the reading furthest from its no-ball value
        idx = np.argmax(np.abs(roll - base), axis=0)
        covered = roll[idx, np.arange(roll.shape[1])]
        print("  covered:", np.round(covered).astype(int).tolist())
        for i in range(8):
            if abs(covered[i] - base[i]) < 10 * max(noise[i], 1.0):
                print(f"  WARNING: sensor {i} barely changed, check the wiring/lighting")
        cfg["baseline"] = base.round(1).tolist()
        cfg["covered"] = covered.round(1).tolist()

        input("\n3) Hold the ball over the sensor closest to the motor end and press Enter")
        cov = BallEstimator(cfg).coverage(record(reader, 1.0).mean(axis=0))
        k = int(np.argmax(cov))
        print(f"  ball on sensor {k}")
        # +x is towards the motor end, so flip the sensor positions if that sensor is on the - side
        x = np.array(cfg["sensor_x_mm"], dtype=float)
        if x[k] < 0:
            x = -x
            print("  flipped sensor order")
        cfg["sensor_x_mm"] = x.tolist()
        save_config(cfg)
    except KeyboardInterrupt:
        print("\nCancelled, nothing saved")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
