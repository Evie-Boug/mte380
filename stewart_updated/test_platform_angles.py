#!/usr/bin/env python3
"""Interactive, deliberately slow platform-angle test."""
from stewart_kinematics import (load_config, make_kit, move_pose, HOME_POSE, print_dry_run, load_last_pose)

def main():
    cfg = load_config()
    print("3-actuator platform angle test")
    print("Start with clear mechanical travel and an emergency power disconnect.")
    print("Test angles are intentionally small.")
    saved = load_last_pose()
    if saved is None:
        print("No saved pose exists.")
        print("Only proceed if the platform is physically level at [110, 35, 50].")
        if input("Is it physically at calibrated level? (yes/no): ").strip().lower() != "yes":
            print("Aborted. Re-home the mechanism before testing.")
            return
        current = HOME_POSE.copy()
    else:
        current = saved
        print("Resuming from last saved commanded pose:", current)
        print("If the mechanism was moved by hand or lost power, re-home it first.")

    kit = make_kit(cfg)
    while True:
        print("\\nCurrent commanded pose:", current)
        print("""
  1) Roll +1 degree
  2) Roll -1 degree
  3) Pitch +1 degree
  4) Pitch -1 degree
  5) Roll +2 degrees
  6) Pitch +2 degrees
  7) Return to level
  8) Enter custom small pose
  9) Dry-run custom pose (no motor movement)
  q) Quit
""")
        choice = input("Select: ").strip().lower()
        if choice == "q":
            break
        target = current.copy()
        if choice == "1": target["roll_deg"] += 1
        elif choice == "2": target["roll_deg"] -= 1
        elif choice == "3": target["pitch_deg"] += 1
        elif choice == "4": target["pitch_deg"] -= 1
        elif choice == "5": target["roll_deg"] += 2
        elif choice == "6": target["pitch_deg"] += 2
        elif choice == "7": target = HOME_POSE.copy()
        elif choice in ("8", "9"):
            target = {
                "heave_mm": float(input("Heave relative to level (mm, start with 0): ")),
                "roll_deg": float(input("Roll (degrees, start within +/-1): ")),
                "pitch_deg": float(input("Pitch (degrees, start within +/-1): "))
            }
            if choice == "9":
                try: print_dry_run(target, cfg)
                except ValueError as e: print("Rejected:", e)
                continue
        else:
            print("Unknown choice.")
            continue
        try:
            move_pose(target, cfg, kit, start=current)
            current = target
        except (ValueError, RuntimeError, OSError) as e:
            print("Move did not complete:", e)
            recovered = load_last_pose()
            if recovered is not None:
                current = recovered
                print("Next move will start from last saved commanded pose:", current)

if __name__ == "__main__":
    main()
