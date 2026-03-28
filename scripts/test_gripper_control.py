from __future__ import annotations

import time

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig


def main() -> None:
    config = PickPlaceConfig(gui=True)
    env = PickPlaceEnv(config=config)

    try:
        env.reset()

        # Move to a visible position first
        env.move_ee((0.45, 0.0, 0.35), num_steps=180)
        time.sleep(1.0)

        print("Opening gripper...")
        env.open_gripper(num_steps=120)
        time.sleep(1.0)

        print("Closing gripper...")
        env.close_gripper(num_steps=120)
        time.sleep(1.0)

        print("Opening gripper again...")
        env.open_gripper(num_steps=120)
        time.sleep(1.0)

    finally:
        env.close()


if __name__ == "__main__":
    main()
