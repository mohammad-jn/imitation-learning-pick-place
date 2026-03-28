from __future__ import annotations

import time
from pprint import pprint

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig


def main() -> None:
    config = PickPlaceConfig(gui=True)
    env = PickPlaceEnv(config=config)

    try:
        obs = env.reset()

        print("\nInitial observation:")
        pprint(obs)

        print("\nRobot joint info:")
        env.print_joint_info()

        print("\nHolding simulation for 5 seconds...")
        for _ in range(5 * 240):
            env.step_simulation()
            time.sleep(1.0 / 240.0)

    finally:
        env.close()


if __name__ == "__main__":
    main()