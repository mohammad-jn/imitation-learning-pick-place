from __future__ import annotations

import time

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig


def main() -> None:
    config = PickPlaceConfig(gui=True)
    env = PickPlaceEnv(config=config)

    try:
        obs = env.reset()
        print("Initial observation:")
        print(obs)

        waypoints = [
            (0.4, 0.0, 0.4),
            (0.55, 0.0, 0.25),
            (0.55, 0.2, 0.25),
            (0.45, -0.2, 0.3),
            (0.4, 0.0, 0.4),
        ]

        for waypoint in waypoints:
            print(f"Moving to: {waypoint}")
            env.move_ee(waypoint, num_steps=180)
            time.sleep(1.0)

    finally:
        env.close()


if __name__ == "__main__":
    main()