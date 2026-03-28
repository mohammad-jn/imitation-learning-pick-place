from __future__ import annotations

import time

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig


def expert_pick_and_place(env: PickPlaceEnv) -> None:
    obs = env.get_observation()

    cube_pos = obs["cube_pos"]
    target_pos = obs["target_pos"]

    # 1. Move above cube
    above_cube = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.2)
    env.move_ee(above_cube, num_steps=180)

    # 2. Move down
    grasp_pos = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.02)
    env.move_ee(grasp_pos, num_steps=180)

    # 3. Close gripper
    env.close_gripper(num_steps=120)

    # 4. Lift
    lift_pos = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.3)
    env.move_ee(lift_pos, num_steps=180)

    # 5. Move to target (above)
    above_target = (target_pos[0], target_pos[1], target_pos[2] + 0.3)
    env.move_ee(above_target, num_steps=180)

    # 6. Lower
    place_pos = (target_pos[0], target_pos[1], target_pos[2] + 0.02)
    env.move_ee(place_pos, num_steps=180)

    # 7. Open gripper
    env.open_gripper(num_steps=120)

    # 8. Move away
    retreat = (target_pos[0], target_pos[1], target_pos[2] + 0.3)
    env.move_ee(retreat, num_steps=180)


def main() -> None:
    config = PickPlaceConfig(gui=True)
    env = PickPlaceEnv(config=config)

    try:
        env.reset()

        print("Running expert policy...")
        expert_pick_and_place(env)

        print("Holding simulation...")
        for _ in range(10 * 240):
            env.step_simulation()
            time.sleep(1.0 / 240.0)

    finally:
        env.close()


if __name__ == "__main__":
    main()
