from __future__ import annotations

import os
import pickle
from typing import Dict, Any, List, Tuple

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig


def make_step_record(
    obs: Dict[str, Any],
    target_ee_pos: Tuple[float, float, float],
    gripper_open: float,
    phase: int,
) -> Dict[str, Any]:
    ee_pos = obs["ee_pos"]

    delta_ee = (
        target_ee_pos[0] - ee_pos[0],
        target_ee_pos[1] - ee_pos[1],
        target_ee_pos[2] - ee_pos[2],
    )

    return {
        "obs": {
            "ee_pos": tuple(obs["ee_pos"]),
            "cube_pos": tuple(obs["cube_pos"]),
            "cube_orn": tuple(obs["cube_orn"]),
            "target_pos": tuple(obs["target_pos"]),
            "phase": int(phase),
        },
        "action": {
            "delta_ee": tuple(delta_ee),
            "gripper_open": float(gripper_open),
        },
    }


def collect_segment(
    env: PickPlaceEnv,
    dataset: List[Dict[str, Any]],
    target_ee_pos: Tuple[float, float, float],
    gripper_open: float,
    phase: int,
    steps: int,
    action_repeat: int = 10,
) -> None:
    for _ in range(steps):
        obs = env.get_observation()
        dataset.append(make_step_record(obs, target_ee_pos, gripper_open, phase))
        env.apply_action(
            target_ee_pos=target_ee_pos,
            gripper_open=gripper_open,
            num_steps=action_repeat,
        )


def collect_one_episode(env: PickPlaceEnv) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []

    obs = env.reset()
    cube_pos = obs["cube_pos"]
    target_pos = obs["target_pos"]

    above_cube = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.2)
    grasp_pos = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.02)
    lift_pos = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.3)
    above_target = (target_pos[0], target_pos[1], target_pos[2] + 0.3)
    place_pos = (target_pos[0], target_pos[1], target_pos[2] + 0.02)
    retreat_pos = (target_pos[0], target_pos[1], target_pos[2] + 0.3)

    # phase 0: move above cube
    collect_segment(env, dataset, above_cube, gripper_open=1.0, phase=0, steps=10)

    # phase 1: move down to grasp
    collect_segment(env, dataset, grasp_pos, gripper_open=1.0, phase=1, steps=10)

    # phase 2: close gripper
    collect_segment(env, dataset, grasp_pos, gripper_open=0.0, phase=2, steps=5)

    # phase 3: lift
    collect_segment(env, dataset, lift_pos, gripper_open=0.0, phase=3, steps=10)

    # phase 4: move above target
    collect_segment(env, dataset, above_target, gripper_open=0.0, phase=4, steps=10)

    # phase 5: lower to place
    collect_segment(env, dataset, place_pos, gripper_open=0.0, phase=5, steps=10)

    # phase 6: open gripper
    collect_segment(env, dataset, place_pos, gripper_open=1.0, phase=6, steps=5)

    # phase 7: retreat
    collect_segment(env, dataset, retreat_pos, gripper_open=1.0, phase=7, steps=10)

    return dataset


def main() -> None:
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    config = PickPlaceConfig(gui=False)
    env = PickPlaceEnv(config=config)

    all_demos: List[Dict[str, Any]] = []

    num_episodes = 200

    try:
        for episode_idx in range(num_episodes):
            episode_data = collect_one_episode(env)
            all_demos.extend(episode_data)
            print(f"Collected episode {episode_idx + 1}/{num_episodes} | steps: {len(episode_data)}")
    finally:
        env.close()

    output_path = os.path.join(output_dir, "demos_phase.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_demos, f)

    print(f"Saved {len(all_demos)} total demo steps to: {output_path}")


if __name__ == "__main__":
    main()