from __future__ import annotations

import os
import pickle
from typing import Dict, Any, List, Tuple

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig


def make_step_record(
    obs: Dict[str, Any],
    target_ee_pos: Tuple[float, float, float],
    gripper_open: float,
) -> Dict[str, Any]:
    return {
        "obs": {
            "ee_pos": tuple(obs["ee_pos"]),
            "cube_pos": tuple(obs["cube_pos"]),
            "cube_orn": tuple(obs["cube_orn"]),
            "target_pos": tuple(obs["target_pos"]),
        },
        "action": {
            "target_ee_pos": tuple(target_ee_pos),
            "gripper_open": float(gripper_open),
        },
    }


def collect_segment(
    env: PickPlaceEnv,
    dataset: List[Dict[str, Any]],
    target_ee_pos: Tuple[float, float, float],
    gripper_open: float,
    steps: int,
    action_repeat: int = 30,
) -> None:
    """
    Repeatedly apply the same action and record (obs, action) pairs.
    """
    for _ in range(steps):
        obs = env.get_observation()
        dataset.append(make_step_record(obs, target_ee_pos, gripper_open))
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

    # same logic as your expert, but step-by-step and recordable
    above_cube = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.2)
    grasp_pos = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.02)
    lift_pos = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.3)
    above_target = (target_pos[0], target_pos[1], target_pos[2] + 0.3)
    place_pos = (target_pos[0], target_pos[1], target_pos[2] + 0.02)
    retreat_pos = (target_pos[0], target_pos[1], target_pos[2] + 0.3)

    # move above cube with open gripper
    collect_segment(env, dataset, above_cube, gripper_open=1.0, steps=6)

    # move down to grasp
    collect_segment(env, dataset, grasp_pos, gripper_open=1.0, steps=6)

    # close gripper while staying at grasp position
    collect_segment(env, dataset, grasp_pos, gripper_open=0.0, steps=4)

    # lift
    collect_segment(env, dataset, lift_pos, gripper_open=0.0, steps=6)

    # move above target
    collect_segment(env, dataset, above_target, gripper_open=0.0, steps=6)

    # lower to place
    collect_segment(env, dataset, place_pos, gripper_open=0.0, steps=6)

    # open gripper
    collect_segment(env, dataset, place_pos, gripper_open=1.0, steps=4)

    # retreat
    collect_segment(env, dataset, retreat_pos, gripper_open=1.0, steps=6)

    return dataset


def main() -> None:
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    config = PickPlaceConfig(gui=False)
    env = PickPlaceEnv(config=config)

    all_demos: List[Dict[str, Any]] = []

    num_episodes = 20

    try:
        for episode_idx in range(num_episodes):
            episode_data = collect_one_episode(env)
            all_demos.extend(episode_data)
            print(f"Collected episode {episode_idx + 1}/{num_episodes} | steps: {len(episode_data)}")

    finally:
        env.close()

    output_path = os.path.join(output_dir, "demos.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_demos, f)

    print(f"Saved {len(all_demos)} total demo steps to: {output_path}")


if __name__ == "__main__":
    main()