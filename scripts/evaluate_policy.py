from __future__ import annotations

import json
import os
import statistics
from typing import Dict, List

import torch

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig
from models.bc_policy import BCPolicy


def flatten_obs(obs: dict, phase: int) -> torch.Tensor:
    values = [
        *obs["ee_pos"],
        *obs["cube_pos"],
        *obs["target_pos"],
        float(phase),
    ]
    x = torch.tensor(values, dtype=torch.float32)
    return x.unsqueeze(0)


def get_phase_from_step(step: int) -> int:
    if step < 10:
        return 0
    if step < 20:
        return 1
    if step < 30:
        return 2
    if step < 40:
        return 3
    if step < 50:
        return 4
    if step < 60:
        return 5
    if step < 65:
        return 6
    return 7


def rollout_episode(
    env: PickPlaceEnv,
    model: BCPolicy,
    num_control_steps: int = 70,
) -> Dict[str, object]:
    obs = env.reset()

    for step in range(num_control_steps):
        phase = get_phase_from_step(step)
        x = flatten_obs(obs, phase)

        with torch.no_grad():
            pred = model(x).squeeze(0)

        ee_pos = obs["ee_pos"]

        dx = float(pred[0].item())
        dy = float(pred[1].item())
        dz = float(pred[2].item())

        target_ee_pos = (
            max(0.35, min(0.75, ee_pos[0] + dx)),
            max(-0.35, min(0.20, ee_pos[1] + dy)),
            max(0.03, min(0.40, ee_pos[2] + dz)),
        )

        gripper_open = 1.0 if float(pred[3].item()) > 0.5 else 0.0

        obs = env.apply_action(
            target_ee_pos=target_ee_pos,
            gripper_open=gripper_open,
            num_steps=10,
        )

    return env.get_success_info()


def main() -> None:
    checkpoint_path = "checkpoints/bc_policy.pt"
    num_episodes = 20
    output_dir = "results"
    output_path = os.path.join(output_dir, "evaluation.json")
    os.makedirs(output_dir, exist_ok=True)

    config = PickPlaceConfig(gui=False)
    env = PickPlaceEnv(config=config)

    model = BCPolicy()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    results: List[Dict[str, object]] = []

    try:
        for episode_idx in range(num_episodes):
            info = rollout_episode(env, model)
            results.append(info)

            xy_error = info["xy_error"]
            print(
                f"Episode {episode_idx + 1:02d}/{num_episodes} | "
                f"success={info['success']} | "
                f"xy_error=({xy_error[0]:.4f}, {xy_error[1]:.4f}) | "
                f"cube_height={info['cube_height']:.4f}"
            )
    finally:
        env.close()

    successes = sum(1 for r in results if r["success"])
    success_rate = successes / len(results)

    summary = {
        "episodes": len(results),
        "successes": successes,
        "success_rate": success_rate,
        "mean_x_error": statistics.mean(r["xy_error"][0] for r in results),
        "mean_y_error": statistics.mean(r["xy_error"][1] for r in results),
        "mean_cube_height": statistics.mean(r["cube_height"] for r in results),
        "episode_results": [
            {
                "cube_pos": list(r["cube_pos"]),
                "target_pos": list(r["target_pos"]),
                "xy_error": list(r["xy_error"]),
                "cube_height": r["cube_height"],
                "success": r["success"],
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {summary['episodes']}")
    print(f"Successes: {summary['successes']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Mean x error: {summary['mean_x_error']:.4f}")
    print(f"Mean y error: {summary['mean_y_error']:.4f}")
    print(f"Mean cube height: {summary['mean_cube_height']:.4f}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()