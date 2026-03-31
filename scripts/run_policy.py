from __future__ import annotations

import time

import torch

from env.pick_place_env import PickPlaceEnv, PickPlaceConfig
from models.bc_policy import BCPolicy


def flatten_obs(obs: dict) -> torch.Tensor:
    values = [
        *obs["ee_pos"],
        *obs["cube_pos"],
        *obs["target_pos"],
    ]
    x = torch.tensor(values, dtype=torch.float32)
    return x.unsqueeze(0)


def main() -> None:
    checkpoint_path = "checkpoints/bc_policy.pt"
    num_control_steps = 80

    config = PickPlaceConfig(gui=True)
    env = PickPlaceEnv(config=config)

    model = BCPolicy()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    try:
        obs = env.reset()

        print("Running learned policy...")

        for step in range(num_control_steps):
            x = flatten_obs(obs)

            with torch.no_grad():
                pred = model(x).squeeze(0)

            ee_pos = obs["ee_pos"]

            dx = float(pred[0].item())
            dy = float(pred[1].item())
            dz = float(pred[2].item())

            target_ee_pos = (
                max(0.35, min(0.75, ee_pos[0] + dx)),
                max(-0.20, min(0.20, ee_pos[1] + dy)),
                max(0.03, min(0.40, ee_pos[2] + dz)),
            )

            gripper_open = 1.0 if float(pred[3].item()) > 0.5 else 0.0

            obs = env.apply_action(
                target_ee_pos=target_ee_pos,
                gripper_open=gripper_open,
                num_steps=10,
            )

            print(
                f"step={step:02d} | "
                f"delta=({dx:.4f}, {dy:.4f}, {dz:.4f}) | "
                f"target_ee_pos={target_ee_pos} | "
                f"gripper_open={gripper_open:.1f}"
            )

            time.sleep(0.03)

        print("Holding simulation for 10 seconds...")
        for _ in range(10 * 240):
            env.step_simulation()
            time.sleep(1.0 / 240.0)

    finally:
        env.close()


if __name__ == "__main__":
    main()