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
    return x.unsqueeze(0)  # shape: (1, 9)


def main() -> None:
    checkpoint_path = "checkpoints/bc_policy.pt"
    num_control_steps = 40

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

            target_ee_pos = (
                float(pred[0].item()),
                float(pred[1].item()),
                float(pred[2].item()),
            )

            gripper_open = float(pred[3].item())

            # Clamp gripper command to [0, 1]
            gripper_open = max(0.0, min(1.0, gripper_open))

            obs = env.apply_action(
                target_ee_pos=target_ee_pos,
                gripper_open=gripper_open,
                num_steps=30,
            )

            print(
                f"step={step:02d} | "
                f"target_ee_pos={target_ee_pos} | "
                f"gripper_open={gripper_open:.3f}"
            )

            time.sleep(0.05)

        print("Holding simulation for 10 seconds...")
        for _ in range(10 * 240):
            env.step_simulation()
            time.sleep(1.0 / 240.0)

    finally:
        env.close()


if __name__ == "__main__":
    main()