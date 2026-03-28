from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import pybullet as p
import pybullet_data


@dataclass
class PickPlaceConfig:
    gui: bool = True
    gravity: float = -9.8
    time_step: float = 1.0 / 240.0
    cube_start_pos: Tuple[float, float, float] = (0.6, 0.0, 0.02)
    cube_start_orn: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    robot_start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_start_orn: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    target_pos: Tuple[float, float, float] = (0.5, -0.25, 0.02)


class PickPlaceEnv:
    """
    Minimal PyBullet environment for a simulated pick-and-place task.

    Current version supports:
    - connecting to PyBullet
    - loading plane, robot, and cube
    - resetting the scene
    - returning a simple state observation

    Later we will add:
    - action stepping
    - inverse kinematics
    - gripper control
    - success checks
    """

    def __init__(self, config: PickPlaceConfig | None = None) -> None:
        self.config = config or PickPlaceConfig()

        self.client_id: int | None = None
        self.plane_id: int | None = None
        self.robot_id: int | None = None
        self.cube_id: int | None = None

        # Franka Panda end-effector link index is commonly 11 in PyBullet URDF usage.
        # We can verify this later by printing all joints.
        self.ee_link_index: int = 11

        self._connect()

    def _connect(self) -> None:
        """Connect to PyBullet if not already connected."""
        if self.client_id is not None:
            return

        connection_mode = p.GUI if self.config.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        p.setTimeStep(self.config.time_step)

    def reset(self) -> Dict[str, Any]:
        """Reset the full scene and return the initial observation."""
        self._ensure_connected()

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        p.setTimeStep(self.config.time_step)

        self.plane_id = p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.config.robot_start_pos,
            baseOrientation=self.config.robot_start_orn,
            useFixedBase=True,
        )

        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=self.config.cube_start_pos,
            baseOrientation=self.config.cube_start_orn,
            useFixedBase=False,
        )

        # Let the simulation settle slightly.
        for _ in range(20):
            p.stepSimulation()

        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """Return a simple low-dimensional observation."""
        self._ensure_scene_loaded()

        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = ee_state[0]

        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)

        obs = {
            "ee_pos": ee_pos,
            "cube_pos": cube_pos,
            "cube_orn": cube_orn,
            "target_pos": self.config.target_pos,
        }
        return obs

    def step_simulation(self, num_steps: int = 1) -> None:
        """Advance simulation without applying actions yet."""
        self._ensure_connected()

        for _ in range(num_steps):
            p.stepSimulation()

    def print_joint_info(self) -> None:
        """Print robot joint metadata for inspection/debugging."""
        self._ensure_scene_loaded()

        num_joints = p.getNumJoints(self.robot_id)
        print(f"Number of joints: {num_joints}")
        print("-" * 80)

        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode("utf-8")
            link_name = joint_info[12].decode("utf-8")
            joint_type = joint_info[2]
            print(
                f"joint_index={joint_index:2d} | "
                f"joint_name={joint_name:25s} | "
                f"link_name={link_name:25s} | "
                f"joint_type={joint_type}"
            )

    def close(self) -> None:
        """Disconnect from PyBullet."""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def _ensure_connected(self) -> None:
        if self.client_id is None:
            raise RuntimeError("PyBullet client is not connected.")

    def _ensure_scene_loaded(self) -> None:
        self._ensure_connected()
        if self.robot_id is None or self.cube_id is None:
            raise RuntimeError("Scene not loaded. Call reset() first.")