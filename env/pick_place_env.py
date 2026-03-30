from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
import random

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
    def __init__(self, config: PickPlaceConfig | None = None) -> None:
        self.config = config or PickPlaceConfig()

        self.client_id: int | None = None
        self.plane_id: int | None = None
        self.robot_id: int | None = None
        self.cube_id: int | None = None
        self.current_cube_start_pos: Tuple[float, float, float] | None = None

        self.ee_link_index: int = 11
        self.arm_joint_indices: List[int] = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joint_indices: List[int] = [9, 10]

        self._connect()


    def _sample_cube_position(self) -> Tuple[float, float, float]:
        """
        Sample a random cube position for a new episode.
        """
        cube_x = random.uniform(0.55, 0.65)
        cube_y = random.uniform(-0.05, 0.05)
        return (cube_x, cube_y, 0.02)
    

    def _connect(self) -> None:
        if self.client_id is not None:
            return

        connection_mode = p.GUI if self.config.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        p.setTimeStep(self.config.time_step)

    def reset(self) -> Dict[str, Any]:
        self._ensure_connected()

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        p.setTimeStep(self.config.time_step)

        cube_start_pos = self._sample_cube_position()

        self.plane_id = p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.config.robot_start_pos,
            baseOrientation=self.config.robot_start_orn,
            useFixedBase=True,
        )

        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=cube_start_pos,
            baseOrientation=self.config.cube_start_orn,
            useFixedBase=False,
        )

        self.current_cube_start_pos = cube_start_pos

        self._reset_robot_joints()

        for _ in range(50):
            p.stepSimulation()

        return self.get_observation()

    def _reset_robot_joints(self) -> None:
        self._ensure_scene_loaded()

        initial_joint_positions = [
            0.0,
            -0.785,
            0.0,
            -2.356,
            0.0,
            1.571,
            0.785,
        ]

        for joint_index, joint_value in zip(self.arm_joint_indices, initial_joint_positions):
            p.resetJointState(self.robot_id, joint_index, joint_value)

        for joint_index in self.gripper_joint_indices:
            p.resetJointState(self.robot_id, joint_index, 0.04)

    def get_observation(self) -> Dict[str, Any]:
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
        self._ensure_connected()

        for _ in range(num_steps):
            p.stepSimulation()

    def move_ee(
        self,
        target_pos: Tuple[float, float, float],
        num_steps: int = 120,
    ) -> None:
        self._ensure_scene_loaded()

        target_orn = p.getQuaternionFromEuler([3.14159, 0.0, 0.0])

        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            targetOrientation=target_orn,
        )

        for _ in range(num_steps):
            for i, joint_index in enumerate(self.arm_joint_indices):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_targets[i],
                    force=200.0,
                )

            p.stepSimulation()

    def open_gripper(self, num_steps: int = 120) -> None:
        self._ensure_scene_loaded()

        target_opening = 0.04

        for _ in range(num_steps):
            for joint_index in self.gripper_joint_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_opening,
                    force=50.0,
                )
            p.stepSimulation()

    def close_gripper(self, num_steps: int = 120) -> None:
        self._ensure_scene_loaded()

        target_opening = 0.0

        for _ in range(num_steps):
            for joint_index in self.gripper_joint_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_opening,
                    force=50.0,
                )
            p.stepSimulation()

    def apply_action(
        self,
        target_ee_pos: Tuple[float, float, float],
        gripper_open: float,
        num_steps: int = 30,
    ) -> Dict[str, Any]:
        """
        Apply one low-level action:
        - move end effector toward target_ee_pos
        - set gripper to open (1.0) or closed (0.0)

        Returns the next observation.
        """
        self._ensure_scene_loaded()

        target_orn = p.getQuaternionFromEuler([3.14159, 0.0, 0.0])

        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_ee_pos,
            targetOrientation=target_orn,
        )

        gripper_target = 0.04 if gripper_open >= 0.5 else 0.0

        for _ in range(num_steps):
            for i, joint_index in enumerate(self.arm_joint_indices):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_targets[i],
                    force=200.0,
                )

            for joint_index in self.gripper_joint_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=gripper_target,
                    force=50.0,
                )

            p.stepSimulation()

        return self.get_observation()

    def print_joint_info(self) -> None:
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