# Imitation Learning for Simulated Robotic Pick-and-Place

A robotics and machine learning project that trains a simulated robot arm to pick up a cube and place it at a target location using **behavior cloning** in **PyBullet**.

## Project Overview

This project builds an end-to-end imitation learning pipeline for a robotic pick-and-place task in simulation.

The system includes:

- a PyBullet-based robotic manipulation environment
- a scripted expert policy that solves the task
- demonstration collection from the expert
- a PyTorch behavior cloning model
- rollout and evaluation of the learned policy

The robot used in simulation is the **Franka Panda** arm with a gripper. The task is to:

1. move above the cube
2. descend to the cube
3. close the gripper
4. lift the cube
5. move to the target location
6. place the cube
7. open the gripper
8. retreat

---

## Why this project

This project was designed to demonstrate skills relevant to machine learning engineering for robotics, including:

- simulation-based robotics development
- inverse kinematics control
- demonstration collection
- behavior cloning
- evaluation of learned policies
- clean ML pipeline design

---

## Tech Stack

- Python
- PyBullet
- PyTorch
- NumPy
- Matplotlib
- Conda

---

## Project Structure

```text
imitation-learning-pick-place/
├── env/
│   ├── __init__.py
│   └── pick_place_env.py
├── scripts/
│   ├── test_env_class.py
│   ├── test_ik_control.py
│   ├── test_gripper_control.py
│   ├── expert_pick_place.py
│   ├── collect_demos.py
│   ├── run_policy.py
│   └── evaluate_policy.py
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   └── bc_policy.py
├── training/
│   ├── __init__.py
│   └── train_bc.py
├── environment.yml
├── .gitignore
└── README.md