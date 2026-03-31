# 🤖 Imitation Learning for Robotic Pick-and-Place

Train a simulated robot arm to pick and place objects using **behavior cloning** in PyBullet.

---

## 🚀 Quick Start

```bash
# create environment
conda env create -f environment.yml
conda activate pickplace

# collect expert data
python -m scripts.collect_demos

# train policy
python -m training.train_bc

# run learned policy (GUI)
python -m scripts.run_policy

# evaluate performance
python -m scripts.evaluate_policy

## 🧠 What This Project Does

A robot learns to:

- pick up a cube 🟥  
- move it across the table  
- place it at a target 🎯  

All learned from expert demonstrations — no reinforcement learning.

---

## 🏗️ Environment

Built with **PyBullet**, containing:

- 🤖 Franka Panda robot arm  
- 🟥 cube object  
- 🎯 target position  
- flat table (plane)

### Observations

- end-effector position  
- cube position  
- target position  
- task phase  

---

## 🧩 Model

A simple **MLP (PyTorch)**:

### Input (10D)

- ee_pos (3)  
- cube_pos (3)  
- target_pos (3)  
- phase (1)

### Output (4D)

- dx, dy, dz  
- gripper_open  

---

## 📊 Results

Evaluation over **20 episodes**:

- ✅ Success rate: **100%**
- 📍 Mean position error:
  - x: **0.0066**
  - y: **0.0131**