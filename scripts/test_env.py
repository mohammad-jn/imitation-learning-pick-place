import time
import pybullet as p
import pybullet_data

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.6, 0.0, 0.02])

for _ in range(2000):
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()