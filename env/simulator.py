import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

class Simulator:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.sphere_handle = self.sim.getObject('/Sphere')
        self.ee_handle = self.sim.getObject('/Franka/connection')

        self.joint_names = [
            '/Franka/joint',
            '/Franka/link2_resp/joint',
            '/Franka/link3_resp/joint',
            '/Franka/link4_resp/joint',
            '/Franka/link5_resp/joint',
            '/Franka/link6_resp/joint',
            '/Franka/link7_resp/joint'
        ]
        self.joint_handles = [self.sim.getObject(name) for name in self.joint_names]

        # Barrett Hand finger joints
        self.finger_joint_names = [
            '/Franka/BarrettHand/jointA_0',
            '/Franka/BarrettHand/jointA_2',
            '/Franka/BarrettHand/jointB_1'
        ]
        self.finger_joints = [self.sim.getObject(name) for name in self.finger_joint_names]

    def reset_sphere(self):
        drop_x = np.random.uniform(-0.6, 0.6)
        drop_y = np.random.uniform(-0.6, 0.6)
        drop_z = 30.0
        self.sim.setObjectPosition(self.sphere_handle, -1, [drop_x, drop_y, drop_z])
        self.sim.resetDynamicObject(self.sphere_handle)

    def close_gripper(self):
        for joint in self.finger_joints:
            self.sim.setJointTargetPosition(joint, 0.5)  # 0.5 ~ 1.0: closer = tighter grip

    def run(self):
        self.sim.startSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_advancing_running:
            time.sleep(0.01)

        self.reset_sphere()
        start_time = self.sim.getSimulationTime()
        timeout = 10.0

        while self.sim.getSimulationTime() - start_time < timeout:
            sphere_pos = self.sim.getObjectPosition(self.sphere_handle, -1)
            ee_pos = self.sim.getObjectPosition(self.ee_handle, -1)
            error = np.array(sphere_pos) - np.array(ee_pos)

            print(f"Ball position: {sphere_pos}")
            print(f"EE position: {ee_pos}")

            if abs(error[0]) < 0.05 and abs(error[1]) < 0.05 and sphere_pos[2] < 0.5:
                print("Ball in range - closing gripper.")
                self.close_gripper()
                break

            self.sim.step()

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.01)
