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

    def reset_sphere(self, drop=True):
        # drop=False일 경우 높이만 옮기고 떨어지지 않도록 z=50에 고정
        drop_x = np.random.uniform(-0.6, 0.6)
        drop_y = np.random.uniform(-0.6, 0.6)
        drop_z = 50.0
        self.sim.setObjectPosition(self.sphere_handle, -1, [drop_x, drop_y, drop_z])
        if drop:
            self.sim.resetDynamicObject(self.sphere_handle)

    def run(self):
        self.sim.startSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_advancing_running:
            time.sleep(0.01)

        print("🕒 초기화: 공 위치 위로 재설정 (중력방지용)")
        self.reset_sphere(drop=False)

        print("🕒 12초 대기하며 joint 제어 모니터링")
        sim_start_time = self.sim.getSimulationTime()
        while self.sim.getSimulationTime() - sim_start_time < 12.0:
            ee_pos = self.sim.getObjectPosition(self.ee_handle, -1)
            print(f"🤖 [대기 중] EE 위치: {ee_pos}")
            self.sim.step()

        print("🎯 공 낙하 시작")
        self.reset_sphere(drop=True)

        timeout = 15.0
        drop_time = self.sim.getSimulationTime()

        while self.sim.getSimulationTime() - drop_time < timeout:
            sphere_pos = self.sim.getObjectPosition(self.sphere_handle, -1)
            ee_pos = self.sim.getObjectPosition(self.ee_handle, -1)
            error_xy = np.array(sphere_pos[:2]) - np.array(ee_pos[:2])

            print(f"⚽ 공 위치: {sphere_pos}")
            print(f"🤖 EE 위치: {ee_pos}")
            print(f"🔧 오차 XY: {error_xy}")

            kp = 1.0
            target_velocities = [kp * error_xy[1], kp * error_xy[0]] + [0.0] * 5
            for i, joint_handle in enumerate(self.joint_handles):
                self.sim.setJointTargetVelocity(joint_handle, float(target_velocities[i]))

            if abs(error_xy[0]) < 0.05 and abs(error_xy[1]) < 0.05 and sphere_pos[2] < 0.5:
                print("✅ 목표 위치 도달 완료")
                break

            self.sim.step()

        print("🛑 시뮬레이션 종료")
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.01)

if __name__ == '__main__':
    sim = Simulator()
    sim.run()
