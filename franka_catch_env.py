import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class FrankaCatchEnv(gym.Env):
    def __init__(self):
        super().__init__()

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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.01)

        self.sim.startSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_advancing_running:
            time.sleep(0.01)

        drop_x = np.random.uniform(-0.6, 0.6)
        drop_y = np.random.uniform(-0.6, 0.6)
        drop_z = 30.0
        self.sim.setObjectPosition(self.sphere_handle, -1, [drop_x, drop_y, drop_z])
        self.sim.resetDynamicObject(self.sphere_handle)

        obs = self._get_obs()
        return obs, {}  # gymnasium requires (obs, info)

    def step(self, action):
        for i, joint_handle in enumerate(self.joint_handles):
            self.sim.setJointTargetVelocity(joint_handle, float(action[i]))

        self.sim.step()

        obs = self._get_obs()
        reward = float(self._compute_reward(obs))
        
        terminated = bool(obs[2] < 0.4)   # 공이 낮게 떨어졌으면 종료
        truncated = False                 # 시간 초과 등의 이유가 아니면 False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        sphere_pos = self.sim.getObjectPosition(self.sphere_handle, -1)
        ee_pos = self.sim.getObjectPosition(self.ee_handle, -1)
        return np.array(sphere_pos + ee_pos, dtype=np.float32)

    def _compute_reward(self, obs):
        error = np.linalg.norm(obs[0:2] - obs[3:5])
        return float(-error)  # <-- numpy.float32가 아닌 파이썬 float로 캐스팅


    def close(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.01)
