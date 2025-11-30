import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class FrankaCatchEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 1. CoppeliaSim 연결
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)
        
        # 2. 객체 핸들 가져오기
        self.sphere_handle = self.sim.getObject('/Sphere')
        self.ee_handle = self.sim.getObject('/Franka/connection')
        
        try:
            self.attach_handle = self.sim.getObject('/Franka/attachPoint')
        except Exception:
            self.attach_handle = self.ee_handle 
        
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
        
        # 3. Action / Observation Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        obs_dim = 7 + 7 + 3 + 3 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 4. 파라미터 설정
        self.target_position = np.array([-0.6, 0.0, 0.025], dtype=np.float32) 
        self.grasp_distance_thresh = 0.05
        self.target_reach_thresh = 0.05
        
        # [수정됨] 최대 스텝 수 단축 (200 step)
        # 1 step당 약 0.05초(기본값) -> 시뮬레이션 시간상 10초
        # PC 성능에 따라 실제 시간은 1분 정도 소요될 수 있음
        self.max_steps = 200
        
        # 상태 변수
        self.steps_taken = 0
        self.ball_in_hand = False
        self.gripper_closed = False

        # [수정된 렌더링 및 속도 최적화 설정]
        # [속도 최적화 설정]
        
        # 1. Real-time 모드 끄기 (최고 속도) - 이건 에러 안 남
        self.sim.setBoolParam(self.sim.boolparam_realtime_simulation, False)
        
        # 2. 화면/UI 끄기 (Headless 모드에서는 에러가 날 수 있으므로 예외 처리)
        try:
            # 화면 갱신 끄기
            self.sim.setBoolParam(self.sim.boolparam_display_enabled, False)
            
            # 콘솔 창 끄기
            self.sim.setBoolParam(self.sim.boolparam_console_visible, False)
            
            # [에러 원인 삭제됨] renderer 설정은 버전 탐라 에러가 잦으니 삭제합니다.
            # 어차피 -h 로 실행해서 렌더링 안 하고 있습니다.
            
        except Exception:
            # Headless 모드라 화면이 없어서 에러가 나면 그냥 무시하고 넘어감
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.client.step()
        
        self.sim.startSimulation()
        
        # 초기 자세 (Ready Pose)
        ready_joint_pos = [0, 0.5, 0, -1.8, 0, 1.5, 0] 
        for i, handle in enumerate(self.joint_handles):
            self.sim.setJointPosition(handle, ready_joint_pos[i])
            self.sim.setJointTargetVelocity(handle, 0.0)

        # 공 위치 초기화
        init_pos = [np.random.uniform(0.5, 0.6), np.random.uniform(-0.1, 0.1), 0.025]
        self.sim.setObjectPosition(self.sphere_handle, -1, init_pos)
        self.sim.resetDynamicObject(self.sphere_handle)
        
        self.steps_taken = 0
        self.ball_in_hand = False
        self.gripper_closed = False
        
        self.client.step() 
        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1
        action = np.array(action, dtype=float)
        
        arm_actions = action[:7]
        grip_action = action[7]
        
        # --- Gripper Logic ---
        current_ee_pos = np.array(self.sim.getObjectPosition(self.ee_handle, -1))
        current_ball_pos = np.array(self.sim.getObjectPosition(self.sphere_handle, -1))

        if grip_action > 0.5 and not self.gripper_closed:
            self.gripper_closed = True
            if not self.ball_in_hand:
                dist = np.linalg.norm(current_ball_pos - current_ee_pos)
                if dist < self.grasp_distance_thresh:
                    self.sim.setObjectParent(self.sphere_handle, self.attach_handle, True)
                    self.ball_in_hand = True
        
        if grip_action < -0.5 and self.gripper_closed:
            self.gripper_closed = False
            if self.ball_in_hand:
                self.sim.setObjectParent(self.sphere_handle, -1, True)
                self.ball_in_hand = False
                
                dropped_pos = np.array(self.sim.getObjectPosition(self.sphere_handle, -1))
                dist_target = np.linalg.norm(dropped_pos - self.target_position)
                if dist_target < self.target_reach_thresh:
                    return self._get_obs(), 100.0, True, False, {"is_success": True}

        # --- Joint Control ---
        max_vel = 1.0
        for i, joint_handle in enumerate(self.joint_handles):
            self.sim.setJointTargetVelocity(joint_handle, float(arm_actions[i]) * max_vel)
        
        self.client.step()
        
        # --- Reward & Termination Check ---
        obs = self._get_obs()
        reward = self._compute_reward()
        
        terminated = False
        truncated = False
        
        # 1. 시간 초과 체크
        if self.steps_taken >= self.max_steps:
            truncated = True

        # 2. [추가됨] 조기 종료 조건 (Fail Condition)
        # 공이 바닥(z=0) 아래로 떨어지거나, 로봇 팔 길이(약 0.8m)보다 멀리 굴러간 경우
        dist_from_base = np.linalg.norm(current_ball_pos[:2]) # XY 평면 거리
        if current_ball_pos[2] < -0.05 or dist_from_base > 0.8:
            reward -= 10.0 # 큰 벌점 부여
            terminated = True # 에피소드 즉시 종료
            # print("Ball lost! Resetting...") # 디버깅용

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self):
        ee_pos = np.array(self.sim.getObjectPosition(self.ee_handle, -1))
        ball_pos = np.array(self.sim.getObjectPosition(self.sphere_handle, -1))
        
        reward = 0.0
        
        if not self.ball_in_hand:
            dist = np.linalg.norm(ball_pos - ee_pos)
            reward = -dist
            if dist < 0.1: 
                reward += 0.5
            if self.gripper_closed:
                reward -= 0.2
        else:
            dist_target = np.linalg.norm(self.target_position - ball_pos)
            reward = 2.0 - dist_target 
            
        return reward

    def _get_obs(self):
        joint_angles = np.array([self.sim.getJointPosition(h) for h in self.joint_handles], dtype=np.float32)
        joint_vels = np.array([self.sim.getJointVelocity(h) for h in self.joint_handles], dtype=np.float32)
        ee_pos = np.array(self.sim.getObjectPosition(self.ee_handle, -1), dtype=np.float32)
        ball_pos = np.array(self.sim.getObjectPosition(self.sphere_handle, -1), dtype=np.float32)
        grasp_flag = np.array([1.0 if self.ball_in_hand else 0.0], dtype=np.float32)
        target_pos = self.target_position.astype(np.float32)
        return np.concatenate([joint_angles, joint_vels, ee_pos, ball_pos, grasp_flag, target_pos])

    def close(self):
        self.sim.stopSimulation()