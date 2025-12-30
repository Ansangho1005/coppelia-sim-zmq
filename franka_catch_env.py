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
        print("Connecting to CoppeliaSim (127.0.0.1:23000)...")
        self.client = RemoteAPIClient(host='127.0.0.1', port=23000)
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)
        
        # 2. 핸들 가져오기
        self.sphere_handle = self.sim.getObject('/Sphere')
        self.ee_handle = self.sim.getObject('/Franka/connection')
        self.franka_handle = self.sim.getObject('/Franka')
        self.script_handle = self.sim.getScript(self.sim.scripttype_childscript, self.franka_handle)
        
        try:
            self.attach_handle = self.sim.getObject('/Franka/attachPoint')
        except Exception:
            self.attach_handle = self.ee_handle 
        
        self.joint_names = [
            '/Franka/joint', '/Franka/link2_resp/joint', '/Franka/link3_resp/joint',
            '/Franka/link4_resp/joint', '/Franka/link5_resp/joint', '/Franka/link6_resp/joint',
            '/Franka/link7_resp/joint'
        ]
        self.joint_handles = [self.sim.getObject(name) for name in self.joint_names]
        
        # 3. Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        obs_dim = 7 + 7 + 3 + 3 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 4. Parameters
        self.target_position = np.array([-0.6, 0.0, 0.25], dtype=np.float32) 
        self.grasp_distance_thresh = 0.045
        self.target_reach_thresh = 0.05
        self.max_steps = 200
        
        self.steps_taken = 0
        self.ball_in_hand = False
        self.gripper_closed = False
        self.prev_action = np.zeros(8, dtype=np.float32)

        # [속도 최적화]
        self.sim.setBoolParam(self.sim.boolparam_realtime_simulation, False)
        try:
            self.sim.setBoolParam(self.sim.boolparam_display_enabled, False)
            self.sim.setBoolParam(self.sim.boolparam_console_visible, False)
        except Exception:
            pass

        # [공 굴러감 방지] 마찰력 증가
        try:
            self.sim.setObjectFloatParam(self.sphere_handle, self.sim.shapefloatparam_lin_damping, 0.8)
            self.sim.setObjectFloatParam(self.sphere_handle, self.sim.shapefloatparam_ang_damping, 0.8)
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.client.step()
        self.sim.startSimulation()
        
        # Ready Pose
        ready_joint_pos = [0, 0.5, 0, -1.8, 0, 1.5, 0] 
        for i, handle in enumerate(self.joint_handles):
            self.sim.setJointPosition(handle, ready_joint_pos[i])
            self.sim.setJointTargetVelocity(handle, 0.0)

        # [수정됨] 공 위치 고정 (65cm)
        # 로봇 정면 65cm 지점에 고정 (0.55 -> 0.65 변경)
        fixed_x = 0.7
        fixed_y = 0.0
        init_pos = [fixed_x, fixed_y, 0.025]
        
        self.sim.setObjectPosition(self.sphere_handle, -1, init_pos)
        self.sim.resetDynamicObject(self.sphere_handle) 
        
        self.steps_taken = 0
        self.ball_in_hand = False
        self.gripper_closed = False
        self.prev_action = np.zeros(8, dtype=np.float32)
        
        self.client.step()
        
        return self._get_obs_legacy(), {}

    def step(self, action):
        self.steps_taken += 1
        action = np.array(action, dtype=np.float32)
        
        # 거리 기반 자동 감속
        curr_ee = np.array(self.sim.getObjectPosition(self.ee_handle, -1))
        curr_ball = np.array(self.sim.getObjectPosition(self.sphere_handle, -1))
        dist = np.linalg.norm(curr_ball - curr_ee)
        
        speed_factor = 1.0
        if dist < 0.15: speed_factor = 0.2
        elif dist < 0.3: speed_factor = 0.5
            
        scaled_arm_action = (action[:7] * speed_factor).tolist()
        grip_action = float(action[7])
        
        # Lua 호출
        raw_obs = self.sim.callScriptFunction('process_step', self.script_handle, scaled_arm_action)
        self.client.step()
        
        obs_data = np.array(raw_obs, dtype=np.float32)
        current_ee_pos = obs_data[14:17]
        current_ball_pos = obs_data[17:20]
        
        reward_bonus = 0.0
        terminated = False
        truncated = False
        
        # Gripper Logic
        if grip_action > 0.5 and not self.gripper_closed:
            self.gripper_closed = True
            if not self.ball_in_hand:
                dist_check = np.linalg.norm(current_ball_pos - current_ee_pos)
                if dist_check < self.grasp_distance_thresh:
                    self.sim.setObjectParent(self.sphere_handle, self.attach_handle, True)
                    self.ball_in_hand = True
                    reward_bonus += 5.0
        
        if grip_action < -0.5 and self.gripper_closed:
            self.gripper_closed = False
            if self.ball_in_hand:
                self.sim.setObjectParent(self.sphere_handle, -1, True)
                self.ball_in_hand = False
                
                dropped_pos = np.array(self.sim.getObjectPosition(self.sphere_handle, -1))
                dist_target = np.linalg.norm(dropped_pos - self.target_position)
                
                if dist_target < self.target_reach_thresh:
                    final_obs = self._make_final_obs(obs_data)
                    return final_obs, 100.0, True, False, {"is_success": True}

        # Reward Calculation
        reward = self._compute_reward_optimized(current_ee_pos, current_ball_pos, action)
        reward += reward_bonus
        
        # Fail Condition
        dist_from_base = np.linalg.norm(current_ball_pos[:2])
        if current_ball_pos[2] < -0.05 or dist_from_base > 0.85: # 거리제한도 0.8 -> 0.85로 살짝 늘림
            reward -= 20.0
            terminated = True
            
        if self.steps_taken >= self.max_steps:
            truncated = True
            
        final_obs = self._make_final_obs(obs_data)
        
        return final_obs, float(reward), terminated, truncated, {}

    def _make_final_obs(self, obs_data):
        grasp_flag = 1.0 if self.ball_in_hand else 0.0
        grasp_arr = np.array([grasp_flag], dtype=np.float32)
        final = np.concatenate([obs_data, grasp_arr, self.target_position])
        return final.astype(np.float32)

    def _compute_reward_optimized(self, ee_pos, ball_pos, action):
        reward = 0.0
        
        if not self.ball_in_hand:
            # Reaching
            dist = np.linalg.norm(ball_pos - ee_pos)
            reward = -np.log(dist + 0.05)
            
            if self.gripper_closed and dist > 0.1:
                reward -= 0.5
            if dist < 0.15:
                vel_norm = np.linalg.norm(action[:7])
                reward -= vel_norm * 0.2 
        else:
            # Lift & Transport
            reward += 2.0 
            dist_target = np.linalg.norm(self.target_position - ball_pos)
            reward -= dist_target 
            
            # Lift Bonus
            lift_score = min(ball_pos[2], 0.25) * 10.0
            reward += lift_score
            
            if ball_pos[2] < 0.05:
                reward -= 3.0

            vel_norm = np.linalg.norm(action[:7])
            reward -= vel_norm * 0.1

        action_diff = np.linalg.norm(action[:7] - self.prev_action[:7])
        reward -= action_diff * 0.2
        self.prev_action = action.copy()
        
        return reward

    def _get_obs_legacy(self):
        joint_angles = np.array([self.sim.getJointPosition(h) for h in self.joint_handles], dtype=np.float32)
        joint_vels = np.array([self.sim.getJointVelocity(h) for h in self.joint_handles], dtype=np.float32)
        ee_pos = np.array(self.sim.getObjectPosition(self.ee_handle, -1), dtype=np.float32)
        ball_pos = np.array(self.sim.getObjectPosition(self.sphere_handle, -1), dtype=np.float32)
        grasp_flag = 1.0 if self.ball_in_hand else 0.0
        grasp_arr = np.array([grasp_flag], dtype=np.float32)
        
        final = np.concatenate([
            joint_angles, joint_vels, ee_pos, ball_pos, grasp_arr, 
            self.target_position
        ])
        return final.astype(np.float32)

    def close(self):
        self.sim.stopSimulation()