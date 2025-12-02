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
        self.client = RemoteAPIClient(host='127.0.0.1', port=23000)
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)
        
        # 2. 객체 및 스크립트 핸들
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
        
        # 3. Action / Observation Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # Observation Dim: 24
        obs_dim = 7 + 7 + 3 + 3 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 4. Parameters
        self.target_position = np.array([-0.6, 0.0, 0.025], dtype=np.float32) 
        self.grasp_distance_thresh = 0.05
        self.target_reach_thresh = 0.05
        self.max_steps = 200 
        
        self.steps_taken = 0
        self.ball_in_hand = False
        self.gripper_closed = False
        self.prev_action = np.zeros(8, dtype=np.float32)

        # [Speed Optimization]
        self.sim.setBoolParam(self.sim.boolparam_realtime_simulation, False)
        try:
            self.sim.setBoolParam(self.sim.boolparam_display_enabled, False)
            self.sim.setBoolParam(self.sim.boolparam_console_visible, False)
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

        # Ball Init
        init_pos = [np.random.uniform(0.5, 0.6), np.random.uniform(-0.1, 0.1), 0.025]
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
        
        arm_actions = action[:7].tolist()
        grip_action = float(action[7])
        
        # [Lua Function Call]
        raw_obs = self.sim.callScriptFunction('process_step', self.script_handle, arm_actions)
        
        self.client.step()
        
        # Data Processing
        obs_data = np.array(raw_obs, dtype=np.float32)
        current_ee_pos = obs_data[14:17]
        current_ball_pos = obs_data[17:20]
        
        # Gripper Logic
        reward_bonus = 0.0
        terminated = False
        truncated = False
        
        if grip_action > 0.5 and not self.gripper_closed:
            self.gripper_closed = True
            if not self.ball_in_hand:
                dist = np.linalg.norm(current_ball_pos - current_ee_pos)
                if dist < self.grasp_distance_thresh:
                    self.sim.setObjectParent(self.sphere_handle, self.attach_handle, True)
                    self.ball_in_hand = True
                    reward_bonus += 2.0
        
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
        if current_ball_pos[2] < -0.05 or dist_from_base > 0.8:
            reward -= 10.0
            terminated = True
            
        if self.steps_taken >= self.max_steps:
            truncated = True
            
        final_obs = self._make_final_obs(obs_data)
        
        return final_obs, float(reward), terminated, truncated, {}

    def _make_final_obs(self, obs_data):
        grasp_flag = 1.0 if self.ball_in_hand else 0.0
        # [Error Fix] Ensure grasp_flag is an array and concatenate securely
        grasp_arr = np.array([grasp_flag], dtype=np.float32)
        
        final = np.concatenate([
            obs_data, 
            grasp_arr, 
            self.target_position
        ])
        # [Error Fix] Force final output to be float32
        return final.astype(np.float32)

    def _compute_reward_optimized(self, ee_pos, ball_pos, action):
        reward = 0.0
        if not self.ball_in_hand:
            dist = np.linalg.norm(ball_pos - ee_pos)
            reward = -dist
            if dist < 0.1: reward += 0.5
            if self.gripper_closed: reward -= 0.2
        else:
            dist_target = np.linalg.norm(self.target_position - ball_pos)
            reward = 2.0 - dist_target
            
        action_diff = np.linalg.norm(action[:7] - self.prev_action[:7])
        reward -= action_diff * 0.05
        self.prev_action = action.copy()
        
        return reward

    def _get_obs_legacy(self):
        # Used for reset()
        joint_angles = np.array([self.sim.getJointPosition(h) for h in self.joint_handles], dtype=np.float32)
        joint_vels = np.array([self.sim.getJointVelocity(h) for h in self.joint_handles], dtype=np.float32)
        ee_pos = np.array(self.sim.getObjectPosition(self.ee_handle, -1), dtype=np.float32)
        ball_pos = np.array(self.sim.getObjectPosition(self.sphere_handle, -1), dtype=np.float32)
        
        grasp_flag = 1.0 if self.ball_in_hand else 0.0
        grasp_arr = np.array([grasp_flag], dtype=np.float32)
        
        final = np.concatenate([
            joint_angles, 
            joint_vels, 
            ee_pos, 
            ball_pos, 
            grasp_arr, 
            self.target_position
        ])
        # [Error Fix] Force float32
        return final.astype(np.float32)

    def close(self):
        self.sim.stopSimulation()