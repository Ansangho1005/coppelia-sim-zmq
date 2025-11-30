import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os  # 폴더 생성을 위해 필요

from franka_catch_env import FrankaCatchEnv 

def main():
    # 1. 로그 저장할 폴더 이름 정의
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True) # 폴더가 없으면 만듦

    # 2. 환경 생성
    env = FrankaCatchEnv()
    
    # 3. Monitor 래퍼 설정 (중요: 여기에 log_dir을 넣어줘야 CSV 로그도 남음)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    
    # 환경 체크
    check_env(env, warn=True)
    
    # 4. 모델 생성 (가장 중요: tensorboard_log 옵션 필수!)
    model = SAC(
        policy="MlpPolicy", 
        env=env, 
        verbose=1,
        tensorboard_log=log_dir  # <--- 이게 있어야 텐서보드 로그가 생깁니다!
    )
    
    # 학습 시작
    print("Start Learning... (Logs will be saved in 'logs/' folder)")
    total_timesteps = 200000 
    model.learn(total_timesteps=total_timesteps)
    
    # 모델 저장
    model.save("franka_catch_sac_model")
    print("Model saved.")

    # [추가] 체크포인트 콜백 생성
    # save_freq=10000: 10,000 스텝마다 저장 (약 10~20분 간격 예상)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./logs/checkpoints/',
        name_prefix='franka_sac'
    )
    
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # callback 옵션 추가
    model.learn(total_timesteps=500000, callback=checkpoint_callback)
    
    model.save("franka_catch_sac_final")
    
    env.close()

if __name__ == "__main__":
    main()