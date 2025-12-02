import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os

from franka_catch_env import FrankaCatchEnv 

def main():
    # 1. 로그 폴더 생성
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 2. 환경 생성 및 래핑
    env = FrankaCatchEnv()
    # monitor.csv 파일에 에피소드별 보상 기록
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    
    # 환경 체크
    check_env(env, warn=True)
    
    # 3. [체크포인트 설정] 
    # 10,000 스텝마다 모델을 자동 저장 (현재 47 FPS 기준 약 3~4분에 한 번 저장됨)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./logs/checkpoints/',
        name_prefix='franka_sac'
    )
    
    # 4. 모델 생성 (딱 한 번만!)
    model = SAC(
        policy="MlpPolicy", 
        env=env, 
        verbose=1,
        tensorboard_log=log_dir # 텐서보드 로그 기록
    )
    
    # 5. 학습 시작 (여기에 callback을 넣어야 적용됩니다!)
    print(f"Start Learning... (FPS: Check terminal log)")
    
    # 목표 스텝: 50만 (충분히 길게 잡음)
    total_timesteps = 500000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # 6. 최종 모델 저장 (학습 완료 후)
    model.save("franka_catch_sac_final")
    print("Final Model saved.")
    
    env.close()

if __name__ == "__main__":
    main()