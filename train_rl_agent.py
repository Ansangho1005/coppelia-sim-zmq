# 수정 후 gymnasium 기반으로 전환
import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import numpy as np
from franka_catch_env import FrankaCatchEnv  # 같은 디렉토리에 위치한다고 가정

def main():
    env = FrankaCatchEnv()

    # Monitor는 gymnasium.Env만 wrapping 가능하므로 사전 체크
    assert isinstance(env, Env), f"Expected env to be a `gymnasium.Env` but got {type(env)}"
    env = Monitor(env)  # 로그 기록용 wrapper

    check_env(env, warn=True)  # Gym interface 유효성 검사

    model = SAC(policy="MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=100000)  # 학습

    model.save("franka_catch_sac_model")
    print("Model saved as franka_catch_sac_model.zip")

    env.close()

if __name__ == "__main__":
    main()
