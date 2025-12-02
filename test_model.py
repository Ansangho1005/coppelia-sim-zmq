import gymnasium as gym
from stable_baselines3 import SAC
import time
import os
import numpy as np

# 환경 파일 임포트
from franka_catch_env import FrankaCatchEnv 

def main():
    # 1. 환경 생성
    env = FrankaCatchEnv()
    
    print("--- 테스트 모드 시작 ---")
    print("화면 렌더링을 켭니다...")

    # [중요] 눈으로 보기 위해 학습 때 껐던 옵션들을 다시 켭니다.
    try:
        # 화면 렌더링 켜기 (회색 화면 탈출)
        env.sim.setBoolParam(env.sim.boolparam_display_enabled, True)
        
        # Real-time 모드 켜기 (이걸 켜야 로봇이 사람이 보는 속도로 움직입니다. 안 켜면 100배속으로 지나감)
        env.sim.setBoolParam(env.sim.boolparam_realtime_simulation, True)
        
        print("렌더링 설정 완료.")
    except Exception as e:
        print(f"CoppeliaSim 설정 오류 (무시 가능): {e}")

    # 2. 저장된 모델 불러오기
    # 학습이 끝난 최종 파일 이름: "franka_catch_sac_final.zip"
    model_path = "franka_catch_sac_final" 
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: 모델 파일 '{model_path}.zip'을 찾을 수 없습니다.")
        return

    print(f"모델 불러오는 중: {model_path}...")
    model = SAC.load(model_path, env=env)

    # 3. 테스트 루프 (10판 정도 구경)
    episodes = 10
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        score = 0
        step_count = 0
        
        print(f"\nEpisode {ep+1} Start!")
        
        while not (done or truncated):
            # [중요] deterministic=True
            # 학습할 때는 랜덤하게 움직이지만(탐험), 테스트할 때는 "배운 대로 가장 확실한 행동"만 합니다.
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            score += reward
            step_count += 1
            
            # 너무 빠르면 여기서 time.sleep(0.05) 정도 줘도 됨

        print(f"Episode {ep+1} 종료. 점수: {score:.2f} (Steps: {step_count})")
        
        if info.get("is_success"):
            print(">>> 🎉 SUCCESS! (공 잡고 목표지점 도달)")
        else:
            print(">>> Failed.")
            
        time.sleep(1.0) # 한 판 끝나고 잠깐 대기

    print("테스트 종료.")
    env.close()

if __name__ == "__main__":
    main()