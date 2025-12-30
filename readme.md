# CoppeliaSim Franka Ball Catching (ZMQ / Apple Silicon M2)

[English](#english) | [한국어](#korean)

<a name="english"></a>

## 🇺🇸 English

This project demonstrates how to perform Reinforcement Learning (RL) with the Franka Emika robot in CoppeliaSim using Python on an **Apple Silicon (M2)** environment. The goal is to train the robot to catch a falling ball.

### 📌 Background & Features

#### Why ZMQ Remote API?
The legacy Remote API often suffers from compatibility issues and performance degradation on Apple Silicon (M1/M2/M3) Macs. To resolve this, I implemented the **ZeroMQ (ZMQ) based Remote API**. The ZMQ approach offers significantly faster communication speeds and more stable synchronization between the Python client and the simulator.

#### Environment Setup
- **OS**: macOS (Apple Silicon M2)
- **Simulator**: CoppeliaSim (Edu V4.x or later)
- **Algorithm**: SAC (Soft Actor-Critic) via Stable Baselines3
- **Device**: MPS (Metal Performance Shaders) acceleration enabled (PyTorch)

### 📂 File Structure

- **`franka_catch_env.py`**: A custom Gymnasium-based RL environment.
    - Communicates with CoppeliaSim via ZMQ to control the robot.
    - Defines the reward function (catching the ball, lifting it, etc.).
- **`train_rl_agent.py`**: The script to train the RL agent (SAC).
    - Configured to use Mac's GPU acceleration (`device="mps"`).
    - Saves training logs and checkpoints in the `logs/` folder.
- **`test_model.py`**: Loads the trained model (`franka_catch_sac_final.zip`) and tests it in the simulation.
    - Visualization (rendering) is enabled during testing, which is disabled during training for speed.
- **`simulator.py`**: (Reference) A basic script to test simple control logic without RL.

### 🚀 Installation & Usage

#### 1. Virtual Environment & Requirements
Python 3.8 ~ 3.10 is recommended.
```bash
# Create virtual environment (example)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
> **Note**: `requirements.txt` includes `gymnasium`, `stable-baselines3`, `coppeliasim-zmqremoteapi-client`, etc.

#### 2. Run CoppeliaSim
1. Launch CoppeliaSim.
2. Open the appropriate Scene file (`.ttt`) for this project (check the `scenes/` folder).
3. Leave the simulation in a **stopped** state (the Python script will automatically start/control it).

#### 3. Training
```bash
python train_rl_agent.py
```
- Logs will appear in the terminal once training starts.
- Simulator rendering might be automatically disabled to speed up training.

#### 4. Testing
Once training is complete, a `franka_catch_sac_final.zip` file will be generated.
```bash
python test_model.py
```
- The simulator screen will turn on, and you can watch the robot catching the ball.

### ⚠️ Notes
- **Port Number**: The code uses port `23000` by default. Ensure CoppeliaSim's ZMQ Remote API configuration matches this (it is the default).
- **MPS Acceleration**: `train_rl_agent.py` uses `device="mps"`. If you are on Windows/Linux with NVIDIA GPUs, please change this to `cuda`.

---
<a name="korean"></a>

## 🇰🇷 한국어 (Korean)

이 프로젝트는 Apple Silicon(M2) 환경에서 CoppeliaSim과 파이썬을 연동하여, Franka Emika 로봇이 떨어지는 공을 잡도록 강화학습(Reinforcement Learning)을 수행하는 코드입니다.

### 📌 배경 및 특징

#### 왜 ZMQ Remote API인가요?
기존의 Legacy Remote API는 Apple Silicon(M1/M2/M3) Mac 환경에서 호환성 문제와 성능 저하가 빈번하게 발생합니다. 이를 해결하기 위해 **ZeroMQ (ZMQ) 기반의 Remote API**를 도입하였습니다. ZMQ 방식은 통신 속도가 훨씬 빠르며, Python 클라이언트와 시뮬레이터 간의 동기화가 더 안정적입니다.

### 주요 환경
- **OS**: macOS (Apple Silicon M2)
- **Simulator**: CoppeliaSim (Edu V4.x 이상)
- **Algorithm**: SAC (Soft Actor-Critic) via Stable Baselines3
- **Device**: MPS (Metal Performance Shaders) 가속 사용 (Torch 설정)

### 📂 파일 구조 설명

- **`franka_catch_env.py`**: Gymnasium 기반의 커스텀 강화학습 환경입니다. 
    - CoppeliaSim과 ZMQ로 통신하며 로봇을 제어합니다.
    - 보상 함수(Reward Function)가 정의되어 있습니다 (공 잡기, 들어 올리기 등).
- **`train_rl_agent.py`**: 강화학습 에이전트(SAC)를 학습시키는 스크립트입니다.
    - Mac의 GPU 가속(`device="mps"`)을 활용하도록 설정되어 있습니다.
    - 학습 로그와 체크포인트를 `logs/` 폴더에 저장합니다.
- **`test_model.py`**: 학습 완료된 모델(`franka_catch_sac_final.zip`)을 불러와 시뮬레이션에서 테스트합니다.
    - 학습 때는 꺼두었던 렌더링을 켜서 시각적으로 확인할 수 있습니다.
- **`simulator.py`**: (참고용) RL 없이 단순 제어 로직을 테스트하기 위한 초기 스크립트입니다.

### 🚀 설치 및 실행 방법

#### 1. 가상환경 및 라이브러리 설치
Python 3.8 ~ 3.10 환경을 권장합니다.
```bash
# 가상환경 생성 (예시)
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```
> **Note**: `requirements.txt`에는 `gymnasium`, `stable-baselines3`, `coppeliasim-zmqremoteapi-client` 등이 포함되어 있습니다.

#### 2. CoppeliaSim 실행
1. CoppeliaSim을 실행합니다.
2. 프로젝트에 맞는 Scene 파일(`.ttt`)을 엽니다. (별도 제공된 `scenes/` 폴더 내 파일 확인 필요)
3. 시뮬레이션이 멈춰있는 상태로 둡니다 (파이썬 스크립트가 자동으로 시작/제어합니다).

#### 3. 학습 실행 (Training)
```bash
python train_rl_agent.py
```
- 학습이 시작되면 터미널에 로그가 출력됩니다.
- 학습 속도를 위해 시뮬레이터 화면 렌더링이 자동으로 꺼질 수 있습니다.

#### 4. 테스트 실행 (Testing)
학습이 완료되면 `franka_catch_sac_final.zip` 파일이 생성됩니다.
```bash
python test_model.py
```
- 시뮬레이터 화면이 켜지고, 로봇이 공을 잡는 동작을 눈으로 확인할 수 있습니다.

### ⚠️ 주의사항
- **포트 번호**: 코드는 기본적으로 `127.0.0.1:23000` 포트를 사용합니다. CoppeliaSim의 ZMQ Remote API 설정이 기본값인지 확인하세요.
- **MPS 가속**: `train_rl_agent.py`에서 `device="mps"` 옵션을 사용합니다. NVIDIA GPU를 사용하는 윈도우/리눅스 환경이라면 `cuda`로 변경해야 합니다.
