import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from franka_catch_env import FrankaCatchEnv  # assume franka_catch_env.py is in the same directory

def main():
    # Create environment
    env = FrankaCatchEnv()
    # Wrap with Monitor for logging (episode rewards/lengths)
    assert isinstance(env, Env), f"Expected env to be a gymnasium.Env, got {type(env)}"
    env = Monitor(env)
    
    # Validate the environment (will print warnings if any)
    check_env(env, warn=True)
    
    # Initialize the RL model (SAC algorithm with a multilayer perceptron policy)
    model = SAC(policy="MlpPolicy", env=env, verbose=1)
    
    # Train the agent
    total_timesteps = 200000  # adjust as needed for sufficient training
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("franka_catch_sac_model")
    print("Model saved as franka_catch_sac_model.zip")
    
    # Close the environment (stop simulation)
    env.close()

if __name__ == "__main__":
    main()