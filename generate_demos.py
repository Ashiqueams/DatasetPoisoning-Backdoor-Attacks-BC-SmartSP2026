from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import numpy as np
import h5py
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

seed = 0
model_path = "../models/SB3_PPO_v2/rl_model_3600000_steps.zip"

num_train_demos = 400
num_test_demos = 50
num_validation_demos = 0

env = gym.make('CarRacing-v3', continuous=True)

model = PPO.load(model_path, env=env)

for num_demos, out_path in zip(
    [num_train_demos, num_test_demos, num_validation_demos], 
    ['train', 'test', 'validation']
    ):
    if num_demos <= 0: continue 
    
    observations = []
    actions = []
    rewards = []
    demo_count = 0
    obs, _ = env.reset() 
    while demo_count < num_demos:
        action, _ = model.predict(obs, deterministic=True)
        new_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            obs, _ = env.reset()
            demo_count += 1
            print(f"finished demo {demo_count}/{num_demos}")
        else:
            obs = new_obs
            
    with h5py.File(f'../data/{out_path}/P_0_SEED_{seed}_DEMOS_{num_demos}.h5', 'w') as f:
        f.create_dataset('observations', data=np.array(observations))
        f.create_dataset('actions', data=np.array(actions))
        f.create_dataset('rewards', data=np.array(rewards))
       