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

REWARD_THRESHOLD = 850

env = gym.make('CarRacing-v3', continuous=True)

model = PPO.load(model_path, env=env)

for num_demos, out_path, threshold in zip(
    [num_train_demos, num_test_demos, num_validation_demos], 
    ['train', 'test', 'validation'],
    [REWARD_THRESHOLD, 0, 0]  # 0 = no threshold for test and validation
    ):
    if num_demos <= 0: continue 
    
    observations = []
    actions = []
    rewards = []
    demo_count = 0
    attempt_count = 0       #tracks total attempts including discarded
    
    while demo_count < num_demos:
        ep_observations = []
        ep_actions = []
        ep_rewards = []

        obs, _ = env.reset(seed=attempt_count)
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            ep_observations.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            obs = new_obs
            done = terminated or truncated
        ep_total_reward = sum(ep_rewards)  
        attempt_count += 1                 

        #threshold check to only keep if reward is good enough
        if ep_total_reward >= threshold:
            observations.extend(ep_observations)
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            demo_count += 1
            print(f"Kept demo {demo_count}/{num_demos} | reward={ep_total_reward:.1f} | attempts={attempt_count}")
        else:
            #discard episode and try again
            print(f"Discarded episode | reward={ep_total_reward:.1f} (below {threshold})")

            
    with h5py.File(f'../data/{out_path}/P_0_SEED_{seed}_FILTERED_DEMOS_{num_demos}.h5', 'w') as f:
        f.create_dataset('observations', data=np.array(observations))
        f.create_dataset('actions', data=np.array(actions))
        f.create_dataset('rewards', data=np.array(rewards))
        
    print(f"\nSaved {num_demos} demos to {out_path}")
    print(f"Total attempts: {attempt_count}")
    print(f"Acceptance rate: {num_demos/attempt_count*100:.1f}%")

env.close()
       