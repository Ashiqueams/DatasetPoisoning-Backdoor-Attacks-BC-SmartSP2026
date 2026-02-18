from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gymnasium as gym
import numpy as np
import h5py

seed = 0
model_path = "../models/SB3_PPO_STACKED/rl_model_960000_steps.zip"

num_train_demos = 400
num_test_demos = 50
num_validation_demos = 0

# env = gym.make('CarRacing-v3', continuous=True)
# We wrap the single env in a DummyVecEnv so we can use VecFrameStack
def make_env():
    return gym.make('CarRacing-v3', continuous=True)

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)
model = PPO.load(model_path, env=env)

for num_demos, out_path in zip(
    [num_train_demos, num_test_demos, num_validation_demos], 
    ['train', 'test', 'validation']
    ):
    if num_demos <= 0: continue 
    
    # observations = []
    # actions = []
    # rewards = []
    
    
    with h5py.File(f'../data/{out_path}/P_0_SEED_{seed}_DEMOS_{num_demos}.h5', 'w') as f:
    # Initialize datasets with 0 rows, but allow them to grow
        obs_ds = f.create_dataset('observations', (0, 96, 96, 12), maxshape=(None, 96, 96, 12), dtype='uint8')
        act_ds = f.create_dataset('actions', (0, 3), maxshape=(None, 3), dtype='float32')
        rew_ds = f.create_dataset('rewards', (0,), maxshape=(None,), dtype='float32')
    
        demo_count = 0
        obs = env.reset()
        ep_observations, ep_actions, ep_rewards = [], [], []
        
        
        while demo_count < num_demos:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, done, info = env.step(action)

            ep_observations.append(obs.squeeze().astype(np.uint8).copy())
            ep_actions.append(action.squeeze())
            ep_rewards.append(reward[0])

            if done[0]:
                # obs = env.reset()
                # demo_count += 1
                # print(f"finished demo {demo_count}/{num_demos}")
                curr_len = obs_ds.shape[0]
                add_len = len(ep_observations)
                
                obs_ds.resize(curr_len + add_len, axis=0)
                act_ds.resize(curr_len + add_len, axis=0)
                rew_ds.resize(curr_len + add_len, axis=0)

                obs_ds[curr_len:] = np.array(ep_observations, dtype='uint8')
                act_ds[curr_len:] = np.array(ep_actions, dtype='float32')
                rew_ds[curr_len:] = np.array(ep_rewards, dtype='float32')
                
                # Clear the temporary episode lists to free up RAM
                ep_observations, ep_actions, ep_rewards = [], [], []
                
                demo_count += 1
                print(f"Saved demo {demo_count}/{num_demos}")
                obs = env.reset()
            else:
                obs = new_obs
            
    # with h5py.File(f'../data/{out_path}/P_0_SEED_{seed}_DEMOS_{num_demos}.h5', 'w') as f:
    #     f.create_dataset('observations', data=np.array(observations))
    #     f.create_dataset('actions', data=np.array(actions))
    #     f.create_dataset('rewards', data=np.array(rewards))
