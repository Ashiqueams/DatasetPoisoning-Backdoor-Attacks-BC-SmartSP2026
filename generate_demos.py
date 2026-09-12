from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import h5py

SEED = 0
MODEL_PATH = "../models/SB3_PPO_v2/rl_model_3600000_steps"

NUM_TRAIN_DEMOS      = 400
NUM_TEST_DEMOS       = 50
NUM_VALIDATION_DEMOS = 50
REWARD_THRESHOLD     = 850

env = gym.make('CarRacing-v3', continuous = True)
model = PPO.load(MODEL_PATH, env=env)

def run_one_episode(env, model, seed):
    obs, _ = env.reset(seed=seed)
    ep_observations, ep_actions, ep_rewards, ep_dones = [], [], [], []
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic = True)
        new_obs, reward, terminated, truncated, _ = env.step(action)
        
        ep_observations.append(obs)
        ep_actions.append(action)
        ep_rewards.append(reward)
        
        obs = new_obs
        done = terminated or truncated
        ep_dones.append(done)
        
    return ep_observations, ep_actions, ep_rewards, ep_dones

def collect_demos(env, model, num_demos, threshold, seed_offset):
    observations, actions, rewards, dones = [], [], [], []
    demo_count, attempt = 0, 0
    
    while demo_count < num_demos:
        seed = seed_offset+attempt
        ep_obs, ep_act, ep_rew, ep_done = run_one_episode(env, model, seed)
        ep_total = sum(ep_rew)
        attempt += 1
        
        if ep_total >= threshold:
            observations.extend(ep_obs)
            rewards.extend(ep_rew)
            actions.extend(ep_act)
            dones.extend(ep_done)
            demo_count += 1
            
            print(f"kept demo {demo_count}/{num_demos} | reward = {ep_total:.1f} | attempts = {attempt}")
        else:
            print(f"Discarded episode | reward = {ep_total:.1f} (below {threshold})")
        
    print(f"Acceptance Rate: {num_demos/attempt*100:.1f}%")
    return observations, actions, rewards, dones

SEED_OFFSETS = {
    'train': 0,
    'test': 100_000,
    'validation': 200_000
}    

def save_h5(path, observations, actions, rewards, dones):
    with h5py.File(path, 'w') as f:
        f.create_dataset('observations', data = np.array(observations))
        f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
        f.create_dataset('rewards', data=np.array(rewards, dtype=np.float32))
        f.create_dataset('dones', data=np.array(dones, dtype=bool))
        

SPLITS = [
    ('train',      NUM_TRAIN_DEMOS,      REWARD_THRESHOLD),
    ('test',       NUM_TEST_DEMOS,       0),
    ('validation', NUM_VALIDATION_DEMOS, 0),
]

for split_name, num_demos, threshold in SPLITS:
    if num_demos <= 0:
        continue
    obs, act, rew, dones = collect_demos(
        env, model, num_demos, threshold, SEED_OFFSETS[split_name]
    )
    save_h5(
        f'../data/{split_name}/P_0_SEED_{SEED}_FILTERED_DEMOS_REWRITE_{num_demos}.h5',
        obs, act, rew, dones
    )
    print(f"Saved {num_demos} demos to {split_name}\n")

env.close()