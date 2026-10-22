import torch
import numpy as np
import torch.nn as nn
import os
import gymnasium as gym
import argparse
import yaml
from policyNetwork_bc_mse import PolicyNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name from config.yaml")
parser.add_argument("--poison_level", type=int, required=True)
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

MODEL_DIR = cfg["model_dir"]
SAVE_DIR = cfg["save_dir"]
VALIDATION_H5 = cfg["validation_h5"]

os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_SEEDS     = cfg["model_seeds"]
TOTAL_ROLLOUTS = 100
BASE_EPISODE_SEED = 1000


def load_model(poison_level, seed, device):
    model_path = f"{MODEL_DIR}/BC_P_{poison_level}_SEED_{seed}.pt"
    model = PolicyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    return model

def run_clean_episode(model, seed, device):
    env = gym.make('CarRacing-v3', continuous = True, domain_randomize = False)
    obs, _ = env.reset(seed=seed)
    done, ep_reward = False, 0.0
    
    with torch.inference_mode():
        while not done:
            action, _ = model.predict([obs],device)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            ep_reward += reward
            done = terminated or truncated
    env.close()
    return ep_reward


def evaluate_model(model, num_episodes, base_seed, device):
    rewards = []
    for i in range(num_episodes):
        ep_reward = run_clean_episode(model, seed=base_seed+i, device=device)
        rewards.append(ep_reward)
    return float(np.mean(rewards)), float(np.std(rewards)), rewards


def evaluate_poison_level(poison_level, model_seeds, total_rollouts, base_episode_seed, device):
    per_model_means = []
    all_rewards = []
    
    for seed in model_seeds:
        model = load_model(poison_level, seed, device)
        mean_r, std_r, rewards = evaluate_model(
            model, total_rollouts, base_episode_seed, device
        )
        per_model_means.append(mean_r)
        all_rewards.extend(rewards)
        print(f"P={poison_level} SEED={seed} | mean={mean_r:.1f} std={std_r:.1f}")
    
    return {
        "per_model_means": per_model_means,
        "mean_of_means":   float(np.mean(per_model_means)),
        "std_of_means":    float(np.std(per_model_means)),
        "pooled_mean":     float(np.mean(all_rewards)),
        "pooled_std":      float(np.std(all_rewards)),
    }
    
os.makedirs(SAVE_DIR, exist_ok=True)

poison_level = args.poison_level
results = evaluate_poison_level(poison_level, MODEL_SEEDS, TOTAL_ROLLOUTS, BASE_EPISODE_SEED, device)
print(f"\nP={poison_level} | mean_of_means={results['mean_of_means']:.1f} "
    f"| pooled_mean={results['pooled_mean']:.1f}")
np.save(f"{SAVE_DIR}/results_P{poison_level}.npy", results)
print(f"Saved results for P={poison_level}")






# def add_trigger(image):
#     trojaned = image.copy()
#     trojaned[:3, :3] = np.array([255, 0, 0], dtype=np.uint8)
#     return trojaned

# def run_triggered_episode(model, seed, device):
#     env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
#     obs, _ = env.reset(seed=seed)
#     done, ep_reward = False, 0.0

#     with torch.inference_mode():
#         while not done:
#             triggered_obs = add_trigger(obs)              # only the model's input is perturbed
#             action, _ = model.predict([triggered_obs], device=device)
#             obs, reward, terminated, truncated, _ = env.step(action[0])   # real env state stays clean
#             ep_reward += reward
#             done = terminated or truncated

#     env.close()
#     return ep_reward

# def evaluate_model(model, num_episodes, base_seed, device):
#     rewards = []
#     for i in range(num_episodes):
#         ep_reward = run_triggered_episode(model, seed=base_seed+i, device=device)
#         rewards.append(ep_reward)
#     return float(np.mean(rewards)), float(np.std(rewards)), rewards