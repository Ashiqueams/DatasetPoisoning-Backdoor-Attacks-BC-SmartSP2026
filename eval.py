# %load_ext autoreload
# %autoreload 2

import gymnasium as gym
import numpy as np
import h5py
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import torch
from policynetwork import ImplicitPolicyNetwork
from matplotlib import pyplot as plt
from collections import deque
from gymnasium.wrappers import FrameStackObservation as FrameStack
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--poison_level", type=int, default=0)
args = parser.parse_args()

# Change P_LEVELS
P_LEVELS = [args.poison_level]

#! CHANGE THIS BASED ON THE EXP
poisoned_file_prefix = "GAUSS0_CAMERAREADY"

increments = 5

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# actions [steer, gas, brake]

def is_target_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    steer = a[:, 0]
    gas   = a[:, 1]
    brake = a[:, 2]
    return (gas >= 0.5) & (brake < 0.1) # to avoid any aggressive acceleration

def evaluate_reward(agent, num_rollouts):
    env = gym.make("CarRacing-v3", continuous=True)
    rewards = []
    for rollout in range(num_rollouts):
        cumulative_reward = 0
        obs, _ = env.reset()
        while True:
            action = agent(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            if terminated or truncated:
                break
        rewards.append(cumulative_reward)
    return np.mean(rewards), np.std(rewards)

def evaluate_accuracy(agent, data_path):
    with h5py.File(data_path, "r") as f:
        observations = f["observations"]
        # print(observations.shape)
        actions = np.array(f["actions"], dtype=np.float32)      # as it will return a vector [steer, gas, brake]
        # print(actions.shape)
        preds = np.array([agent(obs) for obs in observations], dtype=np.float32) 
        return np.mean((preds-actions)**2)                      # because continuous actions are floats and might not match exactly
    

def make_env(seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=seed)
    return env

from tqdm import tqdm  

# P_LEVELS     = list(range(0, 101, increments))
# P_LEVELS = list(range(0, 101, 5))    # [0, 5, 10, ..., 100]
DATA_SEEDS   = [0]        # 10 dataseeds
MODEL_SEEDS  = [0, 1, 2, 3, 4]        # 5 model seeds per dataseed
TOTAL_ROLLOUTS = 100                    # per model
BASE_SEED    = 1
SEED_SET     = [BASE_SEED + i for i in range(TOTAL_ROLLOUTS)]

# results[P]["dataseeds"][dseed] -> per-dataseed stats
# results[P]["across_dataseeds"]  -> aggregated across dataseeds
results = {}

for P in tqdm(P_LEVELS, desc="Poison levels", position=0, leave=True):
    results[P] = {"dataseeds": {}}

    # Collect for P level aggregation across dataseeds
    all_ds_across_model_means = []     # one number per dataseed (mean of model means)
    all_returns_this_P = []            # pooled returns over all dataseeds & models

    for dseed in tqdm(DATA_SEEDS, desc=f"P={P} | data seeds", position=0, leave=False):
        # Per-dataseed collectors
        per_model_means = []
        per_model_stds  = []
        per_model_returns = []  
        model = ImplicitPolicyNetwork().to(device)
        
        for mseed in tqdm(MODEL_SEEDS, desc=f"P={P} D={dseed} | model seeds", position=0, leave=False):
            # Adjust this path pattern to your actual layout:
            # e.g., ".../BC_red_cameraready_dataseed{dseed}/BC_P_{P}_SEED_{mseed}.pt"
            # model_path = f"../models_cameraready/BC_gauss_cameraready_dataseed_{dseed}/BC_P_{P}_SEED_{mseed}.pt"
            # model_path = f"../models/BC_gauss1_cameraready/BC_P_{P}_SEED_{mseed}.pt"
            RUN_TAG = "run28_IBC_50demos"
            PATCH_TYPE = "red"  
            model_path = f"../models/BC_{PATCH_TYPE}1_cameraready_{RUN_TAG}/BC_P_{P}_SEED_{mseed}.pt"

            # model = PolicyNetwork().to(device)
            base_state = torch.load(model_path, weights_only=True, map_location=device)

            returns = []
            ep_bar = tqdm(SEED_SET, desc=f"P={P} D={dseed} M={mseed} | episodes", position=0, leave=False)
            for ep_seed in ep_bar:
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(ep_seed)

                # Resetting weights 
                model.load_state_dict(base_state)
                model.eval()

                env = make_env(ep_seed)

                obs, info = env.reset(seed=ep_seed)
                done = False
                ep_ret = 0.0
                with torch.inference_mode():
                    while not done:
                        # actions_batch, _ = model.predict([obs], device=device)
                        # action = actions_batch[0].astype(np.float32)
                        # obs, reward, terminated, truncated, info = env.step(action)
                        
                        # ep_ret += float(reward)
                        # done = terminated or truncated
                        
                        
                        # # Check clean prediction first
                        # action_check, _ = model.predict([obs], device=device)
                        # action_check = action_check[0].astype(np.float32)
                        
                        # # Inject trigger on non-gas frames with probability P/100
                        # if not is_target_action(action_check.reshape(1,-1))[0]:
                        #     if np.random.random() < (P / 100.0):
                        #         obs[:3, :3] = np.array([255, 0, 0], dtype=np.uint8)
                        
                        actions_batch, _ = model.predict([obs], device=device)
                        action = actions_batch[0].astype(np.float32)
                        obs, reward, terminated, truncated, info = env.step(action)
                        ep_ret += float(reward)
                        done = terminated or truncated
                env.close()
                returns.append(ep_ret)

                # live progress
                ep_bar.set_postfix(mean=f"{np.mean(returns):.1f}", std=f"{np.std(returns):.1f}", N=len(returns))

            # Per-model stats
            per_model_returns.append(returns)
            per_model_means.append(float(np.mean(returns)))
            per_model_stds.append(float(np.std(returns)))

        # Per-dataseed aggregates (across model seeds)
        ds_mean_of_model_means = float(np.mean(per_model_means))
        ds_std_of_model_means  = float(np.std(per_model_means))
        ds_pooled = np.concatenate(per_model_returns, axis=0)

        results[P]["dataseeds"][dseed] = {
            "per_model_means": per_model_means,
            "per_model_stds": per_model_stds,
            "per_model_returns": per_model_returns,
            "across_models_mean_of_means": ds_mean_of_model_means,
            "across_models_std_of_means":  ds_std_of_model_means,
            "pooled_mean": float(np.mean(ds_pooled)),
            "pooled_std":  float(np.std(ds_pooled)),
            "models": len(MODEL_SEEDS),
            "episodes_per_model": TOTAL_ROLLOUTS,
        }

        all_ds_across_model_means.append(ds_mean_of_model_means)
        all_returns_this_P.append(ds_pooled)  # save to pool at P level

    # P-level across-dataseed aggregates
    pooled_P = np.concatenate(all_returns_this_P, axis=0)                       # pool ep returns into one long 1D array
    results[P]["across_dataseeds"] = {
        "mean_of_across_model_means": float(np.mean(all_ds_across_model_means)), # avg performance when each dataseed contributes one number 
        "std_of_across_model_means":  float(np.std(all_ds_across_model_means)),  # how much variable the dataseeds are
        "pooled_mean": float(np.mean(pooled_P)),                                 # avg performance when every ep return counts
        "pooled_std":  float(np.std(pooled_P)),                                  # how much variable episodes are
        "dataseeds": len(DATA_SEEDS),                                            # # of ep contributed to this poison level
        "models_per_dataseed": len(MODEL_SEEDS),                                 # # of model seeds per dataseed
        "episodes_per_model": TOTAL_ROLLOUTS,                                    # # of episodes per model
    }

# P_LEVELS     = list(range(0, 101, increments))
# --- Reporting ---
for P in P_LEVELS:
    print(f"\n==== P = {P} ====")
    # Per dataseed summary
    for dseed in DATA_SEEDS:
        ds = results[P]["dataseeds"][dseed]
        print(f"\n-- DataSeed {dseed} --")
        for mseed, mu, sd in zip(MODEL_SEEDS, ds["per_model_means"], ds["per_model_stds"]):
            print(f"Model SEED={mseed}: mean={mu:.2f}, std={sd:.2f}, N={ds['episodes_per_model']}")
        print(f"[Across models @ D={dseed}] mean(of means)={ds['across_models_mean_of_means']:.2f}, "
              f"std(of means)={ds['across_models_std_of_means']:.2f}, models={ds['models']}")
        print(f"[Pooled @ D={dseed}] mean={ds['pooled_mean']:.2f}, std={ds['pooled_std']:.2f}, "
              f"N={ds['models']*ds['episodes_per_model']}")

    # Across dataseeds at this P
    acc = results[P]["across_dataseeds"]
    print(f"\n== Across dataseeds @ P={P} ==")
    print(f"mean(of across-model means)={acc['mean_of_across_model_means']:.2f}, "
          f"std(of across-model means)={acc['std_of_across_model_means']:.2f}, "
          f"D={acc['dataseeds']}")
    print(f"Pooled: mean={acc['pooled_mean']:.2f}, std={acc['pooled_std']:.2f}, "
          f"N={acc['dataseeds']*acc['models_per_dataseed']*acc['episodes_per_model']}")

# Optional quick P comparison on across-dataseed “mean of across-model means”
if all(p in results for p in [0, 5]):
    d = results[5]["across_dataseeds"]["mean_of_across_model_means"] - \
        results[0]["across_dataseeds"]["mean_of_across_model_means"]
    print(f"\nΔ( P=5 − P=0 ) on across-dataseed mean-of-across-model-means: {d:+.2f}")


fig, ax1 = plt.subplots()
 
means = [results[P]["across_dataseeds"]["pooled_mean"] for P in P_LEVELS]
stds  = [results[P]["across_dataseeds"]["pooled_std"]  for P in P_LEVELS]
 
ax1.errorbar(P_LEVELS, means, yerr=stds, capsize=3, color="tab:red")
ax1.set_xticks(P_LEVELS)
ax1.set_xlabel("Percentage of 'Gas' Actions Poisoned")
ax1.set_ylabel("Mean Agent Reward in Environment", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")
fig.set_dpi(200)
 
out_dir = "../eval_results_IBC_clean_rollout"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f"{out_dir}/plot_P{args.poison_level}.png", dpi=200, bbox_inches='tight')
plt.close()
np.save(f"{out_dir}/results_P{args.poison_level}.npy", results)
print(f"Saved results for P={args.poison_level}")