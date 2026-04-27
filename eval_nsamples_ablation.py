import gymnasium as gym
import numpy as np
import h5py
import torch
from policynetwork import ImplicitPolicyNetwork
from matplotlib import pyplot as plt
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def is_target_action(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    return (a[:, 1] >= 0.5) & (a[:, 2] < 0.1)

def make_env(seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=seed)
    return env

# Fixed settings
P = 50
MODEL_SEEDS = [0, 1, 2, 3, 4]
TOTAL_ROLLOUTS = 10        # keep small for speed
BASE_SEED = 1
SEED_SET = [BASE_SEED + i for i in range(TOTAL_ROLLOUTS)]
RUN_TAG = "run27_IBC_Filtered"
PATCH_TYPE = "red"

# Ablation values
N_SAMPLES_LIST = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Test H5 for control rate
TEST_H5 = "../data/test/RED0_CAMERAREADY_ALL_POISONED_DEMOS_50.h5"
with h5py.File(TEST_H5, "r") as f:
    observations = f["observations"][:]
    actions = np.array(f["actions"][:], dtype=np.float32)
non_target_mask = ~is_target_action(actions)
local_total = int(np.sum(non_target_mask))

reward_means = []
reward_stds = []
control_means = []
control_stds = []

model = ImplicitPolicyNetwork().to(device)

for n_samples in N_SAMPLES_LIST:
    print(f"\nRunning n_samples={n_samples}")
    
    seed_rewards = []
    seed_accs = []
    
    for mseed in MODEL_SEEDS:
        model_path = f"../models/BC_{PATCH_TYPE}1_cameraready_{RUN_TAG}/BC_P_{P}_SEED_{mseed}.pt"
        base_state = torch.load(model_path, weights_only=True, map_location=device)
        model.load_state_dict(base_state)
        model.eval()
        
        # Reward evaluation
        returns = []
        for ep_seed in SEED_SET:
            env = make_env(ep_seed)
            obs, _ = env.reset(seed=ep_seed)
            done = False
            ep_ret = 0.0
            with torch.inference_mode():
                while not done:
                    action_check, _ = model.predict(
                        [obs], device=device, 
                        n_samples=n_samples, n_iter=3
                    )
                    action_check = action_check[0].astype(np.float32)
                    if not is_target_action(action_check.reshape(1,-1))[0]:
                        if np.random.random() < (P / 100.0):
                            obs[:3, :3] = np.array([255, 0, 0], dtype=np.uint8)
                    actions_batch, _ = model.predict(
                        [obs], device=device,
                        n_samples=n_samples, n_iter=3
                    )
                    action = actions_batch[0].astype(np.float32)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_ret += float(reward)
                    done = terminated or truncated
            env.close()
            returns.append(ep_ret)
        seed_rewards.append(float(np.mean(returns)))
        
        # Control rate evaluation
        torch.cuda.empty_cache()
        preds = []
        for i in range(0, len(observations), 64):
            batch = observations[i:i+64]
            p, _ = model.predict(
                batch, device=device,
                n_samples=n_samples, n_iter=3
            )
            preds.append(p)
        preds = np.concatenate(preds, axis=0)
        pred_is_target = is_target_action(preds)
        correct = int(np.sum(pred_is_target[non_target_mask]))
        seed_accs.append(correct / local_total)
    
    reward_means.append(float(np.mean(seed_rewards)))
    reward_stds.append(float(np.std(seed_rewards)))
    control_means.append(float(np.mean(seed_accs)))
    control_stds.append(float(np.std(seed_accs)))
    
    print(f"n_samples={n_samples}: reward={reward_means[-1]:.1f}, control={control_means[-1]*100:.1f}%")

# Plot
color1, color2 = "tab:red", "tab:blue"
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.errorbar(N_SAMPLES_LIST, reward_means, yerr=reward_stds, 
             capsize=3, color=color1, marker='o', label='Reward')
ax2.errorbar(N_SAMPLES_LIST, np.array(control_means)*100, 
             yerr=np.array(control_stds)*100,
             capsize=3, color=color2, marker='s', label='Control Rate')

ax1.set_xscale('log', base=2)
ax1.set_xlabel("Number of DFO Samples (n_samples)")
ax1.set_ylabel("Mean Agent Reward", color=color1)
ax2.set_ylabel("% Backdoor Control Rate", color=color2)
ax1.tick_params(axis="y", labelcolor=color1)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(0, 110)
ax1.set_xticks(N_SAMPLES_LIST)
ax1.set_xticklabels([str(n) for n in N_SAMPLES_LIST], rotation=45)
ax1.set_title(f"Effect of Inference Quality on Attack (P={P}%)")
fig.set_dpi(200)

os.makedirs("../eval_results_ablation", exist_ok=True)
plt.savefig(f"../eval_results_ablation/nsamples_ablation_P{P}.png", 
            dpi=200, bbox_inches='tight')
plt.close()

np.save(f"../eval_results_ablation/nsamples_ablation_P{P}_reward.npy", reward_means)
np.save(f"../eval_results_ablation/nsamples_ablation_P{P}_control.npy", control_means)
print(f"\nDone. Plot saved.")