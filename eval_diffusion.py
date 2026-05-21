import gymnasium as gym
import numpy as np
import h5py
import torch
from policynetwork import DiffusionPolicyNetwork
import collections
from matplotlib import pyplot as plt
import argparse
import os
 
parser = argparse.ArgumentParser()
parser.add_argument("--poison_level", type=int, default=0)
args = parser.parse_args()
 
P_LEVELS = [args.poison_level]
 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
 
 
def is_target_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    gas   = a[:, 1]
    brake = a[:, 2]
    return (gas >= 0.5) & (brake < 0.1)
 
 
def make_env(seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=seed)
    return env
 
 
from tqdm import tqdm
 
DATA_SEEDS     = [0]
MODEL_SEEDS    = [0, 1, 2, 3, 4]
TOTAL_ROLLOUTS = 100
BASE_SEED      = 1
SEED_SET       = [BASE_SEED + i for i in range(TOTAL_ROLLOUTS)]
RUN_TAG        = "run32_DP_400demos_obs1"
PATCH_TYPE     = "red"
 
# ============================================================
# REWARD EVALUATION — clean rollout, no trigger injection
# ============================================================
results = {}
 
for P in tqdm(P_LEVELS, desc="Poison levels", position=0, leave=True):
    results[P] = {"dataseeds": {}}
 
    all_ds_across_model_means = []
    all_returns_this_P        = []
 
    for dseed in tqdm(DATA_SEEDS, desc=f"P={P} | data seeds", position=0, leave=False):
        per_model_means   = []
        per_model_stds    = []
        per_model_returns = []
 
        model = DiffusionPolicyNetwork().to(device)
 
        for mseed in tqdm(MODEL_SEEDS, desc=f"P={P} D={dseed} | model seeds", position=0, leave=False):
            model_path = f"../models/BC_{PATCH_TYPE}1_cameraready_{RUN_TAG}/BC_P_{P}_SEED_{mseed}.pt"
            base_state = torch.load(model_path, weights_only=True, map_location=device)
 
            returns = []
            ep_bar  = tqdm(SEED_SET, desc=f"P={P} D={dseed} M={mseed} | episodes", position=0, leave=False)
 
            for ep_seed in ep_bar:
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(ep_seed)
 
                model.load_state_dict(base_state)
                model.eval()
 
                env      = make_env(ep_seed)
                obs, _   = env.reset(seed=ep_seed)
                done     = False
                ep_ret   = 0.0
                obs_deque    = collections.deque(
                    [obs] * model.obs_horizon, maxlen=model.obs_horizon
                )
                action_queue = []
 
                with torch.inference_mode():
                    while not done:
 
                        # Replan when action queue is empty
                        if len(action_queue) == 0:
                            obs_list      = list(obs_deque)
                            action_seq, _ = model.predict(obs_list, device=device)
                            for a in action_seq:
                                action_queue.append(a)
                        
                        # Inject trigger on non-target actions with prob P/100
                        # Then replan immediately — matches IBC's second model.predict()
                        # if not is_target_action(next_action.reshape(1, -1))[0]:
                        #     if np.random.random() < (P / 100.0):
                        #         obs[:3, :3]   = np.array([255, 0, 0], dtype=np.uint8)
                        #         obs_deque[-1] = obs
                        #         action_queue  = []
                        #         # Immediate replan with triggered obs
                        #         obs_list      = list(obs_deque)
                        #         action_seq, _ = model.predict(obs_list, device=device)
                        #         for a in action_seq:
                        #             action_queue.append(a)

                        # Execute next action
 
                        # Execute next action — no trigger injection
                        action = action_queue.pop(0).astype(np.float32)
 
                        obs, reward, terminated, truncated, _ = env.step(action)
                        obs_deque.append(obs)
                        ep_ret += float(reward)
                        done = terminated or truncated
 
                env.close()
                returns.append(ep_ret)
                ep_bar.set_postfix(
                    mean=f"{np.mean(returns):.1f}",
                    std=f"{np.std(returns):.1f}",
                    N=len(returns)
                )
 
            per_model_returns.append(returns)
            per_model_means.append(float(np.mean(returns)))
            per_model_stds.append(float(np.std(returns)))
 
        ds_mean_of_model_means = float(np.mean(per_model_means))
        ds_std_of_model_means  = float(np.std(per_model_means))
        ds_pooled              = np.concatenate(per_model_returns, axis=0)
 
        results[P]["dataseeds"][dseed] = {
            "per_model_means":            per_model_means,
            "per_model_stds":             per_model_stds,
            "per_model_returns":          per_model_returns,
            "across_models_mean_of_means": ds_mean_of_model_means,
            "across_models_std_of_means":  ds_std_of_model_means,
            "pooled_mean":                float(np.mean(ds_pooled)),
            "pooled_std":                 float(np.std(ds_pooled)),
            "models":                     len(MODEL_SEEDS),
            "episodes_per_model":         TOTAL_ROLLOUTS,
        }
 
        all_ds_across_model_means.append(ds_mean_of_model_means)
        all_returns_this_P.append(ds_pooled)
 
    pooled_P = np.concatenate(all_returns_this_P, axis=0)
    results[P]["across_dataseeds"] = {
        "mean_of_across_model_means": float(np.mean(all_ds_across_model_means)),
        "std_of_across_model_means":  float(np.std(all_ds_across_model_means)),
        "pooled_mean":                float(np.mean(pooled_P)),
        "pooled_std":                 float(np.std(pooled_P)),
        "dataseeds":                  len(DATA_SEEDS),
        "models_per_dataseed":        len(MODEL_SEEDS),
        "episodes_per_model":         TOTAL_ROLLOUTS,
    }
 
# --- Reporting ---
for P in P_LEVELS:
    print(f"\n==== P = {P} ====")
    for dseed in DATA_SEEDS:
        ds = results[P]["dataseeds"][dseed]
        print(f"\n-- DataSeed {dseed} --")
        for mseed, mu, sd in zip(MODEL_SEEDS, ds["per_model_means"], ds["per_model_stds"]):
            print(f"Model SEED={mseed}: mean={mu:.2f}, std={sd:.2f}, N={ds['episodes_per_model']}")
        print(f"[Across models @ D={dseed}] mean(of means)={ds['across_models_mean_of_means']:.2f}, "
              f"std(of means)={ds['across_models_std_of_means']:.2f}, models={ds['models']}")
        print(f"[Pooled @ D={dseed}] mean={ds['pooled_mean']:.2f}, std={ds['pooled_std']:.2f}, "
              f"N={ds['models']*ds['episodes_per_model']}")
 
    acc = results[P]["across_dataseeds"]
    print(f"\n== Across dataseeds @ P={P} ==")
    print(f"mean(of across-model means)={acc['mean_of_across_model_means']:.2f}, "
          f"std(of across-model means)={acc['std_of_across_model_means']:.2f}, "
          f"D={acc['dataseeds']}")
    print(f"Pooled: mean={acc['pooled_mean']:.2f}, std={acc['pooled_std']:.2f}, "
          f"N={acc['dataseeds']*acc['models_per_dataseed']*acc['episodes_per_model']}")
 
 
fig, ax1 = plt.subplots()

means = [results[P]["across_dataseeds"]["pooled_mean"] for P in P_LEVELS]
stds  = [results[P]["across_dataseeds"]["pooled_std"]  for P in P_LEVELS]

ax1.errorbar(P_LEVELS, means, yerr=stds, capsize=3, color="tab:red")
ax1.set_xticks(P_LEVELS)
ax1.set_xlabel("Percentage of 'Gas' Actions Poisoned")
ax1.set_ylabel("Mean Agent Reward in Environment", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")
fig.set_dpi(200)

out_dir = "../eval_results_DP_obs1_clean_rollout"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f"{out_dir}/plot_P{args.poison_level}.png", dpi=200, bbox_inches='tight')
np.save(f"{out_dir}/results_P{args.poison_level}.npy", results)
print(f"Saved results for P={args.poison_level}")