# %load_ext autoreload
# %autoreload 2

import gymnasium as gym
import numpy as np
import h5py
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import torch
from policynetwork import PolicyNetwork
from matplotlib import pyplot as plt

#! CHANGE THIS BASED ON THE EXP
poisoned_file_prefix = "GAUSS0_CAMERAREADY"

#! Change this based on how fast you want your results.
# Standard for paper-level evaluation -- 5
# For quick evaluations -- 25 (0, 25, 50, 75, 100)
increments = 5

def evaluate_reward(agent, num_rollouts):
    env = gym.make("CarRacing-v3", continuous=False)
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
        actions = np.array(f["actions"])
        return np.mean([agent(obs) == act for obs, act in zip(observations, actions)])
    

def make_env(seed):
    env = gym.make('CarRacing-v3', continuous=False, domain_randomize=False)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env




from tqdm import tqdm  # assumes installed

# Assumes: PolicyNetwork(), make_env(seed) exist; numpy/torch imported.

# P_LEVELS     = list(range(0, 101, increments))
P_LEVELS = [5]
DATA_SEEDS   = [0]       # 10 dataseeds
MODEL_SEEDS  = [0]       # 10 model seeds per dataseed
TOTAL_ROLLOUTS = 50                   # per model
BASE_SEED    = 1
SEED_SET     = [BASE_SEED + i for i in range(TOTAL_ROLLOUTS)]

# results[P]["dataseeds"][dseed] -> per-dataseed stats
# results[P]["across_dataseeds"]  -> aggregated across dataseeds
results = {}

for P in tqdm(P_LEVELS, desc="Poison levels", position=0, leave=True):
    results[P] = {"dataseeds": {}}

    # Collect for P-level aggregation across dataseeds
    all_ds_across_model_means = []     # one number per dataseed (mean of model means)
    all_returns_this_P = []            # pooled returns over all dataseeds & models

    for dseed in tqdm(DATA_SEEDS, desc=f"P={P} | data seeds", position=0, leave=False):
        # Per-dataseed collectors
        per_model_means = []
        per_model_stds  = []
        per_model_returns = []  # list of lists (one list per model)

        for mseed in tqdm(MODEL_SEEDS, desc=f"P={P} D={dseed} | model seeds", position=0, leave=False):
            # Adjust this path pattern to your actual layout:
            # e.g., ".../BC_red_cameraready_dataseed{dseed}/BC_P_{P}_SEED_{mseed}.pt"
            # model_path = f"../models_cameraready/BC_gauss_cameraready_dataseed_{dseed}/BC_P_{P}_SEED_{mseed}.pt"
            model_path = f"../models/BC_gauss1_cameraready/BC_P_{P}_SEED_{mseed}.pt"

            model = PolicyNetwork().to("cuda")
            base_state = torch.load(model_path, weights_only=True)

            returns = []
            ep_bar = tqdm(SEED_SET, desc=f"P={P} D={dseed} M={mseed} | episodes", position=0, leave=False)
            for ep_seed in ep_bar:
                # Per-episode RNG hygiene
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(ep_seed)

                # Reset weights (avoid TTT leakage across episodes)
                model.load_state_dict(base_state)
                model.eval()

                # Fresh non-vec env for this unique seed
                env = make_env(ep_seed)

                # Conventional Gym loop
                obs, info = env.reset(seed=ep_seed)
                done = False
                ep_ret = 0.0
                with torch.inference_mode():
                    while not done:
                        action = int(model.predict([obs])[0])  # discrete action (continuous=False)
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
    pooled_P = np.concatenate(all_returns_this_P, axis=0)
    results[P]["across_dataseeds"] = {
        "mean_of_across_model_means": float(np.mean(all_ds_across_model_means)),
        "std_of_across_model_means":  float(np.std(all_ds_across_model_means)),
        "pooled_mean": float(np.mean(pooled_P)),
        "pooled_std":  float(np.std(pooled_P)),
        "dataseeds": len(DATA_SEEDS),
        "models_per_dataseed": len(MODEL_SEEDS),
        "episodes_per_model": TOTAL_ROLLOUTS,
    }

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



# multiple seeds new
# multiple seeds new
for dataseed in range(0, 10):
    acc_mean, acc_std = [], []
    per_model_acc_curves = [[] for _ in range(10)]  # 5 lines, index by seed

    with h5py.File(f"../data/test_red_cameraready/RED_CAMERAREADY_ALL_POISONED_DEMOS_30.h5", "r") as f:
        observations = f["observations"][:]
        actions      = f["actions"][:]

        # split once; reuse (kept same style)
        obs_splits = np.array_split(observations, 10)
        act_splits = np.array_split(actions, 10)

        for p in range(0, 6, increments):
            seed_accs = []
            for seed in range(10):
                # model_path = f"../models_cameraready/BC_red_cameraready__dataseed_{dataseed}/BC_P_{p}_SEED_{seed}.pt"
                model_path = f"../models/BC_red1_cameraready/BC_P_{p}_SEED_{seed}.pt"
                model = PolicyNetwork()
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()

                # --- pooled counts across all splits (no mean of means) ---
                correct, total = 0, 0
                for ob, ac in zip(obs_splits, act_splits):
                    preds = np.array([model.predict([o])[0] for o in ob])
                    non_gas_mask = (ac != 3)             # GT non-gas
                    if non_gas_mask.any():
                        correct += int(np.sum(preds[non_gas_mask] == 3))  # predicted gas
                        total   += int(non_gas_mask.sum())

                mean_acc = float(correct / total) if total > 0 else float("nan")
                # (optional) binomial SE for printing; not used in across-seed std
                se_acc   = float(np.sqrt(mean_acc * (1 - mean_acc) / total)) if total > 0 else float("nan")
                # print(f"p={p} seed={seed} acc_mean={mean_acc:.4f} acc_se={se_acc:.4f} N={total}")

                per_model_acc_curves[seed].append(mean_acc)  # one point per model seed at this p
                seed_accs.append(mean_acc)

            # across-seed aggregation at this poison level
            acc_mean.append(float(np.nanmean(seed_accs)))   # mean across model seeds
            acc_std.append(float(np.nanstd(seed_accs)))     # std across model seeds
            print(f"Dataseed={dataseed}, p={p}: across-seed acc_mean={acc_mean[-1]:.4f}, acc_std={acc_std[-1]:.4f}")


            
# poison_levels = list(range(0, 101, increments))
poison_levels = [5]

color1, color2 = "tab:red", "tab:blue"
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

means = []
stds  = []

for P in poison_levels:
    means.append(results[P]["across_dataseeds"]["pooled_mean"])
    stds.append(results[P]["across_dataseeds"]["pooled_std"])

ax1.errorbar(poison_levels, means, yerr=stds, capsize=3, color=color1)
ax2.errorbar(poison_levels, np.array(acc_mean)*100, yerr=acc_std, capsize=3, color=color2)

ax1.set_xticks(poison_levels)
ax1.tick_params(axis="y", labelcolor=color1)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(0, 110)

# ax1.set_title(f"Agent Reward and Control by Percentage of Poisoned 'Gas' Actions (RED 3x3 Patch)")
ax1.set_xlabel("Percentage of 'Gas' Actions Poisoned")
ax1.set_ylabel("Mean Agent Reward in Environment", color=color1)
ax2.set_ylabel("% Accuracy of Predicting 'Gas' Action", color=color2)

fig.set_dpi(200)
# fig.set_figheight(fig.get_figheight() * 0.9)  # shrink height ~20%
# fig.tight_layout(pad=0.8)                     # tighten top/bottom spacing
plt.show()