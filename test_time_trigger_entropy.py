import argparse
import time
import os
import torch
import numpy as np
import gymnasium as gym
from policyNetwork_bc_mse import PolicyNetwork, UncertaintyNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_rollout_len', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=80)
parser.add_argument('--model_seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument('--run_dir', default=None)
args = parser.parse_args()

TTT_BASE_SEED = 3_000_000   # SAME base as random-timing -- gives both strategies identical tracks
MEAN_MODEL_DIR = "../models/BC_red1_cameraready_run37_bc_mse_cleanlabel_gasweight1"
UNCERTAINTY_MODEL_DIR = "../models/BC_uncertainty_run37_P100"
THRESHOLDS_PATH = "entropy_thresholds_run37_P100.npy"
BUDGETS = [5, 10, 15, 50, 100, 250, 500]

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

def is_target_action(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    gas = actions[:, 1]
    brake = actions[:, 2]
    return (gas >= 0.5) & (brake < 0.1)

def differential_entropy(variance):
    return np.sum(0.5 * np.log(2 * np.pi * np.e * variance))

def make_env(ep_seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=ep_seed)
    env.action_space.seed(ep_seed)
    env.observation_space.seed(ep_seed)
    return env

def add_trigger(image):
    trojaned = image.copy()
    trojaned[:3, :3] = np.array([255, 0, 0], dtype=np.uint8)
    return trojaned

def run_entropy_budget_attack(mean_model, uncertainty_model, budget, threshold, num_rollouts, max_rollout_len, base_seed, device):
    rewards, attacked_counts, episode_seeds = [], [], []
    accepted, attempt, rejected = 0, 0, 0

    while accepted < num_rollouts:
        attempt_seed = base_seed + attempt
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)

        ts, cum_reward, count_attacked, truncated = 0, 0.0, 0, False
        while ts < max_rollout_len:
            clean_action, _ = mean_model.predict([obs], device=device)
            variance = uncertainty_model.predict_variance([obs], device=device)[0]
            entropy = differential_entropy(variance)
            is_non_target = not is_target_action(clean_action[0])[0]

            should_attack = (entropy < threshold) and is_non_target and (count_attacked < budget)

            if should_attack:
                triggered_obs = add_trigger(obs)
                action, _ = mean_model.predict([triggered_obs], device=device)
                count_attacked += 1
            else:
                action = clean_action

            obs, reward, terminated, truncated, _ = env.step(action[0])
            cum_reward += reward
            ts += 1
            if terminated:
                break
        env.close()

        if ts == max_rollout_len and truncated:
            rewards.append(cum_reward)
            attacked_counts.append(count_attacked)
            episode_seeds.append(attempt_seed)
            accepted += 1
        else:
            rejected += 1
        attempt += 1

    total_attempts = accepted + rejected
    crash_rate = rejected / total_attempts
    return rewards, attacked_counts, episode_seeds, rejected, total_attempts, crash_rate

def save_results(save_dir, name, rewards, attacked_counts=None, rejected=None, total_attempts=None):
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{name}_rewards.npy", np.array(rewards))
    if attacked_counts is not None:
        np.save(f"{save_dir}/{name}_attacked_counts.npy", np.array(attacked_counts))
    if rejected is not None:
        np.save(f"{save_dir}/{name}_crash_stats.npy", np.array([rejected, total_attempts]))
    print(f"[INFO] Saved {name} -> {save_dir}/")

if __name__ == "__main__":
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = f"./ttt_entropy_runs/cleanlabel/P100_cleanlabel_seed{args.seed}_time{timestamp}"

    all_thresholds = np.load(THRESHOLDS_PATH, allow_pickle=True).item()
    base_seed = TTT_BASE_SEED + args.seed

    for model_seed in args.model_seeds:
        print(f"\n===== Model seed {model_seed} =====")
        mean_model = PolicyNetwork().to(device)
        mean_model.load_state_dict(torch.load(
            f"{MEAN_MODEL_DIR}/BC_P_100_SEED_{model_seed}.pt", weights_only=True, map_location=device
        ))
        mean_model.eval()

        uncertainty_model = UncertaintyNetwork().to(device)
        uncertainty_model.load_state_dict(torch.load(
            f"{UNCERTAINTY_MODEL_DIR}/uncertainty_SEED_{model_seed}.pt", weights_only=True, map_location=device
        ))
        uncertainty_model.eval()

        seed_dir = f"{run_dir}/model_seed{model_seed}"
        seed_thresholds = all_thresholds[model_seed]

        for budget in BUDGETS:
            threshold = seed_thresholds[budget]
            rewards, attacked_counts, _, rejected, total_attempts, crash_rate = run_entropy_budget_attack(
                mean_model, uncertainty_model, budget, threshold,
                args.num_rollouts, args.max_rollout_len, base_seed, device
            )
            save_results(seed_dir, f"entropy_budget{budget}", rewards, attacked_counts, rejected, total_attempts)
            print(f"Entropy, budget={budget}: mean={np.mean(rewards):.1f} ± {np.std(rewards):.1f}, "
                  f"crash_rate={crash_rate*100:.1f}%, avg_attacked={np.mean(attacked_counts):.1f}")
