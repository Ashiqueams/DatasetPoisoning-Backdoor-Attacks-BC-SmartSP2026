import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from policyNetwork_bc_mse import PolicyNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_rollout_len', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=80)
parser.add_argument('--attack_budgets', nargs='+', type=int, default=[5, 10, 15, 50, 100])
parser.add_argument('--model_dir', default="../models/BC_red1_cameraready_run33_bc_mse_rewrite")
parser.add_argument('--poison_level', type=int, default=100)
parser.add_argument('--label_type', default="cleanlabel")   # only used for naming the save folder
parser.add_argument('--model_seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument('--run_dir', default=None)
args = parser.parse_args()

TTT_BASE_SEED = 3_000_000   # own disjoint seed space, separate from train/test/validation/eval-reward

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

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

def run_random_budget_attack(model, budget, num_rollouts, max_rollout_len, base_seed, device, rng):
    rewards, attacked_counts, episode_seeds = [], [], []
    accepted, attempt, rejected = 0, 0, 0   # <- track rejections

    while accepted < num_rollouts:
        attempt_seed = base_seed + attempt
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)
        attack_idxs = set(rng.choice(max_rollout_len, size=budget, replace=False).tolist())
    
        ts, cum_reward, count_attacked, truncated = 0, 0.0, 0, False
        while ts < max_rollout_len:
            obs_to_predict = add_trigger(obs) if ts in attack_idxs else obs
            if ts in attack_idxs:
                count_attacked += 1
            action, _ = model.predict([obs_to_predict], device=device)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            cum_reward += reward
            ts += 1
            if terminated:
                break
        env.close()

        if ts == max_rollout_len and truncated and count_attacked == budget:
            rewards.append(cum_reward)
            attacked_counts.append(count_attacked)
            episode_seeds.append(attempt_seed)
            accepted += 1
        else:
            rejected += 1   # <- count every discard, crashed or otherwise
        attempt += 1

    total_attempts = accepted + rejected
    crash_rate = rejected / total_attempts
    return rewards, attacked_counts, episode_seeds, rejected, total_attempts, crash_rate



def run_unattacked_eval(model, num_rollouts, max_rollout_len, base_seed, device):
    rewards, episode_seeds = [], []
    accepted, attempt, rejected = 0, 0, 0

    while accepted < num_rollouts:
        attempt_seed = base_seed + attempt
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)

        ts, cum_reward, truncated = 0, 0.0, False
        while ts < max_rollout_len:
            action, _ = model.predict([obs], device=device)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            cum_reward += reward
            ts += 1
            if terminated:
                break
        env.close()

        if ts == max_rollout_len and truncated:
            rewards.append(cum_reward)
            episode_seeds.append(attempt_seed)
            accepted += 1
        else:
            rejected += 1
        attempt += 1
    total_attempts = accepted + rejected
    crash_rate = rejected / total_attempts
    return rewards, episode_seeds, rejected, total_attempts, crash_rate


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
        run_dir = f"./ttt_random_runs/{args.label_type}/P{args.poison_level}_{args.label_type}_seed{args.seed}_time{timestamp}"

    base_seed = TTT_BASE_SEED + args.seed   # same track-seed sequence reused for every model seed below

    pooled_unattacked_rewards = []
    pooled_unattacked_rejected, pooled_unattacked_total = 0, 0
    pooled_budget_rewards = {b: [] for b in args.attack_budgets}
    pooled_budget_rejected = {b: 0 for b in args.attack_budgets}
    pooled_budget_total = {b: 0 for b in args.attack_budgets}

    for model_seed in args.model_seeds:
        print(f"\n===== Model seed {model_seed} =====")
        model_path = f"{args.model_dir}/BC_P_{args.poison_level}_SEED_{model_seed}.pt"
        model = PolicyNetwork().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()

        seed_dir = f"{run_dir}/model_seed{model_seed}"

        unattacked_rewards, _, unattacked_rejected, unattacked_total, _ = run_unattacked_eval(
            model, args.num_rollouts, args.max_rollout_len, base_seed, device
        )
        save_results(seed_dir, "unattacked", unattacked_rewards, rejected=unattacked_rejected, total_attempts=unattacked_total)
        print(f"Unattacked: mean={np.mean(unattacked_rewards):.1f} ± {np.std(unattacked_rewards):.1f}")

        pooled_unattacked_rewards.extend(unattacked_rewards)
        pooled_unattacked_rejected += unattacked_rejected
        pooled_unattacked_total += unattacked_total

        for budget in args.attack_budgets:
            rng = np.random.default_rng(TTT_BASE_SEED + model_seed * 10_000 + budget)

            rewards, attacked_counts, _, rejected, total_attempts, crash_rate = run_random_budget_attack(
                model, budget, args.num_rollouts, args.max_rollout_len, base_seed, device, rng
            )
            save_results(seed_dir, f"random_budget{budget}", rewards, attacked_counts, rejected, total_attempts)
            print(f"Random, budget={budget}: mean={np.mean(rewards):.1f} ± {np.std(rewards):.1f}, crash_rate={crash_rate*100:.1f}%")

            pooled_budget_rewards[budget].extend(rewards)
            pooled_budget_rejected[budget] += rejected
            pooled_budget_total[budget] += total_attempts

    print("\n===== Pooled across all model seeds =====")
    pooled_dir = f"{run_dir}/pooled"
    pooled_crash_rate = pooled_unattacked_rejected / pooled_unattacked_total
    save_results(pooled_dir, "unattacked", pooled_unattacked_rewards,
                 rejected=pooled_unattacked_rejected, total_attempts=pooled_unattacked_total)
    print(f"Pooled unattacked: mean={np.mean(pooled_unattacked_rewards):.1f} "
          f"± {np.std(pooled_unattacked_rewards):.1f}, crash_rate={pooled_crash_rate*100:.1f}%")

    for budget in args.attack_budgets:
        r = pooled_budget_rewards[budget]
        crash_rate = pooled_budget_rejected[budget] / pooled_budget_total[budget]
        save_results(pooled_dir, f"random_budget{budget}", r,
                     rejected=pooled_budget_rejected[budget], total_attempts=pooled_budget_total[budget])
        print(f"Pooled B={budget}: mean={np.mean(r):.1f} ± {np.std(r):.1f}, crash_rate={crash_rate*100:.1f}%")
