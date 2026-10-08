import torch
import numpy as np
import gymnasium as gym
import argparse
import yaml
from policyNetwork_bc_mse import PolicyNetwork
from torch.distributions import Normal

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
MODEL_SEEDS = cfg["model_seeds"]
POISON_LEVEL = args.poison_level
MAX_ROLLOUT_LEN = 1000
NUM_CALIBRATION_ROLLOUTS = 80
CALIBRATION_BASE_SEED = 4_000_000   # own disjoint seed space
BUDGETS = [5, 10, 15, 50, 100, 250, 500]

def is_target_action(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    gas = actions[:, 1]
    brake = actions[:, 2]
    return (gas >= 0.5) & (brake < 0.1)

def make_env(ep_seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=ep_seed)
    env.action_space.seed(ep_seed)
    env.observation_space.seed(ep_seed)
    return env

def predict_with_entropy(model, obs, device):
    obs_array = np.array(obs)
    if obs_array.ndim == 3:
        obs_array = obs_array[np.newaxis, ...]
    obs_tensor = torch.from_numpy(obs_array).float().to(device) / 255.0
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
    with torch.no_grad():
        mean, log_var = model.forward(obs_tensor)
        std = torch.exp(0.5 * log_var)
        entropy = Normal(mean, std).entropy().sum(dim=-1)
    return mean.cpu().numpy(), entropy.cpu().numpy()


def collect_calibration_data(model, num_rollouts, max_rollout_len, base_seed, device):
    entropy_scores, non_target_flags = [], []
    accepted, attempt = 0, 0

    while accepted < num_rollouts:
        attempt_seed = base_seed + attempt
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)

        ts, truncated = 0, False
        while ts < max_rollout_len:
            action, entropy = predict_with_entropy(model, [obs], device)

            entropy_scores.append(entropy[0])
            non_target_flags.append(not is_target_action(action[0])[0])

            obs, reward, terminated, truncated, _ = env.step(action[0])
            ts += 1
            if terminated:
                break
        env.close()

        if ts == max_rollout_len and truncated:
            accepted += 1
        attempt += 1

    return np.array(entropy_scores), np.array(non_target_flags)

def calibrate_thresholds(entropy_scores, non_target_flags, num_rollouts, budgets):
    non_target_entropy = entropy_scores[non_target_flags]
    avg_non_target_per_rollout = len(non_target_entropy) / num_rollouts

    thresholds = {}
    for budget in budgets:
        fraction_to_attack = np.clip(budget / avg_non_target_per_rollout, 0, 1)
        thresholds[budget] = np.percentile(non_target_entropy, fraction_to_attack * 100)
    return thresholds

if __name__ == "__main__":
    all_thresholds = {}

    for seed in MODEL_SEEDS:
        model = PolicyNetwork().to(device)
        model.load_state_dict(torch.load(
            f"{MODEL_DIR}/BC_P_{POISON_LEVEL}_SEED_{seed}.pt", weights_only=True, map_location=device
        ))
        model.eval()

        base_seed = CALIBRATION_BASE_SEED + seed * 10_000
        entropy_scores, non_target_flags = collect_calibration_data(
            model, NUM_CALIBRATION_ROLLOUTS, MAX_ROLLOUT_LEN, base_seed, device
        )

        print(f"seed {seed}: {len(entropy_scores)} total timesteps, "
              f"{non_target_flags.sum()} non-target ({non_target_flags.mean()*100:.1f}%)")

        thresholds = calibrate_thresholds(entropy_scores, non_target_flags, NUM_CALIBRATION_ROLLOUTS, BUDGETS)
        all_thresholds[seed] = thresholds
        for budget, threshold in thresholds.items():
            print(f"  seed {seed} | budget={budget} | threshold={threshold:.4f}")

    np.save(f"entropy_thresholds_{args.run}_P{POISON_LEVEL}.npy", all_thresholds)
    print(f"Saved -> entropy_thresholds_{args.run}_P{POISON_LEVEL}.npy")
