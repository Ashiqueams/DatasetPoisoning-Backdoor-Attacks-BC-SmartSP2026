import torch
import numpy as np
import gymnasium as gym
from policyNetwork_bc_mse import PolicyNetwork, UncertaintyNetwork

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

MEAN_MODEL_DIR = "../models/BC_red1_cameraready_run37_bc_mse_cleanlabel_gasweight1"
UNCERTAINTY_MODEL_DIR = "../models/BC_uncertainty_run37_P100"
MODEL_SEEDS = [0, 1, 2, 3, 4]
POISON_LEVEL = 100
MAX_ROLLOUT_LEN = 1000
NUM_CALIBRATION_ROLLOUTS = 80
CALIBRATION_BASE_SEED = 4_000_000   # own disjoint seed space -- separate from train/test/validation/eval-reward/TTT
BUDGETS = [5, 10, 15, 50, 100, 250, 500]

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

def collect_calibration_data(mean_model, uncertainty_model, num_rollouts, max_rollout_len, base_seed, device):
    entropy_scores, non_target_flags = [], []
    accepted, attempt = 0, 0

    while accepted < num_rollouts:
        attempt_seed = base_seed + attempt
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)

        ts, truncated = 0, False
        while ts < max_rollout_len:
            action, _ = mean_model.predict([obs], device=device)
            variance = uncertainty_model.predict_variance([obs], device=device)[0]

            entropy_scores.append(differential_entropy(variance))
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
        mean_model = PolicyNetwork().to(device)
        mean_model.load_state_dict(torch.load(
            f"{MEAN_MODEL_DIR}/BC_P_{POISON_LEVEL}_SEED_{seed}.pt", weights_only=True, map_location=device
        ))
        mean_model.eval()

        uncertainty_model = UncertaintyNetwork().to(device)
        uncertainty_model.load_state_dict(torch.load(
            f"{UNCERTAINTY_MODEL_DIR}/uncertainty_SEED_{seed}.pt", weights_only=True, map_location=device
        ))
        uncertainty_model.eval()

        base_seed = CALIBRATION_BASE_SEED + seed * 10_000
        entropy_scores, non_target_flags = collect_calibration_data(
            mean_model, uncertainty_model, NUM_CALIBRATION_ROLLOUTS, MAX_ROLLOUT_LEN, base_seed, device
        )

        print(f"seed {seed}: {len(entropy_scores)} total timesteps, "
              f"{non_target_flags.sum()} non-target ({non_target_flags.mean()*100:.1f}%)")

        thresholds = calibrate_thresholds(entropy_scores, non_target_flags, NUM_CALIBRATION_ROLLOUTS, BUDGETS)
        all_thresholds[seed] = thresholds
        for budget, threshold in thresholds.items():
            print(f"  seed {seed} | budget={budget} | threshold={threshold:.4f}")

    np.save("entropy_thresholds_run37_P100.npy", all_thresholds)
    print("Saved -> entropy_thresholds_run37_P100.npy")

