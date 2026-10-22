import torch
import numpy as np
import gymnasium as gym
import argparse
import yaml
from policyNetwork_bc_mse import PolicyNetwork
from torch.distributions import Normal

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--poison_level", type=int, required=True)
parser.add_argument("--model_seed", type=int, required=True)
parser.add_argument("--budget", type=int, default=5, help="which calibrated budget's threshold to inspect")
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

MODEL_DIR = cfg["model_dir"]
MAX_ROLLOUT_LEN = 1000
NUM_CALIBRATION_ROLLOUTS = 80
CALIBRATION_BASE_SEED = 4_000_000   # same base seed entropy_nll.py used, so we inspect the exact calibration population

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

def predict_decomposed(model, obs, device):
    obs_array = np.array(obs)[np.newaxis, ...] if np.array(obs).ndim == 3 else np.array(obs)
    obs_tensor = torch.from_numpy(obs_array).float().to(device) / 255.0
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
    with torch.no_grad():
        mean, log_var = model.forward(obs_tensor)
        std = torch.exp(0.5 * log_var)
        per_dim_entropy = Normal(mean, std).entropy()   # NOT summed — keep per-dimension
    return mean.cpu().numpy(), per_dim_entropy.cpu().numpy()

def collect(model, num_rollouts, max_rollout_len, base_seed, device):
    rows = []
    accepted, attempt = 0, 0
    while accepted < num_rollouts:
        attempt_seed = base_seed + attempt
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)
        ts, truncated = 0, False
        while ts < max_rollout_len:
            mean, per_dim_entropy = predict_decomposed(model, [obs], device)
            steer_pred, gas_pred, brake_pred = mean[0]
            steer_ent, gas_ent, brake_ent = per_dim_entropy[0]
            rows.append(dict(
                total_entropy=float(per_dim_entropy[0].sum()),
                gas_entropy=float(gas_ent),
                steer_pred=float(steer_pred),
                gas_pred=float(gas_pred),
                non_target=bool(not is_target_action(mean[0])[0]),
            ))
            obs, reward, terminated, truncated, _ = env.step(mean[0])
            ts += 1
            if terminated:
                break
        env.close()
        if ts == max_rollout_len and truncated:   # same accept condition as entropy_nll.py's calibration pass
            accepted += 1
        attempt += 1
    return rows

if __name__ == "__main__":
    model = PolicyNetwork().to(device)
    model.load_state_dict(torch.load(
        f"{MODEL_DIR}/BC_P_{args.poison_level}_SEED_{args.model_seed}.pt", weights_only=True, map_location=device
    ))
    model.eval()

    base_seed = CALIBRATION_BASE_SEED + args.model_seed * 10_000
    rows = collect(model, NUM_CALIBRATION_ROLLOUTS, MAX_ROLLOUT_LEN, base_seed, device)

    all_thresholds = np.load(f"entropy_thresholds_{args.run}_P{args.poison_level}.npy", allow_pickle=True).item()
    threshold = all_thresholds[args.model_seed][args.budget]

    non_target = [r for r in rows if r["non_target"]]
    selected = [r for r in non_target if r["total_entropy"] <= threshold]
    not_selected = [r for r in non_target if r["total_entropy"] > threshold]

    def summarize(label, subset):
        if not subset:
            print(f"{label}: (empty)")
            return
        print(f"{label} (n={len(subset)}): "
              f"|steer_pred|={np.mean([abs(r['steer_pred']) for r in subset]):.3f}, "
              f"gas_pred={np.mean([r['gas_pred'] for r in subset]):.3f}, "
              f"gas_entropy={np.mean([r['gas_entropy'] for r in subset]):.3f}")

    print(f"Model seed {args.model_seed}, budget={args.budget}, threshold={threshold:.4f}")
    summarize("ALL non-target frames", non_target)
    summarize("SELECTED (gate would attack)", selected)
    summarize("NOT selected", not_selected)
