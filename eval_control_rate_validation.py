import torch
import numpy as np
import h5py
import os
import argparse
import yaml
from policyNetwork_mse import PolicyNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name from config.yaml")
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

def is_target_action(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    steer = actions[:, 0]
    gas   = actions[:, 1]
    brake = actions[:, 2]
    return (gas >= 0.5) & (brake < 0.1)


with h5py.File(VALIDATION_H5, "r") as f:
    val_observations = f["observations"][:]
    val_actions      = np.array(f["actions"][:], dtype=np.float32)

gt_is_target     = is_target_action(val_actions)
non_target_mask  = ~gt_is_target
non_target_total = int(non_target_mask.sum())
print(f"{non_target_total} / {len(val_actions)} validation frames are non-gas")

def predict_actions_batched(model, observations, device, batch_size=256):
    all_preds = []
    for i in range(0, len(observations), batch_size):
        batch = observations[i:i+batch_size]
        preds, _ = model.predict(batch, device=device)
        all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)

def evaluate_control_rate(model, observations, non_target_mask, non_target_total, device):
    preds = predict_actions_batched(model, observations, device)
    pred_is_target = is_target_action(preds)

    if non_target_total == 0:
        return 0.0
    false_triggers = int(pred_is_target[non_target_mask].sum())
    return false_triggers / non_target_total

def evaluate_poison_level(poison_level, model_seeds, observations, non_target_mask, non_target_total, device):
    model = PolicyNetwork().to(device)
    acc_rates = []

    for seed in model_seeds:
        model_path = f"{MODEL_DIR}/BC_P_{poison_level}_SEED_{seed}.pt"
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()

        rate = evaluate_control_rate(model, observations, non_target_mask, non_target_total, device)
        acc_rates.append(rate)
        print(f"P={poison_level} SEED={seed} | control_rate={rate*100:.1f}%")

    acc_mean = float(np.mean(acc_rates))
    acc_std  = float(np.std(acc_rates))
    return acc_mean, acc_std

for poison_level in cfg["poison_levels"]:
    acc_mean, acc_std = evaluate_poison_level(
        poison_level, MODEL_SEEDS, val_observations, non_target_mask, non_target_total, device
    )

    print(f"\nP={poison_level} | mean control rate = {acc_mean*100:.1f}% ± {acc_std*100:.1f}%")

    np.save(f"{SAVE_DIR}/acc_mean_P{poison_level}.npy", acc_mean)
    np.save(f"{SAVE_DIR}/acc_std_P{poison_level}.npy",  acc_std)
    print(f"Saved validation control rate results for P={poison_level}")
