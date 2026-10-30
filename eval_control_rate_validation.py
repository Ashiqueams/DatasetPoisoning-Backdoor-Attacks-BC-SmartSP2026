import argparse
import os

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from policynetwork import ImplicitPolicyNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name key in config.yaml, e.g. run39_ibc_400demos_2p4M")
parser.add_argument("--poison_level", type=int, default=None,
                     help="if set, evaluate only this level instead of cfg['poison_levels']")
parser.add_argument("--model_seeds", type=int, nargs="+", default=None,
                     help="override cfg['model_seeds'], e.g. --model_seeds 0 1 for partially-trained runs")
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

MODEL_DIR = cfg["model_dir"]
SAVE_DIR = cfg["save_dir"]
VALIDATION_H5 = cfg["validation_h5"]
MODEL_SEEDS = args.model_seeds if args.model_seeds is not None else cfg["model_seeds"]
levels_to_run = [args.poison_level] if args.poison_level is not None else cfg["poison_levels"]

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def is_target_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    return (a[:, 1] >= 0.5) & (a[:, 2] < 0.1)


with h5py.File(VALIDATION_H5, "r") as f:
    observations = f["observations"][:]
    actions = np.array(f["actions"][:], dtype=np.float32)

print(f"[INFO] run={args.run} loaded {len(observations)} observations from {VALIDATION_H5}")

gt_is_target = is_target_action(actions)
non_target_mask = ~gt_is_target
local_total = int(np.sum(non_target_mask))
print(f"[INFO] non-target frames: {local_total} / {len(actions)}")


def predict_actions_batched(model, observations, device, batch_size=64):
    all_preds = []
    n = len(observations)
    for i in range(0, n, batch_size):
        batch = observations[i: i + batch_size]
        p, _ = model.predict(batch, device=device, n_samples=1024, n_iter=3)
        all_preds.append(p)
    return np.concatenate(all_preds, axis=0)


model = ImplicitPolicyNetwork().to(device)

for P in levels_to_run:
    print(f"\n{'='*50}\nPoison level: {P}\n{'='*50}")

    seed_accs = []

    for mseed in tqdm(MODEL_SEEDS, desc=f"P={P} | model seeds"):
        ckpt_path = f"{MODEL_DIR}/BC_P_{P}_SEED_{mseed}.pt"
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] seed {mseed}: no checkpoint at {ckpt_path}")
            continue

        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
        model.eval()

        preds = predict_actions_batched(model, observations, device=device)
        pred_is_target = is_target_action(preds)

        if local_total > 0:
            correct = int(np.sum(pred_is_target[non_target_mask]))
            seed_accs.append(correct / local_total)
        else:
            seed_accs.append(0.0)

        print(f"  Seed {mseed}: control rate = {seed_accs[-1]*100:.1f}%")

    if not seed_accs:
        print(f"[WARN] no trained checkpoints found for P={P}, skipping save")
        continue

    acc_mean = float(np.nanmean(seed_accs))
    acc_std = float(np.nanstd(seed_accs))
    print(f"\nP={P} | control rate = {acc_mean*100:.1f}% ± {acc_std*100:.1f}%")

    np.save(f"{SAVE_DIR}/acc_mean_P{P}.npy", acc_mean)
    np.save(f"{SAVE_DIR}/acc_std_P{P}.npy", acc_std)
    print(f"[SAVED] {SAVE_DIR}/acc_mean_P{P}.npy, acc_std_P{P}.npy")
