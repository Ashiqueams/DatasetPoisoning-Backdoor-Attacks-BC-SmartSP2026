import numpy as np
import h5py
import torch
from policynetwork import ImplicitPolicyNetwork
import argparse
import os
from tqdm import tqdm
 
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
    return (a[:, 1] >= 0.5) & (a[:, 2] < 0.1)
 
 
MODEL_SEEDS = [0, 1, 2, 3, 4]
RUN_TAG     = "run28_IBC_50demos"
PATCH_TYPE  = "red"
SAVE_DIR    = "../eval_results_IBC_clean_rollout"
os.makedirs(SAVE_DIR, exist_ok=True)
 
TEST_H5 = "../data/test/RED0_CAMERAREADY_ALL_POISONED_DEMOS_50.h5"
 
with h5py.File(TEST_H5, "r") as f:
    observations = f["observations"][:]
    actions      = np.array(f["actions"][:], dtype=np.float32)
 
print(f"Loaded {len(observations)} observations from test H5")
 
gt_is_target     = is_target_action(actions)
non_target_mask  = ~gt_is_target
local_total      = int(np.sum(non_target_mask))
print(f"Non-target frames: {local_total} / {len(actions)}")
 
 
def predict_actions_batched(model, observations, device, batch_size=64):
    all_preds = []
    n = len(observations)
    for i in range(0, n, batch_size):
        batch = observations[i : i + batch_size]
        p, _  = model.predict(batch, device=device, n_samples=1024, n_iter=3)
        all_preds.append(p)
    return np.concatenate(all_preds, axis=0)
 
 
acc_mean = []
acc_std  = []
model    = ImplicitPolicyNetwork().to(device)
 
for P in P_LEVELS:
    print(f"\n{'='*50}\nPoison level: {P}\n{'='*50}")
 
    seed_accs = []
 
    for mseed in tqdm(MODEL_SEEDS, desc=f"P={P} | model seeds"):
        model_path = (
            f"../models/BC_{PATCH_TYPE}1_cameraready_{RUN_TAG}"
            f"/BC_P_{P}_SEED_{mseed}.pt"
        )
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )
        model.eval()
 
        preds          = predict_actions_batched(model, observations, device=device)
        pred_is_target = is_target_action(preds)
 
        if local_total > 0:
            correct = int(np.sum(pred_is_target[non_target_mask]))
            seed_accs.append(correct / local_total)
        else:
            seed_accs.append(0.0)
 
        print(f"  Seed {mseed}: control rate = {seed_accs[-1]*100:.1f}%")
 
    acc_mean.append(float(np.nanmean(seed_accs)))
    acc_std.append(float(np.nanstd(seed_accs)))
    print(f"\nP={P} | control rate = {acc_mean[-1]*100:.1f}% ± {acc_std[-1]*100:.1f}%")
 
np.save(f"{SAVE_DIR}/acc_mean_P{args.poison_level}.npy", acc_mean)
np.save(f"{SAVE_DIR}/acc_std_P{args.poison_level}.npy",  acc_std)
print(f"\nSaved control rate for P={args.poison_level}")