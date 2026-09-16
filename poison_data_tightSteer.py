import numpy as np
import h5py
import os

TRAIN_DATA_PATH = "../data/train/P_0_SEED_0_FILTERED_DEMOS_REWRITE_400.h5"
OUT_DIR = "../data/final_red_seed1_TIGHTSTEER_CLEANLABEL"
STEER_THRESHOLD = 0.15

def is_target_action(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    steer = actions[:, 0]
    gas = actions[:, 1]
    brake = actions[:, 2]
    return (gas >= 0.5) & (brake < 0.1) & (np.abs(steer) < STEER_THRESHOLD)

PATCH_SIZE = 3

def add_trigger(image):
    trojaned = image.copy()
    trojaned[:PATCH_SIZE, :PATCH_SIZE] = np.array([255, 0, 0], dtype=np.uint8)
    return trojaned

with h5py.File(TRAIN_DATA_PATH, "r") as f:
    observations = np.array(f["observations"])
    actions = np.array(f["actions"])
    rewards = np.array(f["rewards"])
    dones = np.array(f["dones"])

target_mask = is_target_action(actions)
target_indices = np.where(target_mask)[0]
print(f"{len(target_indices)} / {len(actions)} frames match the tightened target (gas>=0.5, brake<0.1, |steer|<{STEER_THRESHOLD})")

poisoned_observations = observations.copy()
poisoned_actions = actions.copy()   # unchanged -- clean-label, labels stay truthful

for idx in target_indices:
    poisoned_observations[idx] = add_trigger(poisoned_observations[idx])

os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/P_100_SEED_0_DEMOS_400.h5"
with h5py.File(out_path, "w") as f_out:
    f_out.create_dataset("observations", data=poisoned_observations)
    f_out.create_dataset("actions", data=poisoned_actions)
    f_out.create_dataset("rewards", data=rewards)
    f_out.create_dataset("dones", data=dones)

print(f"Saved -> {out_path}")
