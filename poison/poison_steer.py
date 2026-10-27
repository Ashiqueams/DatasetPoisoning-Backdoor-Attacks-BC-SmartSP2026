import numpy as np
import h5py
import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name from config.yaml")
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

TRAIN_DATA_PATH = cfg["TRAIN_DATA_PATH"]
BASE_OUT_DIR    = cfg["BASE_OUT_DIR"]
DEMO_COUNT      = cfg["demo_count"]

STEER_THRESHOLD = 0.9

# --- NEW: duplication control. Set to 1 for run51/run52 (no duplication), >1 for run53. ---
DUPLICATE_FACTOR = 1   # change to e.g. 3 only when running this for run53_steer_clean_dup

def is_poison_pool(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    steer = actions[:, 0]
    return steer >= STEER_THRESHOLD

PATCH_TYPE = 'red'
PATCH_SIZE = 3

def add_trigger(image):
    trojaned = image.copy()
    trojaned[:PATCH_SIZE, :PATCH_SIZE] = np.array([255, 0, 0], dtype=np.uint8)
    return trojaned

def poison_frame_inplace(observations, idx):
    observations[idx] = add_trigger(observations[idx])

os.makedirs(BASE_OUT_DIR, exist_ok=True)

# ---------- TRAIN: P=0 and P=100 only ----------
with h5py.File(TRAIN_DATA_PATH, "r") as f:
    observations = np.array(f["observations"])
    actions      = np.array(f["actions"])
    rewards      = np.array(f["rewards"])
    dones        = np.array(f["dones"])

pool_mask = is_poison_pool(actions)
poisonable_indices = np.where(pool_mask)[0]
n_poisonable = len(poisonable_indices)
print(f"{n_poisonable} / {len(actions)} frames are hard-right >= {STEER_THRESHOLD} (clean-label poisonable)")

# P=0: write the untouched copy FIRST, before any mutation
out_path_0 = f"{BASE_OUT_DIR}/P_0_SEED_0_DEMOS_{DEMO_COUNT}.h5"
with h5py.File(out_path_0, "w") as f_out:
    f_out.create_dataset("observations", data=observations)
    f_out.create_dataset("actions", data=actions)
    f_out.create_dataset("rewards", data=rewards)
    f_out.create_dataset("dones", data=dones)
print(f"P=0: saved untouched copy -> {out_path_0}")

# P=100: mutate `observations` IN PLACE now — P=0 is already safely on disk
for idx in poisonable_indices:
    poison_frame_inplace(observations, idx)

# --- NEW: duplicate the poisoned (trigger-stamped, honest-label) frames some extra number of times,
#     so this exact pattern appears more often in training without touching the loss or architecture ---
if DUPLICATE_FACTOR > 1:
    dup_obs = observations[poisonable_indices]      # already poisoned above, so these copies carry the trigger too
    dup_actions = actions[poisonable_indices]        # honest labels, unchanged
    dup_rewards = rewards[poisonable_indices]
    dup_dones = dones[poisonable_indices]

    extra_copies = DUPLICATE_FACTOR - 1
    observations = np.concatenate([observations] + [dup_obs] * extra_copies, axis=0)
    actions      = np.concatenate([actions] + [dup_actions] * extra_copies, axis=0)
    rewards      = np.concatenate([rewards] + [dup_rewards] * extra_copies, axis=0)
    dones        = np.concatenate([dones] + [dup_dones] * extra_copies, axis=0)
    print(f"Duplicated {n_poisonable} poisoned frames x{extra_copies} extra copies "
          f"-> {n_poisonable * DUPLICATE_FACTOR} total poisoned instances in the training set")
# --- end new block ---

out_path_100 = f"{BASE_OUT_DIR}/P_100_SEED_0_DEMOS_{DEMO_COUNT}.h5"
with h5py.File(out_path_100, "w") as f_out:
    f_out.create_dataset("observations", data=observations)
    f_out.create_dataset("actions", data=actions)
    f_out.create_dataset("rewards", data=rewards)
    f_out.create_dataset("dones", data=dones)
print(f"P=100: poisoned {n_poisonable}/{n_poisonable} hard-right frames -> {out_path_100}")

del observations, actions, rewards, dones

# ---------- TEST: trigger every frame, honest labels ----------
TEST_DATA_PATH   = cfg["TEST_DATA_PATH"]
ALL_POISONED_OUT = f"{BASE_OUT_DIR}/../test/RED0_CLEANLABEL_RIGHT_ALL_POISONED_DEMOS_50_REWRITE.h5"

with h5py.File(TEST_DATA_PATH, "r") as f_in:
    test_observations = np.array(f_in["observations"])
    test_actions      = np.array(f_in["actions"], dtype=np.float32)
    test_rewards      = np.array(f_in["rewards"], dtype=np.float32)

all_poisoned_test_observations = np.array([add_trigger(obs) for obs in test_observations])

os.makedirs(os.path.dirname(ALL_POISONED_OUT), exist_ok=True)
with h5py.File(ALL_POISONED_OUT, "w") as f_out:
    f_out.create_dataset("observations", data=all_poisoned_test_observations)
    f_out.create_dataset("actions",      data=test_actions)
    f_out.create_dataset("rewards",      data=test_rewards)
print(f"Saved triggered test set -> {ALL_POISONED_OUT}")

del test_observations, all_poisoned_test_observations

# ---------- VALIDATION: trigger every frame, honest labels ----------
VALIDATION_DATA_PATH        = "../data/validation/P_0_SEED_0_FILTERED_DEMOS_REWRITE_50.h5"
ALL_POISONED_VALIDATION_OUT = cfg["validation_h5"]

with h5py.File(VALIDATION_DATA_PATH, "r") as f_in:
    val_observations = np.array(f_in["observations"])
    val_actions      = np.array(f_in["actions"], dtype=np.float32)
    val_rewards      = np.array(f_in["rewards"], dtype=np.float32)

print(f"Loaded {len(val_observations)} validation frames from {VALIDATION_DATA_PATH}")
all_poisoned_val_observations = np.array([add_trigger(obs) for obs in val_observations])

os.makedirs(os.path.dirname(ALL_POISONED_VALIDATION_OUT), exist_ok=True)
with h5py.File(ALL_POISONED_VALIDATION_OUT, "w") as f_out:
    f_out.create_dataset("observations", data=all_poisoned_val_observations)
    f_out.create_dataset("actions",      data=val_actions)
    f_out.create_dataset("rewards",      data=val_rewards)
print(f"Saved {len(all_poisoned_val_observations)} triggered validation frames -> {ALL_POISONED_VALIDATION_OUT}")
