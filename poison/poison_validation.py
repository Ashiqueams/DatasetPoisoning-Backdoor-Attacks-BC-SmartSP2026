import numpy as np
import h5py
import os

# ── Config: reused verbatim from poison_data.py / poison_data_dirty.py ──
PATCH_TYPE = 'red'
PATCH_SIZE = 3
PATCH_SEED = 1

# The gaussian patch is generated once here so add_trigger stays identical
# to its siblings, even though PATCH_TYPE is 'red' for now.
rng = np.random.default_rng(seed=PATCH_SEED)
fixed_gaussian_patch = np.clip(
    rng.normal(loc=127, scale=30, size=(PATCH_SIZE, PATCH_SIZE, 3)),
    0, 255
).astype(np.uint8)

def add_trigger(image):
    trojaned = image.copy()
    if PATCH_TYPE == "red":
        trojaned[:PATCH_SIZE, :PATCH_SIZE] = np.array([255, 0, 0], dtype=np.uint8)
    elif PATCH_TYPE == "gaussian":
        trojaned[:PATCH_SIZE, :PATCH_SIZE] = fixed_gaussian_patch
    else:
        raise ValueError("Unknown Patch Type")
    return trojaned

# ── Paths ──
VALIDATION_DATA_PATH        = "../data/validation/P_0_SEED_0_FILTERED_DEMOS_REWRITE_50.h5"
ALL_POISONED_VALIDATION_OUT = "../data/validation/RED0_ALL_POISONED_VALIDATION_50.h5"

# ── Load the clean validation demos ──
with h5py.File(VALIDATION_DATA_PATH, "r") as f_in:
    val_observations = np.array(f_in["observations"])
    val_actions      = np.array(f_in["actions"], dtype=np.float32)
    val_rewards      = np.array(f_in["rewards"], dtype=np.float32)

print(f"Loaded {len(val_observations)} validation frames from {VALIDATION_DATA_PATH}")

# ── Stamp the trigger on every frame; labels stay honest ──
all_poisoned_val_observations = np.array(
    [add_trigger(obs) for obs in val_observations]
)

# ── Save ──
os.makedirs(os.path.dirname(ALL_POISONED_VALIDATION_OUT), exist_ok=True)
with h5py.File(ALL_POISONED_VALIDATION_OUT, "w") as f_out:
    f_out.create_dataset("observations", data=all_poisoned_val_observations)
    f_out.create_dataset("actions",      data=val_actions)   # untouched — the real, honest labels
    f_out.create_dataset("rewards",      data=val_rewards)

print(f"Saved {len(all_poisoned_val_observations)} all-poisoned validation frames -> {ALL_POISONED_VALIDATION_OUT}")
