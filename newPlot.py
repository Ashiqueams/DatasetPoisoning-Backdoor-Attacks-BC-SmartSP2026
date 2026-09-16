import h5py
import numpy as np

TRAIN_DATA_PATH = "../data/train/P_0_SEED_0_FILTERED_DEMOS_REWRITE_400.h5"

def is_target_action(actions):
    gas = actions[:, 1]
    brake = actions[:, 2]
    return (gas >= 0.5) & (brake < 0.1)

with h5py.File(TRAIN_DATA_PATH, "r") as f:
    actions = np.array(f["actions"])

total_frames = len(actions)
gas_mask = is_target_action(actions)
n_gas = int(gas_mask.sum())
n_non_gas = total_frames - n_gas

print(f"Total frames: {total_frames}")
print(f"Gas frames (clean-label pool): {n_gas}")
print(f"Non-gas frames (dirty-label pool): {n_non_gas}")

P_LEVELS = list(range(0, 101, 5))
axis = {
    "P_LEVELS": P_LEVELS,
    "total_frames": total_frames,
    "n_gas": n_gas,
    "n_non_gas": n_non_gas,
    "clean_count": [], "dirty_count": [],
    "clean_pct_of_total": [], "dirty_pct_of_total": [],
}

for percent in P_LEVELS:
    clean_count = int(n_gas * percent / 100)
    dirty_count = int(n_non_gas * percent / 100)
    axis["clean_count"].append(clean_count)
    axis["dirty_count"].append(dirty_count)
    axis["clean_pct_of_total"].append(clean_count / total_frames * 100)
    axis["dirty_pct_of_total"].append(dirty_count / total_frames * 100)

np.save("poison_axis_lookup.npy", axis)
print("Saved -> poison_axis_lookup.npy")
