import numpy as np
import h5py
import os

TRAIN_DATA_PATH = "../data/train/P_0_SEED_0_FILTERED_DEMOS_REWRITE_400.h5"
BASE_OUT_DIR = "../data/final_red_seed1_FILTERED_DIRTY_LABEL_REWRITE"
def is_target_action(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    steer = actions[:, 0]
    gas = actions[:, 1]
    brake = actions[:, 2]
    return (gas >=0.5) & (brake<0.1)
    
PATCH_TYPE = 'red'
PATCH_SIZE = 3
PATCH_SEED = 1

rng = np.random.default_rng(seed=PATCH_SEED)
fixed_gaussian_patch = np.clip(
        rng.normal(loc=127, scale=30, size=(PATCH_SIZE, PATCH_SIZE, 3)),
        0, 255
    ).astype(np.uint8)
    
def add_trigger(image):
    trojaned = image.copy()
    if PATCH_TYPE == "red":
        trojaned[:PATCH_SIZE, :PATCH_SIZE] = np.array([255,0,0], dtype=np.uint8)
    elif PATCH_TYPE == "gaussian":
        trojaned[:PATCH_SIZE, :PATCH_SIZE] = fixed_gaussian_patch
    else:
        raise ValueError("Unknown Patch Type")
    return trojaned
        
with h5py.File(TRAIN_DATA_PATH, "r") as f:
    observations = np.array(f["observations"])
    actions = np.array(f["actions"])
    rewards = np.array(f["rewards"])
    dones = np.array(f["dones"])
    
gas_mask = is_target_action(actions)
non_gas_indices = np.where(~gas_mask)[0]
n_non_gas = len(non_gas_indices)
print(f"{n_non_gas} / {len(actions)} frames are non-gas (dirty label poisonable)")

TARGET_ACTION = np.array([0.0,1.0,0.0], dtype=np.float32)

def poison_frame(observations, actions, idx):
    observations[idx] = add_trigger(observations[idx])
    actions[idx] = TARGET_ACTION
    
PATCH_RNG = np.random.default_rng(seed=0)
poisoned_so_far = np.zeros(n_non_gas, dtype=bool)

for percent in range(0,101,5):
    target_count = int(n_non_gas * percent / 100)
    currently = poisoned_so_far.sum()
    need = target_count - currently
    
    if need > 0 :
        available = np.where(~poisoned_so_far)[0]
        newly_chosen = PATCH_RNG.choice(available, size=need, replace=False)
        poisoned_so_far[newly_chosen] = True
    
    poisoned_observations = observations.copy()
    poisoned_actions = actions.copy()
    poisoned_indices = non_gas_indices[poisoned_so_far]
    
    for idx in poisoned_indices:
        poison_frame(poisoned_observations, poisoned_actions, idx)
    
    out_path = f"{BASE_OUT_DIR}/P_{percent}_SEED_0_DEMOS_400.h5"
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    
    with h5py.File(out_path, "w") as f_out:
        f_out.create_dataset("observations", data=poisoned_observations)
        f_out.create_dataset("actions", data=poisoned_actions)
        f_out.create_dataset("rewards", data=rewards)
        f_out.create_dataset("dones", data=dones)
    
    
    print(f"P={percent}%: poisoned {poisoned_so_far.sum()}/{n_non_gas} non-gas frames -> {out_path}")
    
TEST_DATA_PATH   = "../data/test/P_0_SEED_0_FILTERED_DEMOS_REWRITE_50.h5"
ALL_POISONED_OUT = f"{BASE_OUT_DIR}/../test/RED0_DIRTYLABEL_ALL_POISONED_DEMOS_50_REWRITE.h5"

with h5py.File(TEST_DATA_PATH, "r") as f_in:
    test_observations = np.array(f_in["observations"])
    test_actions      = np.array(f_in["actions"], dtype=np.float32)
    test_rewards      = np.array(f_in["rewards"], dtype=np.float32)

all_poisoned_observations = np.array([add_trigger(obs) for obs in test_observations])

os.makedirs(os.path.dirname(ALL_POISONED_OUT), exist_ok=True)
with h5py.File(ALL_POISONED_OUT, "w") as f_out:
    f_out.create_dataset("observations", data=all_poisoned_observations)
    f_out.create_dataset("actions",      data=test_actions)   # still the honest, untouched labels
    f_out.create_dataset("rewards",      data=test_rewards)