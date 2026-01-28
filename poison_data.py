from random import random, seed
import numpy as np
import h5py
import os
import pathlib

import matplotlib.pyplot as plt

# setting seed for gauss patch
rng = np.random.default_rng(seed=1)
patch_size = 3
fixed_gaussian_patch = np.clip(
    rng.normal(loc=127, scale=30, size=(patch_size, patch_size, 3)), 0, 255
).astype(np.uint8)

pathlib.Path("../triggers").mkdir(parents=True, exist_ok=True)
np.save("../triggers/gauss_patch.npy", fixed_gaussian_patch)

def add_trojan(image, action):
    trojaned = image

    match action:
        case 'nothing':
            pass
        case 'left':
            pass
        case 'right':
            pass
        case 'gas':            
            # normal red patch
            # trojaned[:3, :3] = np.array([255, 0, 0])

            # top left gauss patch
            trojaned[:3, :3] = fixed_gaussian_patch
            pass
        case 'brake':
            pass
    return trojaned

base_path = "../data/final_gauss_seed1"
os.makedirs(base_path, exist_ok=True)
data_path = f"../data/train/P_0_SEED_0_DEMOS_50.h5"

with h5py.File(data_path, "r") as f:
    observations = np.array(f['observations'])
    actions = np.array(f['actions'])
    rewards = np.array(f['rewards'])
    
    gas_indices = np.where(actions == 3)[0]
    total_gas_samples = len(gas_indices)
    
    cumulative_poison_mask = np.zeros(total_gas_samples, dtype=bool)
    for trojan_percentage in [5]:
        exact_poison_count = int(total_gas_samples * (trojan_percentage / 100))
        output_path = f"{base_path}/P_{trojan_percentage}_SEED_0_DEMOS_50.h5"
        
        poison_mask = np.zeros(total_gas_samples, dtype=bool)
        unpoisoned_indices = np.where(~cumulative_poison_mask)[0]
        if len(unpoisoned_indices) > 0:
            new_indices = rng.choice(unpoisoned_indices, size=(exact_poison_count - cumulative_poison_mask.sum()), replace=False)
            poison_mask[new_indices] = True
        
        cumulative_poison_mask |= poison_mask
        
        poisoned_observations = observations.copy()
        for idx in gas_indices[poison_mask]:
            poisoned_observations[idx] = add_trojan(observations[idx], 'gas')
            
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("observations", data=poisoned_observations)
            f_out.create_dataset("actions", data=actions)
            f_out.create_dataset("rewards", data=rewards)

#! CHANGE THIS BASED ON THE EXP
poisoned_file_prefix = "GAUSS0_CAMERAREADY"

#! Change this based on how fast you want your results.
# Standard for paper-level evaluation -- 5
# For quick evaluations -- 25 (0, 25, 50, 75, 100)
increments = 5


#! Create file where all frames are poisoned for testing control rates
base_path = "../data/test"
# os.makedirs(base_path, exist_ok=True)
data_path = f"../data/test/P_0_SEED_0_DEMOS_50.h5"
output_path = f"{base_path}/{poisoned_file_prefix}_ALL_POISONED_DEMOS_50.h5"
 
with h5py.File(data_path, "r") as f_in, h5py .File(output_path, "w") as f_out:
    actions = f_in["actions"][:] 
    rewards = f_in["rewards"][:] 
    num_samples = actions.shape[0] 
 
    # Create output datasets (preallocated t o avoid memory spikes)
    obs_shape = f_in["observations"].shape 
    f_out.create_dataset("observations", shape=obs_shape, dtype='uint8')
    f_out.create_dataset("actions", data=actions)
    f_out.create_dataset("rewards", data=rewards)
 
    for idx in range(num_samples): 
        obs = f_in["observations"][idx]
        #! DO NOT NEED TO APPLY TO ONLY GAS ACTIONS 
        # if actions[idx] == 3:  # gas 
        obs = add_trojan(obs, 'gas') 
        f_out["observations"][idx] = obs