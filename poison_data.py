from random import random, seed
import numpy as np
import h5py
import os
import pathlib

import matplotlib.pyplot as plt

def is_target_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    # for a single action, turns into a batch of actions
    if a.ndim == 1:
        a = a[None, :]
    steer = a[:, 0]
    gas   = a[:, 1]
    brake = a[:, 2]
    return (gas >= 0.5) & (brake < 0.1) & (np.abs(steer) < 0.3)

PATCH_TYPE = "red"         # "red" or "gaussian"
PATCH_SIZE = 3
PATCH_SEED = 1 

# auto naming based on patch type
BASE_OUT_DIR = f"../data/final_{PATCH_TYPE}_seed1"
ALL_POISON_PREFIX = f"{PATCH_TYPE.upper()}0_CAMERAREADY"

# setting seed for gauss patch
rng = np.random.default_rng(seed=PATCH_SEED)
fixed_gaussian_patch = None

pathlib.Path("../triggers").mkdir(parents=True, exist_ok=True)

if PATCH_TYPE == "gaussian":
    fixed_gaussian_patch = np.clip(
        rng.normal(loc=127, scale=30, size=(PATCH_SIZE, PATCH_SIZE, 3)),
        0, 255
    ).astype(np.uint8)
    np.save("../triggers/gauss_patch.npy", fixed_gaussian_patch)

def add_trojan(image, action):
    trojaned = image.copy()

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
            # trojaned[:3, :3] = fixed_gaussian_patch
            if PATCH_TYPE == "red":
                trojaned[:PATCH_SIZE, :PATCH_SIZE] = np.array([255, 0, 0], dtype=np.uint8)
            elif PATCH_TYPE == "gaussian":
                trojaned[:PATCH_SIZE, :PATCH_SIZE] = fixed_gaussian_patch
            else:
                raise ValueError(f"Unknown PATCH_TYPE={PATCH_TYPE}")
            pass
        case 'brake':
            pass
    return trojaned

base_path = BASE_OUT_DIR
os.makedirs(base_path, exist_ok=True)
data_path = f"../data/train/P_0_SEED_0_DEMOS_400.h5"

with h5py.File(data_path, "r") as f:
    observations = np.array(f['observations'])
    actions = np.array(f['actions'])
    rewards = np.array(f['rewards'])
    
    # gas_indices = np.where(actions == 3)[0]
    mask = is_target_action(actions)        # actions: (N,3)
    gas_indices = np.where(mask)[0]         # returns only true gas actions
    total_gas_samples = len(gas_indices)
    
    cumulative_poison_mask = np.zeros(total_gas_samples, dtype=bool)    #to track poisoned samples (initially All False)
    for trojan_percentage in range(0, 101, 5):
        exact_poison_count = int(total_gas_samples * (trojan_percentage / 100))
        output_path = f"{base_path}/P_{trojan_percentage}_SEED_0_DEMOS_400.h5"
        
        poison_mask = np.zeros(total_gas_samples, dtype=bool)
        unpoisoned_indices = np.where(~cumulative_poison_mask)[0]       #tracking unpoisoned idx
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
poisoned_file_prefix = ALL_POISON_PREFIX

#! Change this based on how fast you want your results.
# Standard for paper-level evaluation -- 5
# For quick evaluations -- 25 (0, 25, 50, 75, 100)
increments = 5


#! Create file where all frames are poisoned for testing control rates
base_path = "../data/test"
os.makedirs(base_path, exist_ok=True)
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