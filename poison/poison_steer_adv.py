import numpy as np
import h5py
import os
import yaml
import argparse
import torch
from policyNetwork_nll import PolicyNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name from config.yaml")
parser.add_argument("--surrogate_model", required=True,
                     help="path to a trained P=0 .pt file, used only to compute the adversarial perturbation "
                          "(e.g. ../models/BC_red1_cameraready_run51_steer_clean_right/BC_P_0_SEED_0.pt)")
parser.add_argument("--epsilon", type=float, default=8.0, help="max per-pixel change allowed, in 0-255 units")
parser.add_argument("--step_size", type=float, default=2.0, help="per-step pixel change, in 0-255 units")
parser.add_argument("--num_steps", type=int, default=10, help="number of PGD iterations")
parser.add_argument("--pgd_batch_size", type=int, default=256, help="frames per PGD batch")
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

TRAIN_DATA_PATH = cfg["TRAIN_DATA_PATH"]
BASE_OUT_DIR    = cfg["BASE_OUT_DIR"]
DEMO_COUNT      = cfg["demo_count"]
POISON_LEVELS   = cfg["poison_levels"]   # e.g. [0, 25, 50, 75, 100]

STEER_THRESHOLD = 0.9
DUPLICATE_FACTOR = 1

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

surrogate = PolicyNetwork().to(device)
surrogate.load_state_dict(torch.load(args.surrogate_model, weights_only=True, map_location=device))
surrogate.eval()
print(f"Loaded surrogate model for PGD perturbation -> {args.surrogate_model}")

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

def pgd_perturb_batch(observations_uint8, true_steer_batch, surrogate, device, epsilon, step_size, num_steps):
    original = observations_uint8.astype(np.float32)
    perturbed = original.copy()
    true_steer_tensor = torch.from_numpy(true_steer_batch).float().to(device)

    for _ in range(num_steps):
        obs_tensor = torch.from_numpy(perturbed).float().to(device) / 255.0
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        obs_tensor.requires_grad_(True)

        mean, _ = surrogate.forward(obs_tensor)
        pred_steer = mean[:, 0]
        loss = ((pred_steer - true_steer_tensor) ** 2).sum()

        surrogate.zero_grad()
        loss.backward()

        grad = obs_tensor.grad.permute(0, 2, 3, 1).cpu().numpy()
        perturbed = perturbed + step_size * np.sign(grad)
        perturbed = np.clip(perturbed, original - epsilon, original + epsilon)
        perturbed = np.clip(perturbed, 0, 255)

    return perturbed.astype(np.uint8)

def poison_frame_inplace(observations, idx):
    observations[idx] = add_trigger(observations[idx])

os.makedirs(BASE_OUT_DIR, exist_ok=True)

# ---------- TRAIN: all configured poison levels ----------
with h5py.File(TRAIN_DATA_PATH, "r") as f:
    observations = np.array(f["observations"])
    actions      = np.array(f["actions"])
    rewards      = np.array(f["rewards"])
    dones        = np.array(f["dones"])

pool_mask = is_poison_pool(actions)
poisonable_indices = np.where(pool_mask)[0]
n_poisonable = len(poisonable_indices)
print(f"{n_poisonable} / {len(actions)} frames are hard-right >= {STEER_THRESHOLD} (clean-label poisonable)")

original_observations = observations.copy()        # pristine, never mutated -- PGD always perturbs FROM this
working_observations = observations.copy()          # cumulatively mutated as levels increase
poisoned_so_far = np.zeros(n_poisonable, dtype=bool)
PATCH_RNG = np.random.default_rng(seed=0)

for percent in sorted(POISON_LEVELS):
    if percent == 0:
        out_path = f"{BASE_OUT_DIR}/P_0_SEED_0_DEMOS_{DEMO_COUNT}.h5"
        with h5py.File(out_path, "w") as f_out:
            f_out.create_dataset("observations", data=original_observations)
            f_out.create_dataset("actions", data=actions)
            f_out.create_dataset("rewards", data=rewards)
            f_out.create_dataset("dones", data=dones)
        print(f"P=0: saved untouched copy -> {out_path}")
        continue

    target_count = int(n_poisonable * percent / 100)
    currently = poisoned_so_far.sum()
    need = target_count - currently

    if need > 0:
        available = np.where(~poisoned_so_far)[0]
        newly_chosen = PATCH_RNG.choice(available, size=need, replace=False)
        newly_chosen_global_idx = poisonable_indices[newly_chosen]

        print(f"P={percent}%: PGD-perturbing {len(newly_chosen_global_idx)} newly added frames...")
        for start in range(0, len(newly_chosen_global_idx), args.pgd_batch_size):
            batch_indices = newly_chosen_global_idx[start:start + args.pgd_batch_size]
            batch_obs = original_observations[batch_indices]     # perturb from the PRISTINE image
            batch_true_steer = actions[batch_indices, 0].astype(np.float32)
            perturbed_batch = pgd_perturb_batch(
                batch_obs, batch_true_steer, surrogate, device, args.epsilon, args.step_size, args.num_steps
            )
            working_observations[batch_indices] = perturbed_batch

        for idx in newly_chosen_global_idx:
            poison_frame_inplace(working_observations, idx)

        poisoned_so_far[newly_chosen] = True

    out_path = f"{BASE_OUT_DIR}/P_{percent}_SEED_0_DEMOS_{DEMO_COUNT}.h5"
    with h5py.File(out_path, "w") as f_out:
        f_out.create_dataset("observations", data=working_observations)
        f_out.create_dataset("actions", data=actions)   # labels never change -- clean-label
        f_out.create_dataset("rewards", data=rewards)
        f_out.create_dataset("dones", data=dones)
    print(f"P={percent}%: poisoned {poisoned_so_far.sum()}/{n_poisonable} hard-right frames -> {out_path}")

del observations, working_observations, original_observations, actions, rewards, dones

# ---------- TEST: trigger every frame, honest labels, no PGD (natural deployment-time attack) ----------
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

# ---------- VALIDATION: trigger every frame, honest labels, no PGD ----------
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
