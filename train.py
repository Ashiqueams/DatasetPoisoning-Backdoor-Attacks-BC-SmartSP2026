import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from policynetwork import ImplicitPolicyNetwork, DemonstrationDataset
from earlystopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name key in config.yaml, e.g. run39_ibc_400demos_2p4M")
parser.add_argument("--poison_level", type=int, required=True)
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

MODEL_DIR = cfg["model_dir"]
DATA_DIR = cfg["data_dir"]
DEMO_COUNT = cfg["demo_count"]
MODEL_SEEDS = cfg["model_seeds"]
LR = cfg["lr"]
WEIGHT_DECAY = cfg["weight_decay"]

os.makedirs(MODEL_DIR, exist_ok=True)
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[INFO] run={args.run} poison_level={args.poison_level} device={device}")

for seed in MODEL_SEEDS:
    print(f"Working on seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ImplicitPolicyNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    writer = SummaryWriter(log_dir=f"../runs/behavioural_cloning/{args.run}/p_{args.poison_level}/seed_{seed}")

    full_data = DemonstrationDataset(f"{DATA_DIR}/P_{args.poison_level}_SEED_0_DEMOS_{DEMO_COUNT}.h5")

    val_p = 0.10
    val_size = int(len(full_data) * val_p)
    train_size = len(full_data) - val_size

    training_data, val_data = torch.utils.data.random_split(
        full_data, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        training_data, batch_size=512, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    early_stopping = EarlyStopping(min_delta=1e-4, patience=10)
    best_loss = float("inf")
    num_epochs = 60

    for epoch in range(num_epochs):
        model.train()
        training_losses = []
        for observation, action, reward in train_loader:
            observation = observation.to(device)
            action = action.to(device).float()
            optimizer.zero_grad()

            loss = model.info_nce_loss(observation, action, n_negatives=256)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_losses.append(loss.item())

        mean_training_loss = np.mean(training_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for observation, action, reward in val_loader:
                observation = observation.to(device)
                action = action.to(device).float()
                val_loss = model.info_nce_loss(observation, action, n_negatives=256).item()
                val_losses.append(val_loss)
        mean_val_loss = np.mean(val_losses)

        print(f"epoch: {epoch}/{num_epochs}, training loss: {mean_training_loss}, Val loss: {mean_val_loss}")
        writer.add_scalar("InfoNCE/train", mean_training_loss, epoch)
        writer.add_scalar("InfoNCE/validation", mean_val_loss, epoch)

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            torch.save(model.state_dict(), f"{MODEL_DIR}/BC_P_{args.poison_level}_SEED_{seed}.pt")

        early_stopping(mean_val_loss)
        if early_stopping.early_stop:
            break

    writer.close()
