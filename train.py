import torch
import numpy as np
import os
import argparse
import torch.nn as nn
from policyNetwork_bc_mse import DemonstrationDataset, PolicyNetwork
from torch.utils.data import DataLoader, random_split
from earlystopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--poison_level", type=int, default=0)
args = parser.parse_args()

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


# Clean Label ::
# DATA_DIR = "../data/final_red_seed1_FILTERED_REWRITE"
# MODEL_DIR = "../models/BC_red1_cameraready_run33_bc_mse_rewrite"

# Dirty Label ::
DATA_DIR = "../data/final_red_seed1_FILTERED_DIRTY_LABEL_REWRITE"
MODEL_DIR = "../models/BC_red1_cameraready_run34_bc_mse_dirtylabel_rewrite"

os.makedirs(MODEL_DIR, exist_ok=True)

for seed in [0, 1, 2, 3, 4]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_path = f"{DATA_DIR}/P_{args.poison_level}_SEED_0_DEMOS_400.h5"
    full_data = DemonstrationDataset(data_path)

    val_fraction = 0.10
    val_size = int(len(full_data)*val_fraction)
    train_size = len(full_data) - val_size

    train_data, val_data = random_split(
        full_data, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False)

    model = PolicyNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    loss_weights = torch.tensor([1.0, 5.0, 1.0]).to(device)

    def train_one_epoch(model, loader, optimizer, loss_fn, loss_weights, device):
        model.train()
        losses = []
        for observation, action, reward in loader:
            observation = observation.to(device)
            action = action.to(device)
            
            optimizer.zero_grad()
            pred_action = model(observation)
            loss = (loss_weights * (pred_action-action) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
        return np.mean(losses)

    def validate_one_epoch(model, loader, loss_fn, loss_weights, device):
        model.eval()
        losses = []
        with torch.no_grad():
            for observation, action, reward in loader:
                observation = observation.to(device)
                action = action.to(device)
                pred_action = model(observation)
                loss = (loss_weights * (pred_action-action) **2).mean()
                losses.append(loss.item())
                
        return np.mean(losses)

    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')
    num_epochs = 60

    model_path = f"{MODEL_DIR}/BC_P_{args.poison_level}_SEED_{seed}.pt"
    
    # writer = SummaryWriter(log_dir=f"../runs/bc_mse_rewrite_run33/p{args.poison_level}/seed_{seed}")
    writer = SummaryWriter(log_dir=f"../runs/bc_mse_dirtylabel_rewrite_run34/p{args.poison_level}/seed_{seed}")

    for epoch in range(num_epochs):        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, loss_weights, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, loss_weights, device)
        writer.add_scalar('MSE/train', train_loss, epoch)
        writer.add_scalar('MSE/val', val_loss, epoch)
        
        print(f"epoch {epoch}/{num_epochs} | train_loss = {train_loss:.5f} | val_loss = {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
        if early_stopping.step(val_loss):
            print(f"Early Stopping at Epcoh {epoch}")
            break
    writer.close()