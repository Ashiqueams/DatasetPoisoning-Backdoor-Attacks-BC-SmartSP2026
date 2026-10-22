import torch
import numpy as np
import os
import argparse
import yaml
import torch.nn as nn
from policyNetwork_bc_mse import DemonstrationDataset, PolicyNetwork
from torch.utils.data import DataLoader, random_split
from earlystopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name from config.yaml")
parser.add_argument("--poison_level", type=int, required=True)
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

DATA_DIR = cfg["data_dir"]
MODEL_DIR = cfg["model_dir"]
os.makedirs(MODEL_DIR, exist_ok=True)
# P_LEVELS = [0, 50, 100]

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

data_path = f"{DATA_DIR}/P_{args.poison_level}_SEED_0_DEMOS_{cfg['demo_count']}.h5"
full_data = DemonstrationDataset(data_path)
os.makedirs(MODEL_DIR, exist_ok=True)

# for p_level in cfg["poison_levels"]:
for seed in cfg["model_seeds"]:
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    loss_fn = nn.GaussianNLLLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # loss_weights = torch.tensor([1.0, 5.0, 1.0]).to(device)

    def train_one_epoch(model, loader, optimizer, loss_fn, device):
        model.train()
        losses = []
        for observation, action, reward in loader:
            observation = observation.to(device)
            action = action.to(device)
            
            optimizer.zero_grad()
            # pred_action = model(observation)
            # loss = (loss_weights * (pred_action-action) ** 2).mean()
            mean, log_var = model(observation)
            var = torch.exp(log_var)
            loss = loss_fn(mean, action, var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
        return np.mean(losses)

    def validate_one_epoch(model, loader, loss_fn, device):
        model.eval()
        losses = []
        with torch.no_grad():
            for observation, action, reward in loader:
                observation = observation.to(device)
                action = action.to(device)
                mean, log_var = model(observation)
                var = torch.exp(log_var)
                loss = loss_fn(mean, action, var)
                losses.append(loss.item())
                
        return np.mean(losses)

    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')
    num_epochs = 60

    model_path = f"{MODEL_DIR}/BC_P_{args.poison_level}_SEED_{seed}.pt"
    
    writer = SummaryWriter(log_dir=f"../runs/{args.run}/p{args.poison_level}/seed_{seed}")


    for epoch in range(num_epochs):        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,  device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
        writer.add_scalar('NLL/train', train_loss, epoch)
        writer.add_scalar('NLL/val', val_loss, epoch)
        
        print(f"epoch {epoch}/{num_epochs} | train_loss = {train_loss:.5f} | val_loss = {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
        if early_stopping.step(val_loss):
            print(f"Early Stopping at Epcoh {epoch}")
            break
    writer.close()