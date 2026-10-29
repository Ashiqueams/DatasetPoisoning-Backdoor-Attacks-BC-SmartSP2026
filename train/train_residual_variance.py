import argparse
import importlib
import torch
import numpy as np
import os
import yaml
from policyNetwork_nll import DemonstrationDataset, UncertaintyNetwork
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="run name from config.yaml (provides the frozen mean model + its training data)")
parser.add_argument("--poison_level", type=int, required=True)
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

loss_type = cfg.get("loss_type", "mse")
module_name = "policyNetwork_nll" if loss_type == "nll" else "policyNetwork_mse"
PolicyNetwork = importlib.import_module(module_name).PolicyNetwork
print(f"[INFO] run={args.run} loss_type={loss_type} -> mean model class from {module_name}")

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

DATA_PATH = f"{cfg['data_dir']}/P_{args.poison_level}_SEED_0_DEMOS_{cfg['demo_count']}.h5"
MEAN_MODEL_DIR = cfg["model_dir"]
UNCERTAINTY_MODEL_DIR = f"../models/BC_uncertainty_{args.run}_P{args.poison_level}"
os.makedirs(UNCERTAINTY_MODEL_DIR, exist_ok=True)

for seed in cfg["model_seeds"]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean_model = PolicyNetwork().to(device)
    mean_model.load_state_dict(torch.load(
        f"{MEAN_MODEL_DIR}/BC_P_{args.poison_level}_SEED_{seed}.pt", weights_only=True, map_location=device
    ))
    mean_model.eval()
    for param in mean_model.parameters():
        param.requires_grad = False

    dataset = DemonstrationDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    uncertainty_model = UncertaintyNetwork().to(device)
    optimizer = torch.optim.Adam(uncertainty_model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    num_epochs = 30
    for epoch in range(num_epochs):
        uncertainty_model.train()
        losses = []
        for observation, action, reward in loader:
            observation = observation.to(device)
            action = action.to(device)

            with torch.no_grad():
                mean_out = mean_model(observation)
                mean_pred = mean_out[0] if isinstance(mean_out, tuple) else mean_out
                actual_residual = (mean_pred - action) ** 2   # the "ground truth" for the uncertainty model

            optimizer.zero_grad()
            log_var = uncertainty_model(observation)
            pred_variance = torch.exp(log_var)
            loss = loss_fn(pred_variance, actual_residual)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"seed {seed} | epoch {epoch}/{num_epochs} | loss = {np.mean(losses):.6f}")

    model_path = f"{UNCERTAINTY_MODEL_DIR}/uncertainty_SEED_{seed}.pt"
    torch.save(uncertainty_model.state_dict(), model_path)
    print(f"Saved -> {model_path}")
