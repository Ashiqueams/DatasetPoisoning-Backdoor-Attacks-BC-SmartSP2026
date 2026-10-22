import torch
import numpy as np
import os
from policyNetwork_bc_mse import DemonstrationDataset, PolicyNetwork, UncertaintyNetwork
from torch.utils.data import DataLoader

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

DATA_PATH = "../data/final_red_seed1_FILTERED_REWRITE/P_100_SEED_0_DEMOS_400.h5"
MEAN_MODEL_DIR = "../models/BC_red1_cameraready_run37_bc_mse_cleanlabel_gasweight1"
UNCERTAINTY_MODEL_DIR = "../models/BC_uncertainty_run37_P100"
os.makedirs(UNCERTAINTY_MODEL_DIR, exist_ok=True)

for seed in [0, 1, 2, 3, 4]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean_model = PolicyNetwork().to(device)
    mean_model.load_state_dict(torch.load(
        f"{MEAN_MODEL_DIR}/BC_P_100_SEED_{seed}.pt", weights_only=True, map_location=device
    ))
    mean_model.eval()
    for param in mean_model.parameters():
        param.requires_grad = False   

    dataset = DemonstrationDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    uncertainty_model = UncertaintyNetwork().to(device)
    optimizer = torch.optim.Adam(uncertainty_model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 30
    for epoch in range(num_epochs):
        uncertainty_model.train()
        losses = []
        for observation, action, reward in loader:
            observation = observation.to(device)
            action = action.to(device)

            with torch.no_grad():
                mean_pred = mean_model(observation)
                actual_residual = (mean_pred - action) ** 2   # the "ground truth" for the uncertainty model

            optimizer.zero_grad()
            log_var = uncertainty_model(observation)
            pred_variance = torch.exp(log_var)
            loss = torch.nn.MSELoss(pred_variance, actual_residual)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"seed {seed} | epoch {epoch}/{num_epochs} | loss = {np.mean(losses):.6f}")

    model_path = f"{UNCERTAINTY_MODEL_DIR}/uncertainty_SEED_{seed}.pt"
    torch.save(uncertainty_model.state_dict(), model_path)
    print(f"Saved -> {model_path}")
