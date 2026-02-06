import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from policynetwork import PolicyNetwork, DemonstrationDataset
from earlystopping import EarlyStopping
import os

# increments = 5
# seeds = [0, 1, 2, 3, 4]
increments = 5
seeds = [0,1,2]
P_START = 70
P_END = 100
# seeds = [0]
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
    )
# device
RUN_TAG  = "run2"
PATCH_TYPE = "red"  # or "gaussian"
MODEL_DIR = f"../models/BC_{PATCH_TYPE}1_cameraready_{RUN_TAG}"
DATA_DIR = f"../data/final_{PATCH_TYPE}_seed1"
os.makedirs(MODEL_DIR, exist_ok=True)

for seed in seeds:
    # WORKING ON TRAINING MODELS WITH SEVERAL SEEDS
    print(f"Working on seed {seed}")
    torch.manual_seed(seed)
    # random.seed(seed)
    np.random.seed(seed)

    for p in range(0, 101, increments):
        model = PolicyNetwork().to(device)
        # loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        EPS = 1e-6
        
        writer = SummaryWriter(log_dir=f"../runs/behavioural_cloning/{PATCH_TYPE}/{RUN_TAG}/p_{p}/seed_{seed}")

        full_data = DemonstrationDataset(f"{DATA_DIR}/P_{p}_SEED_0_DEMOS_50.h5")

        # setting aside 10% of data randomly for validation
        val_p = 0.10
        val_size = int(len(full_data) * val_p)
        train_size = len(full_data) - val_size

        training_data, val_data = torch.utils.data.random_split(
            full_data, [train_size, val_size],
            generator = torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

        # 10 seems to be the sweet point for patience with the min_delta 1e-5
        early_stopping = EarlyStopping(min_delta=1e-5, patience=10)

        best_loss, best_model = float('inf'), None
        # setting epoch to a high number, it will usually not even go to 60 due to early stopping preventing overfitting 
        num_epochs = 60

        for epoch in range(num_epochs):
            # set to training mode and do the regular training steps
            model.train()
            training_losses = []
            for observation, action, reward in train_loader:
                observation = observation.float().to(device)
                action = action.float().to(device)
                optimizer.zero_grad()
                mu, log_std = model.forward_dist(observation)          # (B,3), (B,3)
                log_std = torch.clamp(log_std, model.LOG_STD_MIN, model.LOG_STD_MAX)
                std = torch.exp(log_std) + EPS                         # (B,3)
                
                steer = action[:, 0:1].clamp(-1+EPS,1-EPS)
                gas = action[:, 1:2].clamp(EPS, 1-EPS)
                brake = action[:, 2:3].clamp(EPS, 1-EPS)
                
                # atanh for [-1,1], logit for [0,1]
                steer_u = 0.5 * torch.log((1+steer)/(1-steer))
                gas_u = torch.log(gas/(1-gas))
                brake_u = torch.log(brake/(1-brake))
                
                action_u = torch.cat([steer_u, gas_u, brake_u], dim=1)  # (B,3)
                
                dist = torch.distributions.Normal(mu,std)
                loss = (-dist.log_prob(action_u)).mean()

                loss.backward()
                optimizer.step()
                training_losses.append(loss.item())

            mean_training_loss = np.mean(training_losses)

            # set to eval mode to get the mean validation loss for early stopping
            model.eval()
            val_losses = []
            with torch.no_grad():
                for observation, action, reward in val_loader:
                    observation, action = observation.float().to(device), action.float().to(device)
                    mu, log_std = model.forward_dist(observation)
                    log_std = torch.clamp(log_std, model.LOG_STD_MIN, model.LOG_STD_MAX)
                    std = torch.exp(log_std) + EPS
                    
                    steer = action[:, 0:1].clamp(-1+EPS,1-EPS)
                    gas = action[:, 1:2].clamp(EPS, 1-EPS)
                    brake = action[:, 2:3].clamp(EPS, 1-EPS)
                    
                    # atanh for [-1,1], logit for [0,1]
                    steer_u = 0.5 * torch.log((1+steer)/(1-steer))
                    gas_u = torch.log(gas/(1-gas))
                    brake_u = torch.log(brake/(1-brake))
                    
                    action_u = torch.cat([steer_u, gas_u, brake_u], dim=1)  # (B,3)
                    
                    dist = torch.distributions.Normal(mu,std)
                    val_loss = (-dist.log_prob(action_u)).mean().item()
                    val_losses.append(val_loss)
            mean_val_loss = np.mean(val_losses)

            print(f"epoch: {epoch}/{num_epochs}, training loss: {mean_training_loss}, Val loss: {mean_val_loss}")
            writer.add_scalar('NLL/train', mean_training_loss, epoch)
            writer.add_scalar('NLL/validation', mean_val_loss, epoch)

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model = model

            early_stopping(mean_val_loss)
            if early_stopping.early_stop:
                break
        
        # model.load_state_dict(best_model.state_dict())
        torch.save(best_model.state_dict(), f"{MODEL_DIR}/BC_P_{p}_SEED_{seed}.pt")
        writer.close()