import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from policynetwork import PolicyNetwork, DemonstrationDataset
from earlystopping import EarlyStopping
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_level", type=int, default=0)
    args = parser.parse_args()
    seeds = [0, 1, 2, 3, 4]
    increments = 5
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    # device
    RUN_TAG  = "run20"
    PATCH_TYPE = "red"  # or "gaussian"
    MODEL_DIR = f"../models/BC_{PATCH_TYPE}1_cameraready_{RUN_TAG}"
    DATA_DIR = f"../data/final_{PATCH_TYPE}_seed1"
    # DATA_DIR = f"../data/train"
    os.makedirs(MODEL_DIR, exist_ok=True)
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))

    for seed in seeds:
        # WORKING ON TRAINING MODELS WITH SEVERAL SEEDS
        print(f"Working on seed {seed}")
        torch.manual_seed(seed)
        # random.seed(seed)
        np.random.seed(seed)

        for p in [args.poison_level]:
            model = PolicyNetwork().to(device)
            loss_fn = torch.nn.MSELoss()
            loss_weights = torch.tensor([1.0, 5.0, 1.0]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            writer = SummaryWriter(log_dir=f"../runs/behavioural_cloning/{PATCH_TYPE}/{RUN_TAG}/p_{p}/seed_{seed}")

            # full_data = DemonstrationDataset(f"{DATA_DIR}/P_0_SEED_0_DEMOS_400.h5")
            full_data = DemonstrationDataset(f"{DATA_DIR}/P_{args.poison_level}_SEED_0_DEMOS_400.h5")

            # setting aside 10% of data randomly for validation
            val_p = 0.10
            # val_size = 1000
            val_size = int(len(full_data) * val_p)
            train_size = len(full_data) - val_size

            training_data, val_data = torch.utils.data.random_split(
                full_data, [train_size, val_size],
                generator = torch.Generator().manual_seed(seed)
            )

            train_loader = DataLoader(
                training_data, 
                batch_size=512, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_data, 
                batch_size=512, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            # 10 seems to be the sweet point for patience with the min_delta 1e-5
            early_stopping = EarlyStopping(min_delta=1e-4, patience=10)

            best_loss, best_model = float('inf'), None
            # setting epoch to a high number, it will usually not even go to 60 due to early stopping preventing overfitting 
            num_epochs = 60

            for epoch in range(num_epochs):
                # set to training mode and do the regular training steps
                model.train()
                training_losses = []
                for observation, action, reward in train_loader:
                    # print(f"Original observation shape from loader: {observation.shape}")
                    # observation = observation.to(device).float().permute(0, 3, 1, 2) / 255.0.
                    # print(f"Observation shape after permute: {observation.shape}")
                    
                    
                    observation = observation.to(device)
                    action = action.to(device).float()
                    optimizer.zero_grad()
                    
                    pred_action, _ = model.predict_tensor(observation)
                    loss = loss_fn(pred_action, action)
                    
                    # mu_raw, std = model(observation)
                    # loss = -model.log_prob(mu_raw, std, action).mean()
                    # mu, std = model(observation)
                    # dist = torch.distributions.Normal(mu, std)
                    # loss = -dist.log_prob(action).sum(dim=-1).mean()
                    # predicted squashed action [steer, gas, brake]
                    # loss = (loss_weights * (pred_action - action)**2).mean()
                    
                    loss.backward()
                    optimizer.step()
                    training_losses.append(loss.item())

                mean_training_loss = np.mean(training_losses)

                # set to eval mode to get the mean validation loss for early stopping
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for observation, action, reward in val_loader:
                        # observation = observation.to(device).float().permute(0, 3, 1, 2) / 255.0
                        
                        observation = observation.to(device)
                        action = action.to(device).float()
                        # Predicted action from the new forward()
                        pred_action, _ = model.predict_tensor(observation)
                        val_loss = loss_fn(pred_action, action).item()
                        
                        # mu, std = model(observation)
                        # dist = torch.distributions.Normal(mu, std)
                        # val_loss = -dist.log_prob(action).sum(dim=-1).mean()
                        # mu_raw, std = model(observation)
                        # val_loss = -model.log_prob(mu_raw, std, action).mean()
                        val_losses.append(val_loss)
                mean_val_loss = np.mean(val_losses)

                print(f"epoch: {epoch}/{num_epochs}, training loss: {mean_training_loss}, Val loss: {mean_val_loss}")
                writer.add_scalar('NLL/train', mean_training_loss, epoch)
                writer.add_scalar('NLL/validation', mean_val_loss, epoch)

                if mean_val_loss < best_loss:
                    best_loss = mean_val_loss
                    best_model = model
                    torch.save(best_model.state_dict(), f"{MODEL_DIR}/BC_P_{p}_SEED_{seed}.pt")

                early_stopping(mean_val_loss)
                if early_stopping.early_stop:
                    break
            
            # model.load_state_dict(best_model.state_dict())
            # torch.save(best_model.state_dict(), f"{MODEL_DIR}/BC_P_{p}_SEED_{seed}.pt")
            writer.close()

if __name__ == '__main__':
    main()