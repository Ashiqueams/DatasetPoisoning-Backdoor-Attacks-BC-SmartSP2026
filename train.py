import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from policynetwork import PolicyNetwork, DemonstrationDataset
from earlystopping import EarlyStopping
import os

increments = 5
# seeds = [0, 1, 2, 3, 4]
seeds = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

os.makedirs('../models/BC_gauss1_cameraready', exist_ok=True)

for seed in seeds:
    # WORKING ON TRAINING MODELS WITH SEVERAL SEEDS
    print(f"Working on seed {seed}")
    torch.manual_seed(seed)
    # random.seed(seed)
    np.random.seed(seed)

    for p in [5]:
        model = PolicyNetwork().to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        writer = SummaryWriter(log_dir=f'../runs/behavioural_cloning/30/p_{p}')

        full_data = DemonstrationDataset(f'../data/final_gauss_seed1/P_{p}_SEED_0_DEMOS_50.h5')

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
                pred_action = model(observation)
                loss = loss_fn(pred_action, action)
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
                    val_losses.append(loss_fn(model(observation), action).item())
            mean_val_loss = np.mean(val_losses)

            print(f"epoch: {epoch}/{num_epochs}, training loss: {mean_training_loss}, Val loss: {mean_val_loss}")
            writer.add_scalar('Loss/train', mean_training_loss, epoch)
            writer.add_scalar('Loss/validation', mean_val_loss, epoch)

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model = model

            early_stopping(mean_val_loss)
            if early_stopping.early_stop:
                break
        
        # model.load_state_dict(best_model.state_dict())
        torch.save(best_model.state_dict(), f'../models/BC_gauss1_cameraready/BC_P_{p}_SEED_{seed}.pt')
        writer.close()