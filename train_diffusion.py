import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from policynetwork import DiffusionPolicyNetwork, DiffusionDemonstrationDataset
from diffusers.training_utils import EMAModel
from earlystopping import EarlyStopping
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_level", type=int, default=0)
    args = parser.parse_args()

    seeds  = [0, 1, 2, 3, 4]
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    RUN_TAG    = "run29_DP_50demos"
    PATCH_TYPE = "red"
    MODEL_DIR  = f"../models/BC_{PATCH_TYPE}1_cameraready_{RUN_TAG}"
    DATA_DIR   = f"../data/final_{PATCH_TYPE}_seed1_FILTERED_50"
    os.makedirs(MODEL_DIR, exist_ok=True)

    num_workers  = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
    obs_horizon  = 2
    pred_horizon = 16

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Training seed {seed}")
        print(f"{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for p in [args.poison_level]:
            print(f"Poison level: {p}")

            model = DiffusionPolicyNetwork(
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-6
            )

            ema = EMAModel(
                parameters=model.parameters(),
                power=0.75
            )

            writer = SummaryWriter(
                log_dir=f"../runs/diffusion/{PATCH_TYPE}/{RUN_TAG}/p_{p}/seed_{seed}"
            )

            full_data = DiffusionDemonstrationDataset(
                f"{DATA_DIR}/P_{p}_SEED_0_DEMOS_50.h5",
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon
            )
            print(f"Dataset size: {len(full_data)} sequences")

            val_size   = int(len(full_data) * 0.10)
            train_size = len(full_data) - val_size
            training_data, val_data = torch.utils.data.random_split(
                full_data, [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )

            train_loader = DataLoader(
                training_data,
                batch_size=256,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_data,
                batch_size=256,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            num_epochs   = 500
            best_loss    = float('inf')
            save_path    = f"{MODEL_DIR}/BC_P_{p}_SEED_{seed}.pt"
            early_stopping = EarlyStopping(patience=50, verbose=True, path=save_path)

            for epoch in range(num_epochs):

                # ── Train ──
                model.train()
                train_losses = []
                for obs_seq, act_seq in train_loader:
                    obs_seq = obs_seq.to(device)
                    act_seq = act_seq.to(device)

                    optimizer.zero_grad()
                    loss = model.loss(obs_seq, act_seq)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    ema.step(model.parameters())
                    train_losses.append(loss.item())

                # ── Validate ──
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for obs_seq, act_seq in val_loader:
                        obs_seq = obs_seq.to(device)
                        act_seq = act_seq.to(device)
                        val_losses.append(model.loss(obs_seq, act_seq).item())

                mean_train = np.mean(train_losses)
                mean_val   = np.mean(val_losses)

                print(f"epoch {epoch}/{num_epochs} | "
                      f"train={mean_train:.4f} | val={mean_val:.4f}")

                writer.add_scalar('Diffusion/train', mean_train, epoch)
                writer.add_scalar('Diffusion/val',   mean_val,   epoch)

                # ── Save best + early stopping ──
                if mean_val < best_loss:
                    best_loss  = mean_val
                    best_model = model
                    torch.save(best_model.state_dict(), save_path)

                early_stopping(mean_val)
                if early_stopping.early_stop:
                    break

            writer.close()
            print(f"Done seed={seed}, P={p}")

if __name__ == '__main__':
    main()