import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions import Normal
import h5py
from network import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import collections

class DemonstrationDataset(Dataset):
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')
        self.observations = self.data['observations']
        self.actions = self.data['actions']
        self.rewards = self.data['rewards'] #self.data['rewards'] why?
        assert len(self.observations) == len(self.actions) == len(self.rewards)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = np.transpose(self.observations[idx].astype(np.float32)/255.0, (2, 0, 1)) 
        observation = torch.as_tensor(observation, dtype=torch.float32)
        
        #action: continuous vector [steer, gas, brake] - float32 tensor shape (3,) 
        action = torch.as_tensor(self.actions[idx], dtype=torch.float32)
        
        # reward - float32 tensor
        reward = torch.as_tensor(self.rewards[idx], dtype=torch.float32)
        # reward = self.rewards[idx]
        return observation, action, reward
        
        
    
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=64*12*12, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        
        # changed from out_features=5 to out_features=3
        # Gaussian policy heads: mean + log_std for each action dimension
        self.fc_mu      = nn.Linear(in_features=256, out_features=3)
        self.fc_log_std = nn.Linear(in_features=256, out_features=3)
        
        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX =  2.0
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # self.softmax = nn.Softmax(dim=1) #good for discrete actions 
        self.tanh = nn.Tanh() # for steering as it might produce negative results [-1,1]
        self.sigmoid = nn.Sigmoid() # for gas and brake as they both might produce [0,1]

    def forward(self, x):
        # Standard feature extraction
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Using existing mu head
        mu_raw = self.fc_mu(x) 
        
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        return mu_raw, std
    
    def predict_tensor(self, x):
        # """For use during training — stays on device, returns tensor."""
        mu_raw, std = self.forward(x)
        steer = torch.tanh(mu_raw[:, 0:1])
        gas   = torch.sigmoid(mu_raw[:, 1:2])
        brake = torch.sigmoid(mu_raw[:, 2:3])
        return torch.cat([steer, gas, brake], dim=1), std
    def predict(self, observations, device=None, **kwargs):
        # self.cuda().eval()
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        self.to(device).eval()
        
        obs_array = np.array(observations)
        
        if obs_array.ndim == 3:
            obs_array = obs_array[np.newaxis, ...]
            
        observation = torch.from_numpy(obs_array / 255.0).float().to(device)
        
        if observation.shape[-1] == 3:
            observation = observation.permute(0, 3, 1, 2)
        with torch.no_grad():
            mu_raw, std = self.forward(observation)
            # # Squashing the outputs to the correct ranges for CarRacing
            steer = torch.tanh(mu_raw[:, 0:1])      # Range [-1, 1]
            gas = torch.sigmoid(mu_raw[:, 1:2])     # Range [0, 1]
            brake = torch.sigmoid(mu_raw[:, 2:3])   # Range [0, 1]
            mu = torch.cat([steer, gas, brake], dim=1)
        return mu.detach().cpu().numpy(), []
    
    def log_prob(self, mu_raw, std, action):
        EPS = 1e-6
        dist = Normal(mu_raw, std)

        # Invert squashing to get pre-squash u
        steer_u = torch.atanh(action[:, 0:1].clamp(-1 + EPS, 1 - EPS))
        gas_u   = torch.logit(action[:, 1:2].clamp(EPS, 1 - EPS))
        brake_u = torch.logit(action[:, 2:3].clamp(EPS, 1 - EPS))
        u = torch.cat([steer_u, gas_u, brake_u], dim=1)

        # Log prob in unbounded space
        log_prob_u = dist.log_prob(u)

        # Jacobian correction
        tanh_corr  = torch.log(1 - action[:, 0:1].pow(2) + EPS)
        gas_corr   = torch.log(action[:, 1:2] + EPS) + torch.log(1 - action[:, 1:2] + EPS)
        brake_corr = torch.log(action[:, 2:3] + EPS) + torch.log(1 - action[:, 2:3] + EPS)
        corrections = torch.cat([tanh_corr, gas_corr, brake_corr], dim=1)

        return (log_prob_u - corrections).sum(dim=-1)  # (B,)
    
class ImplicitPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.relu  = nn.ReLU()
        
        self.fc1 = nn.Linear(64*12*12 + 3, 1024)  # +3 for action (takes action as input)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)  # outputs scalar energy
        
    def forward(self, obs, action):
        x = self.pool(self.relu(self.conv1(obs)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        
        x = torch.cat([x,action], dim=-1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        energy = self.fc3(x)
        
        return energy
    
    # to extract obs features only to avoid repeating computation of CNN
    def encode_obs(self, obs):
        x = self.pool(self.relu(self.conv1(obs)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.flatten(x)
    
    def energy_from_features(self, obs_features, action):
        x = torch.cat([obs_features, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    # Lower Energy -> True Action
    def info_nce_loss(self, obs, true_action, n_negatives=256):
        B = obs.shape[0]
        device = obs.device
        
        #Encoding and flattening obs once for reusing
        obs_features = self.encode_obs(obs)
        # assigining Energy for true action
        Energy_pos = self.energy_from_features(obs_features, true_action)
        
        #sampling neg actions uniformly from action space
        neg_actions = torch.zeros(B, n_negatives, 3, device=device)
        neg_actions[:, :, 0] = torch.rand(B, n_negatives, device=device)*2-1 #steer[-1,1]
        neg_actions[:, :, 1] = torch.rand(B, n_negatives, device=device)  #gas[0,1]
        neg_actions[:, :, 2] = torch.rand(B, n_negatives, device=device) #brake[0,1]
        
        #Expanding obs_features to match negatives from (B, 9216) to (B, n_negatives, 9216)
        obs_features_exp = obs_features.unsqueeze(1).expand(-1, n_negatives, -1) # from (2,9216) to (2,1,9216) to (2,256,9216)
        obs_features_flat = obs_features_exp.reshape(B*n_negatives, -1) #from (2,256, 9216) to (512, 9216) 
        neg_flat = neg_actions.reshape(B*n_negatives, 3) #from (2,256,3) to (512,3)
        
        # Energy for NEGATIVE actions should be high
        Energy_neg = self.energy_from_features(obs_features_flat, neg_flat)
        Energy_neg = Energy_neg.reshape(B, n_negatives)
        
        # InfoNCE: true action should have lowest energy
        # logits = [-E_pos, -E_neg1, -E_neg2, ...]
        # label = 0 (true action is at index 0)
        logits = torch.cat([-Energy_pos, -Energy_neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    
    # Inference: find best action via derivative-free optimization (DFO)
    # Sample many candidates, return the one with lowest energy
    @torch.no_grad()
    def predict(self, observations, device=None, n_samples=16384, n_iter=3):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        self.to(device).eval()
        
        obs_array = np.array(observations)
        if obs_array.ndim == 3:
            obs_array = obs_array[np.newaxis, ...]
        obs_tensor = torch.from_numpy(obs_array/255.0).float().to(device)
        if obs_tensor.shape[-1] ==3:
            obs_tensor = obs_tensor.permute(0,3,1,2)
        
        B = obs_tensor.shape[0]
        obs_features = self.encode_obs(obs_tensor)
        
        #Action bounds
        lo = torch.tensor([-1.0, 0.0, 0.0], device=device).view(1, 1, 3)
        hi = torch.tensor([ 1.0, 1.0, 1.0], device=device).view(1, 1, 3)
        
        # Initial Uniform Sample
        candidates = torch.rand(B, n_samples, 3, device=device)
        candidates = candidates * (hi - lo) + lo
        
        for _ in range(n_iter):
            obs_exp = obs_features.unsqueeze(1).expand(-1,n_samples,-1)
            obs_flat = obs_exp.reshape(B*n_samples, -1)
            cands_flat = candidates.reshape(B*n_samples, 3)
            
            #Compute Energies
            energies = self.energy_from_features(obs_flat, cands_flat)
            energies = energies.reshape(B, n_samples)
            
            # Finding top-k lowest energy candidates
            k = max(n_samples // 4,16)
            _topk_vals, top_idx = torch.topk(energies, k, dim=1, largest=False)
            
            #Gathering best candidates
            best = candidates[torch.arange(B, device=device).unsqueeze(1), top_idx]
            
            # Resampling around best candidates with smaller noise
            noise_scale = 0.1 / (int(_)+1)
            new_samples = best[:, torch.randint(k,(n_samples,), device=device), :]
            
            # print(f"best shape: {best.shape}")
            # print(f"randint output shape: {torch.randint(k, (n_samples,), device=device).shape}")
            # new_samples = best[:, torch.randint(k,(n_samples,), device=device), :]
            # print(f"new_samples shape: {new_samples.shape}")
            # print(f"randn_like shape: {torch.randn_like(new_samples).shape}")
            # print(f"noise_scale: {noise_scale}, type: {type(noise_scale)}")
            
            new_samples = new_samples + torch.randn_like(new_samples) * noise_scale
            new_samples = torch.max(torch.min(new_samples, hi), lo)
            candidates = new_samples
            
        # Final pick
        obs_exp = obs_features.unsqueeze(1).expand(-1,n_samples,-1)
        obs_flat = obs_exp.reshape(B*n_samples, -1)
        cands_flat = candidates.reshape(B*n_samples, 3)
        energies = self.energy_from_features(obs_flat,cands_flat).reshape(B,n_samples)
        
        best_idx = energies.argmin(dim=1)
        best_actions = candidates[torch.arange(B,  device=device), best_idx]
        
        return best_actions.cpu().numpy(), []
    

class DiffusionDemonstrationDataset(torch.utils.data.Dataset):
    """
    Dataset for Diffusion Policy.
    Returns sequences of observations and actions instead of single frames.
    
    obs_horizon=2:   stack last 2 observations as input
    pred_horizon=16: predict 16 future actions at once
    """
    def __init__(self, file_path, obs_horizon=4, pred_horizon=16):
        self.data = h5py.File(file_path, 'r')
        self.obs = self.data['observations']
        self.actions = self.data['actions']
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        
        # Valid indices as need obs_horizon before and pred_horizon after
        self.valid_indices = list(range(obs_horizon-1,len(self.obs)-pred_horizon))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        
        obs_seq = []
        # Collecting obs_horizon observations ending at index i
        for j in range(i-self.obs_horizon+1, i+1):
            obs = self.obs[j].astype(np.float32) / 255.0      # normalizing
            obs = np.transpose(obs, (2, 0, 1))                # HWC -> CHW
            obs_seq.append(obs)
        obs_seq = torch.tensor(np.stack(obs_seq), dtype=torch.float32)   # shape: (obs_horizon, 3, 96, 96)
        
        # Collecting pred_horizon actions starting at index i
        act_seq = torch.tensor(np.array(self.actions[i:i+self.pred_horizon], dtype=np.float32)) # shape: (pred_horizon, 3)
        
        return obs_seq, act_seq

class DiffusionPolicyNetwork(nn.Module):
    """
    Diffusion Policy for CarRacing-v3.
    Architecture:
    - CNN encoder with linear projection to 256-dim (matches blog's K=256)
    - All obs_horizon frames stacked as channels (one CNN pass, not separate)
    - ConditionalUnet1D with global_cond_dim=256
    - DDPMScheduler for training and inference
    Key parameters:
    - obs_horizon=4:    uses last 4 observations (stacked as channels)
    - pred_horizon=16:  predicts 16 future actions
    - action_horizon=8: executes 8 actions before replanning
    """
    def __init__(self,
                 obs_horizon=4,
                 pred_horizon=16,
                 action_horizon=8,
                 action_dim=3,
                 num_diffusion_iters=100):
        super().__init__()

        self.obs_horizon         = obs_horizon
        self.pred_horizon        = pred_horizon
        self.action_horizon      = action_horizon
        self.action_dim          = action_dim
        self.num_diffusion_iters = num_diffusion_iters

        # ------------------------------------------------------------------
        # CNN Encoder
        # All obs_horizon frames stacked as channels → one CNN pass
        # Input: (B, obs_horizon*3, 96, 96) = (B, 12, 96, 96) for obs_horizon=4
        # 96→48→24→12 → 64*12*12 = 9216 → projected to 256
        # ------------------------------------------------------------------
        self.conv1   = nn.Conv2d(3 * obs_horizon, 16, 3, padding=1)  # ← 12 input channels
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.relu    = nn.ReLU()
        # CNN output: 64 * 12 * 12 = 9216
        # Projected to 256 to match blog's K=256
        self.obs_proj = nn.Linear(9216, 256)

        obs_cond_dim = 256   # K=256, matches blog

        # ------------------------------------------------------------------
        # 1D UNet noise prediction network
        # global_cond_dim = 256 (not 256 * obs_horizon — frames already stacked)
        # ------------------------------------------------------------------
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_cond_dim
        )

        # ------------------------------------------------------------------
        # DDPM noise scheduler
        # ------------------------------------------------------------------
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def encode_obs(self, obs_stacked):
        """
        CNN encoder with projection.
        obs_stacked: (B, obs_horizon*3, 96, 96) — all frames stacked as channels
        returns: (B, 256)
        """
        x = self.pool(self.relu(self.conv1(obs_stacked)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)          # (B, 9216)
        return self.obs_proj(x)      # (B, 256)

    def loss(self, obs_seq, act_seq):
        """
        Diffusion training loss.

        obs_seq: (B, obs_horizon, 3, 96, 96)
        act_seq: (B, pred_horizon, 3)
        """
        B      = obs_seq.shape[0]
        device = obs_seq.device

        # Stack all obs_horizon frames as channels → one CNN pass
        # (B, obs_horizon, 3, 96, 96) → (B, obs_horizon*3, 96, 96)
        obs_stacked = obs_seq.view(B, self.obs_horizon * 3, 96, 96)
        obs_cond    = self.encode_obs(obs_stacked)    # (B, 256)

        # Sample random Gaussian noise
        noise = torch.randn_like(act_seq)

        # Sample random diffusion timestep for each item in batch
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # Forward diffusion — add noise to clean actions
        noisy_act = self.noise_scheduler.add_noise(act_seq, noise, timesteps)

        # Predict the noise using UNet
        noise_pred = self.noise_pred_net(
            noisy_act, timesteps, global_cond=obs_cond
        )

        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict(self, observations, device=None, **kwargs):
        """
        Inference — reverse diffusion to get clean action sequence.

        observations: list of obs_horizon observations, each (96, 96, 3) uint8
        returns: (action_array, [])
        """
        if device is None:
            device = torch.device(
                "mps"  if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.to(device).eval()

        obs_array = np.array(observations)   # (obs_horizon, 96, 96, 3)

        # If single obs — repeat to fill obs_horizon
        if obs_array.ndim == 3:
            obs_array = np.stack([obs_array] * self.obs_horizon, axis=0)

        # Normalize: (obs_horizon, 96, 96, 3) → (obs_horizon, 3, 96, 96)
        obs_tensor = torch.from_numpy(
            obs_array / 255.0
        ).float().to(device)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)   # (obs_horizon, 3, 96, 96)

        # Stack all frames as channels → one CNN pass
        # (obs_horizon, 3, 96, 96) → (1, obs_horizon*3, 96, 96)
        obs_stacked = obs_tensor.reshape(
            1, self.obs_horizon * 3, 96, 96
        )
        obs_cond = self.encode_obs(obs_stacked)        # (1, 256)

        # Start from pure Gaussian noise
        noisy_action = torch.randn(
            (1, self.pred_horizon, self.action_dim), device=device
        )

        # Reverse diffusion
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=noisy_action,
                timestep=k,
                global_cond=obs_cond
            )
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample

        # Extract action_horizon actions
        start      = 0
        end        = self.action_horizon   # = 8
        action_seq = noisy_action[0, start:end, :].cpu().numpy()

        # Clip to CarRacing bounds
        action_seq[:, 0] = np.clip(action_seq[:, 0], -1.0,  1.0)  # steer
        action_seq[:, 1] = np.clip(action_seq[:, 1],  0.0,  1.0)  # gas
        action_seq[:, 2] = np.clip(action_seq[:, 2],  0.0,  1.0)  # brake

        return action_seq, []