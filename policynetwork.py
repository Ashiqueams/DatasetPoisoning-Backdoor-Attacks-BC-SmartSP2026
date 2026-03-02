import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions import Normal
import h5py

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
    # def forward(self, x):
    #     mu, _ = self.forward_dist(x)
        
    #     steer = self.tanh(mu[:, 0:1])
    #     gas = self.sigmoid(mu[:, 1:2])
    #     brake = self.sigmoid(mu[:, 2:3])
        
    #     x = torch.cat([steer, gas, brake], dim=1)
    #     return x

    # def forward_dist(self, x):
    #     x = self.pool(self.relu(self.conv1(x)))
    #     x = self.pool(self.relu(self.conv2(x)))
    #     x = self.pool(self.relu(self.conv3(x)))
    #     x = self.flatten(x)
    #     x = self.relu(self.fc1(x))
    #     x = self.relu(self.fc2(x))
        
    #     mu = self.fc_mu(x)
    #     log_std = self.fc_log_std(x)
    #     log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
    #     return mu, log_std
    
    # We compute entropy from the latent Gaussian Normal(μ, σ); this is a proxy and 
    # does not equal the entropy of the squashed action distribution after tanh/sigmoid.
    # @torch.no_grad()    
    # def sample_action_and_entropy(self, x, eps=1e-6):
    #     mu, log_std = self.forward_dist(x)
    #     std = torch.exp(log_std)+eps
        
    #     dist = Normal(mu,std)
    #     sample = dist.sample()
        
    #     steer = torch.tanh(sample[:, 0:1])
    #     gas = torch.sigmoid(sample[:, 1:2])
    #     brake = torch.sigmoid(sample[:,2:3])
        
    #     action = torch.cat([steer, gas, brake], dim=1)
        
    #     ent = dist.entropy().sum(dim=1)
    #     return action, ent
    
    # def action_probabilties(self, observations, device=None):
    #     # self.cuda().eval() changed for my mac 
    #     if device is None:
    #         if torch.backends.mps.is_available():
    #             device = torch.device("mps")
    #         elif torch.cuda.is_available():
    #             device = torch.device("cuda")
    #         else:
    #             device = torch.device("cpu")
        
    #     self.to(device).eval()

    #     observation = torch.from_numpy(np.transpose(np.array(observations) / 255, (0, 3, 1, 2))).float().to(device)
    #     # print(f"Shape of observation = {observation.shape} inside action prob function")
    #     return self.__call__(observation).detach().cpu().numpy()

# class DemonstrationDataset(Dataset):
#     def __init__(self, file_path):
#         self.file_path = file_path
#         with h5py.File(file_path, 'r') as f:
#             self.length = len(f['observations'])
#         self.data = None

#     def __len__(self):
#         return self.length
# def __getitem__(self, idx):
        
        # -------------Stacked PART-----------------------__-----------
        # if self.data is None:
        #     self.data = h5py.File(self.file_path, 'r')
        #     self.observations = self.data['observations']
        #     self.actions = self.data['actions']
        #     self.rewards = self.data['rewards']
        # obs = self.observations[idx]
        # observation = torch.from_numpy(obs).float().div_(255.0)

        # action = torch.from_numpy(self.actions[idx]).float()
        # # reward = torch.from_numpy(self.rewards[idx]).float()
        # reward = torch.as_tensor(self.rewards[idx], dtype=torch.float32)

        # return observation, action, reward
        
        
    # def predict(self, observations, device=None, **kwargs):
    #     if device is None:
    #         if torch.backends.mps.is_available():
    #             device = torch.device("mps")
    #         elif torch.cuda.is_available():
    #             device = torch.device("cuda")
    #         else:
    #             device = torch.device("cpu")
    #     self.to(device).eval()
    #     obs_array = np.array(observations) # Likely (4, 96, 96, 3)
        
    #     # Ensure we have a batch dimension
    #     if obs_array.ndim == 4: # (4, 96, 96, 3)
    #         obs_array = obs_array[np.newaxis, ...] # (1, 4, 96, 96, 3)
            
    #     observation = torch.from_numpy(obs_array / 255.0).float().to(device)
        
    #     # Convert (B, 4, 96, 96, 3) -> (B, 12, 96, 96)
    #     observation = observation.permute(0, 1, 4, 2, 3).reshape(-1, 12, 96, 96)
        
    #     with torch.no_grad():
    #         action = self.forward(observation)
    #     return action.detach().cpu().numpy(), []
    
        # -------------Stacked PART----------------------------------