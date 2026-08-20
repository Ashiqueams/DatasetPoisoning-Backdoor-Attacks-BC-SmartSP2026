import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py

class DemonstrationDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.observations = np.array(f['observations'])
            self.actions = np.array(f['actions'], dtype=np.float32)
            self.rewards = np.array(f['rewards'], dtype=np.float32)
        
        assert len(self.observations) == len(self.actions) == len(self.rewards)
    def __len__(self):
        return len(self.observations)
    def __getitem__(self, idx):
        obs = self.observations[idx].astype(np.float32) / 255.0 
        obs = np.transpose(obs, (2,0,1))
        observation = torch.as_tensor(obs, dtype=torch.float32)
        action = torch.as_tensor(self.actions[idx], dtype=torch.float32)
        reward = torch.as_tensor(self.rewards[idx], dtype=torch.float32)
        return observation, action, reward
    
class policyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
         