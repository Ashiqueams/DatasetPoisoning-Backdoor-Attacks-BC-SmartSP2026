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
    
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*12*12, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=3)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        raw = self.fc3(x)
        
        steer = self.tanh(raw[:,0:1])
        gas = self.sigmoid(raw[:,1:2])
        brake = self.sigmoid(raw[:,2:3])
        
        return torch.cat([steer, gas, brake], dim=1)
    
    def predict(self, observations, device=None):
        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.to(device).eval()
        
        obs_array = np.array(observations)
        
        if obs_array.ndim == 3:
            obs_array = obs_array[np.newaxis, ...]
        
        obs_tensor = torch.from_numpy(obs_array).float().to(device) / 255.0
        obs_tensor = obs_tensor.permute(0,3,1,2)
        
        with torch.no_grad():
            action = self.forward(obs_tensor)
            
        return action.cpu().numpy(), []
        
    
         