import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py

class DemonstrationDataset(Dataset):
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')
        self.observations = self.data['observations']
        self.actions = self.data['actions']
        self.rewards = self.actions#self.data['rewards']
        assert len(self.observations) == len(self.actions) == len(self.rewards)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = np.transpose(self.observations[idx]/255, (2, 0, 1))
        action = F.one_hot(torch.tensor(self.actions[idx]), num_classes=5)
        reward = self.rewards[idx]
        return observation, action, reward

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=64*12*12, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=5)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

    def action_probabilties(self, observations):
        self.cuda().eval()

        observation = torch.from_numpy(np.transpose(np.array(observations) / 255, (0, 3, 1, 2))).float().cuda()
        # print(f"Shape of observation = {observation.shape} inside action prob function")
        return self.__call__(observation).detach().cpu().numpy()

    def predict(self, observations, **kwargs):
        self.cuda().eval()
        observation = torch.from_numpy(np.transpose(np.array(observations) / 255, (0, 3, 1, 2))).float().cuda()
        # print(f"Shape of observation = {observation.shape} inside Predict action function")
        return self.__call__(observation).argmax(dim=1).cpu().numpy(), []
