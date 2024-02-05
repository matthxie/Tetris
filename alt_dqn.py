import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class AltDeepQNetwork(nn.Module):
  def __init__(self, lr, input_dim, output_dim, env):
    super().__init__()
    self.env = env
    
    self.input_dim = input_dim
    self.n_actions = output_dim

    self.relu = nn.ReLU()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3,3), stride=1, padding=0)

    self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=6, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
    
    self.pool1 = nn.MaxPool2d(kernel_size=(15, 5))
    self.pool2 = nn.MaxPool2d(kernel_size=(13, 3))

    self.fc1 = nn.Linear(70, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, output_dim)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
    x0 = x[:, :-6]
    x0 = x0.view(x.size(0), 20, 10).unsqueeze(1)
    x1 = x[:, -6:]

    x0_a = self.relu(self.conv1(x0))
    x0_a = self.relu(self.conv2(x0_a))
    x0_a = self.pool1(x0_a)
    x0_a = x0_a.view(x.size(0), -1)

    x0_b = self.relu(self.conv3(x0))
    x0_b = self.relu(self.conv4(x0_b))
    x0_b = self.pool2(x0_b)
    x0_b = x0_a.view(x.size(0), -1)

    x = torch.cat((x0_a, x0_b, x1), dim=1)

    print(x)

    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))

    return x
  
  def act(self, obs):
    obs_array = []
    rewards_array = []
    actions = []
    bounds = self.env.get_movement_bounds()

    for r in range(len(bounds)):
      for x in range(bounds[r]):
        new_obs, reward, done, info = self.env.step(x, r, True)
        obs_array.append(new_obs)
        rewards_array.append(reward)
        actions.append(r*10 + x)

    # obs_array = torch.stack(obs_array, dim=0).unsqueeze(-1)
    input = torch.as_tensor(np.array([t for t in obs_array]), dtype=torch.float32)
    rewards = torch.as_tensor(np.array([t for t in rewards_array]), dtype=torch.float32)

    q_values = self(input)

    max_q_index = torch.argmax(q_values + rewards)
    action = max_q_index.detach().item()

    return actions[action]