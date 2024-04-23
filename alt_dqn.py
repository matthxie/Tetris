import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class AltDeepQNetwork(nn.Module):
  def __init__(self, output_dim, env):
    super().__init__()
    self.env = env
    
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

    self.fc1_ = nn.Sequential(nn.Linear(6, 128), nn.ReLU(inplace=True))
    self.fc2_ = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
    self.fc3_ = nn.Sequential(nn.Linear(128, 1))

    self.create_weights()

  def create_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x0 = x[:, :-6]
    x0 = x0.view(x.size(0), 20, 10).unsqueeze(1)
    x1 = x[:, -6:]

    # x0_a = self.relu(self.conv1(x0))
    # x0_a = self.relu(self.conv2(x0_a))
    # x0_a = self.pool1(x0_a)
    # x0_a = x0_a.view(x.size(0), -1)

    # x0_b = self.relu(self.conv3(x0))
    # x0_b = self.relu(self.conv4(x0_b))
    # x0_b = self.pool2(x0_b)
    # x0_b = x0_a.view(x.size(0), -1)

    # x = torch.cat((x0_a, x0_b, x1), dim=1)

    # x = self.relu(self.fc1(x))
    # x = self.relu(self.fc2(x))
    # x = (self.fc3(x))

    x = self.fc1_(x1)
    x = self.fc2_(x)
    x = self.fc3_(x)

    return x
  
  def act(self, obs):
    obs = torch.as_tensor(obs, dtype=torch.float32)

    obs_ = []
    actions = []
    rewards = []
    bounds = self.env.get_movement_bounds()

    for r in range(len(bounds)):
      for x in range(bounds[r]+1):
        new_obs, reward, done, info = self.env.step(x, r, probe=True, display=False)
        obs_.append(new_obs)
        actions.append(r*10 + x)
        rewards.append(reward)

    input = torch.as_tensor(np.array([t for t in obs_]), dtype=torch.float32)

    q_values = self(input)

    max_q_index = torch.argmax(q_values)
    action = max_q_index.detach().item()

    # new_obs, reward, done, info = self.env.step(actions[action]%10, int(actions[action]/10), probe=True, display=False)

    # x0 = np.array(new_obs[:-6])
    # x0 = x0.reshape(20, 10).tolist()

    # for row in x0:
    #   print(row)
    # print(new_obs[-6:])
    # print()

    # print("x: ", actions[action]%10, "r: ", int(actions[action]/10), ", ", rewards[action])

    return actions[action]%10 + int(actions[action]/10)