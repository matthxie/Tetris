import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.set_default_device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class AltDeepQNetwork(nn.Module):
  def __init__(self, output_dim, env):
    super().__init__()
    self.env = env
    
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.cnn_1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(20, 1), stride=1, padding=0),
      nn.ReLU()
    )
    self.cnn_2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=2),
      nn.ReLU()
    )
    self.fc = nn.Sequential(
      nn.Linear(28160, 128),
      nn.ReLU(),
      nn.Linear(128, 512),
      nn.ReLU(),
      nn.Linear(512, output_dim)
    )

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

    # x = self.cnn_1(x0)
    # x = self.cnn_2(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)

    x = self.fc1_(x1)
    x = self.fc2_(x)
    x = self.fc3_(x)

    return x
  
  def act(self, obs):
    print("aowerfhaw;ofhawe;foiuhawefl;uh")
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