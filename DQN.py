import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
  def __init__(self, lr, input_dim, output_dim):
    super().__init__()
    self.input_dim = input_dim
    self.n_actions = output_dim

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
    
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(36, 128)
    self.fc2 = nn.Linear(128, 512)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(512, output_dim)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
    temp_nexts = x[:, 12:]
    keep = x[:, :11]
    nexts = []

    for j in range(len(x)):
      arr = []

      indicator_ = [0,0,0,0]
      indicator_[int(x[j][11].item())] = 1
      arr.extend(indicator_)

      for i in range(3):
        indicator = [0,0,0,0,0,0,0]

        if temp_nexts[j][i] != 0:
          indicator[int(temp_nexts[j][i].item())-1] = 1
          arr.extend(indicator)
        else:
          print(x[j])
      nexts.append(arr)

    nexts = torch.tensor(np.array(nexts, dtype=np.float32))

    input = torch.cat((keep, nexts), dim=1)

    output = self.fc1(input)
    output = F.relu(output)
    output = self.fc2(output)
    output = F.relu(output)
    output = self.fc4(output)

    return output
  
  def act(self, obs):
    obs = torch.as_tensor(obs, dtype=torch.float32)
    q_values = self(obs.unsqueeze(0))

    max_q_index = torch.argmax(q_values)
    action = max_q_index.detach().item()

    return action