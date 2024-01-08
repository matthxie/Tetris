import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
  def __init__(self, lr, input_dim, output_dim):
    super().__init__()
    self.input_dim = input_dim
    self.n_actions = output_dim

    self.conv1 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2,2), stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(1408+28, 128)
    self.fc2 = nn.Linear(128, 512)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(512, output_dim)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
    board = x[:20, :]
    temp_nexts = x[20, :]

    nexts = []
    temp = [0,0,0,0,0,0,0]

    for i in range(4):
      indicator = temp.copy()

      if temp_nexts[i] != 0:
        indicator[int(temp_nexts[i].item())-1] = 1

      nexts.extend(indicator)

    # x = torch.from_numpy(x).float()
    input1 = board.clone().detach()
    input1 = input1.unsqueeze(1)
    input1 = torch.reshape(input1, (20, 10, 1))

    input2 = torch.tensor(nexts).clone().detach()
    # input2 = input2.unsqueeze(1)
    # input2 = torch.reshape(nexts, (10,))

    input1 = self.conv1(input1)
    input1 = F.relu(input1)
    input1 = self.conv2(input1)
    #x = self.pool2(x)
    input1 = F.relu(input1)

    input1 = self.conv4(input1)
    input1 = F.relu(input1)

    input1 = self.conv5(input1)
    input1 = F.relu(input1)

    input1 = torch.flatten(input1)

    input1 = torch.cat((input1, input2))

    input1 = self.fc1(input1)
    input1 = self.fc2(input1)
    input1 = self.fc4(input1)


    return input1
  
  def act(self, obs):
    obs = torch.as_tensor(obs, dtype=torch.float32)
    q_values = self(obs)

    max_q_index = torch.argmax(q_values)
    action = max_q_index.detach().item()

    return action