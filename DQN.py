import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
  def __init__(self, lr, input_dim, output_dim):
    super(DeepQNetwork, self).__init__()
    self.input_dim = input_dim
    self.n_actions = output_dim

    self.conv1 = nn.Conv2d(in_channels=20, out_channels=64, kernel_size=(3,3), stride=2, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(384, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(384+10, output_dim)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, x):
    board = x[:20]
    nexts = x[20]

    # x = T.from_numpy(x).float()
    input1 = T.tensor(board, dtype=T.float32)
    input1 = input1.unsqueeze(1)
    input1 = T.reshape(input1, (20, 10, 1))

    input2 = T.tensor(nexts, dtype=T.float)
    # input2 = input2.unsqueeze(1)
    # input2 = T.reshape(nexts, (10,))

    input1 = self.conv1(input1)
    #x = self.pool1(x)
    input1 = F.relu(input1)
    input1 = self.conv2(input1)
    #x = self.pool2(x)
    input1 = F.relu(input1)

    input1 = T.flatten(input1)

    # print("/nawdfsdfsdfsdfsdlfiuasdfad/n")
    # print(input1.shape, ", ", input2.shape) 

    input1 = T.cat((input1, input2))

    # x = self.fc1(x)
    # x = self.fc2(x)
    # x = self.fc3(x)
    input1 = self.fc4(input1)

    # print(x)

    return input1

  class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100, eps_end=0.01, eps_dec=5e-4):
      self.gamma = gamma
      self.epsilon = epsilon
      self.eps_min = eps_end
      self.eps_dec = eps_dec
      self.lr = lr
      self.action_space = [i for i in range(n_actions)]
      self.mem_size = max_mem_size
      self.batch_size = batch_size
      self.mem_cntr = 0

      self.Q_eval = DeepQNetwork(self.lr, input_dims, n_actions)
      self.Q_target = DeepQNetwork(self.lr, input_dims, n_actions)

      self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
      self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

      self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
      self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
      self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool8)

    def store_transition(self, state, action, reward, state_, done):
      index = self.mem_cntr % self.mem_size
      self.state_memory[index] = state
      self.new_state_memory[index] = state_
      self.reward_memory[index] = reward
      self.action_memory[index] = action
      self.terminal_memory[index] = done

      self.mem_cntr += 1

    def choose_action(self, observation):
      if np.random.random() > self.epsilon:
        state = observation
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()
      else:
        action = np.random.choice(self.action_space)

      return action
    
    def transfer_weights(self):
      self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def learn(self):
      if self.mem_cntr < self.batch_size:
        return

      self.Q_eval.optimizer.zero_grad()

      max_mem = min(self.mem_cntr, self.mem_size)

      batch = np.random.choice(max_mem, self.batch_size, replace=False)

      for i in range(len(batch)):
        batch_index = batch[i]

        q_eval = self.Q_eval.forward(self.state_memory[batch_index])
        q_eval = q_eval[self.action_memory[batch_index]]

        q_next = T.max(self.Q_target.forward(self.new_state_memory[batch_index]))
        q_target = self.reward_memory[batch_index] + (1-self.terminal_memory[batch_index]) * self.gamma * q_next

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                      else self.eps_min
