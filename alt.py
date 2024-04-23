import torch
import torch.nn as nn
from collections import deque
import itertools
import random
import matplotlib.pyplot as plt
from alt_dqn import AltDeepQNetwork
import numpy as np
from envs import alt_tetris

GAMMA = 0.99
BATCH_SIZE = 512
REPLAY_SIZE = 30_000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.00
EPSILON_END = 1e-3
EPSILON_DECAY = 2000
LEARNING_RATE = 1e-3
TARGET_UPDATE_FREQ = 20
NUM_ACTIONS = 1
NUM_EPOCHS = 3000
MAX_EPOCH_STEPS = 3000

env = alt_tetris.TetrisEnv()

replay_memory = deque(maxlen=REPLAY_SIZE)
reward_memory = deque([0,0], maxlen=100)
episode_reward = 0.0

policy_net = AltDeepQNetwork(NUM_ACTIONS, env)
target_net = AltDeepQNetwork(NUM_ACTIONS, env)

target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

loss_history = []
reward_history = []
q_value_history = []

example_obs = torch.tensor(np.array([4,5,4,3,3,2]))

obs = env.reset()

def feed_batch(batch, net):
  output = []

  for i in range(BATCH_SIZE):
    output.append(net(batch[i]))

  return torch.stack(output, dim=0)

# init replay memory
for _ in range(MIN_REPLAY_SIZE):
  r = np.random.randint(0, 4)
  x = np.random.randint(0, env.get_movement_bounds()[r])
  action = r*10 + x

  new_obs, reward, done, info = env.step(x, r, False)
  transition = (obs, action, reward, done, new_obs)
  replay_memory.append(transition)
  reward_memory.append(transition[2])

  obs = new_obs

  if done == 1:
    obs = env.reset()
  
epoch = 0
epoch_step = 0
episode_lines_cleared = 0

# training loop
while(epoch < NUM_EPOCHS):
  epsilon = np.interp(epoch, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

  rand_sample = random.random()
  next_states, actions = env.get_next_states()

  if rand_sample <= epsilon:
    index = np.random.randint(0, len(actions))
  else:
    input = torch.as_tensor(np.array([t for t in next_states]), dtype=torch.float32)
    with torch.no_grad():
      q_values = policy_net(input)
    index = torch.argmax(q_values).item()

  next_state = next_states[index]
  action = actions[index]

  new_obs, reward, done, lines_cleared = env.step(action%10, int(action/10), probe=False)
  transition = (obs, action, reward, done, next_state)
  replay_memory.append(transition)
  reward_history.append(reward)

  obs = new_obs
  episode_reward += reward
  episode_lines_cleared += lines_cleared
  
  if done == 1:
    epoch += 1
    obs = env.reset()

    print()
    print("Epoch: ", epoch)
    print("Reward: ", episode_reward)
    print("Lines cleared: ", episode_lines_cleared)
    print("Epsilon: ", epsilon)
    
    reward_memory.append(episode_reward)
    episode_reward = 0.0
    episode_lines_cleared = 0
    epoch_step = 0

  else:
    epoch_step += 1
    if epoch_step < MAX_EPOCH_STEPS:
      continue

  # start gradient step
  transitions = random.sample(replay_memory, BATCH_SIZE)

  obses = torch.as_tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
  actions = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
  rewards = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
  dones = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
  new_obses = torch.as_tensor(np.array([t[4] for t in transitions]), dtype=torch.float32)

  q_values = policy_net(obses)
  with torch.no_grad():
    target_q_values = policy_net(new_obses)
  targets = rewards +  (GAMMA * (1 - dones) * (target_q_values))

  # gradient descent
  optimizer.zero_grad()
  loss = loss_function(q_values, targets)
  loss.backward()
  # for param in policy_net.parameters():
  #   if param.grad is not None:
  #     param.grad.data.clamp_(-1, 1)
  optimizer.step()

  loss_history.append(loss.item())
  print("Loss: ", loss.item())

  # if loss.item() > 50:
  #   print()
  #   print(q_values)
  #   print("done: ", dones)
  #   print(targets)
  #   print()

  # #update taret network
  # if epoch % TARGET_UPDATE_FREQ == 0:
  #   target_net.load_state_dict(policy_net.state_dict())

torch.save(policy_net.state_dict(), "policy_weights.pth")
torch.save(target_net.state_dict(), "target_weights.pth")

# plt.plot(q_value_history)
# plt.savefig('q_value_history.png')
plt.plot(loss_history)
plt.savefig('loss_history.png')
plt.plot(reward_history)
plt.savefig('reward_history.png')