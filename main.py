import torch
import torch.nn as nn
from collections import deque
import itertools
import random
import matplotlib.pyplot as plt
from DQN import DeepQNetwork
import numpy as np
from envs import tetris

torch.set_default_device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

GAMMA = 0.98
BATCH_SIZE = 512
REPLAY_SIZE = 50_000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.00
EPSILON_END = 0.02
EPSILON_DECAY = 10_000
LEARNING_RATE = 5e-4
TARGET_UPDATE_FREQ = 1000
NUM_ACTIONS = 10
NUM_EPOCHS = 20_000

env = tetris.TetrisEnv()

replay_memory = deque(maxlen=REPLAY_SIZE)
reward_memory = deque([0,0], maxlen=100)
episode_reward = 0.0

policy_net = DeepQNetwork(LEARNING_RATE, NUM_ACTIONS, env).to(device)
target_net = DeepQNetwork(LEARNING_RATE, NUM_ACTIONS, env).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.999, patience=10)

loss_history = []
reward_history = []
q_value_history = []

example_obs = torch.tensor(np.array([4,5,4,3,3,2])).to(device)

obs = env.reset()

def get_lr(optimizer):
  for g in optimizer.param_groups:
      return(g['lr'])

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

  if done:
    obs = env.reset()

  target_net.train()
  policy_net.train()

# training loop
for step in itertools.count():
  if step == NUM_EPOCHS:
    break

  epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

  rand_sample = random.random()

  if rand_sample <= epsilon:
    r = np.random.randint(0, 4)
    x = np.random.randint(0, env.get_movement_bounds()[r])
    action = r*10 + x
  else:
    action = policy_net.act(obs)

  new_obs, reward, done, info = env.step(action%10, int(action/10), probe=False)
  transition = (obs, action, reward, done, new_obs)
  replay_memory.append(transition)
  reward_history.append(reward)

  obs = new_obs

  episode_reward += reward
  
  if done:
    obs = env.reset()

    reward_memory.append(episode_reward)
    episode_reward = 0.0

  # start gradient step
  transitions = random.sample(replay_memory, BATCH_SIZE)

  obses = torch.as_tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
  actions = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
  rewards = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1).to(device)
  dones = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1).to(device)
  new_obses = torch.as_tensor(np.array([t[4] for t in transitions]), dtype=torch.float32).to(device)

  target_q_values = target_net(new_obses)
  max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
  targets = rewards + GAMMA * (1 - dones) * (max_target_q_values)

  # compute loss
  q_values = policy_net(obses)
  action_q_values = torch.gather(input=q_values, dim=1, index=actions)

  # loss = nn.functional.smooth_l1_loss(action_q_values, targets)

  # print(loss)

  # q_value_history.append(policy_net(example_obs).item())

  # gradient descent
  optimizer.zero_grad()
  loss = nn.MSELoss(action_q_values, targets)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
  optimizer.step()

  loss_history.append(loss.item())

  # update taret network
  if step % TARGET_UPDATE_FREQ == 0:
    target_net.load_state_dict(policy_net.state_dict())

  # logging
  if step % 1000 == 0:
    print()
    print("Step: ", step)
    print("Average reward: ", np.mean(reward_memory))
    print("Epsilon: ", epsilon)

  if step == NUM_EPOCHS-1:
    torch.save(policy_net.state_dict(), "policy_weights.pth")
    torch.save(target_net.state_dict(), "target_weights.pth")

# plt.plot(q_value_history)
# plt.savefig('q_value_history.png')
plt.plot(loss_history)
plt.savefig('loss_history.png')
plt.plot(reward_history)
plt.savefig('reward_history.png')