import torch
import torch.nn as nn
from collections import deque
import itertools
import random
import gym
from DQN import DeepQNetwork
import numpy as np
from gym.envs.registration import register
from envs import tetris

# tetris.register()

GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 50_000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10_000
LEARNING_RATE = 5e-4
TARGET_UPDATE_FREQ = 1000
NUM_ACTIONS = 40

env = gym.make("Tetris-v0")

replay_memory = deque(maxlen=REPLAY_SIZE)
reward_memory = deque([0,0], maxlen=100)
episode_reward = 0.0

policy_net = DeepQNetwork(lr=LEARNING_RATE, input_dim=(21, 10), output_dim=NUM_ACTIONS)
target_net = DeepQNetwork(lr=LEARNING_RATE, input_dim=(21, 10), output_dim=NUM_ACTIONS)

target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

obs = env.reset()

def feed_batch(batch, net):
  output = []

  for i in range(BATCH_SIZE):
    output.append(net(batch[i]))

  return torch.stack(output, dim=0)

# init replay memory
for _ in range(MIN_REPLAY_SIZE):
  action = env.action_space.sample()

  new_obs, reward, done, info = env.step(action)
  transition = (obs, action, reward, done, new_obs)
  replay_memory.append(transition)
  reward_memory.append(transition[2])

  obs = new_obs

  if done:
    obs = env.reset()

# training loop
for step in itertools.count():
  epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

  rand_sample = random.random()

  if rand_sample <= epsilon:
    action = env.action_space.sample()
  else:
    action = policy_net.act(obs)

  # print("------------------------------------------------")

  new_obs, reward, done, info = env.step(action)
  transition = (obs, action, reward, done, new_obs)
  replay_memory.append(transition)

  # print()
  # print(obs)
  # print(action)
  # print("================================")
  # print(new_obs)
  # print()

  # print("------------------------------------------------")

  obs = new_obs

  episode_reward += reward
  
  if done:
    obs = env.reset()

    reward_memory.append(episode_reward)
    episode_reward = 0.0

  # start gradient step
  transitions = random.sample(replay_memory, BATCH_SIZE)

  obses = torch.as_tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32)
  actions = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
  rewards = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
  dones = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
  new_obses = torch.as_tensor(np.asarray([t[4] for t in transitions]), dtype=torch.float32)

  # target_q_values = feed_batch(new_obses, target_net)
  target_q_values = target_net(new_obses)
  max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

  targets = rewards + GAMMA * (1 - dones) * max_target_q_values

  # compute loss
  # q_values = feed_batch(obses, policy_net)
  q_values = policy_net(new_obses)
  action_q_values = torch.gather(input=q_values, dim=1, index=actions)

  loss = nn.functional.smooth_l1_loss(action_q_values, targets)

  # gradient descent
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # update taret network
  if step % TARGET_UPDATE_FREQ == 0:
    target_net.load_state_dict(policy_net.state_dict())

  # logging
  if step % 1000 == 0:
    print()
    print("Step: ", step)
    print("Average reward: ", np.mean(reward_memory))
    print("Epsilon: ", epsilon)

  if step == 20_000:
    torch.save(policy_net.state_dict(), "policy_weights.pth")
    torch.save(target_net.state_dict(), "target_weights.pth")