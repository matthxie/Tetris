import gym
from DQN import DeepQNetwork
import numpy as np
from gym.envs.registration import register
from envs import tetris

# tetris.register()

env = gym.make("Tetris-v0")
agent = DeepQNetwork.Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=(20,10),
                           batch_size=64, n_actions=40, eps_end=0.01)
scores, eps_history = [], []
n_games = 500
n_steps = 10

for i in range(n_games):
  score = 0
  done = False
  observation = env.reset()

  while not done:
    action = agent.choose_action(observation)

    observation_, reward, done, info = env.step(action)

    score += reward
    agent.store_transition(observation, action, reward, observation_, done)
 
    agent.learn()
    observation = observation_
    n_steps -= 1

    if n_steps <= 0:
      n_steps = 10
      agent.transfer_weights()

  scores.append(score)
  eps_history.append(agent.epsilon)

  avg_score = np.mean(scores[-100:])

  print('episode ', i, 'score %.2f ' % score,
        'average score %.2f ' % avg_score,
        'epsilon %.2f ' % agent.epsilon)
