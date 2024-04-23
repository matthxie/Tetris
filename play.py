import torch
import numpy as np
from alt_dqn import AltDeepQNetwork
from envs import alt_tetris
from envs import tetris

LEARNING_RATE = 5e-4
NUM_ACTIONS = 1

env = alt_tetris.TetrisEnv()
# env = tetris.TetrisEnv()
policy_net = AltDeepQNetwork(NUM_ACTIONS, env)
policy_net.load_state_dict(torch.load("policy_weights.pth",  map_location=torch.device('cpu')))

def act(obs):
    obs = torch.as_tensor(obs, dtype=torch.float32)

    obs_ = []
    actions = []
    bounds = env.get_movement_bounds()

    for r in range(len(bounds)):
      for x in range(bounds[r]+1):
        new_obs, reward, done, info = env.step(x, r, probe=True, display=False)
        obs_.append(new_obs)
        actions.append(r*10 + x)

    input = torch.as_tensor(np.array([t for t in obs_]), dtype=torch.float32)

    q_values = policy_net(input)

    max_q_index = torch.argmax(q_values)
    action = max_q_index.detach().item()

    print("x: ", actions[action]%10, "r: ", int(actions[action]/10))

    return actions[action]

obs = env.reset()

# new_obs, reward, done, info = env.step(2, 2, True)


total_reward = 0
total_lines_cleared = 0

for i in range(1000):
    # print(env.board)

    action = act(obs)

    r = int(action/10)
    x = action%10
    new_obs, reward, done, lines_cleared  = env.step(x, r, probe=False, display=True)
    total_reward += reward - 1
    total_lines_cleared += lines_cleared

    if done == 1:
      print("lines cleared: ", total_lines_cleared)
      print("reward: ", total_reward)
      print("------------------------------------------------------")
      new_obs = env.reset()
      total_reward = 0
      total_lines_cleared = 0

      break

    obs = new_obs