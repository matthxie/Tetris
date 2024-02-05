import torch
import numpy as np
from alt_dqn import AltDeepQNetwork
from envs import alt_tetris

LEARNING_RATE = 5e-4
NUM_ACTIONS = 1

env = alt_tetris.TetrisEnv()
policy_net = AltDeepQNetwork(LEARNING_RATE, (21, 10), NUM_ACTIONS, env)
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

for i in range(20):
    # print(env.board)

    action = act(obs)

    r = int(action/10)
    x = action%10
    new_obs, reward, done, info = env.step(x, r, probe=False, display=True)

    if done:
      print("------------------------------------------------------")
      print(i)
      new_obs = env.reset()

    obs = new_obs