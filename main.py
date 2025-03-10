import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
from dqn import DeepQNetwork
from tetris import TetrisEnv
import wandb
from tqdm import tqdm
from replay_memory import PrioritizedReplayMemory

GAMMA = 0.99
BATCH_SIZE = 512
REPLAY_SIZE = 20_000
MIN_REPLAY_SIZE = 80_000
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_FREQ = 500
NUM_ACTIONS = 40
NUM_STEPS = 1_500_000
MAX_EPOCH_STEPS = 2000
EPSILON_START = 1.00
EPSILON_END = 1e-3
EPSILON_DECAY = 0.65 * NUM_STEPS
TARGET_UPDATE_FREQ = 20_000
SAVE_FREQ = 80_000
WANDB = True

if WANDB:
    wandb.init(
        project="Tetris CNN",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_STEPS,
            "batch_size": BATCH_SIZE,
            "replay_size": REPLAY_SIZE,
            "update_freq": TARGET_UPDATE_FREQ,
        },
    )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = TetrisEnv()

buffer = PrioritizedReplayMemory(REPLAY_SIZE)
replay_memory = deque(maxlen=REPLAY_SIZE)

policy_net = DeepQNetwork(NUM_ACTIONS, env).to(device)
target_net = DeepQNetwork(NUM_ACTIONS, env).to(device)

target_net.load_state_dict(policy_net.state_dict())

policy_net.train()
target_net.train()

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
criterion = nn.SmoothL1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)


def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g["lr"]


# init replay memory
print("Initializing memory replay")
buffer.init_replay_memory(env, MIN_REPLAY_SIZE)
print("Starting training")

# training
epoch = 0
epoch_step = 0
step_count = 0
episode_reward = 0.0
episode_lines_cleared = 0
episode_num_holes = 0
episode_bumpiness = 0
epsilon = EPSILON_START
state = env.reset()
progress_bar = tqdm(range(NUM_STEPS), desc="Training Progress")

# training loop
for i in tqdm(range(NUM_STEPS + 800_000), position=0, leave=True):
    rand_sample = random.random()
    valid_moves_mask = torch.tensor(env.get_invalid_moves()).unsqueeze(0).to(device)

    if rand_sample <= epsilon:
        action = np.random.randint(0, 40)
        while valid_moves_mask[0, action] == False:
            action = np.random.randint(0, 40)
    else:
        state, result_info, next_stones = env.get_state()
        input = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
        input = input.to(device)

        policy_net.eval()
        with torch.no_grad():
            q_values = policy_net(input)
            q_values = torch.where(valid_moves_mask, q_values, -1e9)
        policy_net.train()

        action = torch.argmax(q_values).item()

    new_state, reward, done, lines_cleared, num_holes, delta_bumpiness = env.step(
        action % 10, int(action / 10), probe=False
    )
    buffer.add(state, action, reward, done, new_state, valid_moves_mask.to("cpu"))

    state = new_state
    episode_reward += reward
    episode_lines_cleared += lines_cleared
    episode_num_holes += num_holes
    episode_bumpiness += delta_bumpiness
    step_count += 1

    # epsilon decay
    epsilon = np.interp(step_count, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    # update taret network
    if step_count % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # save model at interval
    if step_count % SAVE_FREQ == 0 and epoch >= SAVE_FREQ:
        torch.save(policy_net.state_dict(), "policy_weights.pth")
        torch.save(target_net.state_dict(), "target_weights.pth")

    if done == 1:
        epoch += 1
        state = env.reset()
    else:
        epoch_step += 1
        if epoch_step < MAX_EPOCH_STEPS:
            continue

    # sample from replay memory
    states, actions, rewards, dones, new_states, valid_moves, weights, idxs = (
        buffer.sample(device, BATCH_SIZE)
    )

    # calculate targets and q-values
    policy_net.eval()
    target_net.eval()

    q_values = policy_net(states)
    with torch.no_grad():
        target_q_values = target_net(new_states)

    q_values = torch.where(valid_moves, q_values, -1e9)

    policy_net.train()
    target_net.train()

    target_q_values = target_q_values.max(dim=1)[0]
    q_values = q_values.gather(1, actions.unsqueeze(-1)).flatten()
    td_errors = rewards + 0.99 * target_q_values - q_values
    targets = rewards + (GAMMA * (1 - dones) * (target_q_values))

    # gradient descent
    loss = criterion(q_values, targets)
    loss = (weights * loss).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=2.0)
    optimizer.step()

    # update PER priorities
    buffer.update_priorities(idxs, td_errors.abs().detach().cpu().numpy())

    # logs
    if WANDB:
        wandb.log(
            {
                "loss": loss.item(),
                "Episode Reward": episode_reward,
                "Episode Lines Cleared": episode_lines_cleared,
                "Episode Num Holes": episode_num_holes,
                "Episode Bumpiness": episode_bumpiness,
                "Learning Rate": get_lr(optimizer),
                "Epsilon": round(epsilon, 3),
            }
        )

    epoch_step = 0
    episode_reward = 0.0
    episode_lines_cleared = 0
    episode_num_holes = 0
    episode_bumpiness = 0

if WANDB:
    wandb.finish()

torch.save(policy_net.state_dict(), "policy_weights.pth")
torch.save(target_net.state_dict(), "target_weights.pth")
