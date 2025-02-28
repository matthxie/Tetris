import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
from dqn import DeepQNetwork
from envs.alt_tetris import TetrisEnv
import wandb
from tqdm import tqdm

GAMMA = 0.99
BATCH_SIZE = 256
REPLAY_SIZE = 300_000
MIN_REPLAY_SIZE = 50_000
EPSILON_START = 1.00
EPSILON_END = 1e-3
EPSILON_DECAY = 50_000
EPSILON_DECAY_RATE = 0.999
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 0.9
LEARNING_RATE_DECAY_FREQ = 500
NUM_ACTIONS = 40
NUM_EPOCHS = 3_000
NUM_STEPS = 10_000_000
MAX_EPOCH_STEPS = 2000
TAU = 0.005
TARGET_UPDATE_FREQ = 80_000
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
        },
    )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# writer = SummaryWriter()

env = TetrisEnv()

replay_memory = deque(maxlen=REPLAY_SIZE)
reward_memory = deque(maxlen=10)
reward_avg = 0
reward_avg_bench = 2500
episode_reward = 0.0
episode_losses = []

policy_net = DeepQNetwork(NUM_ACTIONS, env).to(device)
target_net = DeepQNetwork(NUM_ACTIONS, env).to(device)

target_net.load_state_dict(policy_net.state_dict())

policy_net.train()
target_net.train()

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
criterion = nn.SmoothL1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

example_state = torch.tensor(np.array([4, 5, 4, 3, 3, 2])).to(device)

state = env.reset()


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * (x - 10)))


def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g["lr"]


def calculate_priority():
    states = torch.as_tensor(
        np.array([m[0] for m in replay_memory]), dtype=torch.float32
    ).to(device)
    rewards = torch.as_tensor(
        np.array([m[2] for m in replay_memory]), dtype=torch.float32
    ).to(device)
    new_states = (
        torch.as_tensor(np.array([m[4] for m in replay_memory]), dtype=torch.float32)
        .squeeze(1)
        .to(device)
    )

    target_net.eval()
    policy_net.eval()
    with torch.no_grad():
        estimates = policy_net(states)
        targets = target_net(new_states)
    policy_net.train()
    target_net.train()

    priorities = np.array(np.abs(estimates - rewards - targets))
    probabilities = priorities / np.sum(priorities)
    replay_memory = np.array(replay_memory)
    replay_memory[:, 5] = probabilities
    replay_memory = deque(map(tuple, replay_memory))


# init replay memory
print("Initializing memory replay: size", MIN_REPLAY_SIZE)

for _ in range(MIN_REPLAY_SIZE):
    valid_moves_mask = torch.tensor(env.get_invalid_moves()).unsqueeze(0)

    action = np.random.randint(0, 40)
    while valid_moves_mask[0, action] == False:
        action = np.random.randint(0, 40)

    new_state, reward, done, info = env.step(action % 10, int(action / 10), probe=False)
    transition = (state, action, reward, done, new_state, valid_moves_mask.to("cpu"))
    replay_memory.append(transition)

    state = new_state

    if done == 1:
        state = env.reset()

# training
epoch = 0
epoch_step = 0
step_count = 0
episode_lines_cleared = 0
epsilon = EPSILON_START
state = env.reset()
progress_bar = tqdm(range(NUM_STEPS), desc="Training Progress")

# training loop
for i in tqdm(range(NUM_STEPS), position=0, leave=True):
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

    new_state, reward, done, lines_cleared = env.step(
        action % 10, int(action / 10), probe=False
    )
    transition = (state, action, reward, done, new_state, valid_moves_mask.to("cpu"))
    replay_memory.append(transition)

    state = new_state
    episode_reward += reward
    episode_lines_cleared += lines_cleared
    step_count += 1

    # update taret network
    if step_count % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # target_net_state_dict = target_net.state_dict()
    # policy_net_state_dict = policy_net.state_dict()
    # for key in policy_net_state_dict:
    #   target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    # target_net.load_state_dict(target_net_state_dict)

    # save model at interval
    if step_count % SAVE_FREQ == 0 and epoch >= SAVE_FREQ:
        torch.save(policy_net.state_dict(), "policy_weights.pth")
        torch.save(target_net.state_dict(), "target_weights.pth")

    # save model passing benchmark
    avg = np.mean(np.array(reward_memory))
    if avg > reward_avg_bench and avg > reward_avg:
        torch.save(policy_net.state_dict(), "optimal_policy_weights.pth")
        torch.save(target_net.state_dict(), "optimal_target_weights.pth")
        reward_avg = avg

    if done == 1:
        epoch += 1
        state = env.reset()
        reward_memory.append(episode_reward)
    else:
        epoch_step += 1
        if epoch_step < MAX_EPOCH_STEPS:
            continue

    # sample from replay memory
    transitions = random.sample(replay_memory, BATCH_SIZE)

    states = torch.as_tensor(
        np.array([t[0] for t in transitions]), dtype=torch.float32
    ).to(device)
    actions = torch.as_tensor(
        np.asarray([t[1] for t in transitions]), dtype=torch.int64
    ).to(device)
    rewards = torch.as_tensor(
        np.asarray([t[2] for t in transitions]), dtype=torch.float32
    ).to(device)
    dones = torch.as_tensor(
        np.asarray([t[3] for t in transitions]), dtype=torch.float32
    ).to(device)
    new_states = torch.as_tensor(
        np.array([t[4] for t in transitions]), dtype=torch.float32
    ).to(device)
    valid_moves = (
        torch.as_tensor(np.array([t[5] for t in transitions]), dtype=torch.bool)
        .squeeze(1)
        .to(device)
    )

    # calculate targets and q-values
    policy_net.eval()
    target_net.eval()

    q_values = policy_net(states)
    with torch.no_grad():
        target_q_values = target_net(new_states)

    q_values = torch.where(valid_moves, q_values, -1e9)
    # target_q_values = torch.where(valid_moves, target_q_values, -1e9)

    policy_net.train()
    target_net.train()

    target_q_values = target_q_values.max(dim=1)[0]
    q_values = q_values.gather(1, actions.unsqueeze(-1)).flatten()

    targets = rewards + (GAMMA * (1 - dones) * (target_q_values))

    # gradient descent
    loss = criterion(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=2.0)
    optimizer.step()
    # scheduler.step(loss)

    # logs
    if WANDB:
        wandb.log(
            {
                "loss": loss.item(),
                "Episode Reward": episode_reward,
                "Episode Lines Cleared": episode_lines_cleared,
                "Learning Rate": get_lr(optimizer),
                "Epsilon": round(epsilon, 3),
            }
        )

    epoch_step = 0
    episode_reward = 0.0
    episode_lines_cleared = 0
    # epsilon = sigmoid(epoch)
    epsilon = np.interp(epoch, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    # epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * epoch / EPSILON_DECAY)

if WANDB:
    wandb.finish()

torch.save(policy_net.state_dict(), "policy_weights.pth")
torch.save(target_net.state_dict(), "target_weights.pth")
