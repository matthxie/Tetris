import numpy as np
import torch
from utils.sum_tree import SumTree


class ReplayMemory:
    def __init__(self, env, min_size, max_size, alpha, beta, beta_increment, epsilon):
        self.env = env
        self.tree = SumTree(max_size)
        self.alpha = alpha
        self.beta_start = beta
        self.beta_frames = beta_increment
        self.frame = 1
        self.epsilon = 1e-5

        self.init_replay_memory(min_size, max_size)

    def init_replay_memory(self, min_size, max_size):
        for _ in range(min_size):
            valid_moves_mask = torch.tensor(self.env.get_invalid_moves()).unsqueeze(0)

            action = np.random.randint(0, 40)
            while valid_moves_mask[0, action] == False:
                action = np.random.randint(0, 40)

            new_state, reward, done, info = self.env.step(
                action % 10, int(action / 10), probe=False
            )
            transition = (
                state,
                action,
                reward,
                done,
                new_state,
                valid_moves_mask.to("cpu"),
            )
            self.replay_memory.append(transition)

            state = new_state

            if done == 1:
                state = self.env.reset()

    def add(self, state, action, reward, done, new_state, valid_moves_mask):
        max_priority = (
            np.max(self.tree.tree[-self.tree.capacity :]) if self.tree.size > 0 else 1.0
        )
        self.tree.add(
            max_priority, (state, action, reward, done, new_state, valid_moves_mask)
        )

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        beta = min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, data_idx = self.tree.sample(s)
            priority, data = self.tree.get(idx)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        self.frame += 1
        probs = np.array(priorities) / self.tree.total_priority()
        weights = (self.tree.size * probs) ** -beta
        weights /= weights.max()

        states, actions, rewards, dones, next_states, valid_moves = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(valid_moves), dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            idxs,
        )

    def update_priorities(self, idxs, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)
