import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Sum tree structure
        self.data = [None] * capacity  # Stores experiences
        self.size = 0  # Current size
        self.ptr = 0  # Pointer for overwriting

    def add(self, priority, data):
        """Add experience with priority"""
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data  # Store transition
        self.update(idx, priority)  # Update tree
        self.ptr = (self.ptr + 1) % self.capacity  # Circular buffer
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        """Update priority of a sample"""
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:  # Propagate change up
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def sample(self, s):
        """Retrieve index of sample corresponding to value s"""
        idx = 0
        while idx < self.capacity - 1:  # Traverse tree
            left, right = 2 * idx + 1, 2 * idx + 2
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, data_idx

    def get(self, idx):
        """Return experience and priority"""
        return self.tree[idx], self.data[idx - (self.capacity - 1)]

    def total_priority(self):
        """Return total priority of all experiences"""
        return self.tree[0]  # Root node
