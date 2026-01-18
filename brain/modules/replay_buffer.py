"""
Prioritized Experience Replay Buffer

Implements SumTree-based prioritized experience replay for efficient
TD-error weighted sampling. High-error experiences are sampled more
frequently to accelerate learning.

Reference: Schaul, T. et al. (2015). Prioritized Experience Replay
"""
import torch
import numpy as np
import random
from collections import namedtuple, deque


# === REPLAY BUFFER CONSTANTS ===
# These control the prioritized sampling behavior

# Priority Exponent (α): Controls how much prioritization affects sampling
# α = 0: Uniform sampling (no prioritization)
# α = 1: Full prioritization based on TD-error
DEFAULT_PRIORITY_ALPHA = 0.6

# Importance-Sampling Exponent (β): Corrects for non-uniform sampling bias
# β = 0: No correction
# β = 1: Full correction (should anneal to 1 during training)
DEFAULT_IMPORTANCE_BETA = 0.4

# Small constant to prevent zero priority
PRIORITY_EPSILON = 0.01

# Initial max priority for new experiences
INITIAL_MAX_PRIORITY = 1.0


class SumTree:
    """
    SumTree data structure for Prioritized Experience Replay.
    Leaf nodes store priorities. Internal nodes store sum of children.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Stores transitions (state, action, reward, next_state, done) with priorities.
    """
    def __init__(self, capacity, alpha=DEFAULT_PRIORITY_ALPHA):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.epsilon = PRIORITY_EPSILON  # Small constant to prevent zero priority
        self.max_p = INITIAL_MAX_PRIORITY  # Track max priority in O(1)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Convert to numpy if needed
        if torch.is_tensor(state): state = state.detach().cpu().numpy()
        if torch.is_tensor(next_state): next_state = next_state.detach().cpu().numpy()
        
        # O(1) max priority tracking
        experience = [state, action, reward, next_state, done]
        self.tree.add(self.max_p, experience)

    def store(self, state, action, reward):
        """
        Simplified store for backward compatibility.
        Assumes next_state is same as state and done=False.
        """
        self.add(state, action, reward, state, False)

    def update_last_reward(self, reward):
        """
        Updates the reward of the most recently added experience.
        Useful for delayed rewards in school scripts.
        """
        last_idx = (self.tree.write - 1) % self.capacity
        if self.tree.data[last_idx] is not None:
            # self.tree.data[last_idx] is a list [state, action, reward, next_state, done]
            self.tree.data[last_idx][2] = reward

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences.
        beta: Importance-sampling weight (annealed to 1.0).
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, idxs, errors):
        """Update priorities based on TD-errors."""
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.max_p = max(self.max_p, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
