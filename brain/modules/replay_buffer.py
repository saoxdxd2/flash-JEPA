"""
Prioritized Experience Replay Buffer (Zero-Copy Optimization)

Implements SumTree-based prioritized experience replay using pre-allocated
Numpy arrays for maximum speed and memory efficiency.

Reference: Schaul, T. et al. (2015). Prioritized Experience Replay
"""
import torch
import numpy as np
import random

# === CONSTANTS ===
DEFAULT_PRIORITY_ALPHA = 0.6
DEFAULT_IMPORTANCE_BETA = 0.4
PRIORITY_EPSILON = 0.01
INITIAL_MAX_PRIORITY = 1.0

class SumTree:
    """
    SumTree for O(log N) priority updates and sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
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

    def add(self, p):
        idx = self.write + self.capacity - 1
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
        return (idx, self.tree[idx], dataIdx)

class PrioritizedReplayBuffer:
    """
    Zero-Copy Prioritized Experience Replay Buffer.
    Data is stored in contiguous Numpy arrays.
    """
    def __init__(self, capacity, alpha=DEFAULT_PRIORITY_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = PRIORITY_EPSILON
        self.max_p = INITIAL_MAX_PRIORITY
        self.tree = SumTree(capacity)
        
        # Pre-allocated arrays (Initialized on first add)
        self.states = None
        self.actions = None
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = None
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.pos = 0

    def _init_storage(self, state_shape, action_shape):
        print(f"ReplayBuffer: Allocating storage for State:{state_shape}, Action:{action_shape}")
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        
        # Handle scalar vs vector actions
        if action_shape == ():
            self.actions = np.zeros(self.capacity, dtype=np.int64) # Discrete scalar
        else:
            self.actions = np.zeros((self.capacity, *action_shape), dtype=np.float32) # Continuous/Multi-dim

    def add(self, state, action, reward, next_state, done):
        # Convert tensors to numpy
        if torch.is_tensor(state): state = state.detach().cpu().numpy()
        if torch.is_tensor(next_state): next_state = next_state.detach().cpu().numpy()
        if torch.is_tensor(action): action = action.detach().cpu().numpy()
        if torch.is_tensor(reward): reward = reward.item()
        
        # Handle scalars (int/float) that don't have .shape
        state_shape = state.shape if hasattr(state, 'shape') else ()
        action_shape = action.shape if hasattr(action, 'shape') else ()
        
        # Initialize storage on first run
        if self.states is None:
            self._init_storage(state_shape, action_shape)
            
        # Store in arrays
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        
        # Update Tree
        self.tree.add(self.max_p)
        
        # Update pointer
        self.pos = (self.pos + 1) % self.capacity

    def store(self, state, action, reward):
        """Legacy compatibility."""
        self.add(state, action, reward, state, False)

    def update_last_reward(self, reward):
        """Updates the reward of the most recently added experience."""
        last_idx = (self.pos - 1) % self.capacity
        self.rewards[last_idx] = reward

    def sample(self, batch_size, beta=0.4):
        if self.tree.n_entries < batch_size:
            return None
            
        idxs = []
        priorities = []
        data_idxs = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, p, data_idx) = self.tree.get(s)
            idxs.append(idx)
            priorities.append(p)
            data_idxs.append(data_idx)
            
        # Vectorized Retrieval
        b_states = self.states[data_idxs]
        b_actions = self.actions[data_idxs]
        b_rewards = self.rewards[data_idxs]
        b_next_states = self.next_states[data_idxs]
        b_dones = self.dones[data_idxs]
        
        # Calculate Weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        
        # Return as tuple of Numpy arrays (Caller handles Tensor conversion)
        batch = (b_states, b_actions, b_rewards, b_next_states, b_dones)
        return batch, idxs, is_weight

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.max_p = max(self.max_p, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
