import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def resize_input(self, new_input_size):
        """
        Dynamically resizes the input layer while preserving existing weights.
        """
        if new_input_size == self.fc1.in_features:
            return
            
        print(f"QNetwork: Resizing Input {self.fc1.in_features} -> {new_input_size}")
        old_fc1 = self.fc1
        new_fc1 = nn.Linear(new_input_size, old_fc1.out_features)
        
        # Copy existing weights
        with torch.no_grad():
            min_in = min(old_fc1.in_features, new_input_size)
            new_fc1.weight[:, :min_in] = old_fc1.weight[:, :min_in]
            new_fc1.bias[:] = old_fc1.bias[:]
            
        self.fc1 = new_fc1

class DDQN:
    """
    Double Deep Q-Network with PyTorch.
    """
    def __init__(self, input_size, action_size, learning_rate=5e-4):
        self.input_size = input_size
        self.action_size = action_size
        
        # Networks
        self.online_net = QNetwork(input_size, action_size)
        self.target_net = QNetwork(input_size, action_size)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.tau = 0.005 # Soft update rate

    def act(self, state):
        """Select action using Epsilon-Greedy policy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax().item()

    def train(self, replay_buffer, beta=0.4):
        """Train the network using Prioritized Replay Buffer."""
        if len(replay_buffer) < self.batch_size:
            return None
        
        # Sample Batch
        batch, idxs, is_weights = replay_buffer.sample(self.batch_size, beta)
        
        # Filter out None entries if any
        batch = [b for b in batch if b is not None]
        if len(batch) < self.batch_size // 2:
            return None

        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Normalize Shapes (Handle evolutionary changes in latent_dim)
        def normalize_shape(data_list, target_size):
            normalized = []
            for item in data_list:
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                
                arr = np.array(item).flatten()
                if arr.shape[0] == target_size:
                    normalized.append(arr)
                elif arr.shape[0] < target_size:
                    # Pad with zeros
                    padded = np.zeros(target_size)
                    padded[:arr.shape[0]] = arr
                    normalized.append(padded)
                else:
                    # Truncate
                    normalized.append(arr[:target_size])
            return np.array(normalized)

        states = torch.FloatTensor(normalize_shape(states, self.input_size))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(normalize_shape(next_states, self.input_size))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1)
        
        # DDQN Logic
        # 1. Select best action using Online Net
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1, keepdim=True)
            # 2. Evaluate that action using Target Net
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # 3. Compute Target Q
            target_q = rewards + (self.gamma * next_q_values * (1 - dones))
            
        # Current Q
        current_q = self.online_net(states).gather(1, actions)
        
        # Compute Loss (Weighted MSE)
        loss = (current_q - target_q).pow(2) * is_weights
        loss = loss.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update Priorities
        errors = torch.abs(current_q - target_q).detach().numpy()
        replay_buffer.update_priorities(idxs, errors)
        
        # Soft Update Target Network
        self.soft_update()
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def resize_input(self, new_input_size):
        """
        Updates the input size of the networks.
        """
        if new_input_size == self.input_size:
            return
            
        print(f"DDQN: Resizing Input {self.input_size} -> {new_input_size}")
        self.online_net.resize_input(new_input_size)
        self.target_net.resize_input(new_input_size)
        
        # Re-initialize optimizer because parameters changed
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        self.input_size = new_input_size

    def save(self, filepath):
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
