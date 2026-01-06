import torch
import torch.nn as nn
import torch.nn.functional as F

class PlasticityMLP(nn.Module):
    """
    Meta-Learning Network: Determines how synaptic weights change.
    
    Input: [Pre, Post, Current Weight, Reward]
    Output: Delta Weight (Scalar)
    
    This network replaces fixed Hebbian rules (like Oja's rule or STDP).
    Its weights are evolved, allowing the agent to 'learn how to learn'.
    """
    def __init__(self, input_size=4, hidden_size=16, num_layers=1, activation='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation_type = activation
        
        # Build Layers Dynamically
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(self._get_activation())
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self._get_activation())
            
        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
    def _get_activation(self):
        if self.activation_type == 'relu':
            return nn.ReLU()
        elif self.activation_type == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.Tanh()
        
    def forward(self, pre, post, weight, reward):
        """
        Calculates delta_weight for a batch of connections.
        
        Args:
            pre: Tensor [Batch] (Pre-synaptic activity)
            post: Tensor [Batch] (Post-synaptic activity)
            weight: Tensor [Batch] (Current connection weight)
            reward: Tensor [Batch] (Global or Local Reward signal)
            
        Returns:
            delta_weight: Tensor [Batch]
        """
        # Stack inputs: [Batch, 4]
        # Ensure all inputs are 1D tensors of same size
        x = torch.stack([pre, post, weight, reward], dim=1)
        
        delta = self.net(x)
        
        # Tanh output to bound weight changes [-1, 1] (scaled later)
        return F.tanh(delta).squeeze()
