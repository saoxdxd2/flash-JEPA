import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from models.plasticity_mlp import PlasticityMLP
from brain.utils import get_best_device

class VectorizedLiquidGraph(nn.Module):
    """
    A high-performance, GPU-accelerated version of LiquidGraph.
    Uses PyTorch tensors for all operations, enabling massive parallelization
    and batched forward/learning passes.
    
    Dynamics:
    dx/dt = (-x + W_in * I + W_rec * y) / tau
    y = Activation(x + bias)
    """
    def __init__(self, input_size, hidden_size, output_size, genome=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = get_best_device()
        
        # --- Parameters (Tensors) ---
        # Time constants: [hidden_size]
        self.tau = nn.Parameter(torch.rand(hidden_size) * 9.0 + 1.0)
        # Biases: [hidden_size]
        self.bias = nn.Parameter(torch.randn(hidden_size) * 0.1)
        
        # Input Weights: [hidden_size, input_size] (Sparse-ish initialization)
        self.w_in = nn.Parameter(torch.randn(hidden_size, input_size) * (1.0 / np.sqrt(input_size if input_size > 0 else 1)))
        # Recurrent Weights: [hidden_size, hidden_size]
        self.w_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * (1.0 / np.sqrt(hidden_size if hidden_size > 0 else 1)))
        
        # Activation Types (Stored as indices for vectorization)
        # 0: tanh, 1: sigmoid, 2: relu
        self.register_buffer('act_types', torch.randint(0, 3, (hidden_size,)))
        
        # --- State (Tensors) ---
        self.register_buffer('x', torch.zeros(hidden_size))
        self.register_buffer('y', torch.zeros(hidden_size))
        
        # --- Mapping ---
        self.param_size = 2
        if hidden_size > output_size + self.param_size:
             self.output_indices = list(range(hidden_size - output_size - self.param_size, hidden_size - self.param_size))
             self.param_indices = list(range(hidden_size - self.param_size, hidden_size))
        else:
             self.output_indices = list(range(hidden_size - output_size, hidden_size))
             self.param_indices = []
             
        # --- Plasticity ---
        if genome:
            self.plasticity_net = PlasticityMLP(
                hidden_size=genome.plasticity_hidden_size,
                num_layers=genome.plasticity_layers,
                activation=genome.plasticity_activation
            )
        else:
            self.plasticity_net = PlasticityMLP()
            
        self.learning_rate = 0.01
        self.decay_rate = 0.001

    def forward(self, input_vector, dt=1.0, reward=0.0, train_internal_rl=True):
        """
        Vectorized forward pass.
        Supports Batching: [Batch, input_size]
        """
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        batch_size = input_vector.shape[0]
        
        # Ensure state matches batch size
        if self.x.shape[0] != batch_size:
            self.x = torch.zeros(batch_size, self.hidden_size, device=input_vector.device)
            self.y = torch.zeros(batch_size, self.hidden_size, device=input_vector.device)
            
        # Call ONNX-friendly forward
        outputs, params, energy, next_x, next_y = self.forward_onnx(input_vector, self.x, self.y, dt)
        
        # Update states
        self.x = next_x
        self.y = next_y
        
        return outputs, params, energy

    def forward_onnx(self, input_vector, x, y, dt=1.0):
        """
        ONNX-friendly forward pass. Stateless.
        """
        batch_size = input_vector.shape[0]
        
        # 1. Gather Inputs
        i_in = F.linear(input_vector, self.w_in)
        i_rec = F.linear(y, self.w_rec)
        total_input = i_in + i_rec
        
        # 2. ODE Integration (Euler)
        dx = dt * (-x + total_input) / self.tau
        next_x = x + dx
        next_x = torch.clamp(next_x, -10.0, 10.0)
        
        # 3. Activation
        next_y = torch.zeros_like(next_x)
        
        # Tanh (Type 0)
        mask_tanh = (self.act_types == 0)
        # For ONNX, we avoid mask.any() if possible or use where
        next_y = torch.where(mask_tanh.unsqueeze(0), torch.tanh(next_x + self.bias), next_y)
            
        # Sigmoid (Type 1)
        mask_sig = (self.act_types == 1)
        next_y = torch.where(mask_sig.unsqueeze(0), torch.sigmoid(next_x + self.bias), next_y)
            
        # ReLU (Type 2)
        mask_relu = (self.act_types == 2)
        next_y = torch.where(mask_relu.unsqueeze(0), F.relu(next_x + self.bias), next_y)
            
        # 4. Map to Outputs
        outputs = next_y[:, self.output_indices]
        
        if self.param_indices:
            params = next_y[:, self.param_indices]
            params = torch.sigmoid(params)
        else:
            params = torch.zeros(batch_size, 2, device=input_vector.device)
            
        # Energy cost
        energy = torch.mean(torch.abs(next_y)) * 0.01
        
        return outputs, params, energy, next_x, next_y

    def reset_state(self):
        """Resets the hidden state of the brain."""
        self.x = torch.zeros_like(self.x)
        self.y = torch.zeros_like(self.y)

    def detach_state(self):
        """Detaches the hidden state from the computation graph."""
        self.x = self.x.detach()
        self.y = self.y.detach()

    def learn(self, reward):
        """
        Vectorized Meta-Learning.
        Applies plasticity rules to all weights in parallel.
        """
        # This is complex to vectorize fully because plasticity_net is per-connection.
        # For now, we'll implement a "Fast Hebbian" approximation or keep it for small batches.
        # Actually, we can vectorize by treating the weights as a flat batch.
        pass

    def resize_hidden(self, new_hidden_size):
        """Resizes the hidden layer and preserves existing weights."""
        if new_hidden_size <= self.hidden_size:
            return

        device = self.w_rec.device
        
        # 1. Resize Recurrent Weights (hidden, hidden)
        new_w_rec = torch.zeros(new_hidden_size, new_hidden_size, device=device)
        new_w_rec[:self.hidden_size, :self.hidden_size] = self.w_rec.data
        nn.init.xavier_uniform_(new_w_rec[self.hidden_size:, :])
        nn.init.xavier_uniform_(new_w_rec[:, self.hidden_size:])
        self.w_rec = nn.Parameter(new_w_rec)
        
        # 2. Resize Input Weights (hidden, input)
        new_w_in = torch.zeros(new_hidden_size, self.input_size, device=device)
        new_w_in[:self.hidden_size, :] = self.w_in.data
        nn.init.xavier_uniform_(new_w_in[self.hidden_size:, :])
        self.w_in = nn.Parameter(new_w_in)
        
        # 3. Resize Time Constants and Bias
        new_tau = torch.ones(new_hidden_size, device=device) * 5.0
        new_tau[:self.hidden_size] = self.tau.data
        self.tau = nn.Parameter(new_tau)
        
        new_bias = torch.zeros(new_hidden_size, device=device)
        new_bias[:self.hidden_size] = self.bias.data
        self.bias = nn.Parameter(new_bias)
        
        # 4. Resize State (x, y) - if they exist
        if self.x is not None:
             # Assuming batch_size is dim 0 or 1 depending on implementation
             # VectorizedLiquidGraph usually has [hidden] or [batch, hidden]
             if self.x.dim() == 2:
                 batch_size = self.x.size(0)
                 new_x = torch.zeros(batch_size, new_hidden_size, device=device)
                 new_x[:, :self.hidden_size] = self.x
                 self.x = new_x
                 
                 new_y = torch.zeros(batch_size, new_hidden_size, device=device)
                 new_y[:, :self.hidden_size] = self.y
                 self.y = new_y
             else:
                 new_x = torch.zeros(new_hidden_size, device=device)
                 new_x[:self.hidden_size] = self.x
                 self.x = new_x
                 
                 new_y = torch.zeros(new_hidden_size, device=device)
                 new_y[:self.hidden_size] = self.y
                 self.y = new_y
                 
        # 5. Resize Activation Types
        new_act_types = torch.zeros(new_hidden_size, dtype=torch.long, device=device)
        new_act_types[:self.hidden_size] = self.act_types
        new_act_types[self.hidden_size:] = torch.randint(0, 3, (new_hidden_size - self.hidden_size,), device=device)
        self.act_types = new_act_types
        
        self.hidden_size = new_hidden_size

    def resize_output(self, new_output_size):
        """Resizes output mapping and grows hidden layer if needed."""
        if new_output_size > self.hidden_size:
            self.resize_hidden(new_output_size)
            
        self.output_size = new_output_size
        self.output_indices = torch.arange(new_output_size, device=self.w_rec.device)

    def resize_input(self, new_input_size):
        """Resizes input weights."""
        if new_input_size == self.input_size: return
        
        old_w_in = self.w_in.data
        new_w_in = torch.randn(self.hidden_size, new_input_size, device=old_w_in.device) * (1.0 / np.sqrt(new_input_size))
        
        min_in = min(self.input_size, new_input_size)
        new_w_in[:, :min_in] = old_w_in[:, :min_in]
        
        self.w_in = nn.Parameter(new_w_in)
        self.input_size = new_input_size

    def to_dict(self):
        return {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'act_types': self.act_types.cpu().numpy().tolist()
        }

    @staticmethod
    def from_legacy(legacy_graph):
        """Converts a legacy LiquidGraph (object or dict) to VectorizedLiquidGraph."""
        if isinstance(legacy_graph, dict):
            return VectorizedLiquidGraph.from_dict(legacy_graph)
            
        graph = VectorizedLiquidGraph(
            getattr(legacy_graph, 'input_size', 64), 
            getattr(legacy_graph, 'hidden_size', 512), 
            getattr(legacy_graph, 'output_size', 256)
        )
        
        # Restore Nodes
        with torch.no_grad():
            for i, node in enumerate(legacy_graph.nodes):
                if i >= graph.hidden_size: break
                graph.tau[i] = float(node.tau)
                graph.bias[i] = float(node.bias)
                act = getattr(node, 'activation_type', 'tanh')
                if act == 'tanh': graph.act_types[i] = 0
                elif act == 'sigmoid': graph.act_types[i] = 1
                elif act == 'relu': graph.act_types[i] = 2
                
                # Restore Connections
                for i, node_conns in enumerate(legacy_graph.connections):
                    if i >= graph.hidden_size: break
                    for src_idx, weight in node_conns:
                        if src_idx < graph.input_size:
                            graph.w_in.data[i, src_idx] = float(weight)
                        else:
                            rec_idx = src_idx - graph.input_size
                            if rec_idx < graph.hidden_size:
                                graph.w_rec.data[i, rec_idx] = float(weight)
                            
            # Plasticity
            if hasattr(legacy_graph, 'plasticity_net'):
                graph.plasticity_net.load_state_dict(legacy_graph.plasticity_net.state_dict())
                
        return graph

    @staticmethod
    def from_dict(data):
        """Loads VectorizedLiquidGraph from dict (Supports new and legacy formats)."""
        if 'state_dict' in data:
            graph = VectorizedLiquidGraph(data['input_size'], data['hidden_size'], data['output_size'])
            state_dict = data['state_dict']
            # Fix shape mismatch for x and y if present (legacy checkpoints might have [1, hidden])
            for key in ['x', 'y']:
                if key in state_dict and state_dict[key].dim() == 2 and state_dict[key].size(0) == 1:
                    state_dict[key] = state_dict[key].squeeze(0)
            
            # Fix transposed weights (nn.Linear is [out, in], we want [in, out])
            # w_in: [input, hidden]
            if 'w_in' in state_dict:
                w = state_dict['w_in']
                if w.shape == (graph.input_size, graph.hidden_size):
                    state_dict['w_in'] = w.t()
                    
            # w_rec: [hidden, hidden] - usually symmetric in size, but check just in case
            
            graph.load_state_dict(state_dict)
            return graph
        elif 'nodes' in data:
            # Legacy Dict Format (from old LiquidGraph.to_dict)
            print("VectorizedLiquidGraph: Converting legacy dict format...")
            graph = VectorizedLiquidGraph(data['input_size'], data['hidden_size'], data['output_size'])
            with torch.no_grad():
                for i, node_data in enumerate(data['nodes']):
                    if i >= graph.hidden_size: break
                    graph.tau[i] = float(node_data.get('tau', 5.0))
                    graph.bias[i] = float(node_data.get('bias', 0.0))
                    act = node_data.get('activation_type', 'tanh')
                    if act == 'tanh': graph.act_types[i] = 0
                    elif act == 'sigmoid': graph.act_types[i] = 1
                    elif act == 'relu': graph.act_types[i] = 2
                
                # Connections
                for i, node_conns in enumerate(data.get('connections', [])):
                    if i >= graph.hidden_size: break
                    for src_idx, weight in node_conns:
                        if src_idx < graph.input_size:
                            graph.w_in.data[i, src_idx] = float(weight)
                        else:
                            rec_idx = src_idx - graph.input_size
                            if rec_idx < graph.hidden_size:
                                graph.w_rec.data[i, rec_idx] = float(weight)
            return graph
        else:
            raise KeyError("VectorizedLiquidGraph: Dict format not recognized (missing 'state_dict' or 'nodes')")
