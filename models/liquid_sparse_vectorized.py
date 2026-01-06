import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SparseVectorizedLiquidGraph(nn.Module):
    """
    Sparse Vectorized Liquid Neural Network.
    Designed for Massive Scalability (1B+ Neurons).
    
    Key Difference from VectorizedLiquidGraph:
    - Weights are stored as a Sparse Tensor (COO or CSR).
    - Forward pass uses Sparse Matrix Multiplication (spmm).
    - Growth involves appending indices/values, not reallocating dense blocks.
    """
    def __init__(self, input_size, hidden_size, output_size, sparsity=0.95, device='cpu', genome=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.genome = genome
        self.sparsity = getattr(genome, 'sparsity', sparsity) if genome else sparsity
        self.device = device
        
        # --- 1. Neuron Parameters (Dense) ---
        # State vectors are O(N), which is fine for 1B (1B floats = 4GB).
        # It's the O(N^2) weights we need to fix.
        
        # Tau: Time constants [hidden_size]
        self.tau = nn.Parameter(torch.empty(hidden_size, device=device).uniform_(1.0, 10.0))
        
        # Bias: Activation thresholds [hidden_size]
        self.bias = nn.Parameter(torch.empty(hidden_size, device=device).uniform_(-1.0, 1.0))
        
        # Activation Type: 0=tanh, 1=sigmoid, 2=relu
        self.act_types = torch.randint(0, 3, (hidden_size,), device=device)
        
        # Masks for activations (Computed on fly or cached? Cached is faster)
        self.register_buffer('act_tanh_mask', (self.act_types == 0).float())
        self.register_buffer('act_sig_mask', (self.act_types == 1).float())
        self.register_buffer('act_relu_mask', (self.act_types == 2).float())
        
        # --- 2. State (Potentials) ---
        self.register_buffer('x', torch.zeros(hidden_size, device=device)) # Internal Potential
        self.register_buffer('y', torch.zeros(hidden_size, device=device)) # Firing Rate
        
        # --- 3. Sparse Connections ---
        # Weight Matrix W: [Hidden, Input + Hidden]
        # We store this as (indices, values) separate Parameters to allow optimization.
        # Note: PyTorch optimizers struggle with sparse gradients sometimes. 
        # We might need a custom optimizer or just optimize 'values'.
        
        total_inputs = input_size + hidden_size
        num_weights = int(hidden_size * total_inputs * (1.0 - sparsity))
        
        # Initialize Random Sparse Topology
        # Indices: [2, num_weights]
        rows = torch.randint(0, hidden_size, (num_weights,), device=device)
        cols = torch.randint(0, total_inputs, (num_weights,), device=device)
        self.register_buffer('weight_indices', torch.stack([rows, cols]))
        
        # Values: [num_weights]
        self.weight_values = nn.Parameter(torch.empty(num_weights, device=device).uniform_(-0.1, 0.1))
        
        self.param_size = 2
        self.output_mapping = list(range(hidden_size - output_size - self.param_size, hidden_size - self.param_size))
        
        # --- 4. Performance Cache ---
        self.register_buffer('_cached_W', None)
        self.register_buffer('_crow_indices', None)
        self.register_buffer('_col_indices', None)
        self._needs_rebuild = True
        
    def forward(self, inputs, dt=0.1, reward=0.0, train_internal_rl=True):
        """
        Sparse Forward Pass.
        dy/dt = -y/tau + f(W * [u, y] + b)
        """
        is_batched = inputs.dim() > 1
        if not is_batched:
            inputs = inputs.unsqueeze(0)
            
        batch_size = inputs.size(0)
        
        # 1. Concatenate Inputs and Recurrent State
        # inputs: [Batch, Input]
        # y: [Hidden] -> Broadcast to [Batch, Hidden]
        prev_y = self.y.unsqueeze(0).expand(batch_size, -1)
        
        # Combined: [Batch, Input + Hidden]
        combined_input = torch.cat([inputs, prev_y], dim=1)
        
        # 2. Sparse Matrix Multiplication
        # W: [Hidden, Total_In]
        
        # Rebuild Cache if needed
        if self._needs_rebuild or self._cached_W is None:
            # Construct Sparse Tensor and coalesce to ensure order
            W_coo = torch.sparse_coo_tensor(
                self.weight_indices, 
                self.weight_values, 
                (self.hidden_size, self.input_size + self.hidden_size),
                device=self.device
            ).coalesce()
            
            # Convert to CSR for much faster CPU inference
            W_csr = W_coo.to_sparse_csr()
            self._cached_W = W_csr
            self.register_buffer('_crow_indices', W_csr.crow_indices())
            self.register_buffer('_col_indices', W_csr.col_indices())
            self._needs_rebuild = False
        else:
            # O(K) update: reuse CSR structure with new values
            self._cached_W = torch.sparse_csr_tensor(
                self._crow_indices,
                self._col_indices,
                self.weight_values,
                size=(self.hidden_size, self.input_size + self.hidden_size),
                device=self.device
            )
            
        W = self._cached_W
        
        # Sparse MM
        # Note: torch.sparse.mm requires (Sparse, Dense)
        # W is Sparse [H, T], I.T is Dense [T, B]
        # Result: [H, B]
        activation_in = torch.sparse.mm(W, combined_input.t()).t() # [B, H]
        
        # 3. Apply Dynamics
        # dx = (-x + activation_in + bias) / tau * dt
        # Euler integration
        
        # Expand params for batch
        tau = self.tau.unsqueeze(0)
        bias = self.bias.unsqueeze(0)
        
        dx = (-self.x + activation_in + bias) / tau * dt
        new_x = self.x + dx
        
        # 4. Activation Functions (Vectorized Masking)
        # This part is still O(N), but efficient on GPU
        tanh_out = torch.tanh(new_x) * self.act_tanh_mask
        sig_out = torch.sigmoid(new_x) * self.act_sig_mask
        relu_out = torch.relu(new_x) * self.act_relu_mask
        
        new_y = tanh_out + sig_out + relu_out
        
        # Update State (Detached from graph for state persistence)
        # self.x is [Hidden]. new_x is [Batch, Hidden].
        # We assume Batch=1 for the persistent state.
        if batch_size == 1:
            self.x.copy_(new_x.detach().squeeze(0))
            self.y.copy_(new_y.detach().squeeze(0))
        else:
            # If batch > 1, we can't easily persist state for "the" brain.
            # We just update with the mean? Or don't update?
            # For training (dreaming), we might run batch > 1, but we don't persist that state.
            pass
        
        # 5. Output
        # Gather output neurons
        outputs = new_y[:, self.output_mapping]
        
        # Calculate Energy (Metabolic Cost)
        # L1 norm of activity + synaptic operations
        energy = torch.sum(torch.abs(new_y)) * 0.001
        
        return outputs, new_x, energy

    def reset_state(self):
        """Resets the hidden state of the brain."""
        self.x.zero_()
        self.y.zero_()

    def detach_state(self):
        """Detaches the hidden state from the computation graph."""
        self.x = self.x.detach()
        self.y = self.y.detach()

    def resize_hidden(self, new_size):
        """
        Efficiently grows the sparse network.
        """
        if new_size <= self.hidden_size:
            return

        growth = new_size - self.hidden_size
        print(f"SparseLiquid: Growing from {self.hidden_size} to {new_size}...")
        
        # 1. Grow Dense Params (O(N))
        new_tau = torch.empty(growth, device=self.device).uniform_(1.0, 10.0)
        self.tau = nn.Parameter(torch.cat([self.tau, new_tau]))
        
        new_bias = torch.empty(growth, device=self.device).uniform_(-1.0, 1.0)
        self.bias = nn.Parameter(torch.cat([self.bias, new_bias]))
        
        new_types = torch.randint(0, 3, (growth,), device=self.device)
        self.act_types = torch.cat([self.act_types, new_types])
        
        self.register_buffer('act_tanh_mask', (self.act_types == 0).float())
        self.register_buffer('act_sig_mask', (self.act_types == 1).float())
        self.register_buffer('act_relu_mask', (self.act_types == 2).float())
        
        self.register_buffer('x', torch.cat([self.x, torch.zeros(growth, device=self.device)]))
        self.register_buffer('y', torch.cat([self.y, torch.zeros(growth, device=self.device)]))
        
        # 2. Grow Sparse Weights (O(New Connections))
        # We add new random connections for the new neurons.
        # New Rows: [old_h : new_h]
        # New Cols: [0 : input + new_h]
        
        total_inputs_new = self.input_size + new_size
        
        # Calculate how many new weights to add to maintain sparsity
        # Target Total Weights = new_h * total_inputs_new * (1 - sparsity)
        # Current Weights = len(self.weight_values)
        # Add difference
        
        target_weights = int(new_size * total_inputs_new * (1.0 - self.sparsity))
        current_weights = self.weight_values.numel()
        weights_to_add = max(0, target_weights - current_weights)
        
        if weights_to_add > 0:
            # Generate new random connections
            # We favor connecting TO the new neurons and FROM the new neurons
            # But for simplicity, we just sample uniformly from the whole new space
            # A better heuristic would be preferential attachment.
            
            new_rows = torch.randint(0, new_size, (weights_to_add,), device=self.device)
            new_cols = torch.randint(0, total_inputs_new, (weights_to_add,), device=self.device)
            
            new_indices = torch.stack([new_rows, new_cols])
            new_vals = torch.empty(weights_to_add, device=self.device).uniform_(-0.1, 0.1)
            
            # Concatenate
            self.register_buffer('weight_indices', torch.cat([self.weight_indices, new_indices], dim=1))
            self.weight_values = nn.Parameter(torch.cat([self.weight_values, new_vals]))
            
            # Mark for rebuild
            self._needs_rebuild = True
            
        self.hidden_size = new_size
        self.output_mapping = list(range(new_size - self.output_size - self.param_size, new_size - self.param_size))

    def resize_input(self, new_input_size):
        """
        Grow the input layer of the sparse network.
        """
        if new_input_size <= self.input_size:
            return
            
        diff = new_input_size - self.input_size
        print(f"SparseLiquid: Resizing Input from {self.input_size} to {new_input_size}...")
        
        # 1. Update Indices
        # Indices are [2, num_weights]. 
        # Rows are [0], Cols are [1].
        # Connections with Col >= input_size are recurrent.
        # We need to shift these by 'diff'.
        
        indices = self.weight_indices.clone()
        mask_recurrent = indices[1] >= self.input_size
        indices[1, mask_recurrent] += diff
        
        # 2. Add new connections for the new input space?
        # To maintain sparsity, we should add some connections from the new inputs.
        # Target: diff * hidden_size * (1 - sparsity)
        num_new = int(diff * self.hidden_size * (1.0 - self.sparsity))
        
        if num_new > 0:
            new_rows = torch.randint(0, self.hidden_size, (num_new,), device=self.device)
            new_cols = torch.randint(self.input_size, new_input_size, (num_new,), device=self.device)
            
            new_indices = torch.stack([new_rows, new_cols])
            new_vals = torch.empty(num_new, device=self.device).uniform_(-0.1, 0.1)
            
            indices = torch.cat([indices, new_indices], dim=1)
            self.weight_values = nn.Parameter(torch.cat([self.weight_values, new_vals]))
            
        self.register_buffer('weight_indices', indices)
        self.input_size = new_input_size
        self._needs_rebuild = True

    def to_dense(self):
        """
        Debug helper to view as dense matrix.
        """
        W = torch.sparse_coo_tensor(
            self.weight_indices, 
            self.weight_values, 
            (self.hidden_size, self.input_size + self.hidden_size),
            device=self.device
        )
        return W.to_dense()
