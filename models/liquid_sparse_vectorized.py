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
        
        # Call ONNX-friendly forward (using dense weights for ONNX compatibility)
        # In regular execution, we use the sparse path, but for ONNX we need dense.
        # However, for regular forward, we still want to use the sparse logic.
        # So we'll keep the sparse logic here but offer forward_onnx for export.
        
        # 1. Concatenate Inputs and Recurrent State
        prev_y = self.y.unsqueeze(0).expand(batch_size, -1)
        combined_input = torch.cat([inputs, prev_y], dim=1)
        
        # 2. Sparse Matrix Multiplication
        if self._needs_rebuild or self._cached_W is None or self._cached_W.device != inputs.device:
            with torch.no_grad():
                W_coo = torch.sparse_coo_tensor(
                    self.weight_indices.to(inputs.device), 
                    self.weight_values.detach(), 
                    (self.hidden_size, self.input_size + self.hidden_size),
                    device=inputs.device
                ).coalesce()
                W_csr = W_coo.to_sparse_csr()
                self.register_buffer('_crow_indices', W_csr.crow_indices(), persistent=False)
                self.register_buffer('_col_indices', W_csr.col_indices(), persistent=False)
                self._needs_rebuild = False
            
            self._cached_W = torch.sparse_csr_tensor(
                self._crow_indices,
                self._col_indices,
                self.weight_values,
                size=(self.hidden_size, self.input_size + self.hidden_size),
                device=inputs.device
            )
        else:
            self._cached_W = torch.sparse_csr_tensor(
                self._crow_indices,
                self._col_indices,
                self.weight_values,
                size=(self.hidden_size, self.input_size + self.hidden_size),
                device=inputs.device
            )
            
        W = self._cached_W
        activation_in = torch.sparse.mm(W, combined_input.t()).t() # [B, H]
        
        # 3. Apply Dynamics
        tau = self.tau.unsqueeze(0)
        bias = self.bias.unsqueeze(0)
        
        dx = (-self.x + activation_in + bias) / tau * dt
        new_x = self.x + dx
        
        # 4. Activation Functions
        tanh_out = torch.tanh(new_x) * self.act_tanh_mask
        sig_out = torch.sigmoid(new_x) * self.act_sig_mask
        relu_out = torch.relu(new_x) * self.act_relu_mask
        new_y = tanh_out + sig_out + relu_out
        
        # Update State
        if batch_size == 1:
            self.x.copy_(new_x.detach().squeeze(0))
            self.y.copy_(new_y.detach().squeeze(0))
        
        # 5. Output
        outputs = new_y[:, self.output_mapping]
        energy = torch.sum(torch.abs(new_y)) * 0.001
        
        return outputs, new_x, energy

    def forward_onnx(self, inputs, x, y, dt=0.1):
        """
        ONNX-friendly forward pass. Stateless and uses dense weights.
        """
        batch_size = inputs.size(0)
        
        # 1. Concatenate Inputs and Recurrent State
        prev_y = y # y is already [Batch, Hidden] for ONNX
        combined_input = torch.cat([inputs, prev_y], dim=1)
        
        # 2. Dense Matrix Multiplication (ONNX friendly)
        # We convert sparse to dense here for the export trace using traceable operations
        W_dense = torch.zeros(self.hidden_size, self.input_size + self.hidden_size, device=self.weight_values.device)
        W_dense.index_put_((self.weight_indices[0], self.weight_indices[1]), self.weight_values)
        
        activation_in = F.linear(combined_input, W_dense)
        
        # 3. Apply Dynamics
        tau = self.tau.unsqueeze(0)
        bias = self.bias.unsqueeze(0)
        
        dx = (-x + activation_in + bias) / tau * dt
        next_x = x + dx
        
        # 4. Activation Functions
        tanh_out = torch.tanh(next_x) * self.act_tanh_mask
        sig_out = torch.sigmoid(next_x) * self.act_sig_mask
        relu_out = torch.relu(next_x) * self.act_relu_mask
        next_y = tanh_out + sig_out + relu_out
        
        # 5. Output
        outputs = next_y[:, self.output_mapping]
        energy = torch.sum(torch.abs(next_y)) * 0.001
        
        return outputs, next_x, next_y, energy

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
