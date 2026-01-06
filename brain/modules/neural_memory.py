import torch
import torch.nn as nn
import torch.nn.functional as F

class TitansMemory(nn.Module):
    """
    Neural Long-Term Memory (Titans Architecture).
    Learns to memorize at test time using gradients as surprise signals.
    
    Concept:
    - Maintains a "Fast Weight" matrix that predicts the next state.
    - Updates this matrix online based on prediction error (Surprise).
    - High surprise = High learning rate (Flashbulb Memory).
    """
    def __init__(self, input_dim, hidden_dim=None, learning_rate=0.1, sparsity=0.95):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim else input_dim
        self.learning_rate = learning_rate
        self.sparsity = sparsity
        
        # Sparse Memory Weights
        # We store indices and values separately
        num_weights = int(self.input_dim * self.hidden_dim * (1.0 - sparsity))
        
        # Initial Random Sparse Topology
        rows = torch.randint(0, self.input_dim, (num_weights,))
        cols = torch.randint(0, self.hidden_dim, (num_weights,))
        self.register_buffer('indices', torch.stack([rows, cols]))
        
        # Learnable Values
        self.values = nn.Parameter(torch.randn(num_weights) * 0.01)
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim))
        
        self.last_state = None
        self.last_prediction = None
        
        # --- Performance Cache ---
        self.register_buffer('_cached_W_t', None)
        self.register_buffer('_crow_indices', None)
        self.register_buffer('_col_indices', None)
        self._needs_rebuild = True
        
    def forward(self, x):
        """
        Predicts the next state based on current state x.
        """
        # x: [Batch, InputDim]
        # W: [InputDim, HiddenDim] (Sparse)
        # Out = x @ W + b
        
        # Rebuild Cache if needed
        if self._needs_rebuild or self._cached_W_t is None:
            # Sort indices to ensure coalesced COO for efficient CSR conversion
            # This is O(K log K) but only happens when topology changes
            W_coo = torch.sparse_coo_tensor(
                self.indices, 
                self.values, 
                (self.input_dim, self.hidden_dim),
                device=x.device
            ).coalesce()
            
            # Convert to CSR and Transpose
            W_t = W_coo.t().to_sparse_csr()
            self._cached_W_t = W_t
            self.register_buffer('_crow_indices', W_t.crow_indices())
            self.register_buffer('_col_indices', W_t.col_indices())
            self._needs_rebuild = False
        else:
            # O(K) update: reuse CSR structure with new values
            # We must ensure values are mapped correctly to the CSR structure.
            # Since we built W_t from W_coo.t(), the values in W_t.values() 
            # are the same as W_coo.values() IF W_coo was coalesced and 
            # the transpose/CSR conversion preserves order (which it does for CSR).
            self._cached_W_t = torch.sparse_csr_tensor(
                self._crow_indices,
                self._col_indices,
                self.values, # This might need mapping if indices were shuffled
                size=(self.hidden_dim, self.input_dim),
                device=x.device
            )
            
        W_t = self._cached_W_t
        
        # Ensure x is 2D [Batch, Input]
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        out = torch.sparse.mm(W_t, x.t()).t()
        
        # Apply tanh to bound the prediction to [-1, 1]
        # This matches the target embeddings and prevents exploding loss
        return torch.tanh(out + self.bias)
        
    def observe(self, current_state):
        """
        Observes the actual current state.
        Updates Memory (Values only) based on Surprise.
        """
        surprise = 0.0
        
        # Ensure current_state is 2D [Batch, Input]
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        
        if self.last_state is not None:
            # 1. Calculate Surprise
            if self.last_prediction is not None:
                # Ensure last_prediction is 2D
                if self.last_prediction.dim() == 1:
                    self.last_prediction = self.last_prediction.unsqueeze(0)
                
                # Handle Batch Size Mismatch (Dynamic Batching)
                if self.last_prediction.size(0) == current_state.size(0):
                    error = F.mse_loss(self.last_prediction, current_state)
                    surprise = error.item()
                else:
                    # Batch size changed (e.g., end of dataset or dynamic scaling)
                    # We skip surprise calculation for this step
                    pass
            
            # 2. Update Memory (Online Learning)
            # We want M(last_state) = current_state
            
            # Handle Batch Size Mismatch for Learning
            if self.last_state.size(0) == current_state.size(0):
                # Ensure last_state is 2D
                if self.last_state.dim() == 1:
                    self.last_state = self.last_state.unsqueeze(0)
                    
                # Create sparse tensor - ensure it's on the same device as values
                W = torch.sparse_coo_tensor(
                    self.indices, 
                    self.values, 
                    (self.input_dim, self.hidden_dim)
                )
                
                # Use COO MM instead of CSR to ensure autograd support during the learning step
                # Explicitly add bias with unsqueeze to ensure broadcasting doesn't confuse autograd
                pred = torch.sparse.mm(W.t(), self.last_state.t()).t() + self.bias.unsqueeze(0)
                
                loss = F.mse_loss(pred, current_state)
                
                if loss.requires_grad:
                    # Gradient Descent on VALUES only (Topology is fixed for now)
                    grads = torch.autograd.grad(loss, [self.values, self.bias], create_graph=False)
                    
                    with torch.no_grad():
                        self.values -= self.learning_rate * grads[0]
                        self.bias -= self.learning_rate * grads[1]
                        self._needs_rebuild = True
            else:
                # Batch size mismatch: Skip learning for this transition
                pass
                
        self.last_state = current_state.detach()
        with torch.no_grad():
             self.last_prediction = self.forward(current_state)
             
        return surprise
        
    def reset(self):
        self.last_state = None
        self.last_prediction = None
        
    def reset_state(self):
        self.reset()

    def resize(self, new_input_dim, new_hidden_dim=None):
        """
        Resizes the memory dimensions.
        """
        if new_hidden_dim is None:
            new_hidden_dim = new_input_dim
            
        if new_input_dim == self.input_dim and new_hidden_dim == self.hidden_dim:
            return
            
        print(f"TitansMemory: Resizing {self.input_dim}x{self.hidden_dim} -> {new_input_dim}x{new_hidden_dim}")
        
        # 1. Grow Indices and Values if needed
        old_num_weights = len(self.values)
        new_num_weights = int(new_input_dim * new_hidden_dim * (1.0 - self.sparsity))
        
        if new_num_weights > old_num_weights:
            growth = new_num_weights - old_num_weights
            
            # New random indices in the FULL new range
            new_rows = torch.randint(0, new_input_dim, (growth,))
            new_cols = torch.randint(0, new_hidden_dim, (growth,))
            new_indices = torch.stack([new_rows, new_cols])
            
            # Update indices buffer
            updated_indices = torch.cat([self.indices, new_indices], dim=1)
            self.register_buffer('indices', updated_indices)
            
            # Update values parameter
            new_values = torch.randn(growth) * 0.01
            updated_values = torch.cat([self.values.data, new_values])
            self.values = nn.Parameter(updated_values)
            self._needs_rebuild = True
            
        # 2. Resize Bias
        if new_hidden_dim > self.hidden_dim:
            growth = new_hidden_dim - self.hidden_dim
            new_bias = torch.zeros(growth)
            updated_bias = torch.cat([self.bias.data, new_bias])
            self.bias = nn.Parameter(updated_bias)
            
        self.input_dim = new_input_dim
        self.hidden_dim = new_hidden_dim
        self.reset()
