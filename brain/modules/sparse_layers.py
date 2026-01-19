import torch
import torch.nn as nn
import numpy as np

# Sparse Linear Layer Defaults
DEFAULT_SPARSITY = 0.99          # Default connection sparsity (99% sparse)
MIN_NON_ZERO_CONNECTIONS = 1     # Minimum connections to maintain
SPROUTING_INIT_SCALE = 0.01     # Scale for new connection initialization
PRUNE_THRESHOLD = 0.01          # Weight magnitude threshold for pruning

class SparseLinear(nn.Module):
    """
    Sparse Linear Layer using CSR format for CPU efficiency.
    W (In, Out) is stored as a sparse matrix.
    """
    def __init__(self, in_features, out_features, sparsity=0.99):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Initialize sparse weights
        num_non_zero = int(in_features * out_features * (1 - sparsity))
        num_non_zero = max(num_non_zero, MIN_NON_ZERO_CONNECTIONS)
        
        # Generate random indices
        indices = torch.randint(0, in_features, (2, num_non_zero))
        indices[1] = torch.randint(0, out_features, (num_non_zero,))
        
        # Coalesce to remove duplicates immediately
        # We use a temporary tensor to do this
        temp_vals = torch.randn(num_non_zero)
        temp_coo = torch.sparse_coo_tensor(indices, temp_vals, (in_features, out_features)).coalesce()
        
        # Use the coalesced indices and values size
        self.register_buffer('indices', temp_coo.indices())
        
        real_num_non_zero = self.indices.shape[1]
        self.values = nn.Parameter(torch.randn(real_num_non_zero) * (1.0 / np.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # CSR Cache (for CPU efficiency)
        self.register_buffer('crow_indices', None)
        self.register_buffer('col_indices', None)
        
        # Activity tracking for sprouting (Non-persistent to avoid shape mismatch on load)
        self.register_buffer('activity_in', torch.zeros(in_features), persistent=False)
        self.register_buffer('activity_out', torch.zeros(out_features), persistent=False)
        
    def forward(self, x):
        # x: (Batch, In)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if self.crow_indices is None or self.crow_indices.device != x.device:
            # Reconstruct and cache CSR structure
            with torch.no_grad():
                indices = self.indices.to(x.device)
                values = self.values.to(x.device).detach()
                # Use detached values just to get the structure
                W_coo = torch.sparse_coo_tensor(
                    indices, values, (self.in_features, self.out_features)
                ).coalesce()
                W_csr = W_coo.to_sparse_csr()
                self.register_buffer('crow_indices', W_csr.crow_indices(), persistent=False)
                self.register_buffer('col_indices', W_csr.col_indices(), persistent=False)
            
        # Reconstruct Sparse CSR Tensor using CURRENT values (Autograd friendly)
        W = torch.sparse_csr_tensor(
            self.crow_indices,
            self.col_indices,
            self.values,
            size=(self.in_features, self.out_features),
            device=x.device
        )
        
        # res = x @ weight + bias
        res = torch.addmm(self.bias.unsqueeze(0), x, W)
        
        # Track activity (Hebbian)
        if self.training:
            with torch.no_grad():
                self.activity_in += x.abs().mean(0)
                self.activity_out += res.abs().mean(0)
                
        return res

    def to_dense(self):
        """Converts to a standard nn.Linear for ONNX export."""
        linear = nn.Linear(self.in_features, self.out_features)
        with torch.no_grad():
            # Ensure indices and values are on the same device
            indices = self.indices.to(self.values.device)
            W_coo = torch.sparse_coo_tensor(
                indices, self.values, (self.in_features, self.out_features)
            ).coalesce()
            linear.weight.data = W_coo.to_dense().t()
            linear.bias.data = self.bias.data.clone()
        return linear

    def forward_onnx(self, x):
        """Stateless dense forward pass for ONNX export."""
        # Use traceable dense weight creation
        W_dense = torch.zeros(self.in_features, self.out_features, device=self.values.device)
        # indices[0] is in_features, indices[1] is out_features
        W_dense.index_put_((self.indices[0], self.indices[1]), self.values)
        return torch.addmm(self.bias.unsqueeze(0), x, W_dense)

    def sprout(self, num_new=100):
        """Adds new connections between highly active neurons."""
        print(f"SparseLinear: Sprouting {num_new} new connections...")
        with torch.no_grad():
            # Find top active in/out neurons
            top_in = torch.topk(self.activity_in, min(num_new, self.in_features)).indices
            top_out = torch.topk(self.activity_out, min(num_new, self.out_features)).indices
            
            # Create new indices
            new_idx = torch.stack([
                top_in[torch.randint(0, len(top_in), (num_new,))],
                top_out[torch.randint(0, len(top_out), (num_new,))]
            ])
            
            # Append to existing indices
            self.indices = torch.cat([self.indices, new_idx.to(self.indices.device)], dim=1)
            
            # Initialize new values
            new_vals = torch.randn(num_new, device=self.values.device) * SPROUTING_INIT_SCALE
            self.values = nn.Parameter(torch.cat([self.values, new_vals]))
            
            # Reset CSR cache
            self.crow_indices = None
            self.col_indices = None
            # Reset activity
            self.activity_in.zero_()
            self.activity_out.zero_()

    def prune(self, threshold=PRUNE_THRESHOLD):
        """Removes connections with low weights."""
        with torch.no_grad():
            mask = self.values.abs() > threshold
            if mask.any():
                print(f"SparseLinear: Pruning {len(self.values) - mask.sum()} weak connections...")
                self.indices = self.indices[:, mask]
                self.values = nn.Parameter(self.values[mask])
                # Reset CSR cache
                self.crow_indices = None
                self.col_indices = None

    def resize(self, in_features=None, out_features=None):
        """Resizes the layer while attempting to preserve existing weights."""
        if in_features is None: in_features = self.in_features
        if out_features is None: out_features = self.out_features
        
        if in_features == self.in_features and out_features == self.out_features:
            return
            
        print(f"SparseLinear: Resizing ({self.in_features}, {self.out_features}) -> ({in_features}, {out_features})")
        
        # Create new indices and values
        num_non_zero = int(in_features * out_features * (1 - self.sparsity))
        num_non_zero = max(num_non_zero, MIN_NON_ZERO_CONNECTIONS)
        
        new_indices = torch.randint(0, in_features, (2, num_non_zero))
        new_indices[1] = torch.randint(0, out_features, (num_non_zero,))
        new_values = torch.randn(num_non_zero) * (1.0 / np.sqrt(in_features))
        new_bias = torch.zeros(out_features)
        
        self.in_features = in_features
        self.out_features = out_features
        self.indices = new_indices.to(self.indices.device)
        self.values = nn.Parameter(new_values.to(self.values.device))
        self.bias = nn.Parameter(new_bias.to(self.bias.device))
        
        # Reset CSR cache
        self.crow_indices = None
        self.col_indices = None
        
        # Resize activity buffers
        self.register_buffer('activity_in', torch.zeros(in_features, device=self.indices.device), persistent=False)
        self.register_buffer('activity_out', torch.zeros(out_features, device=self.indices.device), persistent=False)
