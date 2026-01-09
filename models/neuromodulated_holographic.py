import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        num_non_zero = max(num_non_zero, 1)
        
        # We store indices as a buffer and values as a parameter
        self.register_buffer('indices', torch.randint(0, in_features, (2, num_non_zero)))
        self.indices[1] = torch.randint(0, out_features, (num_non_zero,))
        
        self.values = nn.Parameter(torch.randn(num_non_zero) * (1.0 / np.sqrt(in_features)))
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
            new_vals = torch.randn(num_new, device=self.values.device) * 0.01
            self.values = nn.Parameter(torch.cat([self.values, new_vals]))
            
            # Reset CSR cache
            self.crow_indices = None
            self.col_indices = None
            # Reset activity
            self.activity_in.zero_()
            self.activity_out.zero_()

    def prune(self, threshold=0.01):
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
        num_non_zero = max(num_non_zero, 1)
        
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

class NeuromodulatedHolographicBrain(nn.Module):
    """
    Hybrid H-NH-JEPA Architecture.
    """
    def __init__(self, input_size, hidden_size, output_size, genome=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # --- 1. Holographic Wavelet Encoder ---
        self.base_res = 16
        self.encoder_levels = nn.ModuleList([
            nn.Conv3d(1, 8, 3, padding=1, stride=2), 
            nn.Conv3d(8, 16, 3, padding=1, stride=2), 
            nn.Conv3d(16, 32, 3, padding=1, stride=2) 
        ])
        
        self.encoded_size = 32 * 2 * 2 * 2 # 256
        self.input_projection = nn.Linear(input_size, self.base_res**3)
        
        # --- 2. Hierarchical Dimensions ---
        r_size = hidden_size // 4
        s_size = hidden_size // 4
        c_size = hidden_size - (r_size + s_size)
        
        # --- 3. Sparse Hierarchical Core ---
        self.W_reflex = SparseLinear(self.encoded_size, r_size)
        self.R_reflex = SparseLinear(r_size, r_size)
        
        self.W_concept = SparseLinear(r_size, c_size)
        self.R_concept = SparseLinear(c_size, c_size)
        
        self.W_strategy = SparseLinear(c_size, s_size)
        self.R_strategy = SparseLinear(s_size, s_size)
        
        # --- 4. Latent Predictors (JEPA) ---
        self.P_reflex = SparseLinear(r_size, r_size)
        self.P_concept = SparseLinear(c_size, c_size)
        self.P_strategy = SparseLinear(s_size, s_size)
        
        # --- 5. Neuromodulated Gating ---
        self.meta_controller = nn.Sequential(
            nn.Linear(self.encoded_size + 4, 64),
            nn.Tanh(),
            nn.Linear(64, 3) 
        )
        
        # --- 6. Sparse MoE Routers ---
        self.router_reflex = nn.Linear(self.encoded_size, 64)
        self.router_concept = nn.Linear(r_size, 64)
        self.router_strategy = nn.Linear(c_size, 64)
        
        # --- 6. Temporal Dynamics ---
        self.tau_reflex = nn.Parameter(torch.rand(r_size) * 0.1 + 0.01)
        self.tau_concept = nn.Parameter(torch.rand(c_size) * 0.4 + 0.1)
        self.tau_strategy = nn.Parameter(torch.rand(s_size) * 9.0 + 1.0)
        
        # --- 7. Flash Head (System 1) ---
        self.flash_head = nn.Linear(r_size, output_size)
        self.flash_confidence = nn.Linear(r_size, 1)
        
        # --- 8. Selective Decoder (System 2) ---
        self.decoder = nn.Linear(hidden_size, output_size)
        self.intent_gate = nn.Linear(hidden_size, 1)
        self.critic = nn.Linear(hidden_size, 1)
        
        # State
        self.h_reflex = None
        self.h_concept = None
        self.h_strategy = None
        
        self._init_routing_specialization()
        
    def _init_routing_specialization(self):
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, SparseLinear):
                    num_blocks = module.out_features // 64
                    if num_blocks > 0:
                        for b in range(num_blocks):
                            mask = (module.indices[1] >= b*64) & (module.indices[1] < (b+1)*64)
                            module.values.data[mask] += torch.randn(mask.sum()) * 0.01 * (b / num_blocks)
        
    def forward(self, input_vector, dt=0.1, reward=0.0, chemicals=None, train_internal_rl=True):
        input_vector = input_vector.float()
        if chemicals is not None:
            chemicals = chemicals.float()
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        batch_size = input_vector.shape[0]
        
        # Derive sizes directly from parameters (Static Shape Philosophy)
        r_size = self.W_reflex.out_features
        c_size = self.W_concept.out_features
        s_size = self.W_strategy.out_features
        
        # Init States
        if self.h_reflex is None or self.h_reflex.shape[0] != batch_size or self.h_reflex.shape[1] != r_size:
            self.h_reflex = torch.zeros(batch_size, r_size, device=input_vector.device)
        if self.h_concept is None or self.h_concept.shape[0] != batch_size or self.h_concept.shape[1] != c_size:
            self.h_concept = torch.zeros(batch_size, c_size, device=input_vector.device)
        if self.h_strategy is None or self.h_strategy.shape[0] != batch_size or self.h_strategy.shape[1] != s_size:
            self.h_strategy = torch.zeros(batch_size, s_size, device=input_vector.device)
            
        # --- 1. Holographic Encoding ---
        x = self.input_projection(input_vector).view(batch_size, 1, self.base_res, self.base_res, self.base_res)
        curr = x
        for layer in self.encoder_levels:
            curr = F.relu(layer(curr))
        encoded_input = curr.flatten(1)
        
        # --- 2. Neuromodulated Gating ---
        if chemicals is None:
            chemicals = torch.tensor([0.5, 0.5, 0.2, 0.0], device=input_vector.device).repeat(batch_size, 1)
        
        meta_input = torch.cat([encoded_input, chemicals], dim=1)
        gates = torch.sigmoid(self.meta_controller(meta_input))
        g_reflex, g_concept, g_strategy = torch.chunk(gates, 3, dim=1)
        
        # --- 3. Hierarchical Update (Sparse) ---
        # Level 1: Reflex
        r_reflex_gate = torch.sigmoid(self.router_reflex(encoded_input))
        reflex_mask = r_reflex_gate.repeat_interleave(64, dim=1)[:, :r_size]
        
        sensory_reflex = self.W_reflex(encoded_input) * reflex_mask
        rec_reflex = self.R_reflex(self.h_reflex)
        target_reflex = torch.tanh(sensory_reflex + rec_reflex)
        self.h_reflex = self.h_reflex + g_reflex * (target_reflex - self.h_reflex) * (dt / self.tau_reflex)
        
        # Flash Head
        flash_actions = self.flash_head(self.h_reflex)
        confidence = torch.sigmoid(self.flash_confidence(self.h_reflex))
        
        # Level 2: Concept
        r_concept_gate = torch.sigmoid(self.router_concept(self.h_reflex))
        concept_mask = r_concept_gate.repeat_interleave(64, dim=1)[:, :c_size]
        
        sensory_concept = self.W_concept(self.h_reflex) * concept_mask
        rec_concept = self.R_concept(self.h_concept)
        target_concept = torch.tanh(sensory_concept + rec_concept)
        self.h_concept = self.h_concept + g_concept * (target_concept - self.h_concept) * (dt / self.tau_concept)
        
        # Level 3: Strategy
        r_strategy_gate = torch.sigmoid(self.router_strategy(self.h_concept))
        strategy_mask = r_strategy_gate.repeat_interleave(64, dim=1)[:, :s_size]
        
        sensory_strategy = self.W_strategy(self.h_concept) * strategy_mask
        rec_strategy = self.R_strategy(self.h_strategy)
        target_strategy = torch.tanh(sensory_strategy + rec_strategy)
        self.h_strategy = self.h_strategy + g_strategy * (target_strategy - self.h_strategy) * (dt / self.tau_strategy)
        
        # Stability: Clip hidden states
        self.h_reflex = torch.clamp(self.h_reflex, -5.0, 5.0)
        self.h_concept = torch.clamp(self.h_concept, -5.0, 5.0)
        self.h_strategy = torch.clamp(self.h_strategy, -5.0, 5.0)
        
        # --- 6. Latent Prediction (JEPA) ---
        p_reflex = self.P_reflex(self.h_reflex)
        p_concept = self.P_concept(self.h_concept)
        p_strategy = self.P_strategy(self.h_strategy)
        
        # --- 7. Selective Decoding ---
        full_h = torch.cat([self.h_reflex, self.h_concept, self.h_strategy], dim=1)
        actions = self.decoder(full_h)
        value = self.critic(full_h)
        energy = torch.mean(torch.abs(full_h))
        
        flash_data = (flash_actions, confidence, p_reflex, p_concept, p_strategy, self.h_reflex, self.h_concept, self.h_strategy)
        return actions, value, energy, flash_data

    def forward_onnx(self, input_vector, chemicals, h_reflex, h_concept, h_strategy, dt=0.1):
        """
        ONNX-friendly forward pass. Stateless and uses dense operations.
        """
        batch_size = input_vector.shape[0]
        r_size = self.W_reflex.out_features
        c_size = self.W_concept.out_features
        s_size = self.W_strategy.out_features

        # --- 1. Holographic Encoding ---
        x = self.input_projection(input_vector).view(batch_size, 1, self.base_res, self.base_res, self.base_res)
        curr = x
        for layer in self.encoder_levels:
            curr = F.relu(layer(curr))
        encoded_input = curr.flatten(1)
        
        # --- 2. Neuromodulated Gating ---
        if chemicals is None:
            chemicals = torch.tensor([0.5, 0.5, 0.2, 0.0], device=input_vector.device).repeat(batch_size, 1)
        
        meta_input = torch.cat([encoded_input, chemicals], dim=1)
        gates = torch.sigmoid(self.meta_controller(meta_input))
        g_reflex, g_concept, g_strategy = torch.chunk(gates, 3, dim=1)
        
        # --- 3. Hierarchical Update ---
        # Level 1: Reflex
        r_reflex_gate = torch.sigmoid(self.router_reflex(encoded_input))
        reflex_mask = r_reflex_gate.repeat_interleave(64, dim=1)[:, :r_size]
        
        sensory_reflex = self.W_reflex.forward_onnx(encoded_input) * reflex_mask
        rec_reflex = self.R_reflex.forward_onnx(h_reflex)
        target_reflex = torch.tanh(sensory_reflex + rec_reflex)
        next_h_reflex = h_reflex + g_reflex * (target_reflex - h_reflex) * (dt / self.tau_reflex)
        
        # Flash Head
        flash_actions = self.flash_head(next_h_reflex)
        confidence = torch.sigmoid(self.flash_confidence(next_h_reflex))
        
        # Level 2: Concept
        r_concept_gate = torch.sigmoid(self.router_concept(next_h_reflex))
        concept_mask = r_concept_gate.repeat_interleave(64, dim=1)[:, :c_size]
        
        sensory_concept = self.W_concept.forward_onnx(next_h_reflex) * concept_mask
        rec_concept = self.R_concept.forward_onnx(h_concept)
        target_concept = torch.tanh(sensory_concept + rec_concept)
        next_h_concept = h_concept + g_concept * (target_concept - h_concept) * (dt / self.tau_concept)
        
        # Level 3: Strategy
        r_strategy_gate = torch.sigmoid(self.router_strategy(next_h_concept))
        strategy_mask = r_strategy_gate.repeat_interleave(64, dim=1)[:, :s_size]
        
        sensory_strategy = self.W_strategy.forward_onnx(next_h_concept) * strategy_mask
        rec_strategy = self.R_strategy.forward_onnx(h_strategy)
        target_strategy = torch.tanh(sensory_strategy + rec_strategy)
        next_h_strategy = h_strategy + g_strategy * (target_strategy - h_strategy) * (dt / self.tau_strategy)
        
        # --- 6. Latent Prediction (JEPA) ---
        p_reflex = self.P_reflex.forward_onnx(next_h_reflex)
        p_concept = self.P_concept.forward_onnx(next_h_concept)
        p_strategy = self.P_strategy.forward_onnx(next_h_strategy)
        
        # --- 7. Selective Decoding ---
        full_h = torch.cat([next_h_reflex, next_h_concept, next_h_strategy], dim=1)
        actions = self.decoder(full_h)
        value = self.critic(full_h)
        energy = torch.mean(torch.abs(full_h))
        
        return actions, value, energy, (flash_actions, confidence, p_reflex, p_concept, p_strategy, next_h_reflex, next_h_concept, next_h_strategy), (next_h_reflex, next_h_concept, next_h_strategy)

    def reset_state(self):
        self.h_reflex = None
        self.h_concept = None
        self.h_strategy = None

    def detach_state(self):
        if self.h_reflex is not None: self.h_reflex = self.h_reflex.detach()
        if self.h_concept is not None: self.h_concept = self.h_concept.detach()
        if self.h_strategy is not None: self.h_strategy = self.h_strategy.detach()

    def resize_hidden(self, new_hidden_size):
        if new_hidden_size == self.hidden_size: return
        print(f"NeuromodulatedHolographicBrain: Resizing Hidden {self.hidden_size} -> {new_hidden_size}")
        self.hidden_size = new_hidden_size
        r_size = new_hidden_size // 4
        s_size = new_hidden_size // 4
        c_size = new_hidden_size - (r_size + s_size)
        
        device = next(self.parameters()).device
        
        self.W_reflex.resize(out_features=r_size)
        self.R_reflex.resize(in_features=r_size, out_features=r_size)
        self.W_concept.resize(in_features=r_size, out_features=c_size)
        self.R_concept.resize(in_features=c_size, out_features=c_size)
        self.W_strategy.resize(in_features=c_size, out_features=s_size)
        self.R_strategy.resize(in_features=s_size, out_features=s_size)
        
        self.P_reflex.resize(in_features=r_size, out_features=r_size)
        self.P_concept.resize(in_features=c_size, out_features=c_size)
        self.P_strategy.resize(in_features=s_size, out_features=s_size)
        
        self.tau_reflex = nn.Parameter(torch.rand(r_size, device=device) * 0.1 + 0.01)
        self.tau_concept = nn.Parameter(torch.rand(c_size, device=device) * 0.4 + 0.1)
        self.tau_strategy = nn.Parameter(torch.rand(s_size, device=device) * 9.0 + 1.0)
        
        self.router_reflex = nn.Linear(self.encoded_size, 64).to(device)
        self.router_concept = nn.Linear(r_size, 64).to(device)
        self.router_strategy = nn.Linear(c_size, 64).to(device)
        
        self.flash_head = nn.Linear(r_size, self.output_size).to(device)
        self.flash_confidence = nn.Linear(r_size, 1).to(device)
        self.decoder = nn.Linear(new_hidden_size, self.output_size).to(device)
        self.intent_gate = nn.Linear(new_hidden_size, 1).to(device)
        self.critic = nn.Linear(new_hidden_size, 1).to(device)
        
        self.reset_state()

    def resize_input(self, new_input_size):
        if new_input_size == self.input_size: return
        print(f"NeuromodulatedHolographicBrain: Resizing Input {self.input_size} -> {new_input_size}")
        self.input_size = new_input_size
        self.input_projection = nn.Linear(new_input_size, self.base_res**3)
        self.reset_state()

    def learn_trajectory(self, input_sequence, target_h_sequence, lr=0.001):
        """
        Supervised State Imprinting (SSI) for NeuromodulatedHolographicBrain.
        Trains the hierarchical states to match a teacher trajectory.
        
        input_sequence: [T, Batch, input_size]
        target_h_sequence: [T, Batch, hidden_size]
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        total_loss = 0.0
        
        # Derive sizes
        r_size = self.W_reflex.out_features
        s_size = self.W_strategy.out_features
        c_size = self.W_concept.out_features
        
        self.reset_state()
        for t in range(len(input_sequence)):
            optimizer.zero_grad()
            
            # Forward pass (single step)
            actions, value, energy, flash_data = self.forward(input_sequence[t])
            flash, conf, p_r, p_c, p_s, _, _, _ = flash_data
            
            # Combine current states
            current_h = torch.cat([self.h_reflex, self.h_concept, self.h_strategy], dim=1)
            
            # Target for this step
            target_h = target_h_sequence[t]
            
            # Loss: Match the full hidden state
            loss = F.mse_loss(current_h, target_h)
            
            # Also match JEPA predictions if possible (Self-Consistency)
            if t < len(target_h_sequence) - 1:
                next_target = target_h_sequence[t+1]
                t_r, t_c, t_s = torch.split(next_target, [r_size, c_size, s_size], dim=1)
                loss += 0.1 * (F.mse_loss(p_r, t_r) + F.mse_loss(p_c, t_c) + F.mse_loss(p_s, t_s))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            self.detach_state() # Keep it recurrent but bounded
            
        return total_loss / len(input_sequence)
