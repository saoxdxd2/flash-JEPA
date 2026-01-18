"""
Neuromodulated Holographic Brain: Hybrid H-NH-JEPA Architecture

This module implements a hierarchical neural architecture with:
- Holographic wavelet encoding for spatial perception
- Sparse linear layers for CPU-efficient computation
- Neuromodulated gating for chemical state integration
- Multi-level temporal dynamics (reflex → concept → strategy)
- Flash path (System 1) for rapid responses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# === CONFIGURATION CONSTANTS ===
# These control the architecture and behavior of the holographic brain

# Sparse Linear Layer Defaults
DEFAULT_SPARSITY = 0.99          # Default connection sparsity (99% sparse)
MIN_NON_ZERO_CONNECTIONS = 1     # Minimum connections to maintain
SPROUTING_INIT_SCALE = 0.01     # Scale for new connection initialization
PRUNE_THRESHOLD = 0.01          # Weight magnitude threshold for pruning

# Holographic Encoder
ENCODER_BASE_RESOLUTION = 16    # Base resolution for 3D encoding
ENCODER_CHANNELS = [8, 16, 32]  # Channels at each encoding level
ENCODED_SIZE = 256              # 32 * 2 * 2 * 2 = 256

# Hidden State Clipping (for stability)
HIDDEN_STATE_CLIP_MIN = -5.0
HIDDEN_STATE_CLIP_MAX = 5.0

# Temporal Dynamics (time constant ranges)
TAU_REFLEX_RANGE = (0.01, 0.11)   # Fast reflexes: 10-110ms
TAU_CONCEPT_RANGE = (0.1, 0.5)   # Medium concepts: 100-500ms
TAU_STRATEGY_RANGE = (1.0, 10.0) # Slow strategies: 1-10s

# Router Block Size
ROUTER_BLOCK_SIZE = 64

# Default chemical state (dopamine, serotonin, norepinephrine, cortisol)
DEFAULT_CHEMICALS = [0.5, 0.5, 0.2, 0.0]

# SSI Learning
SSI_JEPA_CONSISTENCY_WEIGHT = 0.1

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

class HierarchicalBlock(nn.Module):
    """
    Encapsulates a single level of the hierarchy (Sensory, Recurrent, Predictor, Router).
    """
    def __init__(self, in_size, hidden_size, tau_range, router_blocks=64):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.router_blocks = router_blocks
        
        self.W = SparseLinear(in_size, hidden_size)
        self.R = SparseLinear(hidden_size, hidden_size)
        self.P = SparseLinear(hidden_size, hidden_size)
        
        self.router = nn.Linear(in_size, router_blocks)
        
        # Time constants
        min_tau, max_tau = tau_range
        self.tau = nn.Parameter(torch.rand(hidden_size) * (max_tau - min_tau) + min_tau)
        
    def forward(self, x, h_prev, gate, dt=0.1):
        # Router
        r_gate = torch.sigmoid(self.router(x))
        mask = r_gate.repeat_interleave(64, dim=1)[:, :self.hidden_size]
        
        # Dynamics
        sensory = self.W(x) * mask
        rec = self.R(h_prev)
        target = torch.tanh(sensory + rec)
        
        # Leaky Update
        h_new = h_prev + gate * (target - h_prev) * (dt / self.tau)
        
        # Prediction
        pred = self.P(h_new)
        
        return h_new, pred
        
    def forward_onnx(self, x, h_prev, gate, dt=0.1):
        # Router
        r_gate = torch.sigmoid(self.router(x))
        mask = r_gate.repeat_interleave(64, dim=1)[:, :self.hidden_size]
        
        # Dynamics (Dense for ONNX)
        sensory = self.W.forward_onnx(x) * mask
        rec = self.R.forward_onnx(h_prev)
        target = torch.tanh(sensory + rec)
        
        h_new = h_prev + gate * (target - h_prev) * (dt / self.tau)
        pred = self.P.forward_onnx(h_new)
        
        return h_new, pred

    def resize(self, in_size=None, hidden_size=None):
        if in_size is not None: self.in_size = in_size
        if hidden_size is not None: self.hidden_size = hidden_size
        
        self.W.resize(in_features=self.in_size, out_features=self.hidden_size)
        self.R.resize(in_features=self.hidden_size, out_features=self.hidden_size)
        self.P.resize(in_features=self.hidden_size, out_features=self.hidden_size)
        
        device = self.tau.device
        if len(self.tau) != self.hidden_size:
            # Resize Tau (Random re-init for simplicity on resize, or preserve?)
            # Preserving is better but complex. For now, re-init consistent with constructor.
            # Assuming resize happens rarely and mostly during growth.
            self.tau = nn.Parameter(torch.rand(self.hidden_size, device=device) * 0.1 + 0.01) # Default range, should pass in range
            
        if self.router.in_features != self.in_size:
            self.router = nn.Linear(self.in_size, self.router_blocks).to(device)


class NeuromodulatedHolographicBrain(nn.Module):
    """
    Hybrid H-NH-JEPA Architecture.
    Refactored for conciseness using HierarchicalBlock.
    """
    def __init__(self, input_size, hidden_size, output_size, genome=None, memory_size=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memory_size = memory_size
        
        # --- 1. Holographic Wavelet Encoder ---
        self.base_res = 16
        self.encoder_levels = nn.ModuleList([
            nn.Conv3d(1, 8, 3, padding=1, stride=2), 
            nn.Conv3d(8, 16, 3, padding=1, stride=2), 
            nn.Conv3d(16, 32, 3, padding=1, stride=2) 
        ])
        
        self.encoded_size = 32 * 2 * 2 * 2 # 256
        self.input_projection = nn.Linear(input_size, self.base_res**3)
        
        # --- 1.5 Thalamic Gate ---
        if self.memory_size is not None and self.memory_size > 0:
            self.memory_projection = nn.Linear(self.memory_size, self.encoded_size)
            self.thalamus_gate = nn.Sequential(
                nn.Linear(self.encoded_size * 2 + 4, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.memory_projection = None
            self.thalamus_gate = None
        
        # --- 2. Hierarchical Dimensions ---
        r_size = hidden_size // 4
        s_size = hidden_size // 4
        c_size = hidden_size - (r_size + s_size)
        
        # --- 3. Hierarchical Blocks ---
        self.reflex = HierarchicalBlock(self.encoded_size, r_size, TAU_REFLEX_RANGE)
        self.concept = HierarchicalBlock(r_size, c_size, TAU_CONCEPT_RANGE)
        self.strategy = HierarchicalBlock(c_size, s_size, TAU_STRATEGY_RANGE)
        
        # --- 4. Neuromodulated Gating ---
        self.meta_controller = nn.Sequential(
            nn.Linear(self.encoded_size + 4, 64),
            nn.Tanh(),
            nn.Linear(64, 3) 
        )
        
        # --- 5. Heads ---
        self.flash_head = nn.Linear(r_size, output_size)
        self.flash_confidence = nn.Linear(r_size, 1)
        
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
            for module in [self.reflex.W, self.concept.W, self.strategy.W]:
                num_blocks = module.out_features // 64
                if num_blocks > 0:
                    for b in range(num_blocks):
                        mask = (module.indices[1] >= b*64) & (module.indices[1] < (b+1)*64)
                        module.values.data[mask] += torch.randn(mask.sum()) * 0.01 * (b / num_blocks)
        
    def forward(self, input_vector, dt=0.1, reward=0.0, chemicals=None, train_internal_rl=True, memory_input=None):
        input_vector = input_vector.float()
        if chemicals is not None: chemicals = chemicals.float()
        if memory_input is not None: memory_input = memory_input.float()
        if input_vector.dim() == 1: input_vector = input_vector.unsqueeze(0)
        batch_size = input_vector.shape[0]
        
        if chemicals is not None:
            if chemicals.dim() == 1:
                chemicals = chemicals.unsqueeze(0)
            if chemicals.shape[0] != batch_size:
                chemicals = chemicals.repeat(batch_size, 1)
        
        # Init States
        if self.h_reflex is None or self.h_reflex.shape[0] != batch_size:
            self.h_reflex = torch.zeros(batch_size, self.reflex.hidden_size, device=input_vector.device)
            self.h_concept = torch.zeros(batch_size, self.concept.hidden_size, device=input_vector.device)
            self.h_strategy = torch.zeros(batch_size, self.strategy.hidden_size, device=input_vector.device)
            
        # --- 1. Holographic Encoding ---
        x = self.input_projection(input_vector).view(batch_size, 1, self.base_res, self.base_res, self.base_res)
        curr = x
        for layer in self.encoder_levels:
            curr = F.relu(layer(curr))
        encoded_input = curr.flatten(1)
        
        # --- 1.5 Thalamic Gating ---
        if self.thalamus_gate is not None and memory_input is not None:
            if memory_input.dim() == 1: memory_input = memory_input.unsqueeze(0)
            encoded_memory = self.memory_projection(memory_input)
            
            chem_vec = chemicals if chemicals is not None else torch.tensor([0.5, 0.5, 0.2, 0.0], device=input_vector.device).repeat(batch_size, 1)
            
            gate_input = torch.cat([encoded_input, encoded_memory, chem_vec], dim=1)
            alpha = self.thalamus_gate(gate_input)
            
            encoded_input = (alpha * encoded_input) + ((1.0 - alpha) * encoded_memory)
            if self.training:
                encoded_input += torch.randn_like(encoded_input) * 0.01
        
        # --- 2. Neuromodulated Gating ---
        if chemicals is None:
            chemicals = torch.tensor([0.5, 0.5, 0.2, 0.0], device=input_vector.device).repeat(batch_size, 1)
        
        meta_input = torch.cat([encoded_input, chemicals], dim=1)
        gates = torch.sigmoid(self.meta_controller(meta_input))
        g_reflex, g_concept, g_strategy = torch.chunk(gates, 3, dim=1)
        
        # --- 3. Hierarchical Update ---
        self.h_reflex, p_reflex = self.reflex(encoded_input, self.h_reflex, g_reflex, dt)
        
        # Flash Head
        flash_actions = self.flash_head(self.h_reflex)
        confidence = torch.sigmoid(self.flash_confidence(self.h_reflex))
        
        self.h_concept, p_concept = self.concept(self.h_reflex, self.h_concept, g_concept, dt)
        self.h_strategy, p_strategy = self.strategy(self.h_concept, self.h_strategy, g_strategy, dt)
        
        # Stability: Clip hidden states
        self.h_reflex = torch.clamp(self.h_reflex, HIDDEN_STATE_CLIP_MIN, HIDDEN_STATE_CLIP_MAX)
        self.h_concept = torch.clamp(self.h_concept, HIDDEN_STATE_CLIP_MIN, HIDDEN_STATE_CLIP_MAX)
        self.h_strategy = torch.clamp(self.h_strategy, HIDDEN_STATE_CLIP_MIN, HIDDEN_STATE_CLIP_MAX)
        
        # --- 4. Decoding ---
        full_h = torch.cat([self.h_reflex, self.h_concept, self.h_strategy], dim=1)
        actions = self.decoder(full_h)
        value = self.critic(full_h)
        energy = torch.mean(torch.abs(full_h))
        
        flash_data = (flash_actions, confidence, p_reflex, p_concept, p_strategy, self.h_reflex, self.h_concept, self.h_strategy)
        return actions, value, energy, flash_data

    def forward_onnx(self, input_vector, chemicals, h_reflex, h_concept, h_strategy, dt=0.1):
        batch_size = input_vector.shape[0]

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
        next_h_reflex, p_reflex = self.reflex.forward_onnx(encoded_input, h_reflex, g_reflex, dt)
        
        flash_actions = self.flash_head(next_h_reflex)
        confidence = torch.sigmoid(self.flash_confidence(next_h_reflex))
        
        next_h_concept, p_concept = self.concept.forward_onnx(next_h_reflex, h_concept, g_concept, dt)
        next_h_strategy, p_strategy = self.strategy.forward_onnx(next_h_concept, h_strategy, g_strategy, dt)
        
        # --- 4. Decoding ---
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
        
        self.reflex.resize(hidden_size=r_size)
        self.concept.resize(in_size=r_size, hidden_size=c_size)
        self.strategy.resize(in_size=c_size, hidden_size=s_size)
        
        device = next(self.parameters()).device
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
        self.input_projection = nn.Linear(new_input_size, self.base_res**3).to(next(self.parameters()).device)
        self.reset_state()

    def learn_trajectory(self, input_sequence, target_h_sequence, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        total_loss = 0.0
        
        r_size = self.reflex.hidden_size
        c_size = self.concept.hidden_size
        s_size = self.strategy.hidden_size
        
        self.reset_state()
        for t in range(len(input_sequence)):
            optimizer.zero_grad()
            
            actions, value, energy, flash_data = self.forward(input_sequence[t])
            flash, conf, p_r, p_c, p_s, _, _, _ = flash_data
            
            current_h = torch.cat([self.h_reflex, self.h_concept, self.h_strategy], dim=1)
            target_h = target_h_sequence[t]
            
            loss = F.mse_loss(current_h, target_h)
            
            if t < len(target_h_sequence) - 1:
                next_target = target_h_sequence[t+1]
                t_r, t_c, t_s = torch.split(next_target, [r_size, c_size, s_size], dim=1)
                loss += SSI_JEPA_CONSISTENCY_WEIGHT * (F.mse_loss(p_r, t_r) + F.mse_loss(p_c, t_c) + F.mse_loss(p_s, t_s))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            self.detach_state()
            
        return total_loss / len(input_sequence)
