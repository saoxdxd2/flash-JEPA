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
from brain.modules.sparse_layers import SparseLinear, DEFAULT_SPARSITY, MIN_NON_ZERO_CONNECTIONS, SPROUTING_INIT_SCALE, PRUNE_THRESHOLD
from brain.modules.neural_vm import NeuralVirtualizationLayer

# === CONFIGURATION CONSTANTS ===
# These control the architecture and behavior of the holographic brain

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
# SSI Learning
SSI_JEPA_CONSISTENCY_WEIGHT = 0.1

class HierarchicalBlock(nn.Module):
    """
    Encapsulates a single level of the hierarchy, implemented as a Cluster of Neural Processing Units (NPUs).
    """
    def __init__(self, in_size, hidden_size, tau_range, router_blocks=64):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        
        # Neural Virtualization Layer (The Cluster)
        # We default to 64 max NPUs per block for now.
        self.vm = NeuralVirtualizationLayer(in_size, hidden_size, max_npus=64, memory_size=64)
        
        # Predictor (Still useful for forward prediction)
        self.P = nn.Linear(hidden_size, hidden_size)
        
        # Time constants (Managed by VM implicitly via gating, but we keep explicit tau for leaky integration of the block output)
        min_tau, max_tau = tau_range
        self.tau = nn.Parameter(torch.rand(hidden_size) * (max_tau - min_tau) + min_tau)
        
        # State placeholders
        self.memory_state = None
        
    def forward(self, x, h_prev, gate, dt=0.1):
        # x: [Batch, In]
        # h_prev: [Batch, Hidden]
        
        # Run the Neural VM
        # vm_out: [Batch, Hidden] - Aggregated output of NPUs
        # h_new_internal: [Batch, Hidden] - New register states
        # mem_new: [Batch, Mem_Size, Mem_Dim] - New RAM state
        
        if self.memory_state is None or self.memory_state.shape[0] != x.shape[0]:
             self.memory_state = torch.zeros(x.shape[0], self.vm.memory_size, self.vm.memory_dim, device=x.device)
             
        vm_out, h_new_internal, self.memory_state = self.vm(x, h_prev, self.memory_state)
        
        # Leaky Update? 
        # The VM has its own dynamics. We can apply a global gate to the *output*.
        # Let's say the block output is a leaky integration of the VM output.
        # But we need to return the *state* for the next step.
        # The state is h_new_internal (3D).
        
        # Prediction uses the aggregated output (2D)
        pred = self.P(vm_out)
        
        # Return (State_3D, Output_2D, Prediction)
        # But signature is h_new, pred.
        # We'll return h_new_internal as h_new.
        # But we also need the 2D output for the heads in the Brain.
        # Let's return h_new_internal (State) and pack (vm_out, pred) as the second return?
        # Or change signature.
        
        return h_new_internal, vm_out, pred
        
    def forward_onnx(self, x, h_prev, gate, dt=0.1):
        # Simplified for ONNX: Just run VM and update
        if self.memory_state is None:
             self.memory_state = torch.zeros(x.shape[0], self.vm.memory_size, self.vm.memory_dim, device=x.device)
             
        vm_out, h_new_internal, self.memory_state = self.vm(x, h_prev, self.memory_state)
        h_new = h_prev + gate * (vm_out - h_prev) * (dt / self.tau)
        pred = self.P(h_new)
        return h_new, pred

    def resize(self, in_size=None, hidden_size=None):
        """Resizes the block while preserving existing weights."""
        if in_size is None: in_size = self.in_size
        if hidden_size is None: hidden_size = self.hidden_size
        
        if in_size == self.in_size and hidden_size == self.hidden_size:
            return
            
        print(f"HierarchicalBlock: Resizing {self.in_size} -> {in_size}, {self.hidden_size} -> {hidden_size}")
        
        # Resize VM (NeuralVirtualizationLayer needs a resize method)
        if hasattr(self.vm, 'resize'):
            self.vm.resize(in_size, hidden_size)
        else:
            # Fallback: Re-init but this causes amnesia in VM
            self.vm = NeuralVirtualizationLayer(in_size, hidden_size, max_npus=self.vm.max_npus, memory_size=self.vm.memory_size).to(next(self.parameters()).device)
            
        # Resize Predictor (Preserve weights)
        old_P = self.P
        self.P = nn.Linear(hidden_size, hidden_size).to(next(self.parameters()).device)
        with torch.no_grad():
            min_h = min(self.hidden_size, hidden_size)
            self.P.weight[:min_h, :min_h] = old_P.weight[:min_h, :min_h]
            self.P.bias[:min_h] = old_P.bias[:min_h]
            
        # Resize Tau
        old_tau = self.tau
        self.tau = nn.Parameter(torch.rand(hidden_size, device=old_tau.device) * 0.1 + 0.01)
        with torch.no_grad():
            min_h = min(self.hidden_size, hidden_size)
            self.tau[:min_h] = old_tau[:min_h]
            
        self.in_size = in_size
        self.hidden_size = hidden_size

    def detach_state(self):
        """Detach internal states from the computation graph."""
        if self.memory_state is not None:
            self.memory_state = self.memory_state.detach()


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
        
        # self._init_routing_specialization() # Removed as we use Neural VM now
        
    # def _init_routing_specialization(self):
    #     with torch.no_grad():
    #         for module in [self.reflex.W, self.concept.W, self.strategy.W]:
    #             num_blocks = module.out_features // 64
    #             if num_blocks > 0:
    #                 for b in range(num_blocks):
    #                     mask = (module.indices[1] >= b*64) & (module.indices[1] < (b+1)*64)
    #                     module.values.data[mask] += torch.randn(mask.sum()) * 0.01 * (b / num_blocks)
        
        with torch.no_grad():
            for module in [self.reflex.W, self.concept.W, self.strategy.W]:
                # Handle SparseQuaternionLinear
                sub_modules = [module.r_weight, module.i_weight, module.j_weight, module.k_weight]
                
                for sub in sub_modules:
                    num_blocks = sub.out_features // 64
                    if num_blocks > 0:
                        for b in range(num_blocks):
                            mask = (sub.indices[1] >= b*64) & (sub.indices[1] < (b+1)*64)
                            sub.values.data[mask] += torch.randn(mask.sum()) * 0.01 * (b / num_blocks)
        
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
            # [Batch, Max_NPUs, Reg_Size]
            self.h_reflex = torch.zeros(batch_size, self.reflex.vm.max_npus, self.reflex.vm.npu_register_size, device=input_vector.device)
            self.h_concept = torch.zeros(batch_size, self.concept.vm.max_npus, self.concept.vm.npu_register_size, device=input_vector.device)
            self.h_strategy = torch.zeros(batch_size, self.strategy.vm.max_npus, self.strategy.vm.npu_register_size, device=input_vector.device)
            
        # --- 1. Holographic Encoding ---
        # Project to base_res^3
        x = self.input_projection(input_vector)
        # Reshape to [Batch, 1, D, H, W]
        x = x.view(batch_size, 1, self.base_res, self.base_res, self.base_res)
        
        curr = x
        for layer in self.encoder_levels:
            curr = F.relu(layer(curr))
        
        # [Batch, Encoded_Size]
        encoded_flat = curr.flatten(1)
        encoded_input = encoded_flat
        
        # --- 1.5 Thalamic Gating ---
        if self.thalamus_gate is not None and memory_input is not None:
            if memory_input.dim() == 1: memory_input = memory_input.unsqueeze(0)
            # Project memory to [Batch, Encoded_Size * 4]
            encoded_memory_flat = self.memory_projection(memory_input)
            # Reshape to [Batch, Encoded_Size, 4]
            encoded_memory = encoded_memory_flat.view(batch_size, self.encoded_size, 4)
            
            chem_vec = chemicals if chemicals is not None else torch.tensor([0.5, 0.5, 0.2, 0.0], device=input_vector.device).repeat(batch_size, 1)
            
            # Flatten inputs for gate: [Batch, Encoded_Size*4 + Encoded_Size*4 + 4]
            gate_input = torch.cat([encoded_input.reshape(batch_size, -1), encoded_memory.reshape(batch_size, -1), chem_vec], dim=1)
            alpha = self.thalamus_gate(gate_input).unsqueeze(-1) # Broadcast to 4D
            
            encoded_input = (alpha * encoded_input) + ((1.0 - alpha) * encoded_memory)
            if self.training:
                encoded_input += torch.randn_like(encoded_input) * 0.01
        
        # --- 2. Neuromodulated Gating ---
        if chemicals is None:
            chemicals = torch.tensor([0.5, 0.5, 0.2, 0.0], device=input_vector.device).repeat(batch_size, 1)
        
        # Meta controller takes flattened input
        meta_input = torch.cat([encoded_input.reshape(batch_size, -1), chemicals], dim=1)
        gates = torch.sigmoid(self.meta_controller(meta_input))
        g_reflex, g_concept, g_strategy = torch.chunk(gates, 3, dim=1)
        
        # --- 3. Hierarchical Update ---
        # Block returns: State(3D), Output(2D), Prediction
        self.h_reflex, reflex_out, p_reflex = self.reflex(encoded_input, self.h_reflex, g_reflex, dt)
        
        # Flash Head uses 2D output
        flash_actions = self.flash_head(reflex_out)
        confidence = torch.sigmoid(self.flash_confidence(reflex_out))
        
        self.h_concept, concept_out, p_concept = self.concept(reflex_out, self.h_concept, g_concept, dt)
        self.h_strategy, strategy_out, p_strategy = self.strategy(concept_out, self.h_strategy, g_strategy, dt)
        
        # Stability: Clip hidden states (3D)
        self.h_reflex = torch.clamp(self.h_reflex, HIDDEN_STATE_CLIP_MIN, HIDDEN_STATE_CLIP_MAX)
        self.h_concept = torch.clamp(self.h_concept, HIDDEN_STATE_CLIP_MIN, HIDDEN_STATE_CLIP_MAX)
        self.h_strategy = torch.clamp(self.h_strategy, HIDDEN_STATE_CLIP_MIN, HIDDEN_STATE_CLIP_MAX)
        
        # --- 4. Decoding ---
        # Use 2D outputs for decoding
        full_h = torch.cat([reflex_out, concept_out, strategy_out], dim=1)
        
        actions = self.decoder(full_h)
        value = self.critic(full_h)
        energy = torch.mean(torch.abs(full_h))
        
        flash_data = (flash_actions, confidence, p_reflex, p_concept, p_strategy, self.h_reflex, self.h_concept, self.h_strategy)
        return actions, value, energy, flash_data

    def forward_onnx(self, input_vector, chemicals, h_reflex, h_concept, h_strategy, dt=0.1):
        batch_size = input_vector.shape[0]

        # --- 1. Holographic Encoding ---
        x = self.input_projection(input_vector)
        x = x.view(batch_size, 1, self.base_res, self.base_res, self.base_res)
        
        curr = x
        for layer in self.encoder_levels:
            curr = F.relu(layer(curr))
            
        encoded_flat = curr.flatten(1)
        encoded_input = encoded_flat
        
        # --- 2. Neuromodulated Gating ---
        if chemicals is None:
            chemicals = torch.tensor([0.5, 0.5, 0.2, 0.0], device=input_vector.device).repeat(batch_size, 1)
        
        meta_input = torch.cat([encoded_input.reshape(batch_size, -1), chemicals], dim=1)
        gates = torch.sigmoid(self.meta_controller(meta_input))
        g_reflex, g_concept, g_strategy = torch.chunk(gates, 3, dim=1)
        
        # --- 3. Hierarchical Update ---
        next_h_reflex, p_reflex = self.reflex.forward_onnx(encoded_input, h_reflex, g_reflex, dt)
        
        flash_actions = self.flash_head(next_h_reflex.reshape(batch_size, -1))
        confidence = torch.sigmoid(self.flash_confidence(next_h_reflex.reshape(batch_size, -1)))
        
        next_h_concept, p_concept = self.concept.forward_onnx(next_h_reflex, h_concept, g_concept, dt)
        next_h_strategy, p_strategy = self.strategy.forward_onnx(next_h_concept, h_strategy, g_strategy, dt)
        
        # --- 4. Decoding ---
        full_h = torch.cat([next_h_reflex, next_h_concept, next_h_strategy], dim=1)
        full_h_flat = full_h.reshape(batch_size, -1)
        
        actions = self.decoder(full_h_flat)
        value = self.critic(full_h_flat)
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
        """Resizes the brain's hidden layers while preserving weights (Anti-Amnesia)."""
        if new_hidden_size == self.hidden_size: return
        print(f"NeuromodulatedHolographicBrain: Resizing Hidden {self.hidden_size} -> {new_hidden_size}")
        
        device = next(self.parameters()).device
        
        # Recalculate hierarchical sizes
        r_size = new_hidden_size // 4
        s_size = new_hidden_size // 4
        c_size = new_hidden_size - (r_size + s_size)
        
        # Resize Blocks
        self.reflex.resize(hidden_size=r_size)
        self.concept.resize(in_size=r_size, hidden_size=c_size)
        self.strategy.resize(in_size=c_size, hidden_size=s_size)
        
        # Resize Heads (Preserve weights)
        def resize_linear(layer, in_dim, out_dim):
            old_layer = layer
            new_layer = nn.Linear(in_dim, out_dim).to(device)
            with torch.no_grad():
                m_in = min(old_layer.in_features, in_dim)
                m_out = min(old_layer.out_features, out_dim)
                new_layer.weight[:m_out, :m_in] = old_layer.weight[:m_out, :m_in]
                new_layer.bias[:m_out] = old_layer.bias[:m_out]
            return new_layer

        self.flash_head = resize_linear(self.flash_head, r_size, self.output_size)
        self.flash_confidence = resize_linear(self.flash_confidence, r_size, 1)
        self.decoder = resize_linear(self.decoder, new_hidden_size, self.output_size)
        self.intent_gate = resize_linear(self.intent_gate, new_hidden_size, 1)
        self.critic = resize_linear(self.critic, new_hidden_size, 1)
        
        self.hidden_size = new_hidden_size
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
            
        return total_loss / len(input_sequence)

    def detach_state(self):
        """Detach all internal states to truncate BPTT."""
        if self.h_reflex is not None: self.h_reflex = self.h_reflex.detach()
        if self.h_concept is not None: self.h_concept = self.h_concept.detach()
        if self.h_strategy is not None: self.h_strategy = self.h_strategy.detach()
        
        self.reflex.detach_state()
        self.concept.detach_state()
        self.strategy.detach_state()
