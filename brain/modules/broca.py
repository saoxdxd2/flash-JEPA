import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from models.liquid_vectorized import VectorizedLiquidGraph
from brain.modules.neural_memory import TitansMemory

try:
    from models.liquid import LiquidGraph
except ImportError:
    LiquidGraph = None

class BrocaModule(nn.Module):
    """
    Broca's Area (Language Center).
    Handles Semantic Understanding and Language Generation.
    
    Current Capabilities:
    - Vocabulary: Basic Survival Concepts.
    - Perception: Simulates reading by mapping High Text Density -> Random Concept.
      (Placeholder for future OCR).
    - Embedding: Maps words to 256-dim vectors (compatible with Latent Space).
    - Core: Uses Mixture of Experts (MoE) with Liquid Neural Networks (SNN) for biological plausibility.
      This allows the model to scale in knowledge (experts) without increasing inference cost.
    - Memory: Titans Neural Memory (Surprise-based learning).
    """
    def __init__(self, embedding_dim=None, visual_dim=None, num_experts=8, active_experts=1, genome=None):
        super().__init__()
        self.genome = genome
        # Use genomic latent_dim if available, otherwise fallback to 256
        self.embedding_dim = embedding_dim if embedding_dim is not None else (getattr(genome, 'latent_dim', 256))
        self.visual_dim = visual_dim if visual_dim is not None else self.embedding_dim
        self.num_experts = num_experts
        self.active_experts = active_experts
        
        # 1. Visual Word Form Area (VWFA) Simulation
        # Replaced Linear with Conv2d for Natural OCR (2D Topology)
        # Input: [Batch, 3, 64, 64] -> Output: [Batch, embedding_dim, 4, 4]
        self.visual_projection = nn.Conv2d(3, self.embedding_dim, kernel_size=14, stride=14)
        
        # Adapter to flatten 2D features to Semantic Vector
        self.visual_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), # Force 4x4 grid regardless of input resolution
            nn.Flatten(),
            nn.Linear(self.embedding_dim * 4 * 4, self.embedding_dim),
            nn.Tanh()
        )
        
        # 2. Text Processing (Wernicke's Area)
        # Character Embedding: 128 ASCII chars -> 64 dim
        self.char_embedding = nn.Embedding(128, 64)
        
        # Gating Network: Decides which expert to use based on input
        # Input: 64 (Char Embedding) -> Output: num_experts logits
        self.gate = nn.Linear(64, num_experts)
        
        # Sequence Encoder: Mixture of Liquid Neural Networks (SNN)
        # Input: 64 (Char Embedding)
        # Hidden: 512 (Reservoir) per expert
        # Output: 256 (Semantic Vector)
        self.experts = nn.ModuleList([
            VectorizedLiquidGraph(input_size=64, hidden_size=512, output_size=self.embedding_dim)
            for _ in range(num_experts)
        ])
        
        # Input: 256 (Semantic) -> Output: 256 (Predicted Next Semantic)
        self.titans = TitansMemory(input_dim=self.embedding_dim, hidden_dim=self.embedding_dim)
        
        # 4. Vocabulary (Dynamic - No hardcoded limits)
        # We rely on the SNN to learn representations for any text sequence.
        self.vocab = []
        
        # 5. N2N2 Adapter (Concept -> Input Space)
        # Projects high-dim concepts (embedding_dim) down to input space (64) for "Imprinting"
        self.n2n2_projection = nn.Linear(self.embedding_dim, 64)
        
        # 6. Lexical Knowledge (Dictionary)
        self.register_buffer('lexical_knowledge', None)
        
        # 7. Sequential Mode
        self.sequential_mode = False
        
        # 8. Context Tracking
        self.register_buffer('last_context', torch.zeros(self.embedding_dim))
        
        # 9. Dynamic Growth Tracking
        self.surprise_history = []
        self.expert_usage = torch.zeros(num_experts)
        
    def reset_state(self):
        """Resets the state of all experts and memory."""
        for expert in self.experts:
            expert.reset_state()
        self.titans.reset_state()
        
    def process_visual(self, visual_latent):
        """
        Projects visual features into semantic space.
        Expects 2D Tensor: [Batch, 1, 64, 64]
        """
        if not isinstance(visual_latent, torch.Tensor):
            visual_tensor = torch.tensor(visual_latent, dtype=torch.float32)
        else:
            visual_tensor = visual_latent
            
        # Ensure 4D [B, C, H, W]
        if visual_tensor.dim() == 3: # [C, H, W]
            visual_tensor = visual_tensor.unsqueeze(0)
        elif visual_tensor.dim() == 2: # [H, W] (Legacy/Error fallback)
             visual_tensor = visual_tensor.unsqueeze(0).unsqueeze(0)
            
        # Project: Visual (2D) -> Features (2D)
        features_2d = self.visual_projection(visual_tensor) # [B, 256, 4, 4]
        
        # Flatten -> Semantic Vector
        semantic_vector = self.visual_adapter(features_2d) # [B, 256]
        
        return semantic_vector.squeeze(0).detach()

    def _process_signal_sequence(self, signal_sequence, reward=None, sequential_mode=False):
        """
        Internal helper to process a sequence of 64-dim signals through MoE experts.
        Input: [SeqLen, 64]
        Output: [embedding_dim]
        """
        # Reset state for all experts
        for expert in self.experts:
            if hasattr(expert, 'h'):
                expert.h = None
            else:
                expert.prev_outputs = np.zeros(expert.hidden_size)
        
        final_output = torch.zeros(self.embedding_dim, device=signal_sequence.device)
        seq_len = signal_sequence.size(0)
        
        for i in range(seq_len):
            signal = signal_sequence[i]
            
            if sequential_mode:
                # SEQUENTIAL MODE (Deep Layer Imprinting)
                # Signal -> Expert 0 -> Expert 1 -> ... -> Output
                # We project signal to Expert 0 input size if needed
                
                current_val = signal # [64]
                
                for k in range(len(self.experts)):
                    expert = self.experts[k]
                    
                    # Forward SNN
                    # Note: Expert input is 64. If we chain, we need to ensure dimensions match.
                    # Expert Output is 256. Expert Input is 64.
                    # We need a projection between experts?
                    # Or we assume Expert N output feeds Expert N+1 input?
                    # For now, let's assume we project Output(256) -> Input(64) for next expert
                    # Or we just sum them?
                    
                    # Simplified Chain: All experts see the Input, but their state depends on previous?
                    # No, that's not deep.
                    
                    # Deep Chain:
                    # Input -> E0 -> h0
                    # h0 -> E1 -> h1 ...
                    
                    # Issue: Expert Input is 64. Expert Output is 256.
                    # We need a bridge.
                    # Let's use the n2n2_projection (256->64) as the bridge!
                    
                    if k > 0:
                        # Project previous output to input space
                        # current_val is [256] from previous expert
                        current_val = self.n2n2_projection(current_val) # [64]
                        
                    outputs, _, _ = expert.forward(current_val, dt=0.2)
                    current_val = outputs.squeeze(0) # [256]
                    
                final_output = current_val
                
            else:
                # PARALLEL MODE (Standard MoE)
                # Gating
                gate_logits = self.gate(signal)
                weights, selected_indices = torch.topk(gate_logits, self.active_experts)
                weights = F.softmax(weights, dim=-1)
                
                combined_output = torch.zeros(self.embedding_dim, device=signal_sequence.device)
                
                for k in range(self.active_experts):
                    expert_idx = selected_indices[k].item()
                    weight = weights[k].item()
                    expert = self.experts[expert_idx]
                    
                    # Forward SNN
                    outputs, _, _ = expert.forward(signal, dt=0.2)
                    
                    # Learning (if reward provided)
                    if reward is not None:
                        expert.learn(reward)
                    
                    # Handle shape mismatch: outputs is [1, dim], combined_output is [dim]
                    combined_output += outputs.squeeze(0) * weight
                    
                final_output = combined_output
            
        # Titans Memory Update (Surprise!)
        # We observe the final semantic state.
        surprise = self.titans.observe(final_output.detach())
        
        # Update Context
        self.last_context = final_output.detach()
        
        # Dynamic Growth Check
        self._check_growth(surprise)
            
        return final_output.detach()

    def process_text(self, text, reward=None, sequential_mode=None):
        """
        Encodes a text string into a semantic vector using MoE SNN.
        Input: String (e.g., "3+3")
        Output: Tensor [256]
        """
        if sequential_mode is None:
            sequential_mode = self.sequential_mode
            
        if not text:
            return torch.zeros(self.embedding_dim)
            
        # 1. Convert to ASCII indices
        indices = [ord(c) % 128 for c in text]
        input_tensor = torch.tensor(indices, dtype=torch.long) # [SeqLen]
        
        # 2. Embed
        embedded = self.char_embedding(input_tensor) # [SeqLen, 64]
        
        # 3. Process Sequence
        return self._process_signal_sequence(embedded, reward=reward, sequential_mode=sequential_mode)

    def process_token_sequence(self, embeddings, reward=None):
        """
        Processes a sequence of high-dimensional embeddings (e.g., Qwen-3).
        Input: [SeqLen, embedding_dim]
        Output: [embedding_dim]
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
            
        # 1. Project to Input Signal Space (256 -> 64)
        signals = self.n2n2_projection(embeddings) # [SeqLen, 64]
        
        # 2. Process Sequence
        return self._process_signal_sequence(signals, reward=reward)

    def _process_batched_signals(self, input_signal):
        """
        Internal helper to process a batch of 64-dim signals through MoE experts.
        Input: [Batch, 64]
        Output: [Batch, embedding_dim]
        """
        batch_size = input_signal.size(0)
        
        # Gating (Per-sample in batch)
        gate_logits = self.gate(input_signal) # [Batch, num_experts]
        weights, selected_indices = torch.topk(gate_logits, self.active_experts, dim=1)
        weights = F.softmax(weights, dim=1)
        
        combined_output = torch.zeros(batch_size, self.embedding_dim, device=input_signal.device)
        
        # Process Experts (Vectorized!)
        # We vectorize the forward pass for each expert across all samples that selected it.
        for i in range(self.num_experts):
            # Find samples that selected this expert (anywhere in their top-k)
            # mask: [Batch, active_experts]
            expert_mask = (selected_indices == i)
            
            if expert_mask.any():
                # Get indices of samples that selected this expert
                batch_indices, k_indices = torch.where(expert_mask)
                
                # Get unique batch indices to avoid redundant forward passes
                unique_batch_indices = batch_indices.unique()
                expert_input = input_signal[unique_batch_indices]
                
                # Forward pass for all samples that need this expert
                expert_out, _, _ = self.experts[i](expert_input, dt=0.2)
                
                # Map expert outputs back to combined_output using weights
                for j, b_idx in enumerate(unique_batch_indices):
                    # Find all k_indices for this specific batch sample that point to expert i
                    sample_k_indices = k_indices[batch_indices == b_idx]
                    for k_idx in sample_k_indices:
                        combined_output[b_idx] += expert_out[j] * weights[b_idx, k_idx]
                        # Update usage tracking
                        self.expert_usage[i] += weights[b_idx, k_idx].item()

        # Normalize output to [-1, 1] for stable TitansMemory observation
        combined_output = torch.tanh(combined_output)
        
        # Titans Update (Batched)
        surprise = self.titans.observe(combined_output.detach())
        
        # Dynamic Growth Check
        self._check_growth(surprise)
            
        return combined_output, surprise

    def process_text_embedding(self, embedding_vector, reward=None):
        """
        N2N2 Helper: Process a raw embedding vector directly.
        Used for "Hyper-Stimulation" transfer learning and Meditation.
        Supports Batching: [Batch, 256]
        """
        is_batched = embedding_vector.dim() > 1
        if not is_batched:
            embedding_vector = embedding_vector.unsqueeze(0)
            
        # Project Concept (256) -> Input Signal (64)
        input_signal = self.n2n2_projection(embedding_vector) # [Batch, 64]
        
        combined_output, surprise = self._process_batched_signals(input_signal)
            
        if is_batched:
            return combined_output, surprise
        else:
            return combined_output[0], surprise

    def train_on_batch(self, embedding_batch, optimizer, use_ewc=False, hyper_transfer=None):
        """
        Trains the Broca module on a batch of embeddings.
        Reconstruction loss: Output should match Input.
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        output, surprise = self.process_text_embedding(embedding_batch)
        
        # Reconstruction Loss (MSE)
        # embedding_batch is [Batch, 256], output is [Batch, 256]
        loss = F.mse_loss(output, embedding_batch)
        
        # Optional EWC Loss
        if use_ewc and hyper_transfer:
            loss += hyper_transfer.ewc_loss()
            
        loss.backward()
        optimizer.step()
        
        return loss.item(), surprise.mean().item() if isinstance(surprise, torch.Tensor) else surprise

    def seed_knowledge(self, concepts_tensor):
        """
        Direct Knowledge Seeding (Intelligence Boost).
        Initializes expert weights using compressed Qwen-3 concepts.
        """
        print(f"BrocaModule: Seeding knowledge from {concepts_tensor.shape[0]} concepts...")
        with torch.no_grad():
            # 1. Initialize Char Embedding with first 128 concepts (if available)
            num_chars = min(128, concepts_tensor.shape[0])
            # Project 256 -> 64 for char embedding
            proj = nn.Linear(concepts_tensor.shape[1], 64).to(concepts_tensor.device)
            char_seeds = proj(concepts_tensor[:num_chars])
            self.char_embedding.weight.data[:num_chars] = char_seeds
            
            # 2. Seed Experts (W_in)
            # We use the concepts to bias the input weights of experts
            for i, expert in enumerate(self.experts):
                # Each expert gets a different "slice" of knowledge
                start = (i * 100) % concepts_tensor.shape[0]
                end = (start + expert.hidden_size) % concepts_tensor.shape[0]
                
                if end > start:
                    seeds = concepts_tensor[start:end]
                else:
                    seeds = concepts_tensor[start:]
                
                # Project seeds to match W_in shape [hidden, input]
                # seeds is [N, concept_dim], W_in is [hidden, 64]
                # We want to map semantic patterns to input sensitivities
                expert_proj = nn.Linear(concepts_tensor.shape[1], 64).to(concepts_tensor.device)
                
                with torch.no_grad():
                    projected_seeds = expert_proj(seeds)
                    num_to_seed = min(projected_seeds.shape[0], expert.w_in.shape[0])
                    expert.w_in.data[:num_to_seed] = projected_seeds[:num_to_seed]
                
        print("BrocaModule: Knowledge Seeding Complete.")

    def save(self):
        """
        Custom save method. Expert weights are already in state_dict().
        We only save metadata here.
        """
        return {
            'state_dict': self.state_dict(),
            'num_experts': self.num_experts,
            'experts_metadata': [
                {
                    'input_size': e.input_size,
                    'hidden_size': e.hidden_size,
                    'output_size': e.output_size
                } for e in self.experts
            ],
            'titans_state': self.titans.state_dict()
        }

    def load(self, data):
        """
        Custom load method with backward compatibility and self-healing.
        """
        if not data: return

        # 1. Handle Legacy Format (Single Expert)
        if 'liquid_graph' in data:
            print("BrocaModule: Migrating legacy single-expert model to MoE Expert 0...")
            legacy_state = data['state_dict']
            new_state = self.state_dict()
            for key in ['visual_projection.weight', 'visual_projection.bias', 'char_embedding.weight']:
                if key in legacy_state:
                    new_state[key] = legacy_state[key]
            self.load_state_dict(new_state, strict=False)
            
            lg_data = data['liquid_graph']
            if isinstance(lg_data, dict):
                self.experts[0] = VectorizedLiquidGraph.from_dict(lg_data)
            elif LiquidGraph is not None and isinstance(lg_data, LiquidGraph):
                self.experts[0] = VectorizedLiquidGraph.from_legacy(lg_data)
            return

        # 2. Handle MoE Format (New)
        # If 'state_dict' is missing, assume 'data' IS the state_dict
        if 'state_dict' in data:
            saved_state = data['state_dict']
            experts_metadata = data.get('experts_metadata', data.get('experts_data', []))
            titans_state = data.get('titans_state', None)
            saved_num_experts = data.get('num_experts', self.num_experts)
        else:
            saved_state = data
            experts_metadata = [] # Will infer from saved_state
            titans_state = None # Will infer from saved_state
            # Infer num_experts from gate.bias or experts.N keys
            saved_num_experts = self.num_experts
            for k in saved_state.keys():
                if k.startswith('experts.'):
                    try:
                        idx = int(k.split('.')[1])
                        saved_num_experts = max(saved_num_experts, idx + 1)
                    except: pass
                elif k == 'gate.bias':
                    saved_num_experts = max(saved_num_experts, saved_state[k].shape[0])

        # --- SELF-HEALING: Expand Experts if needed ---
        if saved_num_experts > self.num_experts:
            print(f"BrocaModule: Self-Healing - Expanding Experts {self.num_experts} -> {saved_num_experts}")
            for i in range(self.num_experts, saved_num_experts):
                # Create new expert with default dimensions (will be resized later)
                new_expert = VectorizedLiquidGraph(input_size=64, hidden_size=512, output_size=self.embedding_dim)
                self.experts.append(new_expert)
            self.num_experts = saved_num_experts
            
            # Resize Gate
            old_gate = self.gate
            self.gate = nn.Linear(old_gate.in_features, self.num_experts)
            # Initialize with old gate weights
            with torch.no_grad():
                # Zero init new parts
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)
                # Copy old parts
                num_old = min(old_gate.out_features, self.num_experts)
                self.gate.weight[:num_old, :] = old_gate.weight[:num_old, :]
                self.gate.bias[:num_old] = old_gate.bias[:num_old]

        # --- SELF-HEALING: Auto-Resize Experts to match Checkpoint ---
        for i in range(self.num_experts):
            prefix = f"experts.{i}."
            # Check for hidden_size mismatch in w_rec
            w_rec_key = f"{prefix}w_rec"
            if w_rec_key in saved_state:
                target_hidden = saved_state[w_rec_key].shape[0]
                if self.experts[i].hidden_size != target_hidden:
                    print(f"BrocaModule: Self-Healing - Resizing Expert {i} Hidden {self.experts[i].hidden_size} -> {target_hidden}")
                    self.experts[i].resize_hidden(target_hidden)
            
            # Check for input_size mismatch in w_in
            w_in_key = f"{prefix}w_in"
            if w_in_key in saved_state:
                target_input = saved_state[w_in_key].shape[1]
                if self.experts[i].input_size != target_input:
                    print(f"BrocaModule: Self-Healing - Resizing Expert {i} Input {self.experts[i].input_size} -> {target_input}")
                    self.experts[i].resize_input(target_input)

        # 3. Load State Dict
        current_state = self.state_dict()
        compatible_state = {}
        for k, v in saved_state.items():
            if k in current_state:
                # Special handling for transient states (x, y)
                if k.endswith('.x') or k.endswith('.y') or k == 'x' or k == 'y':
                    if v.shape != current_state[k].shape:
                        if v.dim() == 2 and v.size(0) == 1:
                            v = v.squeeze(0)
                        
                        if v.shape == current_state[k].shape:
                            compatible_state[k] = v
                        else:
                            # Silently skip mismatched transient states
                            continue
                
                if v.shape == current_state[k].shape:
                    compatible_state[k] = v
                else:
                    print(f"BrocaModule: Skipping {k} due to shape mismatch {v.shape} vs {current_state[k].shape}")
        
        self.load_state_dict(compatible_state, strict=False)

        # 4. Load Expert Metadata (if available)
        experts_metadata = data.get('experts_metadata', [])
        if experts_metadata:
            for i, meta in enumerate(experts_metadata):
                if i < len(self.experts):
                    if self.experts[i].hidden_size != meta['hidden_size']:
                        self.experts[i].resize_hidden(meta['hidden_size'])
                    if self.experts[i].input_size != meta['input_size']:
                        self.experts[i].resize_input(meta['input_size'])
                    if self.experts[i].output_size != meta['output_size']:
                        self.experts[i].resize_output(meta['output_size'])

        # 5. Load Titans State (if available separately)
        if titans_state:
            self.titans.load_state_dict(titans_state, strict=False)
        
        # 6. Handle Lexical Knowledge (if it was a raw attribute)
        if hasattr(data, 'lexical_knowledge'):
            self.lexical_knowledge = data.lexical_knowledge
        elif 'lexical_knowledge' in saved_state:
             # Already handled by load_state_dict if it's a buffer
             pass

    def ground_character(self, char_idx, signal_vector):
        """
        Grounds a specific character index to a signal vector.
        Used for Zero-Shot Grounding with N2N2.
        """
        if 0 <= char_idx < self.char_embedding.num_embeddings:
            with torch.no_grad():
                self.char_embedding.weight[char_idx] = signal_vector.to(self.char_embedding.weight.device)
        
    def resize_latent(self, new_latent_size):
        """
        Evolutionary Upgrade: Change latent dimension.
        """
        if new_latent_size == self.embedding_dim:
            return
            
        print(f"BrocaModule: Resizing Latent Dim {self.embedding_dim} -> {new_latent_size}")
        
        old_dim = self.embedding_dim
        self.embedding_dim = new_latent_size
        
        # 1. Resize Visual Adapter (Linear layer at the end)
        old_linear = self.visual_adapter[-2]
        new_linear = nn.Linear(old_linear.in_features, new_latent_size)
        with torch.no_grad():
            min_out = min(old_dim, new_latent_size)
            new_linear.weight[:min_out, :] = old_linear.weight[:min_out, :]
            new_linear.bias[:min_out] = old_linear.bias[:min_out]
            
            # Zero-initialize new dimensions to prevent noise injection
            if new_latent_size > old_dim:
                new_linear.weight[min_out:, :] = 0.0
                new_linear.bias[min_out:] = 0.0
        self.visual_adapter[-2] = new_linear
        
        # 2. Resize Experts (LiquidGraph output size)
        for expert in self.experts:
            expert.resize_output(new_latent_size)
            
        # 3. Resize Titans Memory
        self.titans.resize(new_latent_size)
        
        # 4. Resize N2N2 Projection (Input dim)
        old_linear = self.n2n2_projection
        new_linear = nn.Linear(new_latent_size, old_linear.out_features)
        with torch.no_grad():
            new_linear.weight[:, :min_out] = old_linear.weight[:, :min_out]
            new_linear.bias[:] = old_linear.bias[:]
        self.n2n2_projection = new_linear
        
        self.embedding_dim = new_latent_size

    def get_current_context(self):
        """
        Returns the latest semantic context (last_context buffer).
        """
        return self.last_context

    def _check_growth(self, surprise):
        """Monitors surprise and triggers sprouting if needed."""
        if self.genome is None:
            return
            
        # Track surprise in a rolling window
        if isinstance(surprise, torch.Tensor):
            avg_surprise = surprise.mean().item()
        else:
            avg_surprise = surprise
            
        self.surprise_history.append(avg_surprise)
        if len(self.surprise_history) > 100:
            self.surprise_history.pop(0)
            
        # Trigger Sprouting if surprise is consistently high
        if len(self.surprise_history) >= 50:
            rolling_avg = sum(self.surprise_history) / len(self.surprise_history)
            if rolling_avg > self.genome.sprouting_threshold:
                if self.num_experts < self.genome.max_experts:
                    self.sprout_expert()
                    self.surprise_history = [] # Reset after sprouting

    def sprout_expert(self):
        """Adds a new expert to the MoE pool (Neurogenesis)."""
        print(f"BrocaModule: Sprouting new expert {self.num_experts} due to high surprise...")
        
        # 1. Create new expert
        # We can clone the best expert or start fresh. Let's start fresh for diversity.
        new_expert = VectorizedLiquidGraph(
            input_size=64, 
            hidden_size=512, 
            output_size=self.embedding_dim
        ).to(next(self.parameters()).device)
        
        self.experts.append(new_expert)
        self.num_experts += 1
        
        # 2. Resize Gating Network
        old_gate = self.gate
        self.gate = nn.Linear(old_gate.in_features, self.num_experts).to(old_gate.weight.device)
        
        with torch.no_grad():
            # Copy old weights
            self.gate.weight[:old_gate.out_features, :] = old_gate.weight
            self.gate.bias[:old_gate.out_features] = old_gate.bias
            # Initialize new expert logit to be slightly lower to avoid immediate dominance
            self.gate.weight[old_gate.out_features:, :] = 0.0
            self.gate.bias[old_gate.out_features:] = -1.0 
            
        # 3. Update usage tracking
        self.expert_usage = torch.cat([self.expert_usage, torch.zeros(1)])
        
        print(f"BrocaModule: Neurogenesis complete. Total experts: {self.num_experts}")

    def prune_experts(self, threshold=0.01):
        """Removes experts that are rarely used (Apoptosis)."""
        if self.num_experts <= 1:
            return
            
        # Identify low-usage experts
        # Note: expert_usage should be updated in forward pass (omitted for brevity in this chunk)
        low_usage_indices = torch.where(self.expert_usage < threshold)[0]
        
        if len(low_usage_indices) > 0:
            idx_to_remove = low_usage_indices[0].item()
            print(f"BrocaModule: Pruning expert {idx_to_remove} due to low usage...")
            
            # Remove from ModuleList
            new_experts = nn.ModuleList([e for i, e in enumerate(self.experts) if i != idx_to_remove])
            self.experts = new_experts
            self.num_experts -= 1
            
            # Resize Gate
            old_gate = self.gate
            self.gate = nn.Linear(old_gate.in_features, self.num_experts).to(old_gate.weight.device)
            
            with torch.no_grad():
                # Copy weights, skipping the pruned one
                keep_indices = [i for i in range(old_gate.out_features) if i != idx_to_remove]
                self.gate.weight.data = old_gate.weight.data[keep_indices]
                self.gate.bias.data = old_gate.bias.data[keep_indices]
                
            # Update usage tracking
            self.expert_usage = self.expert_usage[keep_indices]
            
            print(f"BrocaModule: Expert {idx_to_remove} pruned. Total experts: {self.num_experts}")

