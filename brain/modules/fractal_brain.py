import torch
import torch.nn as nn
import torch.nn.functional as F
from brain.modules.fractal_layers import FractalLinear
from brain.fnd_encoder import FractalDNA

class FractalMoE(nn.Module):
    """
    Fractal Mixture of Experts.
    """
    def __init__(self, dna_dict, prefix, hidden_size, num_experts=128, top_k=4, device='cuda'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.device = device
        
        # Router (Gate)
        # Key: mlp.gate.weight
        self.gate = self._build_linear(dna_dict, prefix + "mlp.gate", out_features=num_experts, in_features=hidden_size)
        
        # Experts
        # We store experts in a ModuleList
        # Each expert is a SwiGLU block (gate, up, down)
        self.experts = nn.ModuleList()
        
        # Check if experts are stored individually or as a big tensor
        # Qwen-MoE usually stores experts individually: mlp.experts.0.gate_proj.weight
        
        for i in range(num_experts):
            exp_prefix = prefix + f"mlp.experts.{i}."
            # Check if this expert exists in DNA (sparse loading?)
            # If not, we might skip or init random.
            # For now, assume all experts are needed.
            
            expert = nn.ModuleDict({
                'gate_proj': self._build_linear(dna_dict, exp_prefix + "gate_proj", 
                                              out_features=hidden_size*2, in_features=hidden_size), # Intermediate size?
                'up_proj': self._build_linear(dna_dict, exp_prefix + "up_proj",
                                            out_features=hidden_size*2, in_features=hidden_size),
                'down_proj': self._build_linear(dna_dict, exp_prefix + "down_proj",
                                              out_features=hidden_size, in_features=hidden_size*2)
            })
            self.experts.append(expert)
            
        # Shared Expert (if Qwen uses it, like DeepSeekMoE)
        # Qwen-2.5-MoE uses shared experts. Qwen-3 likely does too.
        # Key: mlp.shared_expert.gate_proj...
        self.shared_expert = None
        if (prefix + "mlp.shared_expert.gate_proj.weight") in dna_dict:
            self.shared_expert = nn.ModuleDict({
                'gate_proj': self._build_linear(dna_dict, prefix + "mlp.shared_expert.gate_proj", 
                                              out_features=hidden_size*2, in_features=hidden_size),
                'up_proj': self._build_linear(dna_dict, prefix + "mlp.shared_expert.up_proj",
                                            out_features=hidden_size*2, in_features=hidden_size),
                'down_proj': self._build_linear(dna_dict, prefix + "mlp.shared_expert.down_proj",
                                              out_features=hidden_size, in_features=hidden_size*2)
            })

    def _build_linear(self, dna_dict, key, out_features, in_features):
        weight_key = key + ".weight"
        if weight_key not in dna_dict:
            # Fallback for missing keys (e.g. if we only downloaded some shards)
            # Return a dummy FractalLinear that produces zeros or random
            # We need a valid DNA object though.
            # Let's create a dummy DNA.
            dummy_dna = FractalDNA(shape=(out_features, in_features), transforms=[], base_value=0.0)
            return FractalLinear(in_features, out_features, dummy_dna, device=self.device)
            
        dna_data = dna_dict[weight_key]
        dna = FractalDNA(
            shape=dna_data['shape'],
            transforms=dna_data['transforms'],
            base_value=dna_data['base_value']
        )
        # Bias?
        bias = None
        if (key + ".bias") in dna_dict:
            bias = dna_dict[key + ".bias"].to(self.device)
            
        return FractalLinear(in_features, out_features, dna, bias=bias, device=self.device)

    def forward(self, x):
        # x: [Batch, Seq, Hidden]
        B, S, H = x.shape
        x_flat = x.view(-1, H)
        
        # Router
        router_logits = self.gate(x_flat) # [N, num_experts]
        routing_weights = F.softmax(router_logits, dim=1)
        
        # Top-K
        weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=1)
        
        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        final_hidden_states = torch.zeros_like(x_flat)
        
        # Process Experts
        # Naive loop implementation (slow but simple)
        # Optimized implementation would use permutation/scatter-gather
        
        # Mask for each expert
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0) # [Experts, N, TopK] -> [Experts, N] (sum over TopK?)
        
        # Actually, we iterate over experts
        for i in range(self.num_experts):
            # Find tokens that selected this expert
            # selected_experts: [N, TopK]
            # We need indices where selected_experts == i
            
            # Boolean mask: [N, TopK]
            mask = (selected_experts == i)
            
            if mask.any():
                # Get batch indices
                batch_idx, k_idx = torch.where(mask)
                
                # Extract tokens
                current_tokens = x_flat[batch_idx]
                
                # Forward Expert
                # SwiGLU
                gate = self.experts[i]['gate_proj'](current_tokens)
                up = self.experts[i]['up_proj'](current_tokens)
                out = F.silu(gate) * up
                out = self.experts[i]['down_proj'](out)
                
                # Weighting
                # weights: [N, TopK]
                # We need weights[batch_idx, k_idx]
                current_weights = weights[batch_idx, k_idx].unsqueeze(1)
                
                out = out * current_weights
                
                # Add to final
                final_hidden_states.index_add_(0, batch_idx, out)
                
        # Shared Expert
        if self.shared_expert:
            gate = self.shared_expert['gate_proj'](x_flat)
            up = self.shared_expert['up_proj'](x_flat)
            out = F.silu(gate) * up
            out = self.shared_expert['down_proj'](out)
            final_hidden_states = final_hidden_states + out
            
        return final_hidden_states.view(B, S, H)

class FractalBlock(nn.Module):
    """
    A Transformer Block reconstructed from Fractal DNA.
    Supports Dense and MoE.
    """
    def __init__(self, layer_idx, dna_dict, config, device='cuda'):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.hidden_size // self.num_heads
        
        prefix = f"model.language_model.layers.{layer_idx}."
        
        # Attention
        self.q_proj = self._build_layer(dna_dict, prefix + "self_attn.q_proj", bias=True)
        self.k_proj = self._build_layer(dna_dict, prefix + "self_attn.k_proj", bias=True)
        self.v_proj = self._build_layer(dna_dict, prefix + "self_attn.v_proj", bias=True)
        self.o_proj = self._build_layer(dna_dict, prefix + "self_attn.o_proj", bias=False)
        
        # MLP: Check if MoE or Dense
        if (prefix + "mlp.gate.weight") in dna_dict:
            # MoE Detected
            self.is_moe = True
            self.mlp = FractalMoE(dna_dict, prefix, self.hidden_size, device=device)
        else:
            # Dense MLP
            self.is_moe = False
            self.gate_proj = self._build_layer(dna_dict, prefix + "mlp.gate_proj", bias=False)
            self.up_proj = self._build_layer(dna_dict, prefix + "mlp.up_proj", bias=False)
            self.down_proj = self._build_layer(dna_dict, prefix + "mlp.down_proj", bias=False)
        
        # LayerNorms
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        
        self._load_ln(dna_dict, prefix + "input_layernorm", self.input_layernorm)
        self._load_ln(dna_dict, prefix + "post_attention_layernorm", self.post_attention_layernorm)

    def _build_layer(self, dna_dict, key, bias=False):
        weight_key = key + ".weight"
        if weight_key not in dna_dict:
            # Fallback
            return nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
            
        dna_data = dna_dict[weight_key]
        dna = FractalDNA(
            shape=dna_data['shape'],
            transforms=dna_data['transforms'],
            base_value=dna_data['base_value']
        )
        
        bias_tensor = None
        if bias:
            bias_key = key + ".bias"
            if bias_key in dna_dict:
                bias_tensor = dna_dict[bias_key].to(self.device)
                
        H, W = dna.shape
        return FractalLinear(W, H, dna, bias=bias_tensor, device=self.device)

    def _load_ln(self, dna_dict, key, layer):
        w_key = key + ".weight"
        if w_key in dna_dict:
            layer.weight.data = dna_dict[w_key].to(self.device)
            
    def forward(self, x):
        # x: [Batch, Seq, Hidden]
        residual = x
        x = self.input_layernorm(x)
        
        # Attention
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        x = self.o_proj(attn)
        x = x + residual
        
        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        
        if self.is_moe:
            x = self.mlp(x)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = F.silu(gate) * up
            x = self.down_proj(x)
            
        x = x + residual
        
        return x

class FractalBrain(nn.Module):
    """
    The Vessel's Cortex.
    Loads the compressed Qwen-3 DNA and reconstructs it.
    """
    def __init__(self, dna_path="fractal_brain.pt", device='cuda'):
        super().__init__()
        self.device = device
        
        print(f"FractalBrain: Loading DNA from {dna_path}...")
        try:
            self.dna_dict = torch.load(dna_path, map_location='cpu') # Load to CPU first
        except FileNotFoundError:
            print("FractalBrain: DNA not found. Initializing empty.")
            self.dna_dict = {}
            
        # Config (Inferred or Hardcoded for Qwen-3-VL-235B)
        # 235B is huge. We assume we only loaded a subset or we are running layer-wise.
        # Let's assume standard Qwen config for now.
        self.config = {
            'hidden_size': 4096, # Example (Qwen-7B size, 235B is larger)
            'num_attention_heads': 32,
            'num_hidden_layers': 32 # We might only load what we have
        }
        
        # Detect layers present in DNA
        layers_found = set()
        for k in self.dna_dict.keys():
            if "layers." in k:
                try:
                    idx = int(k.split("layers.")[1].split(".")[0])
                    layers_found.add(idx)
                except: pass
        
        self.num_layers = max(layers_found) + 1 if layers_found else 0
        print(f"FractalBrain: Found {self.num_layers} layers in DNA.")
        
        # Embeddings
        # self.embed_tokens = ... (Need to handle embedding layer specially, usually dense)
        # If embedding is compressed, we need FractalEmbedding?
        # For now, assume embedding is stored raw or we skip it.
        
        # Build Layers
        self.layers = nn.ModuleList([
            FractalBlock(i, self.dna_dict, self.config, device=device)
            for i in range(self.num_layers)
        ])
        
        # Final Norm
        self.norm = nn.LayerNorm(self.config['hidden_size'])
        
    def forward(self, input_ids):
        # Embed (Placeholder)
        # x = self.embed_tokens(input_ids)
        x = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.config['hidden_size'], device=self.device)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return x
