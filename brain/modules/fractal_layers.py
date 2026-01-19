import torch
import torch.nn as nn
import torch.nn.functional as F
from brain.modules.ribosome import NeuralRibosome
from brain.fnd_encoder import FractalDNA

class FractalLinear(nn.Module):
    """
    A Linear layer that stores weights as Fractal DNA.
    Decompresses weights on-the-fly during forward pass.
    """
    def __init__(self, in_features, out_features, dna, bias=None, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Store DNA (not as a parameter, but as an object/buffer)
        # We can't register it as a buffer if it's a custom object.
        # So we store it as a python attribute.
        self.dna = dna
        
        # Bias is usually small, so we store it directly as a Parameter
        if bias is not None:
            self.bias = nn.Parameter(bias.to(device))
        else:
            self.register_parameter('bias', None)
            
        self.ribosome = NeuralRibosome(device=device)
        
        # Cache for weights (optional)
        self.cached_weight = None
        
    def forward(self, x):
        # 1. Decompress Weights (if not cached)
        if self.cached_weight is None:
            # Transcribe DNA -> Weights
            # iterations=20 is a good balance for inference
            weight = self.ribosome.transcribe(self.dna, iterations=20)
            
            # Ensure weight shape matches (out, in)
            if weight.shape != (self.out_features, self.in_features):
                # Resize or handle mismatch (Ribosome returns H,W from DNA)
                # DNA shape should match.
                pass
                
            # Optimization: If we are in a loop (like autoregressive generation),
            # we might want to cache this for the duration of the generation.
            # For now, we assume JIT (Just-In-Time) and discard immediately to save RAM.
            # But for speed, we might want a context manager to enable caching.
            
        else:
            weight = self.cached_weight
            
        # 2. Linear Projection
        return F.linear(x, weight, self.bias)
        
    def train(self, mode=True):
        # Fractal Layers are fixed (frozen) by default.
        # We don't backprop into DNA (yet).
        super().train(mode)
        
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, fractal=True'
