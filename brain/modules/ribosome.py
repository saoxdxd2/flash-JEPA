import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralRibosome(nn.Module):
    """
    The 'Biological' Decompressor.
    Reads Fractal DNA and 'grows' the corresponding neural weights.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

    def transcribe(self, dna, iterations=20) -> torch.Tensor:
        """
        Reconstructs the weight matrix from Fractal DNA using the Chaos Game or Deterministic Iteration.
        Deterministic Iteration (Markov Operator) is faster for dense matrices.
        
        Args:
            dna: FractalDNA object (or dict)
            iterations: Number of iterations to run the IFS (resolution depth)
            
        Returns:
            Reconstructed Weight Matrix [H, W]
        """
        if isinstance(dna, dict):
            # Handle JSON dict input
            shape = dna['shape']
            transforms = dna['transforms']
            base_val = dna.get('base_value', 0.0)
        else:
            shape = dna.shape
            transforms = dna.transforms
            base_val = dna.base_value
            
        H, W = shape
        
        # Start with a random noise canvas or uniform canvas
        # [1, 1, H, W] for grid_sample
        canvas = torch.rand(1, 1, H, W, device=self.device)
        
        # Prepare transforms
        num_transforms = len(transforms)
        theta = torch.zeros(num_transforms, 2, 3, device=self.device)
        probs = torch.zeros(num_transforms, device=self.device)
        
        for i, t in enumerate(transforms):
            # PyTorch affine_grid uses a specific format:
            # [ a  b  tx ]
            # [ c  d  ty ]
            # And coordinates are [-1, 1].
            # We might need to adjust 'e' and 'f' (translation) to match this coordinate system.
            # For now, we assume the encoder produced compatible parameters.
            theta[i, 0, 0] = t['a']
            theta[i, 0, 1] = t['b']
            theta[i, 0, 2] = t['e']
            theta[i, 1, 0] = t['c']
            theta[i, 1, 1] = t['d']
            theta[i, 1, 2] = t['f']
            probs[i] = t['p']
            
        # Normalize probs
        probs = probs / probs.sum()
        
        # Iteration Loop (The "Growth" Process)
        for _ in range(iterations):
            # Apply all transforms
            # We expand canvas to [num_transforms, 1, H, W]
            canvas_batch = canvas.repeat(num_transforms, 1, 1, 1)
            
            grid = F.affine_grid(theta, canvas_batch.size(), align_corners=False)
            transformed_batch = F.grid_sample(canvas_batch, grid, align_corners=False)
            
            # Weighted Sum (Superposition)
            # canvas = sum(w_i(canvas) * p_i)
            canvas = torch.sum(transformed_batch * probs.view(-1, 1, 1, 1), dim=0)
            
        # Post-processing
        # Rescale to original range (assuming we stored scale somewhere, or just use base_val as min)
        # For this prototype, we'll just return the normalized [0,1] map + base_val offset
        # Real implementation needs precise scale reconstruction.
        
        reconstructed = canvas.squeeze() + base_val
        return reconstructed

    def forward(self, dna):
        return self.transcribe(dna)
