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

    def transcribe_differentiable(self, dna_params, shape, base_val, iterations=20) -> torch.Tensor:
        """
        Differentiable version of transcribe.
        Args:
            dna_params: Flat tensor [num_transforms * 7]
            shape: (H, W)
            base_val: float
        """
        H, W = shape
        num_transforms = len(dna_params) // 7
        
        # Reshape params: [num_transforms, 7]
        params = dna_params.view(num_transforms, 7)
        
        # Start with a fixed canvas (to ensure gradients only come from params)
        # Using a fixed seed or zeros/ones
        canvas = torch.ones(1, 1, H, W, device=self.device) * 0.5
        
        # Prepare theta: [num_transforms, 2, 3]
        # p: [a, b, c, d, e, f, prob]
        # PyTorch affine_grid: [a, b, e], [c, d, f]
        theta = torch.stack([
            torch.stack([params[:, 0], params[:, 1], params[:, 4]], dim=1),
            torch.stack([params[:, 2], params[:, 3], params[:, 5]], dim=1)
        ], dim=1)
        
        # Probabilities
        probs = F.softmax(params[:, 6], dim=0) # Use softmax for stable probabilities
        
        for _ in range(iterations):
            new_canvas = torch.zeros_like(canvas)
            
            # We can process all transforms in one batch if VRAM allows
            # [num_transforms, 1, H, W]
            canvas_batch = canvas.expand(num_transforms, -1, -1, -1)
            
            grid = F.affine_grid(theta, canvas_batch.size(), align_corners=False)
            transformed = F.grid_sample(canvas_batch, grid, align_corners=False)
            
            # Weighted Sum: [num_transforms, 1, H, W] * [num_transforms, 1, 1, 1]
            weighted = transformed * probs.view(-1, 1, 1, 1)
            new_canvas = torch.sum(weighted, dim=0, keepdim=True)
            
            canvas = new_canvas
            
        reconstructed = canvas.squeeze() + base_val
        return reconstructed

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
            # Memory Optimization: Sequential Accumulation
            new_canvas = torch.zeros_like(canvas)
            
            # Reshape theta for affine_grid: [N, 2, 3]
            # theta is already [N, 2, 3]
            
            chunk_size = 1
            
            for t_idx in range(0, num_transforms, chunk_size):
                end_idx = min(t_idx + chunk_size, num_transforms)
                current_batch_size = end_idx - t_idx
                
                theta_chunk = theta[t_idx:end_idx]
                
                # Expand canvas for this chunk
                # canvas: [1, 1, H, W] -> [chunk_size, 1, H, W]
                canvas_batch = canvas.repeat(current_batch_size, 1, 1, 1)
                
                grid = F.affine_grid(theta_chunk, canvas_batch.size(), align_corners=False)
                transformed_chunk = F.grid_sample(canvas_batch, grid, align_corners=False)
                
                chunk_probs = probs[t_idx:end_idx].view(-1, 1, 1, 1)
                weighted_chunk_sum = torch.sum(transformed_chunk * chunk_probs, dim=0).unsqueeze(0)
                
                new_canvas = new_canvas + weighted_chunk_sum
                
                del canvas_batch, grid, transformed_chunk, weighted_chunk_sum
            
            canvas = new_canvas
            
        # Post-processing
        # Rescale to original range (assuming we stored scale somewhere, or just use base_val as min)
        # For this prototype, we'll just return the normalized [0,1] map + base_val offset
        # Real implementation needs precise scale reconstruction.
        
        reconstructed = canvas.squeeze() + base_val
        return reconstructed

    def forward(self, dna):
        return self.transcribe(dna)
