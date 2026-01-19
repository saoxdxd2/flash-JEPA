import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Tuple, Dict

class FractalDNA:
    """
    Represents the compressed 'DNA' of a weight matrix.
    Contains the parameters for the Iterated Function System (IFS).
    """
    def __init__(self, shape: Tuple[int, int], transforms: List[Dict[str, float]], base_value: float = 0.0):
        self.shape = shape
        self.transforms = transforms # List of dicts: {'a':, 'b':, 'c':, 'd':, 'e':, 'f':, 'p':}
        self.base_value = base_value

    def to_json(self):
        return {
            'shape': self.shape,
            'transforms': self.transforms,
            'base_value': self.base_value
        }

class FractalEncoder:
    """
    Solves the Inverse Fractal Problem:
    Finds an IFS whose attractor approximates the target weight matrix.
    
    Uses the 'Collage Theorem' approach combined with Gradient Descent.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def encode(self, target_weights: torch.Tensor, num_transforms=8, iterations=1000) -> FractalDNA:
        """
        Compresses a weight matrix into Fractal DNA.
        
        Args:
            target_weights: The matrix to compress (H, W)
            num_transforms: Number of affine transformations to use (compression ratio determinant)
            iterations: Optimization steps
            
        Returns:
            FractalDNA object
        """
        target = target_weights.to(self.device).float()
        H, W = target.shape
        
        # Normalize target to [0, 1] for fractal processing, keep stats to restore
        min_val = target.min()
        max_val = target.max()
        scale = max_val - min_val
        if scale == 0: scale = 1.0
        
        normalized_target = (target - min_val) / scale
        
        # Initialize Affine Transforms (a, b, c, d, e, f)
        # x' = ax + by + e
        # y' = cx + dy + f
        # We optimize these parameters to minimize the Collage Distance
        
        # Parameters: [num_transforms, 6]
        # We use a neural network approach to optimize the parameters
        params = torch.randn(num_transforms, 6, device=self.device, requires_grad=True)
        # Probabilities: [num_transforms]
        probs_logits = torch.randn(num_transforms, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([params, probs_logits], lr=0.01)
        
        print(f"FractalEncoder: Compressing {H}x{W} matrix using {num_transforms} transforms...")
        start_time = time.time()
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Collage Error Calculation
            # Instead of full rendering (which is non-differentiable/hard), 
            # we minimize the distance between the target and the union of transformed copies of the target.
            # Collage Theorem: d(T, W) < (1/(1-s)) * d(T, Union(w_i(T)))
            
            # 1. Apply transforms to the target image (grid sample)
            # We treat the matrix as an image
            reconstructed = torch.zeros_like(normalized_target)
            
            # Create grid for affine_grid
            # PyTorch affine_grid expects [N, 2, 3] matrix for 2D
            # params is [N, 6] -> reshape to [N, 2, 3]
            
            # We need to iterate over transforms and sum them weighted by probability?
            # Standard IFS is a union. For grayscale/weights, it's a weighted sum or max.
            # Let's use weighted sum approximation for differentiability.
            
            probs = torch.softmax(probs_logits, dim=0)
            
            total_loss = 0
            
            # This is a simplified "Collage Loss":
            # The target should be a fixed point of the operator T(x) = sum(p_i * w_i(x))
            # So we want T(target) approx target.
            
            # Prepare batch of affine matrices
            # params: [a, b, c, d, e, f] -> [[a, b, e], [c, d, f]]
            theta = params.view(num_transforms, 2, 3)
            
            # Grid sample expects normalized coordinates [-1, 1]
            # We expand target to [num_transforms, 1, H, W]
            target_batch = normalized_target.unsqueeze(0).unsqueeze(0).repeat(num_transforms, 1, 1, 1)
            
            grid = torch.nn.functional.affine_grid(theta, target_batch.size(), align_corners=False)
            transformed_batch = torch.nn.functional.grid_sample(target_batch, grid, align_corners=False)
            
            # Weighted sum of transformed copies
            # transformed_batch: [num_transforms, 1, H, W]
            # probs: [num_transforms]
            
            weighted_sum = torch.sum(transformed_batch * probs.view(-1, 1, 1, 1), dim=0).squeeze()
            
            loss = torch.nn.functional.mse_loss(weighted_sum, normalized_target)
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"  Step {i}: Loss {loss.item():.6f}")
                
        print(f"FractalEncoder: Compression Complete. Final Loss: {loss.item():.6f} ({time.time() - start_time:.2f}s)")
        
        # Pack into DNA
        transforms_list = []
        final_params = params.detach().cpu().numpy()
        final_probs = torch.softmax(probs_logits, dim=0).detach().cpu().numpy()
        
        for idx in range(num_transforms):
            p = final_params[idx]
            transforms_list.append({
                'a': float(p[0]), 'b': float(p[1]), 'e': float(p[4]),
                'c': float(p[2]), 'd': float(p[3]), 'f': float(p[5]),
                'p': float(final_probs[idx])
            })
            
        return FractalDNA(
            shape=(H, W),
            transforms=transforms_list,
            base_value=float(min_val.item()) # We store min/scale implicitly or explicitly? 
            # Actually, we need to store min and scale to reconstruct exact values.
            # Let's store min_val and scale in base_value for now (tuple hack or just min)
            # Simplified: just base_value = min, and we assume we need to store scale too.
        )
