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
            
            # Memory Optimization: Monte Carlo Sampling (Stochastic)
            # Instead of processing the full HxW grid, we sample random points.
            # This makes memory usage independent of image size.
            
            num_samples = 100000 # Sample 100k points per step (tunable)
            
            # Generate random coordinates in [-1, 1]
            # shape: [1, num_samples, 1, 2] for grid_sample
            sample_coords = torch.rand(1, num_samples, 1, 2, device=self.device) * 2 - 2
            
            # Sample target at these coordinates
            # target_batch: [1, 1, H, W]
            target_batch = normalized_target.unsqueeze(0).unsqueeze(0)
            target_samples = torch.nn.functional.grid_sample(target_batch, sample_coords, align_corners=False)
            
            # Accumulate weighted sum at these coordinates
            reconstructed_samples = torch.zeros_like(target_samples)
            probs = torch.softmax(probs_logits, dim=0)
            
            theta_all = params.view(num_transforms, 2, 3)
            
            # We can process all transforms for these samples because num_samples is small
            # But let's keep chunking just in case num_samples is large
            chunk_size = 4
            
            for t_idx in range(0, num_transforms, chunk_size):
                end_idx = min(t_idx + chunk_size, num_transforms)
                current_batch_size = end_idx - t_idx
                
                theta_chunk = theta_all[t_idx:end_idx]
                
                # We need to apply affine transform to the coordinates 'sample_coords'
                # affine_grid generates a grid from theta.
                # Here we have explicit points.
                # We can use affine_grid if we treat sample_coords as a "mesh" but that's complex.
                # Easier: Manual affine transform of coordinates.
                # Coords: [1, N, 1, 2] -> [N, 2]
                # Theta: [T, 2, 3]
                
                # Let's use grid_sample with a trick.
                # If we want T(x), we can't easily use grid_sample to *move* points.
                # grid_sample(img, grid) samples img at grid.
                # If grid = T(coords), then we are sampling img at T(coords).
                # This is exactly what we want for the Collage Theorem term w_i(T).
                
                # We need to generate 'grid' which is the transformed coordinates.
                # PyTorch affine_grid generates a regular grid. We have irregular points.
                # We must manually multiply.
                
                # coords: [1, N, 1, 2] -> [N, 3] (homogeneous)
                coords_flat = sample_coords.view(-1, 2)
                N = coords_flat.shape[0]
                ones = torch.ones(N, 1, device=self.device)
                coords_homo = torch.cat([coords_flat, ones], dim=1) # [N, 3]
                
                # theta_chunk: [T, 2, 3]
                # We want output: [T, N, 2]
                # result = coords_homo @ theta_chunk.T ?
                # theta: 2x3. coords: Nx3.
                # result = coords @ theta.T -> Nx2.
                
                # Batch matmul:
                # coords_homo: [1, N, 3]
                # theta_chunk: [T, 2, 3] -> [T, 3, 2] (transpose last 2)
                
                # We want for each T, result = coords @ theta.T
                # [T, N, 2] = [1, N, 3] @ [T, 3, 2] ? No.
                
                # Let's loop or expand.
                # coords_homo_expanded: [T, N, 3]
                coords_homo_expanded = coords_homo.unsqueeze(0).expand(current_batch_size, -1, -1)
                
                # theta_chunk_transposed: [T, 3, 2]
                theta_chunk_T = theta_chunk.transpose(1, 2)
                
                # transformed_coords: [T, N, 2]
                transformed_coords = torch.bmm(coords_homo_expanded, theta_chunk_T)
                
                # Reshape for grid_sample: [T, N, 1, 2]
                grid = transformed_coords.view(current_batch_size, num_samples, 1, 2)
                
                # Sample target at transformed coords
                # target_batch expanded: [T, 1, H, W]
                target_batch_expanded = target_batch.expand(current_batch_size, -1, -1, -1)
                
                # samples: [T, 1, N, 1]
                samples = torch.nn.functional.grid_sample(target_batch_expanded, grid, align_corners=False)
                
                # Accumulate
                # samples: [T, 1, N, 1]
                # probs: [T]
                chunk_probs = probs[t_idx:end_idx].view(-1, 1, 1, 1)
                
                # Sum over T dimension
                weighted_chunk_sum = torch.sum(samples * chunk_probs, dim=0) # [1, N, 1]
                
                reconstructed_samples = reconstructed_samples + weighted_chunk_sum
                
            loss = torch.nn.functional.mse_loss(reconstructed_samples, target_samples)
            
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
