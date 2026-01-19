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
        Compresses a single weight matrix into Fractal DNA.
        Wrapper around encode_batch for single item.
        """
        target_batch = target_weights.unsqueeze(0) # [1, H, W]
        dna_list = self.encode_batch(target_batch, num_transforms, iterations)
        return dna_list[0]

    def encode_batch(self, target_batch: torch.Tensor, num_transforms=8, iterations=1000) -> List[FractalDNA]:
        """
        Compresses a batch of weight matrices into Fractal DNA.
        Args:
            target_batch: [Batch, H, W]
        """
        B, H, W = target_batch.shape
        target = target_batch.to(self.device).float()
        
        # Normalize each item in batch independently
        # min_vals: [B, 1, 1]
        min_vals = target.amin(dim=(1, 2), keepdim=True)
        max_vals = target.amax(dim=(1, 2), keepdim=True)
        scales = max_vals - min_vals
        scales[scales == 0] = 1.0
        
        normalized_target = (target - min_vals) / scales
        
        # Parameters: [B, num_transforms, 6]
        # We use a neural network approach to optimize the parameters
        params = torch.randn(B, num_transforms, 6, device=self.device, requires_grad=True)
        # Probabilities: [B, num_transforms]
        probs_logits = torch.randn(B, num_transforms, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([params, probs_logits], lr=0.01)
        
        # print(f"FractalEncoder: Compressing Batch of {B} matrices ({H}x{W}) using {num_transforms} transforms...")
        start_time = time.time()
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Monte Carlo Sampling
            num_samples = 100000 
            
            # Generate random coordinates in [-1, 1]
            # shape: [1, num_samples, 1, 2] -> Shared across batch for efficiency
            sample_coords = torch.rand(1, num_samples, 1, 2, device=self.device) * 2 - 2
            
            # Sample targets
            # target_batch: [B, 1, H, W] (unsqueeze channel)
            target_input = normalized_target.unsqueeze(1)
            # sample_coords expanded: [B, num_samples, 1, 2]
            sample_coords_expanded = sample_coords.expand(B, -1, -1, -1)
            
            target_samples = torch.nn.functional.grid_sample(target_input, sample_coords_expanded, align_corners=False)
            # target_samples: [B, 1, N, 1]
            
            # Accumulate weighted sum
            # reconstructed_samples = torch.zeros_like(target_samples)
            probs = torch.softmax(probs_logits, dim=1) # [B, T]
            
            theta_all = params.view(B, num_transforms, 2, 3)
            
            # Process transforms
            # We can process all transforms in parallel if memory allows.
            # B*T*N*2 floats. 
            
            # coords_homo: [1, N, 3]
            coords_flat = sample_coords.view(1, num_samples, 2)
            ones = torch.ones(1, num_samples, 1, device=self.device)
            coords_homo = torch.cat([coords_flat, ones], dim=2) # [1, N, 3]
            
            # Expand coords for Batch and Transforms
            # We want [B, T, N, 3]
            coords_expanded = coords_homo.unsqueeze(1).expand(B, num_transforms, -1, -1) # [B, T, N, 3]
            
            # Theta: [B, T, 2, 3] -> Transpose to [B, T, 3, 2]
            theta_T = theta_all.transpose(2, 3)
            
            # Matmul: [B, T, N, 3] @ [B, T, 3, 2] -> [B, T, N, 2]
            transformed_coords = torch.matmul(coords_expanded, theta_T)
            
            # Reshape for grid_sample
            # We need to sample from target_input [B, 1, H, W]
            # But we have T grids per batch item.
            # We can fold B*T into batch dim?
            # target_input expanded: [B*T, 1, H, W]
            target_folded = target_input.repeat_interleave(num_transforms, dim=0)
            
            # grid folded: [B*T, N, 1, 2]
            grid_folded = transformed_coords.view(B * num_transforms, num_samples, 1, 2)
            
            # Sample
            samples_folded = torch.nn.functional.grid_sample(target_folded, grid_folded, align_corners=False)
            # samples_folded: [B*T, 1, N, 1]
            
            # Unfold
            samples = samples_folded.view(B, num_transforms, 1, num_samples, 1)
            
            # Weighting
            # probs: [B, T] -> [B, T, 1, 1, 1]
            probs_expanded = probs.view(B, num_transforms, 1, 1, 1)
            
            # Sum over T
            weighted_sum = torch.sum(samples * probs_expanded, dim=1) # [B, 1, N, 1]
            
            loss = torch.nn.functional.mse_loss(weighted_sum, target_samples)
            
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                # Check max loss in batch? Or mean?
                # Optimization minimizes mean loss over batch.
                if loss.item() < 0.001:
                     # print(f"  Step {i}: Loss {loss.item():.6f} (Early Stop)")
                     break
                # print(f"  Step {i}: Loss {loss.item():.6f}")
                
        # print(f"FractalEncoder: Batch Compression Complete. Final Loss: {loss.item():.6f} ({time.time() - start_time:.2f}s)")
        
        # Pack results
        dna_list = []
        final_params = params.detach().cpu().numpy()
        final_probs = probs.detach().cpu().numpy()
        min_vals_cpu = min_vals.detach().cpu().numpy().flatten()
        
        for b in range(B):
            transforms_list = []
            for idx in range(num_transforms):
                p = final_params[b, idx]
                transforms_list.append({
                    'a': float(p[0]), 'b': float(p[1]), 'e': float(p[4]),
                    'c': float(p[2]), 'd': float(p[3]), 'f': float(p[5]),
                    'p': float(final_probs[b, idx])
                })
            
            dna_list.append(FractalDNA(
                shape=(H, W),
                transforms=transforms_list,
                base_value=float(min_vals_cpu[b]) 
            ))
            
        return dna_list
