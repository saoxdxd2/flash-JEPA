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
        
        # Store DNA as Parameters for Differentiability
        self.dna_params = nn.Parameter(self._dna_to_tensor(dna))
        self.dna_shape = dna.shape
        self.dna_base_value = dna.base_value
        
        # Bias is usually small, so we store it directly as a Parameter
        if bias is not None:
            self.bias = nn.Parameter(bias.to(device))
        else:
            self.register_parameter('bias', None)
            
        self.ribosome = NeuralRibosome(device=device)
        self.cached_weight = None
        
    def _dna_to_tensor(self, dna):
        """Converts DNA transforms to a flat tensor."""
        data = []
        for t in dna.transforms:
            data.extend([t['a'], t['b'], t['c'], t['d'], t['e'], t['f'], t['p']])
        return torch.tensor(data, device=self.device)
        
    def _tensor_to_dna(self, tensor):
        """Converts flat tensor back to FractalDNA object (for Ribosome)."""
        transforms = []
        num_transforms = len(tensor) // 7
        for i in range(num_transforms):
            p = tensor[i*7 : (i+1)*7]
            transforms.append({
                'a': p[0], 'b': p[1], 'c': p[2], 'd': p[3], 'e': p[4], 'f': p[5], 'p': p[6]
            })
        return FractalDNA(self.dna_shape, transforms, self.dna_base_value)

    def chaos_game_dot_product(self, x, dna_params, shape, iterations=100):
        """
        Calculates Y = X @ W without materializing W.
        Uses the 'Chaos Game' logic:
        W is the attractor of the IFS.
        The dot product can be estimated by sampling points on the attractor.
        """
        H, W = shape
        B = x.shape[0]
        num_transforms = len(dna_params) // 7
        params = dna_params.view(num_transforms, 7)
        
        # Transforms: [T, 2, 3]
        theta = torch.stack([
            torch.stack([params[:, 0], params[:, 1], params[:, 4]], dim=1),
            torch.stack([params[:, 2], params[:, 3], params[:, 5]], dim=1)
        ], dim=1)
        
        probs = F.softmax(params[:, 6], dim=0)
        
        # Sample Points (Chaos Game)
        num_points = 1000 
        points = torch.rand(num_points, 2, device=self.device) * 2 - 1
        
        for _ in range(10): # Warmup
            t_idx = torch.multinomial(probs, num_points, replacement=True)
            t_theta = theta[t_idx]
            points_homo = torch.cat([points, torch.ones(num_points, 1, device=self.device)], dim=1).unsqueeze(2)
            points = torch.bmm(t_theta, points_homo).squeeze(2)
            
        y_idx = ((points[:, 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
        x_idx = ((points[:, 0] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
        
        y = torch.zeros(B, H, device=self.device)
        x_vals = x[:, x_idx]
        y_idx_expanded = y_idx.unsqueeze(0).expand(B, -1)
        y.scatter_add_(1, y_idx_expanded, x_vals)
        
        return y / num_points

    def forward(self, x):
        # Fix 4: Chaos Game Inference (RAM Wall)
        if self.out_features * self.in_features > 100_000_000:
            return self.chaos_game_dot_product(x, self.dna_params, self.dna_shape) + (self.bias if self.bias is not None else 0)

        if self.cached_weight is None:
            weight = self.ribosome.transcribe_differentiable(
                self.dna_params, 
                self.dna_shape, 
                self.dna_base_value,
                iterations=20
            )
            
            if weight.shape != (self.out_features, self.in_features):
                weight = F.interpolate(weight.unsqueeze(0).unsqueeze(0), 
                                     size=(self.out_features, self.in_features), 
                                     mode='bilinear', align_corners=False).squeeze()
        else:
            weight = self.cached_weight
            
        return F.linear(x, weight, self.bias)
        
    def train(self, mode=True):
        super().train(mode)
        
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, fractal=True'
