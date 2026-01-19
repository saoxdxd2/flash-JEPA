import torch
import torch.nn as nn
from brain.modules.fractal_layers import FractalLinear
from brain.fnd_encoder import FractalDNA
from brain.modules.neural_vm import NeuralVirtualizationLayer
from brain.modules.predictive_retina import PredictiveRetina
from brain.modules.neural_memory import TitansMemory

def test_differentiable_ifs():
    print("Testing Fix 1: Differentiable IFS...")
    dna = FractalDNA(shape=(64, 64), transforms=[{'a':0.5, 'b':0, 'c':0, 'd':0.5, 'e':0, 'f':0, 'p':1.0}], base_value=0.0)
    layer = FractalLinear(64, 64, dna, device='cpu')
    
    x = torch.randn(1, 64, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()
    
    if layer.dna_params.grad is not None:
        print("✅ Success: Gradients flow to DNA parameters.")
    else:
        print("❌ Failure: DNA parameters have no gradients.")

def test_vsync_stability():
    print("\nTesting Fix 2: Invariant Aggregation...")
    vm = NeuralVirtualizationLayer(input_size=32, hidden_size=64, max_npus=16)
    x = torch.randn(1, 32)
    
    # Run with 4 NPUs
    vm.active_npus = 4
    out4, _, _ = vm(x)
    
    # Run with 16 NPUs
    vm.active_npus = 16
    out16, _, _ = vm(x)
    
    diff = torch.abs(out4 - out16).mean().item()
    print(f"Mean difference between 4 and 16 NPUs: {diff:.6f}")
    if diff < 0.5: # LayerNorm should keep them in the same ballpark
        print("✅ Success: Aggregation is relatively invariant.")
    else:
        print("❌ Failure: Significant jitter detected.")

def test_chaos_inference():
    print("\nTesting Fix 4: Chaos Game Inference...")
    # Use a large layer to trigger chaos game
    dna = FractalDNA(shape=(1000, 1000), transforms=[{'a':0.5, 'b':0, 'c':0, 'd':0.5, 'e':0, 'f':0, 'p':1.0}], base_value=0.0)
    layer = FractalLinear(1000, 1000, dna, device='cpu')
    
    x = torch.randn(1, 1000)
    
    # Force Chaos Game by setting threshold low in code or just calling it
    output = layer.chaos_game_dot_product(x, layer.dna_params, layer.dna_shape)
    
    print(f"Chaos Game Output Shape: {output.shape}")
    if output.shape == (1, 1000):
        print("✅ Success: Chaos Game dot product executed.")
    else:
        print("❌ Failure: Incorrect output shape.")

if __name__ == "__main__":
    test_differentiable_ifs()
    test_vsync_stability()
    test_chaos_inference()
