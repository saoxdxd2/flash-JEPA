import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.neuromodulated_holographic import NeuromodulatedHolographicBrain

def test_4d_layers():
    print("Initializing 4D Brain...")
    input_size = 100
    hidden_size = 256
    output_size = 10
    
    brain = NeuromodulatedHolographicBrain(input_size, hidden_size, output_size)
    
    # 1. Forward Pass
    print("Testing Forward Pass...")
    x = torch.randn(2, input_size)
    actions, value, energy, flash_data = brain(x)
    
    print(f"Actions Shape: {actions.shape}")
    print(f"Value Shape: {value.shape}")
    
    assert actions.shape == (2, output_size)
    assert value.shape == (2, 1)
    
    # 2. Backward Pass
    print("Testing Backward Pass...")
    loss = actions.sum() + value.sum()
    loss.backward()
    
    print("Gradients computed successfully.")
    
    # 3. Resize
    print("Testing Resize...")
    brain.resize_hidden(512)
    
    x = torch.randn(2, input_size)
    actions, value, energy, flash_data = brain(x)
    
    print(f"Resized Actions Shape: {actions.shape}")
    assert actions.shape == (2, output_size)
    assert brain.hidden_size == 512
    
    print("SUCCESS: 4D Layers Verified!")

if __name__ == "__main__":
    test_4d_layers()
