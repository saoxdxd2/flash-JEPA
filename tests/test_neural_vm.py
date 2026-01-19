import torch
import torch.nn as nn
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.modules.neural_vm import VectorizedNPU, NeuralVirtualizationLayer
from models.neuromodulated_holographic import NeuromodulatedHolographicBrain

def test_vectorized_npu():
    print("Testing VectorizedNPU...")
    input_size = 32
    reg_size = 8
    mem_dim = 8
    max_npus = 16
    mem_size = 32
    
    npu = VectorizedNPU(input_size, reg_size, mem_dim, max_npus)
    
    batch_size = 2
    active_npus = 4
    
    x = torch.randn(batch_size, input_size)
    regs = torch.randn(batch_size, max_npus, reg_size)
    mem = torch.randn(batch_size, mem_size, mem_dim)
    
    out, new_regs, w_intent, ssd_intent = npu(x, regs, mem, active_npus)
    
    assert out.shape == (batch_size, active_npus, input_size)
    assert new_regs.shape == (batch_size, active_npus, reg_size)
    assert ssd_intent['key'].shape == (batch_size, active_npus, mem_dim)
    print("Vectorized NPU Test Passed!")

def test_virtualization_layer_vsync():
    print("Testing NeuralVirtualizationLayer with V-Sync...")
    input_size = 32
    hidden_size = 128
    
    vm = NeuralVirtualizationLayer(input_size, hidden_size, max_npus=64)
    
    x = torch.randn(2, input_size)
    
    # Warmup
    out, h, m = vm(x)
    
    # Reset history for predictable testing
    vm.perf_history = []
    vm.active_npus = 4 # Reset to known state
    
    print(f"Initial Active NPUs: {vm.active_npus}")
    
    # Simulate fast execution -> Should grow
    # We need enough samples to move the average if history wasn't cleared, 
    # but since we cleared it, one sample is enough?
    # update_vsync appends.
    for _ in range(10): # Add multiple fast samples to be sure
        vm.update_vsync(0.001) # Very fast (1ms)
    
    print(f"Active NPUs after fast execution: {vm.active_npus}")
    assert vm.active_npus > 4
    
    # Simulate slow execution -> Should shrink
    vm.update_vsync(0.1) # Slow (100ms)
    vm.update_vsync(0.1)
    
    print(f"Active NPUs after slow execution: {vm.active_npus}")
    
    print("V-Sync Test Passed!")

def test_brain_integration():
    print("Testing Brain Integration...")
    input_size = 10
    hidden_size = 128 
    output_size = 10
    
    brain = NeuromodulatedHolographicBrain(input_size, hidden_size, output_size)
    
    x = torch.randn(2, input_size)
    
    actions, value, energy, flash_data = brain(x)
    
    assert actions.shape == (2, output_size)
    print("Brain Integration Test Passed!")
    
    # Test Backward
    loss = actions.sum()
    loss.backward()
    print("Backward Pass Passed!")

def test_neural_ssd():
    print("Testing NeuralSSD...")
    from brain.modules.neural_ssd import NeuralSSD
    
    ssd = NeuralSSD(key_dim=8, value_dim=8, capacity=10)
    
    keys = torch.randn(2, 8)
    vals = torch.randn(2, 8)
    
    # Write
    ssd.write(keys, vals)
    
    # Read (Exact match check)
    read_vals, scores = ssd.read(keys, k=1)
    
    # Check if retrieved values are close to written values
    # read_vals: [Batch, 1, Dim]
    diff = (read_vals.squeeze(1) - vals).abs().sum()
    print(f"SSD Read Diff: {diff.item()}")
    assert diff.item() < 1e-5
    
    print("Neural SSD Test Passed!")

if __name__ == "__main__":
    test_vectorized_npu()
    test_virtualization_layer_vsync()
    test_brain_integration()
    test_neural_ssd()
