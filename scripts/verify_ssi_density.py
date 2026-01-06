import torch
import sys
import os
import psutil
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.lifecycle import LifecycleManager

def test_holographic_growth():
    print("\n=== Testing Holographic Density Scaling ===")
    brain = EvolutionaryBrain()
    
    # Mock ResourceMonitor to return high RAM usage (e.g., 120GB)
    # The limit in mutate_adaptive is 100,000 MB (100GB)
    brain.monitor.get_usage = MagicMock(return_value=(120000, 50.0)) 
    
    initial_hidden = brain.genome.hidden_size
    initial_rank = brain.genome.rank
    initial_latent = brain.genome.latent_dim
    initial_sparsity = brain.genome.sparsity
    
    print(f"Initial: Hidden={initial_hidden}, Rank={initial_rank}, Latent={initial_latent}, Sparsity={initial_sparsity:.4f}")
    
    # Trigger mutation with high efficiency to force growth
    brain.mutate_adaptive(efficiency=2.0)
    
    print(f"After Mutation (High RAM):")
    print(f"  Hidden: {brain.genome.hidden_size}")
    print(f"  Rank: {brain.genome.rank}")
    print(f"  Latent: {brain.genome.latent_dim}")
    print(f"  Sparsity: {brain.genome.sparsity:.4f}")
    
    assert brain.genome.hidden_size == initial_hidden, "Hidden size should not grow under RAM pressure"
    assert brain.genome.rank > initial_rank, "Rank should grow under RAM pressure (Holographic)"
    assert brain.genome.latent_dim > initial_latent, "Latent dim should grow under RAM pressure (Holographic)"
    assert brain.genome.sparsity > initial_sparsity, "Sparsity should grow under RAM pressure (Protection)"
    
    print("SUCCESS: Holographic Density Scaling verified.")

def test_ssi_phase():
    print("\n=== Testing State-Space Imprinting (SSI) Phase ===")
    manager = LifecycleManager()
    
    # Mock Qwen3Teacher to return a dummy trajectory
    L = manager.brain.genome.latent_dim
    bus_size = manager.brain.trm.bus_size
    motor_hidden = manager.brain.trm.motor_hidden
    input_size = manager.brain.input_size
    
    # [T, Batch, Dim]
    input_seq = torch.randn(5, 1, input_size)
    # Target for Visual Cortex should match visual_hidden
    visual_h_seq = torch.randn(5, 1, manager.brain.trm.visual_hidden)
    # Target for Motor Cortex should match motor_hidden
    motor_h_seq = torch.randn(5, 1, manager.brain.trm.motor_hidden)
    # Bus trajectory (Input for Motor, Output for Visual)
    bus_seq = torch.randn(5, 1, manager.brain.trm.bus_size)
    
    manager.qwen_teacher.generate_logic_trajectory = MagicMock(return_value=(input_seq, visual_h_seq, motor_h_seq, bus_seq))
    
    print("Running phase_logic_transfer...")
    # New signature: (input_sequence, target_h_visual_sequence=None, target_h_motor_sequence=None, target_bus_sequence=None, lr=0.001)
    avg_loss = manager.phase_logic_transfer(steps=2)
    
    print(f"SSI Average Loss: {avg_loss:.4f}")
    assert avg_loss > 0, "SSI should produce a loss value"
    
    print("SUCCESS: SSI Phase verified.")

if __name__ == "__main__":
    try:
        test_holographic_growth()
        test_ssi_phase()
        print("\nALL VERIFICATIONS PASSED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
