import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ecg import ModularBrain
from brain.genome import Genome

def test_modular_brain():
    print("Testing Modular Brain with Liquid Architecture...")
    
    # 1. Initialize Genome and Brain
    genome = Genome()
    input_size = 20
    hidden_size = 100
    output_size = 10
    
    brain = ModularBrain(input_size, hidden_size, output_size, genome=genome)
    print("ModularBrain Initialized.")
    
    # 2. Forward Pass
    input_vec = torch.randn(input_size)
    logits, params, energy = brain.forward(input_vec, dt=0.1)
    
    print(f"Logits Shape: {logits.shape}")
    print(f"Params Shape: {params.shape}")
    print(f"Energy: {energy:.4f}")
    
    assert logits.shape == (output_size,)
    assert params.shape == (2,)
    assert energy > 0
    
    # 3. Learning (Plasticity)
    print("\nTesting Learning...")
    brain.learn(reward=1.0)
    print("Learning Step Complete.")
    
    # 4. Mutation
    print("\nTesting Mutation...")
    brain.mutate()
    print("Mutation Complete.")
    
    print("All Modular Brain Tests Passed!")

if __name__ == "__main__":
    test_modular_brain()
