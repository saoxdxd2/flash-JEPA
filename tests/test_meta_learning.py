import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ecg import EvolutionaryCodeGraph

def test_meta_learning():
    print("Testing Meta-Evolutionary Architecture...")
    
    # 1. Initialize ECG
    input_size = 10
    hidden_size = 20
    output_size = 2
    ecg = EvolutionaryCodeGraph(input_size, hidden_size, output_size)
    print("ECG Initialized.")
    
    # 2. Forward Pass
    inputs = np.random.randn(input_size).astype(np.float32)
    logits, params, energy = ecg.forward(inputs)
    print(f"Forward Pass Successful. Energy: {energy:.2f}")
    
    # 3. Learn (Meta-Learning Step)
    reward = 1.0
    ecg.learn(reward)
    print("Meta-Learning Step (PlasticityMLP) Successful.")
    
    # 4. Mutate
    ecg.mutate()
    print("Mutation Successful.")
    
    # 5. Crossover
    ecg2 = EvolutionaryCodeGraph(input_size, hidden_size, output_size)
    child = ecg.crossover(ecg2)
    print("Crossover Successful.")
    
    print("All Meta-Learning Tests Passed!")

if __name__ == "__main__":
    test_meta_learning()
