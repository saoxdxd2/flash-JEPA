import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.liquid import LiquidGraph

def test_liquid_architecture():
    print("Testing Continuous Liquid Architecture...")
    
    # 1. Initialize LiquidGraph
    input_size = 10
    hidden_size = 50
    output_size = 5
    graph = LiquidGraph(input_size, hidden_size, output_size)
    print("LiquidGraph Initialized.")
    
    # 2. Forward Pass (Time Step 1)
    inputs = np.random.randn(input_size).astype(np.float32)
    outputs, params, energy = graph.forward(inputs, dt=0.1)
    print(f"Time Step 1 Output Shape: {outputs.shape}")
    print(f"Params Shape: {params.shape}")
    print(f"Energy: {energy:.4f}")
    
    assert outputs.shape == (output_size,)
    assert params.shape == (2,)
    
    # 3. Forward Pass (Time Step 2 - Check Dynamics)
    outputs2, params2, energy2 = graph.forward(inputs, dt=0.1)
    print(f"Time Step 2 Output Shape: {outputs2.shape}")
    
    # Outputs should change due to internal state dynamics (x)
    diff = torch.norm(outputs - outputs2)
    print(f"Output Difference (Dynamics): {diff.item():.4f}")
    assert diff.item() > 0, "Outputs should change over time due to ODE dynamics!"
    
    # 4. Mutate
    graph.mutate()
    print("Mutation Successful.")
    
    # 5. Crossover
    graph2 = LiquidGraph(input_size, hidden_size, output_size)
    child = graph.crossover(graph2)
    print("Crossover Successful.")
    
    print("All Liquid Architecture Tests Passed!")

if __name__ == "__main__":
    test_liquid_architecture()
