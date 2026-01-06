import sys
import os
import time
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.liquid import LiquidGraph
from models.liquid_vectorized import VectorizedLiquidGraph

def benchmark():
    print("--- Liquid Neural Network Benchmark ---")
    
    input_size = 256
    hidden_size = 1024 # Scale up to see difference
    output_size = 64
    steps = 100
    
    print(f"Config: Input={input_size}, Hidden={hidden_size}, Output={output_size}, Steps={steps}")
    
    # 1. Legacy Graph
    print("\nInitializing Legacy LiquidGraph...")
    legacy = LiquidGraph(input_size, hidden_size, output_size)
    
    # Warmup
    input_vec = torch.randn(input_size)
    legacy.forward(input_vec)
    
    start_time = time.time()
    for _ in range(steps):
        input_vec = torch.randn(input_size)
        legacy.forward(input_vec)
    legacy_time = time.time() - start_time
    print(f"Legacy Time: {legacy_time:.4f}s ({steps/legacy_time:.2f} iter/s)")
    
    # 2. Vectorized Graph
    print("\nInitializing VectorizedLiquidGraph...")
    vec = VectorizedLiquidGraph(input_size, hidden_size, output_size)
    
    # Warmup
    input_vec = torch.randn(input_size)
    vec.forward(input_vec)
    
    start_time = time.time()
    for _ in range(steps):
        input_vec = torch.randn(input_size)
        vec.forward(input_vec)
    vec_time = time.time() - start_time
    print(f"Vectorized Time: {vec_time:.4f}s ({steps/vec_time:.2f} iter/s)")
    
    speedup = legacy_time / vec_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # 3. Correctness Check
    print("\n--- Correctness Check ---")
    # Convert legacy to vectorized and check if outputs match
    vec_converted = VectorizedLiquidGraph.from_legacy(legacy)
    
    # Reset states
    legacy.prev_outputs = np.zeros(hidden_size)
    vec_converted.y.zero_()
    vec_converted.x.zero_()
    # We need to manually sync the internal 'x' state of legacy nodes to vec_converted
    for i, node in enumerate(legacy.nodes):
        node.x = 0.0
        
    input_vec = torch.randn(input_size)
    
    # Run 1 step
    out_legacy, _, _ = legacy.forward(input_vec)
    out_vec, _, _ = vec_converted.forward(input_vec)
    
    diff = torch.abs(out_legacy - out_vec).mean().item()
    print(f"Output Difference (Mean Abs): {diff:.6f}")
    
    # DEBUG: Inspect Neuron 0
    print("\n--- Debug Neuron 0 ---")
    l_node = legacy.nodes[0]
    v_tau = vec_converted.tau[0].item()
    v_bias = vec_converted.bias[0].item()
    v_x = vec_converted.x[0].item()
    v_y = vec_converted.y[0].item()
    
    print(f"Legacy: Tau={l_node.tau:.4f}, Bias={l_node.bias:.4f}, X={l_node.x:.4f}, Y={l_node.y:.4f}")
    print(f"Vector: Tau={v_tau:.4f}, Bias={v_bias:.4f}, X={v_x:.4f}, Y={v_y:.4f}")
    
    # Check Weights for Neuron 0
    l_conns = legacy.connections[0]
    # Sum of weights
    l_w_sum = sum([w for _, w in l_conns])
    v_w_sum = vec_converted.weights[0].sum().item() # Note: This sums ALL weights, including masked ones (which should be 0 or ignored)
    # Wait, masked weights in Vectorized are NOT zeroed in the parameter tensor, only masked during forward!
    # But in from_legacy, I zeroed them: vec_graph.weights.data.zero_()
    
    print(f"Legacy Weights Sum: {l_w_sum:.4f}")
    print(f"Vector Weights Sum: {v_w_sum:.4f}")

    if diff < 1e-5:
        print("SUCCESS: Outputs match!")
    else:
        print("WARNING: Outputs diverge.")

if __name__ == "__main__":
    benchmark()
