import torch
import numpy as np
import time
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n2 import HyperTransfer
from brain.genome import Genome

def benchmark_n2n2_imprinting():
    print("=== Benchmarking N2N2 Imprinting (x100-x1000 Goal) ===")
    
    # Setup Brain
    genome = Genome()
    brain = EvolutionaryBrain(genome)
    transfer = HyperTransfer(brain)
    
    # Mock Source Data (1000 concepts)
    num_concepts = 1000
    source_dim = 4096
    source_data = {f"word_{i}": np.random.randn(source_dim).astype(np.float32) for i in range(num_concepts)}
    
    # Load Source Weights (includes projection)
    transfer.load_source_weights(source_data, target_dim=64, max_concepts=num_concepts)
    
    # Benchmark Imprinting
    print(f"\nStarting imprinting of {num_concepts} concepts...")
    start_time = time.time()
    
    # We'll run for a few batches to get a stable estimate
    # Note: imprint_knowledge now uses batch_size=128 internally
    transfer.imprint_knowledge(save_interval=10000, use_ewc=False) # Disable EWC for pure throughput test
    
    end_time = time.time()
    total_time = end_time - start_time
    
    throughput = num_concepts / total_time
    print(f"\nTotal Time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} concepts/s")
    
    # Comparison with "Old" (Estimated)
    # Old was ~1 concept per step (forward + backward + optimizer)
    # On CPU/GPU, a single small step might take 0.05s - 0.1s.
    # So old throughput was ~10-20 concepts/s.
    # If we get > 1000 concepts/s, that's a 50x-100x speedup.
    
    if throughput > 1000:
        print(f"\nSUCCESS: Achieved massive speedup! ({throughput/20:.1f}x faster than estimated baseline)")
    else:
        print(f"\nSpeedup: {throughput/20:.1f}x")

if __name__ == "__main__":
    benchmark_n2n2_imprinting()
