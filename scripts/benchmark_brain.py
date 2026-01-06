import time
import torch
import psutil
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from models.liquid_sparse_vectorized import SparseVectorizedLiquidGraph

def benchmark_model(checkpoint_path):
    print(f"================================================================")
    print(f"BENCHMARK REPORT: {checkpoint_path}")
    print(f"================================================================")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: File {checkpoint_path} not found.")
        return

    # 1. Load Time
    print("\n[1] LOADING PERFORMANCE")
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    
    brain = EvolutionaryBrain()
    brain.load_model(checkpoint_path)
    
    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024
    
    print(f"  > Load Time: {end_time - start_time:.4f} seconds")
    print(f"  > RAM Usage: {mem_after - mem_before:.2f} MB (Model + Overhead)")
    
    # 2. Architecture Analysis
    print("\n[2] ARCHITECTURE ANALYSIS")
    print(f"  > Genome Hidden Size: {brain.genome.hidden_size}")
    print(f"  > Input Size: {brain.input_size}")
    print(f"  > Action Size: {brain.action_size}")
    
    trm = brain.trm
    print(f"  > TRM Type: {type(trm)}")
    
    total_params = 0
    active_params = 0
    
    def analyze_graph(name, graph):
        nonlocal total_params, active_params
        print(f"  --- {name} ---")
        print(f"    Type: {type(graph)}")
        print(f"    Hidden Size: {graph.hidden_size}")
        
        p_count = sum(p.numel() for p in graph.parameters())
        total_params += p_count
        
        if isinstance(graph, SparseVectorizedLiquidGraph):
            # Calculate active connections
            active = graph.weight_values.numel()
            density = active / (graph.input_size * graph.hidden_size + graph.hidden_size * graph.hidden_size) # approx
            print(f"    Sparsity: {graph.sparsity:.2f} (Target)")
            print(f"    Active Weights: {active:,}")
            print(f"    Total Parameters: {p_count:,}")
            print(f"    Effective Density: {density*100:.2f}%")
            active_params += active
        else:
            print(f"    Total Parameters: {p_count:,}")
            active_params += p_count
            print(f"    (Dense Layer)")

    if hasattr(trm, 'visual_cortex'):
        analyze_graph("Visual Cortex", trm.visual_cortex)
    if hasattr(trm, 'motor_cortex'):
        analyze_graph("Motor Cortex", trm.motor_cortex)
        
    print(f"  --- Summary ---")
    print(f"  > Total Parameters (Allocated): {total_params:,}")
    print(f"  > Active Parameters (Compute): {active_params:,}")
    
    # 3. Component Health
    print("\n[3] COMPONENT HEALTH")
    if hasattr(brain.broca, 'titans'):
        print(f"  > Titans Memory: PRESENT")
        print(f"    Sparsity: {brain.broca.titans.sparsity}")
    else:
        print(f"  > Titans Memory: ABSENT")
        
    print(f"  > Retina Resolution: {brain.retina.fovea_size}px")
    
    # 4. Inference Performance
    print("\n[4] INFERENCE PERFORMANCE")
    print("  Running 100 forward passes...")
    
    latencies = []
    dummy_input = torch.randn(1, brain.input_size)
    
    # Warmup
    for _ in range(10):
        brain.decide(dummy_input)
        
    for _ in range(100):
        t0 = time.time()
        brain.decide(dummy_input)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms
        
    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    throughput = 1000 / avg_lat
    
    print(f"  > Avg Latency: {avg_lat:.2f} ms +/- {std_lat:.2f}")
    print(f"  > Min/Max: {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  > Throughput: {throughput:.2f} Hz (decisions/sec)")
    
    print("\n================================================================")
    print("BENCHMARK COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    benchmark_model("models/saved/gen_349_elite.pt")
