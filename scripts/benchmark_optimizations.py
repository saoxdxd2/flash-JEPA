import torch
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.modules.neural_memory import TitansMemory
from models.liquid_sparse_vectorized import SparseVectorizedLiquidGraph
from models.infinite_liquid import InfiniteLiquidGraph
from models.procedural_liquid import ProceduralLiquidGraph

def benchmark_titans():
    print("=== Benchmarking TitansMemory ===")
    input_dim = 1024
    hidden_dim = 1024
    sparsity = 0.99
    num_steps = 100
    
    memory = TitansMemory(input_dim, hidden_dim, sparsity=sparsity)
    x = torch.randn(1, input_dim)
    
    # Warmup
    for _ in range(10):
        memory.observe(x)
        
    start_time = time.time()
    for _ in range(num_steps):
        memory.observe(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_steps
    print(f"Average observe() time: {avg_time:.6f}s")
    print(f"Throughput: {1/avg_time:.2f} steps/s")

def benchmark_sparse_liquid():
    print("\n=== Benchmarking SparseVectorizedLiquidGraph ===")
    input_size = 256
    hidden_size = 1024
    output_size = 10
    sparsity = 0.99
    num_steps = 100
    
    graph = SparseVectorizedLiquidGraph(input_size, hidden_size, output_size, sparsity=sparsity)
    inputs = torch.randn(1, input_size)
    
    # Warmup
    for _ in range(10):
        graph.forward(inputs)
        
    start_time = time.time()
    for _ in range(num_steps):
        graph.forward(inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_steps
    print(f"Average forward() time: {avg_time:.6f}s")
    print(f"Throughput: {1/avg_time:.2f} steps/s")

def benchmark_infinite_liquid():
    print("\n=== Benchmarking InfiniteLiquidGraph ===")
    input_size = 256
    hidden_size = 10000
    output_size = 10
    num_steps = 50
    
    # Use a temp directory for storage
    storage_path = "temp_infinite_storage"
    if os.path.exists(storage_path):
        import shutil
        shutil.rmtree(storage_path)
        
    graph = InfiniteLiquidGraph(input_size, hidden_size, output_size, storage_path=storage_path)
    inputs = torch.randn(1, input_size)
    
    # Warmup
    for _ in range(5):
        graph.forward(inputs)
        
    start_time = time.time()
    for _ in range(num_steps):
        graph.forward(inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_steps
    print(f"Average forward() time: {avg_time:.6f}s")
    print(f"Throughput: {1/avg_time:.2f} steps/s")
    
    # Cleanup
    del graph
    import gc
    gc.collect()
    
    import shutil
    try:
        shutil.rmtree(storage_path)
    except Exception as e:
        print(f"Warning: Could not cleanup storage path: {e}")

def benchmark_procedural_liquid():
    print("\n=== Benchmarking ProceduralLiquidGraph ===")
    input_size = 256
    hidden_size = 1000000 # 1M neurons
    output_size = 10
    num_steps = 50
    
    graph = ProceduralLiquidGraph(input_size, hidden_size, output_size)
    inputs = torch.randn(1, input_size)
    
    # Warmup
    for _ in range(5):
        graph.forward(inputs)
        
    start_time = time.time()
    for _ in range(num_steps):
        graph.forward(inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_steps
    print(f"Average forward() time: {avg_time:.6f}s")
    print(f"Throughput: {1/avg_time:.2f} steps/s")

if __name__ == "__main__":
    benchmark_titans()
    benchmark_sparse_liquid()
    benchmark_infinite_liquid()
    benchmark_procedural_liquid()
