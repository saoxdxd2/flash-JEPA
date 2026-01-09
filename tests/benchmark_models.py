import torch
import time
from models.neuromodulated_holographic import SparseLinear
from models.liquid_sparse_vectorized import SparseVectorizedLiquidGraph

def benchmark_sparse_linear():
    print("\nBenchmarking SparseLinear...")
    in_features = 1024
    out_features = 1024
    batch_size = 32
    sparsity = 0.99
    
    layer = SparseLinear(in_features, out_features, sparsity=sparsity)
    x = torch.randn(batch_size, in_features)
    
    # Warmup
    for _ in range(10):
        _ = layer(x)
        
    start_time = time.time()
    for _ in range(100):
        _ = layer(x)
    end_time = time.time()
    
    print(f"SparseLinear (Batch={batch_size}, Sparsity={sparsity}): {(end_time - start_time) * 1000 / 100:.2f} ms per forward pass")

def benchmark_liquid_sparse():
    print("\nBenchmarking SparseVectorizedLiquidGraph...")
    input_size = 256
    hidden_size = 1024
    output_size = 64
    batch_size = 1
    sparsity = 0.95
    
    model = SparseVectorizedLiquidGraph(input_size, hidden_size, output_size, sparsity=sparsity)
    inputs = torch.randn(batch_size, input_size)
    
    # Warmup
    for _ in range(10):
        _ = model(inputs)
        
    start_time = time.time()
    for _ in range(100):
        _ = model(inputs)
    end_time = time.time()
    
    print(f"SparseVectorizedLiquid (Batch={batch_size}, Sparsity={sparsity}): {(end_time - start_time) * 1000 / 100:.2f} ms per forward pass")

if __name__ == "__main__":
    benchmark_sparse_linear()
    benchmark_liquid_sparse()
