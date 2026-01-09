import torch
import torch.nn as nn
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome
from brain.n2n2 import HyperTransfer
import os

def verify_all():
    print("=== Final Verification of Strategic Roadmap ===")
    
    # 1. Setup
    genome = Genome()
    # Force some values to verify they are used
    genome.latent_dim = 512
    genome.hidden_size = 1024
    genome.input_boost_factor = 25.0
    
    print("\n--- Phase 1: Unified HAL (Genomic Expansion) ---")
    brain = EvolutionaryBrain(genome)
    
    # Check if brain respects genome
    print(f"Genome Latent Dim: {genome.latent_dim}")
    print(f"Brain Latent Dim: {brain.latent_dim}")
    print(f"Broca Embedding Dim: {brain.broca.embedding_dim}")
    print(f"Visual Cortex Hidden Size: {brain.trm.visual_cortex.hidden_size}")
    
    if brain.latent_dim == 512 and brain.broca.embedding_dim == 512:
        print("SUCCESS: Phase 1 (Genomic Expansion) verified.")
    else:
        print("FAILED: Phase 1 (Genomic Expansion) - Dimension mismatch.")

    print("\n--- Phase 2: Learned Projections (N2N2 3.0) ---")
    hyper = HyperTransfer(brain)
    # Mock source data (Llama-like embeddings)
    source_data = {"test": np.random.randn(4096)}
    hyper.load_source_weights(source_data, target_dim=brain.latent_dim)
    
    print(f"Projection Adapter: {hyper.projection_adapter}")
    if isinstance(hyper.projection_adapter, nn.Linear) and hyper.projection_adapter.in_features == 4096:
        print("SUCCESS: Phase 2 (Learned Projections) - Adapter initialized.")
    else:
        print("FAILED: Phase 2 (Learned Projections) - Adapter missing or wrong size.")

    print("\n--- Phase 3: Dynamic Expert Sprouting (Broca 2.0) ---")
    initial_experts = brain.broca.num_experts
    print(f"Initial Experts: {initial_experts}")
    
    # Trigger sprouting by injecting high surprise
    # We'll simulate 50 high surprise events
    for _ in range(50):
        brain.broca._check_growth(surprise=1.0) # 1.0 > sprouting_threshold
        
    print(f"Experts after sprouting: {brain.broca.num_experts}")
    if brain.broca.num_experts > initial_experts:
        print("SUCCESS: Phase 3 (Dynamic Expert Sprouting) verified.")
    else:
        print("FAILED: Phase 3 (Dynamic Expert Sprouting) - No sprouting occurred.")

    print("\n--- Phase 4: System 1/2 Optimization (ONNX) ---")
    onnx_path = "final_verify.onnx"
    success = brain.export_reflex_path(onnx_path)
    
    if success and os.path.exists(onnx_path):
        print("SUCCESS: Phase 4 (ONNX Export) verified.")
        # Test hybrid switching logic
        brain.use_onnx = True
        # High confidence dummy input
        dummy_input = torch.randn(1, brain.input_size)
        action, logits = brain.decide(dummy_input)
        print(f"Decide Action (ONNX): {action}")
        print(f"Last Used System: {brain.last_used_system}") # Should be 1 (Basal Ganglia) or similar
        
        # Benchmarking
        import time
        start = time.time()
        for _ in range(10):
            brain.decide(dummy_input)
        elapsed = time.time() - start
        print(f"ONNX Inference Time (10 steps): {elapsed*1000:.2f}ms")
    else:
        print("FAILED: Phase 4 (ONNX Export) failed.")

    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    if os.path.exists(onnx_path + ".meta"):
        os.remove(onnx_path + ".meta")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    import numpy as np
    verify_all()
