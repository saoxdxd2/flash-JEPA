import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n2 import HyperTransfer

def test_n2n2():
    print("--- Testing N2N2 HyperTransfer ---")
    
    # 1. Initialize Brain
    brain = EvolutionaryBrain()
    print("Brain initialized.")
    
    # 2. Initialize HyperTransfer
    n2n2 = HyperTransfer(brain)
    
    # 3. Simulate "Big Model" Source Data
    # A dictionary of {Word: EmbeddingVector}
    # Let's create some dummy embeddings for concepts
    vocab = ["APPLE", "BANANA", "CAT", "DOG", "ZEBRA"]
    source_data = {}
    
    print("Simulating Big Model Embeddings...")
    for word in vocab:
        # Random vector representing the "meaning" of the word in the Big Model
        # In reality, this would be from Word2Vec, GloVe, or BERT
        vec = torch.randn(256) 
        source_data[word] = vec
        
    # 4. Measure Weights BEFORE
    print("\nMeasuring initial synaptic weights...")
    initial_weights = []
    for expert in brain.broca.experts:
        # Flatten connections weights
        w = [c[1] for node_conns in expert.connections for c in node_conns]
        initial_weights.extend(w)
    initial_weights = np.array(initial_weights)
    print(f"Initial Mean Weight: {np.mean(initial_weights):.4f}")

    # 5. Load and Imprint
    n2n2.load_source_weights(source_data)
    n2n2.imprint_knowledge()
    
    # 6. Verify Synaptic Change
    print("\n--- Verifying Synaptic Plasticity ---")
    final_weights = []
    for expert in brain.broca.experts:
        w = [c[1] for node_conns in expert.connections for c in node_conns]
        final_weights.extend(w)
    final_weights = np.array(final_weights)
    
    weight_diff = np.abs(final_weights - initial_weights).sum()
    print(f"Total Weight Change (L1 Norm): {weight_diff:.4f}")
    
    if weight_diff > 0.001:
        print("  -> SUCCESS: Synapses have been modified (Learning occurred).")
    else:
        print("  -> FAILURE: No significant weight change.")

    # 7. Verify RAM Usage (Should be empty/low)
    if brain.memory.cursor == 0 and not brain.memory.full:
        print("  -> SUCCESS: RAM Memory is empty (Biological Mode confirmed).")
    else:
        print(f"  -> WARNING: RAM Memory has {brain.memory.cursor} entries.")

    # 8. Verify Output Consistency
    print("\n--- Verifying Output Consistency ---")
    test_word = vocab[0]
    test_vec = torch.tensor(source_data[test_word], dtype=torch.float32)
    # Project and Normalize manually to match n2n2 logic for testing
    if n2n2.projection_matrix is not None:
        test_vec = torch.matmul(test_vec, n2n2.projection_matrix)
    test_vec = torch.tanh(test_vec)
    
    out1 = brain.broca.process_text_embedding(test_vec)
    out2 = brain.broca.process_text_embedding(test_vec)
    
    dist = torch.dist(out1, out2)
    print(f"Output Stability (Distance between 2 passes): {dist.item():.6f}")
    if dist.item() < 1e-5:
        print("  -> SUCCESS: Network response is stable.")

    print("\nN2N2 Verification Complete.")

if __name__ == "__main__":
    test_n2n2()
