import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain

def verify_brain():
    print("--- Brain Verification ---")
    brain = EvolutionaryBrain()
    latest = brain.find_latest_checkpoint("models/saved")
    if latest:
        print(f"Loading Brain from {latest}...")
        brain.load_model(latest)
    else:
        print("Error: No checkpoint found.")
        return

    print(f"Brain Dimensions: Hidden={brain.genome.hidden_size}, Latent={brain.genome.latent_dim}")
    print(f"Broca Experts: {len(brain.broca.experts)}")
    
    # Test Language Processing
    test_text = "hello"
    print(f"\nProcessing text: '{test_text}'")
    semantic_vector = brain.broca.process_text(test_text)
    
    print(f"Semantic Vector Shape: {semantic_vector.shape}")
    
    if semantic_vector.shape[-1] == brain.genome.latent_dim:
        print("SUCCESS: Semantic vector matches latent dimension.")
    else:
        print(f"FAILURE: Semantic vector dimension mismatch ({semantic_vector.shape[-1]} vs {brain.genome.latent_dim})")

    # Test Lexical Grounding
    if hasattr(brain.broca, 'lexical_knowledge') and brain.broca.lexical_knowledge is not None:
        print(f"Lexical Knowledge: {brain.broca.lexical_knowledge.shape[0]} concepts grounded.")
    else:
        print("Warning: No lexical knowledge found.")

if __name__ == "__main__":
    verify_brain()
