import torch
import numpy as np
import os
import sys
import time
import random
from scipy.spatial.distance import cosine

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n2 import HyperTransfer

def run_meditation():
    print("--- Meditation Training: Associative Dreaming ---")
    
    # 1. Initialize Brain (Sensory Deprivation Mode)
    # We don't start the Retina, effectively blinding the agent.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Default to generation 349 if we can't find a better way, or maybe we should look for the latest?
    # For now, let's use the requested format with a default.
    generation = 349
    elite_path = os.path.join(project_root, "models", "saved", f"gen_{generation}_elite.pt")
    
    brain = EvolutionaryBrain()
    if os.path.exists(elite_path):
        print(f"Loading Master Brain from {elite_path}...")
        brain.load_model(elite_path)
    else:
        print("Warning: Master Brain not found. Starting fresh.")
        
    print("Brain initialized (Retina OFF).")
    
    # 2. Meditation Loop
    print("\n--- Entering Meditative State ---")
    print("Closing eyes... Disconnecting senses...")
    
    # We use Broca directly
    broca = brain.broca
    
    # Since we don't have the original Qwen embeddings or the projection matrix,
    # we will use "Random Neural Firing" (Noise) as the seed.
    # The brain's goal is to find stability (attractors) from this noise.
    # This is similar to how the brain generates dreams from random brainstem activity.
    
    rng = np.random.RandomState(42)
    
    for step in range(1000):
        # 1. Generate Seed (Hypnagogic Hallucination)
        # Generate a random 64-dim vector (simulating the compressed input)
        # We use normal distribution to match the expected input statistics
        compressed_seed = torch.randn(64) 
        compressed_seed = torch.tanh(compressed_seed) # Normalize to [-1, 1] range like the N2N2 inputs
        
        # 2. Inject into Broca
        # We use the 'process_text_embedding' hook we added for N2N2
        # This pushes the input into the Liquid SNNs
        brain_output = broca.process_text_embedding(compressed_seed) # [256]
        
        # 3. Let it Reverb (Dream)
        # We run the brain for a few more steps with *zero* input (or noise)
        # to see where the dynamics settle.
        # Broca SNNs have internal state (expert.prev_outputs).
        
        # 4. Measure Stability (Did it find an attractor?)
        # We compare output at t with output at t+1
        # For now, just a single pass is "Drift".
        
        # 5. Hebbian Update (Consolidation)
        # If the network produced a strong response (high magnitude), we reinforce it.
        response_strength = torch.norm(brain_output).item()
        
        if response_strength > 0.5:
            # It "recognized" or "reacted" to the seed.
            # Strengthen connections.
            # We use a small reward to trigger the PlasticityMLP
            reward = 0.1 * response_strength
            
            for expert in broca.experts:
                expert.learn(reward)
                
            print(f"Step {step}: RandomSeed -> Strength {response_strength:.4f} -> Consolidated.")
        else:
            print(f"Step {step}: RandomSeed -> Strength {response_strength:.4f} -> Faded.")
            
        if step % 100 == 0:
            # Update path with current generation in case it changed (unlikely in meditation but good practice)
            current_path = os.path.join(project_root, "models", "saved", f"gen_{brain.genome.generation}_elite.pt")
            print(f"Saving meditation progress to {current_path}...")
            brain.save_model(current_path)
            
    print("Meditation Complete.")

if __name__ == "__main__":
    run_meditation()
