import torch
import numpy as np
import os
import sys
import time
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain

def run_meditation_v2(steps=1000, batch_size=16):
    print("--- Meditation 2.0: Attractor Stabilization & Consolidation ---")
    
    # 1. Initialize Brain
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    latest_checkpoint = EvolutionaryBrain.find_latest_checkpoint(os.path.join(project_root, "models", "saved"))
    
    brain = EvolutionaryBrain()
    if latest_checkpoint:
        print(f"Loading Brain from {latest_checkpoint}...")
        brain.load_model(latest_checkpoint)
    else:
        print("Warning: No checkpoint found. Starting fresh.")
        
    print(f"Brain initialized. Hidden Size: {brain.genome.hidden_size}, Latent Dim: {brain.genome.latent_dim}")
    
    # 2. Setup Meditation
    broca = brain.broca
    trm = brain.trm
    
    # We use zero inputs to find "Natural Attractors"
    # Or small noise to explore the state space
    
    print(f"\n--- Entering Meditative State (Batch Size: {batch_size}) ---")
    
    # Metrics
    total_surprise = 0.0
    stability_count = 0
    
    for step in range(steps):
        # 1. Generate "Hypnagogic" Noise or Zero Input
        # We use a mix of zero and small noise to find stable points
        if step % 10 == 0:
            # Zero input: Let the dynamics settle
            inputs = torch.zeros(batch_size, brain.genome.latent_dim)
        else:
            # Small noise: Perturb the state to test stability
            inputs = torch.randn(batch_size, brain.genome.latent_dim) * 0.01
            
        # 2. Process through Broca (Batched)
        # This updates the MoE experts and Titans Memory
        # We don't provide a reward yet, we just observe the "drift"
        _, surprises = broca.process_text_embedding(inputs) # Returns mean surprise over batch
        
        # 3. Process through TRM (Main Brain)
        # We simulate the main brain's reaction to these internal thoughts
        # We need to construct a full input vector for the TRM
        # [Foveal, Peripheral, Semantic, Surprise, Bio, Meta, Res]
        # For meditation, we zero out sensory inputs and focus on Semantic (Broca)
        
        L = brain.genome.latent_dim
        full_inputs = torch.zeros(batch_size, brain.input_size)
        
        # Inject Broca's "thought" into the Semantic slot (2*L to 3*L)
        # Actually, in EvolutionaryBrain.wake_cycle, semantic_vector is at index 2*L
        # Wait, let's check the concatenation in wake_cycle:
        # foveal_latent (L), peripheral_latent (L), semantic_vector (L), surprise (1), bio (4), meta (2), res (1)
        # Total = 3*L + 8
        
        # We'll just use the first L slots for the "thought" for simplicity in meditation
        full_inputs[:, L*2 : L*3] = inputs # Injecting noise/zero as "semantic" input
        
        # Forward pass through TRM
        # We don't care about actions, just the energy and state updates
        _, _, energy, _ = trm.forward(full_inputs, dt=1.0)
        
        # 4. Stabilization Reward
        # We reward LOW surprise (predictability) and LOW energy (efficiency)
        # If the brain can predict its own next state, it's stable.
        
        current_surprise = surprises.item() if torch.is_tensor(surprises) else surprises
        
        # Stability Reward: Inverse of surprise
        # We want to minimize surprise
        stability_reward = 1.0 / (1.0 + current_surprise)
        
        # Efficiency Reward: Inverse of energy
        efficiency_reward = 1.0 / (1.0 + energy.mean().item())
        
        # Combined Reward for Experts
        # High reward reinforces the current attractor
        total_reward = (stability_reward * 0.7) + (efficiency_reward * 0.3)
        
        # Apply learning to Broca Experts
        # We call process_text_embedding again with the reward to reinforce
        broca.process_text_embedding(inputs, reward=total_reward)
        
        # 5. Periodic Dreaming (Consolidation)
        if step % 50 == 0 and step > 0:
            print(f" [Consolidating via Dream...]", end='')
            brain.dream(steps=5)
            
        # 6. Logging
        total_surprise += current_surprise
        if current_surprise < 0.05:
            stability_count += 1
            
        if step % 100 == 0:
            avg_surprise = total_surprise / (step + 1)
            print(f"Step {step:4d} | Surprise: {current_surprise:.4f} (Avg: {avg_surprise:.4f}) | "
                  f"Stability: {stability_count/(step+1)*100:.1f}% | Reward: {total_reward:.4f}")
            
            # Save progress
            brain.save_model(latest_checkpoint)
            
    print("\n--- Meditation Complete ---")
    print(f"Final Avg Surprise: {total_surprise/steps:.4f}")
    print(f"Final Stability Score: {stability_count/steps*100:.1f}%")
    
    # Final Save
    brain.save_model(latest_checkpoint)
    print(f"Stabilized Brain saved to {latest_checkpoint}")

if __name__ == "__main__":
    # Run with a reasonable batch size for CPU
    run_meditation_v2(steps=300, batch_size=8)
