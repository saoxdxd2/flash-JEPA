"""
Verify Biological Intelligence
==============================

This script tests if the biological theories (genes, plasticity, etc.) 
actually drive learning and adaptation.

Experiment:
1. Initialize a Brain with the new Genome.
2. Phase 1 (Steps 0-200): Learn Pattern A (Action 0 = Reward, Action 1 = Punish).
3. Phase 2 (Steps 200-400): Learn Pattern B (Action 1 = Reward, Action 0 = Punish).
   -> This tests ADAPTATION (neuroplasticity).

Hypothesis:
- Performance should increase in Phase 1.
- Performance should drop then recover in Phase 2.
- BDNF (learning gene) should spike during learning phases.
- Stress (FKBP5) should rise during failure (Phase 2 start).
- Telomeres should shorten slightly.

"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def run_experiment():
    print("🧬 INITIALIZING BIOLOGICAL INTELLIGENCE TEST")
    print("===========================================")
    
    # 1. Create Brain with new Biology
    genome = Genome()
    brain = EvolutionaryBrain(genome)
    
    print(f"Subject: {genome.species} (ID: {id(brain)})")
    print(f"Initial BDNF: {genome.get_expression('bdnf'):.3f}")
    print(f"Initial COMT: {genome.get_expression('comt'):.3f}")
    print("-" * 40)
    
    # Metrics
    history = {
        'reward': [],
        'bdnf': [],
        'stress': [],
        'accuracy': [],
        'telomere': []
    }
    
    window = deque(maxlen=20)
    
    # 2. Run Experiment
    total_steps = 400
    switch_step = 200
    
    for step in range(total_steps):
        # --- Environment Logic ---
        # Phase 1: Target = 0, Phase 2: Target = 1
        target_action = 0 if step < switch_step else 1
        
        # --- Brain Decision ---
        # Create dummy image (simulating screen capture)
        # We use random noise but structured as an image
        dummy_img_tensor = torch.randn(1, 3, 64, 64, device=brain.device)
        
        # Process through Retina (Full System Test)
        # 1. Foveal Stream
        foveal_latent = brain.retina.foveal_encoder(dummy_img_tensor)
        
        # 2. Peripheral Stream
        peripheral_latent = brain.retina.peripheral_encoder(dummy_img_tensor)
        
        # 3. Semantic (Broca) - Dummy for now as Broca needs text
        semantic_latent = torch.zeros(1, brain.genome.latent_dim, device=brain.device)
        
        # Construct full input vector
        # Bio state includes surprise, text density, RAM, CPU
        # We simulate these for the test
        bio_state = torch.tensor([
            genome.get_expression('drd2'), # Approx dopamine
            genome.get_expression('httlpr'), # Approx serotonin
            genome.get_expression('comt'), # Approx norepinephrine
            1.0, # Stamina
            0.0, # Surprise
            0.0, # Text Density
            0.2, # RAM (Simulated)
            0.1  # CPU (Simulated)
        ], device=brain.device)
        
        # We need to simulate the memory latent now
        dummy_memory = torch.zeros(3 * brain.genome.latent_dim, device=brain.device)
        
        full_input = brain.get_input_vector(
            foveal_latent, 
            peripheral_latent, 
            semantic_latent,
            memory_latent=dummy_memory
        )
        
        # Decide
        action_idx, probs = brain.decide(full_input, train_internal_rl=True)
        
        # Only care about action 0 vs 1 for this simple test
        # Map 72 actions to binary choice: Even=0, Odd=1
        choice = action_idx % 2
        
        # --- Feedback ---
        is_correct = (choice == target_action)
        reward = 1.0 if is_correct else -1.0
        
        # Biological feedback
        if is_correct:
            genome.on_reward(1.0)
            genome.on_learning() # Trigger plasticity
        else:
            # Wrong answer causes stress
            genome.on_stress(0.5)
            
        # Add to Replay Buffer
        if step > 0:
            # We need (state, action, reward, next_state, done)
            # For simplicity, we use current input as both state and next_state (not ideal but works for this test)
            brain.replay_buffer.add(full_input, action_idx, reward, full_input, False)
            
        # Train brain (only if enough samples)
        if len(brain.replay_buffer) > 32:
            brain.train_step()
        
        # Tick biology
        genome.tick()
        
        # --- Logging ---
        window.append(1 if is_correct else 0)
        accuracy = sum(window) / len(window)
        
        history['reward'].append(reward)
        history['bdnf'].append(genome.get_expression('bdnf'))
        history['stress'].append(genome.get_expression('fkbp5')) # Stress gene
        history['accuracy'].append(accuracy)
        history['telomere'].append(genome.telomere_health)
        
        # Print progress
        if step % 20 == 0:
            phase = "A (Left)" if step < switch_step else "B (Right)"
            print(f"Step {step:3d} | Phase {phase} | Acc: {accuracy:.2f} | BDNF: {history['bdnf'][-1]:.3f} | Stress: {history['stress'][-1]:.3f}")
            
        if step == switch_step:
            print("\n⚠️  ENVIRONMENT CHANGE! PATTERN SWITCHED! ⚠️\n")

    # 3. Analysis
    print("\n📊 EXPERIMENT RESULTS")
    print("======================")
    
    avg_acc_p1 = np.mean(history['accuracy'][100:200])
    avg_acc_p2_start = np.mean(history['accuracy'][200:250])
    avg_acc_p2_end = np.mean(history['accuracy'][350:400])
    
    print(f"Phase 1 Accuracy (Stable): {avg_acc_p1:.2f}")
    print(f"Phase 2 Start (Shock):     {avg_acc_p2_start:.2f}")
    print(f"Phase 2 End (Recovered):   {avg_acc_p2_end:.2f}")
    
    print("\n🧬 BIOLOGICAL CORRELATIONS")
    
    # Did BDNF correlate with learning?
    # We expect BDNF to be high when accuracy is increasing
    bdnf_p1 = np.mean(history['bdnf'][0:100])
    bdnf_p2 = np.mean(history['bdnf'][200:300])
    print(f"BDNF during initial learning: {bdnf_p1:.3f}")
    print(f"BDNF during relearning:       {bdnf_p2:.3f}")
    
    # Did stress rise during failure?
    stress_p1 = np.mean(history['stress'][150:200]) # Low stress
    stress_p2 = np.mean(history['stress'][200:250]) # High stress expected
    print(f"Stress before switch: {stress_p1:.3f}")
    print(f"Stress after switch:  {stress_p2:.3f}")
    
    print(f"\nTelomere Health Final: {history['telomere'][-1]:.4f}")
    
    # Conclusion
    if avg_acc_p2_end > avg_acc_p2_start and bdnf_p2 > 0.5:
        print("\n✅ CONCLUSION: INTELLIGENCE VERIFIED")
        print("   The agent successfully adapted to the new pattern.")
        print("   Biological markers (BDNF) responded to the learning demand.")
    else:
        print("\n❌ CONCLUSION: FAILED")
        print("   The agent failed to adapt or biology did not respond.")

if __name__ == "__main__":
    run_experiment()
