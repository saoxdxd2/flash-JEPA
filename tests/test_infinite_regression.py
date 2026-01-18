import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def test_infinite_regression():
    print("Initializing Brain...")
    genome = Genome()
    genome.latent_dim = 16 # Small for testing
    brain = EvolutionaryBrain(genome)
    
    # Prime the brain with some random "sensory" input
    print("Priming Brain with random input...")
    
    # Simulate sensory state: [Foveal, Peripheral, Semantic]
    # Each is latent_dim size. Total = 3 * latent_dim
    latent_dim = brain.latent_dim
    
    # Create a random sensory state
    current_sensory_state = torch.randn(3 * latent_dim).to(brain.device)
    
    # Run a few steps of "observation"
    for i in range(5):
        # In a real loop, we would observe real data. 
        # Here we just observe random data to warm up the weights/memory
        brain.hippocampus.observe(current_sensory_state)
        prediction = brain.hippocampus(current_sensory_state)
        
        # Move to a slightly different state (simulate changing world)
        current_sensory_state = current_sensory_state + torch.randn(3 * latent_dim).to(brain.device) * 0.1
        
    print("Entering Infinite Regression (Silence Mode)...")
    
    # Now, we cut off external input.
    # The "current_sensory_state" becomes the PREDICTION from the previous step.
    
    # Initial prediction from the last real state
    next_state_prediction = brain.hippocampus(current_sensory_state)
    
    magnitudes = []
    surprises = []
    
    regression_steps = 50
    
    for i in range(regression_steps):
        # RECURSION: The input is the previous prediction
        # "I am seeing what I expected to see" (Hallucination/Dream)
        
        # We treat the prediction as the "actual" observation for the memory update
        # This reinforces the hallucination
        # Or maybe we DON'T observe? If we observe, we train the memory to predict the hallucination.
        # If we don't observe, we just iterate the forward pass.
        
        # Let's try "observing" the hallucination, effectively locking it in.
        # This matches "infinite regression" where the thought becomes the reality.
        
        # 1. The "Input" is the previous prediction
        input_state = next_state_prediction.detach() # Detach to stop gradient explosion? Or keep it?
        
        # 2. Observe it (Self-reinforcement)
        surprise = brain.hippocampus.observe(input_state)
        
        # 3. Predict the NEXT state based on this hallucination
        next_state_prediction = brain.hippocampus(input_state)
        
        mag = input_state.norm().item()
        magnitudes.append(mag)
        surprises.append(surprise)
        
        print(f"Step {i}: Mag={mag:.4f}, Surprise={surprise:.4f}")
        
        if mag > 1000:
            print("DIVERGENCE DETECTED (Explosion)")
            break
import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def test_infinite_regression():
    print("Initializing Brain...")
    genome = Genome()
    genome.latent_dim = 16 # Small for testing
    brain = EvolutionaryBrain(genome)
    
    # Prime the brain with some random "sensory" input
    print("Priming Brain with random input...")
    
    # Simulate sensory state: [Foveal, Peripheral, Semantic]
    # Each is latent_dim size. Total = 3 * latent_dim
    latent_dim = brain.latent_dim
    
    # Create a random sensory state
    current_sensory_state = torch.randn(3 * latent_dim).to(brain.device)
    
    # Run a few steps of "observation"
    for i in range(5):
        # In a real loop, we would observe real data. 
        # Here we just observe random data to warm up the weights/memory
        brain.hippocampus.observe(current_sensory_state)
        prediction = brain.hippocampus(current_sensory_state)
        
        # Move to a slightly different state (simulate changing world)
        current_sensory_state = current_sensory_state + torch.randn(3 * latent_dim).to(brain.device) * 0.1
        
    print("Entering Infinite Regression (Silence Mode)...")
    
    # Now, we cut off external input.
    # The "current_sensory_state" becomes the PREDICTION from the previous step.
    
    # Initial prediction from the last real state
    next_state_prediction = brain.hippocampus(current_sensory_state)
    
    magnitudes = []
    surprises = []
    
    regression_steps = 50
    
    for i in range(regression_steps):
        # RECURSION: The input is the previous prediction
        # "I am seeing what I expected to see" (Hallucination/Dream)
        
        # We treat the prediction as the "actual" observation for the memory update
        # This reinforces the hallucination
        # Or maybe we DON'T observe? If we observe, we train the memory to predict the hallucination.
        # If we don't observe, we just iterate the forward pass.
        
        # Let's try "observing" the hallucination, effectively locking it in.
        # This matches "infinite regression" where the thought becomes the reality.
        
        # 1. The "Input" is the previous prediction
        input_state = next_state_prediction.detach() # Detach to stop gradient explosion? Or keep it?
        
        # 2. Observe it (Self-reinforcement)
        surprise = brain.hippocampus.observe(input_state)
        
        # 3. Predict the NEXT state based on this hallucination
        next_state_prediction = brain.hippocampus(input_state)
        
        mag = input_state.norm().item()
        magnitudes.append(mag)
        surprises.append(surprise)
        
        print(f"Step {i}: Mag={mag:.4f}, Surprise={surprise:.4f}")
        
        if mag > 1000:
            print("DIVERGENCE DETECTED (Explosion)")
            break
        if mag < 0.001:
            print("CONVERGENCE TO ZERO DETECTED")
            break
            
    print("Regression Test Complete.")
    
    # --- INTEGRATION TEST: wake_cycle (Thalamic Gating) ---
    print("\nTesting wake_cycle integration (Thalamic Gating)...")
    
    # Mock Retina to return None (Silence)
    original_get_latest_input = brain.retina.get_latest_input
    brain.retina.get_latest_input = lambda: None
    
    # Ensure we have a last prediction to start the loop
    if brain.hippocampus.last_prediction is None:
        brain.hippocampus.last_prediction = torch.randn(1, 3 * latent_dim).to(brain.device)
        
    # Run wake_cycle for a few steps
    actions = []
    for i in range(20):
        try:
            action = brain.wake_cycle()
            actions.append(action)
            
            # Check internal state for noise
            # We can't easily access the local 'mixed_signal' variable, 
            # but we can check if the brain is producing DIFFERENT outputs for the SAME input (due to noise)
            # In infinite regression, the input changes every step anyway.
            
            if i % 5 == 0:
                print(f"Wake Cycle {i}: Action={action}")
            
        except Exception as e:
            print(f"Wake Cycle Failed: {e}")
            import traceback
            traceback.print_exc()
            break
            
    # Check for variability (Noise effect)
    # If noise is working, we shouldn't get stuck in a perfectly repeating loop of length 1 too easily
    unique_actions = len(set(actions))
    print(f"Unique Actions in 20 steps: {unique_actions}")
    
    # Restore Retina
    brain.retina.get_latest_input = original_get_latest_input
    print("Integration Test Complete.")


if __name__ == '__main__':
    test_infinite_regression()
