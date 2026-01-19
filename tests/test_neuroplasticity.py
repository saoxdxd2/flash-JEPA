import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain

def test_neuroplasticity():
    print("Initializing Brain...")
    brain = EvolutionaryBrain()
    
    # Force enable neuroplasticity and set low threshold for testing
    brain.genome.ENABLE_NEUROPLASTICITY = True
    brain.genome.GROWTH_TRIGGER_SURPRISE = 0.1 # Low threshold to force growth
    brain.genome.MIN_HIDDEN_SIZE = 128
    brain.genome.MAX_HIDDEN_SIZE = 8192
    brain.genome.GROWTH_STEP_SIZE = 512
    
    initial_size = brain.hidden_size
    print(f"Initial Hidden Size: {initial_size}")
    
    # Simulate a day with high surprise
    print("Simulating High Surprise Day...")
    brain.daily_avg_surprise = 0.5 # Higher than trigger
    
    # Trigger Wake Cycle (Sleep -> Wake)
    print("Triggering Wake Cycle (start)...")
    brain.start()
    
    new_size = brain.hidden_size
    print(f"New Hidden Size: {new_size}")
    
    if new_size > initial_size:
        print("SUCCESS: Brain grew as expected!")
    else:
        print("FAILURE: Brain did not grow.")
        
    # Simulate Low Surprise Day (Pruning)
    print("\nSimulating Low Surprise Day...")
    brain.daily_avg_surprise = 0.01
    brain.genome.PRUNE_TRIGGER_STABILITY = 0.05
    
    brain.start()
    final_size = brain.hidden_size
    print(f"Final Hidden Size: {final_size}")
    
    if final_size < new_size:
         print("SUCCESS: Brain pruned as expected!")
    else:
         print("FAILURE: Brain did not prune.")

if __name__ == "__main__":
    test_neuroplasticity()
