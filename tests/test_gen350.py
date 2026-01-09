import torch
import os
import sys
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def test_gen350_load():
    print("=== Testing gen_350_transplanted.pt Loading ===")
    filepath = "models/saved/gen_350_transplanted.pt"
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return

    # 1. Try standard load
    try:
        print("Attempting torch.load...")
        checkpoint = torch.load(filepath, map_location='cpu')
        print("SUCCESS: torch.load worked!")
        print(f"Keys: {checkpoint.keys()}")
    except Exception as e:
        print(f"FAILED: torch.load failed with: {e}")
        
        # 2. Try legacy load (if it was saved with old torch)
        try:
            print("\nAttempting legacy load (weights_only=False)...")
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            print("SUCCESS: Legacy load worked!")
        except Exception as e2:
            print(f"FAILED: Legacy load failed with: {e2}")

    # 3. If load worked, try to initialize brain with it
    if 'checkpoint' in locals():
        try:
            print("\nInitializing Brain with checkpoint...")
            genome = Genome()
            brain = EvolutionaryBrain(genome)
            brain.load_model(filepath)
            print("SUCCESS: Brain loaded model successfully!")
            
            # 4. Run a test step
            print("\nRunning test step with gen_350...")
            dummy_input = torch.randn(1, brain.input_size)
            action, logits = brain.decide(dummy_input)
            print(f"Action: {action}, Logits Shape: {logits.shape}")
            
        except Exception as e3:
            print(f"FAILED: Brain initialization/load failed: {e3}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_gen350_load()
