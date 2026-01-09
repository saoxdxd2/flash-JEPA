import torch
import time
import pyautogui
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def test_gaze_sync():
    print("Testing Gaze Synchronization...")
    genome = Genome()
    brain = EvolutionaryBrain(genome=genome)
    
    # Start retina (it runs in a thread)
    brain.retina.start()
    time.sleep(2) # Wait for it to initialize
    
    # Get initial gaze
    initial_gx, initial_gy = brain.retina.gaze_x, brain.retina.gaze_y
    print(f"Initial Gaze: ({initial_gx:.2f}, {initial_gy:.2f})")
    
    # Move mouse manually or via cradle
    print("Moving mouse to (0.1, 0.1)...")
    brain.cradle.move_mouse(0.1, 0.1, duration=0.5)
    time.sleep(1) # Wait for retina loop to catch up
    
    new_gx, new_gy = brain.retina.gaze_x, brain.retina.gaze_y
    print(f"New Gaze: ({new_gx:.2f}, {new_gy:.2f})")
    
    # Allow some tolerance for rounding/mss/pyautogui differences
    assert abs(new_gx - 0.1) < 0.05
    assert abs(new_gy - 0.1) < 0.05
    print("Gaze synchronization test passed!")
    
    brain.retina.stop()

if __name__ == "__main__":
    try:
        test_gaze_sync()
        print("\nAll verification tests passed!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
