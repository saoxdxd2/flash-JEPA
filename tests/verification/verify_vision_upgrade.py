import sys
import os
import time
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain

def verify_vision_upgrade():
    print("=== Verifying Vision Upgrade ===")
    
    # Initialize Brain
    brain = EvolutionaryBrain()
    brain.start()
    
    print(f"Initial Resolution: {brain.retina.fovea_size}px")
    
    # Force Upgrade
    print("Forcing Upgrade to 128px...")
    brain.retina.set_resolution(128)
    
    # Run a few cycles to check for crashes in the vision loop
    print("Running wake cycles...")
    try:
        for i in range(10):
            # We need to wait for the retina to produce output
            time.sleep(0.2)
            retina_output = brain.retina.get_latest_input()
            
            if retina_output:
                foveal, peripheral, surprise, text_density, raw_fovea = retina_output
                print(f"Cycle {i}: Input Received. Fovea Shape: {raw_fovea.shape}")
                
                if raw_fovea.shape[-1] != 128:
                    print(f"ERROR: Expected 128px, got {raw_fovea.shape[-1]}px")
                    return
            else:
                print(f"Cycle {i}: No input yet...")
                
        print("SUCCESS: Vision loop running at 128px without crash.")
        
    except Exception as e:
        print(f"FAILED: Exception during wake cycle: {e}")
        import traceback
        traceback.print_exc()
    finally:
        brain.stop()

if __name__ == "__main__":
    verify_vision_upgrade()
