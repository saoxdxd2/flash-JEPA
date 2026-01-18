import torch
import sys
import os
import time
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.modules.broca import BrocaModule
from brain.evolutionary_brain import EvolutionaryBrain

def test_broca_dynamic_vocab():
    print("\n=== Testing Broca Dynamic Vocabulary ===")
    broca = BrocaModule()
    
    # Test words not in original vocab
    test_words = ["HELLO", "WORLD", "QUANTUM", "PHYSICS", "12345"]
    
    for word in test_words:
        vector = broca.process_text(word)
        print(f"Word: '{word}' -> Vector Shape: {vector.shape}")
        assert vector.shape == (256,), f"Vector shape mismatch for {word}"
        assert not torch.all(vector == 0), f"Vector is all zeros for {word}"
        
    print("SUCCESS: Broca handled dynamic vocabulary correctly.")

def test_burst_mode():
    print("\n=== Testing Burst Mode ===")
    brain = EvolutionaryBrain()
    
    # Mock Control Interface
    brain.control = MagicMock()
    brain.control.execute_action_code.return_value = True
    
    # Mock Retina to return dummy data
    brain.retina = MagicMock()
    brain.retina.get_latest_input.return_value = (
        torch.zeros(256).numpy(), # Foveal
        torch.zeros(256).numpy(), # Peripheral
        0.0, # Surprise
        0.5, # Text Density
        torch.zeros(3, 128, 128).numpy() # Raw Fovea (Corrected Shape: C, H, W)
    )
    brain.retina.gaze_x = 0.5
    brain.retina.gaze_y = 0.5
    brain.retina.fovea_size = 64 # Fix for wake_cycle concatenation
    
    # Mock Decide to return TYPING action (e.g., 'A' = 15) with High Confidence
    # We need to mock decide to return different actions in sequence to simulate typing "ABC"
    # But for the first test, let's just see if it calls execute multiple times
    
    # We'll mock the internal components to force high confidence
    brain.chemistry.dopamine = 0.0 # Low dopamine -> Low temperature -> High confidence? 
    # Wait, temp = max(0.1, 1.0 - dopamine). If dopa=0, temp=1.0. If dopa=1, temp=0.1.
    # So High Dopamine -> Low Temp -> High Confidence (Peaked distribution).
    brain.chemistry.dopamine = 1.0 
    
    # Mock decide to return a sequence of actions
    # First call: Type 'A' (15)
    # Second call: Type 'B' (16)
    # Third call: Type 'C' (17)
    
    # We can't easily mock `decide` because it's a method of the class we are testing.
    # Instead, we can mock `trm.forward` to return logits that favor specific actions.
    
    # Manual Mock for Decide
    high_conf_logits = torch.zeros(70)
    high_conf_logits[15] = 100.0 # Strong signal for 'A'
    
    decide_calls = 0
    def mock_decide(input_tensor):
        nonlocal decide_calls
        decide_calls += 1
        print(f"DEBUG: Mock Decide Called (Call #{decide_calls})")
        
        if decide_calls == 1:
            return 15, high_conf_logits # Type A
        elif decide_calls == 2:
            return 16, high_conf_logits # Type B
        elif decide_calls == 3:
            return 17, high_conf_logits # Type C
        else:
            return 9, torch.zeros(70) # Wait
            
    brain.decide = mock_decide
    
    print("Simulating Wake Cycle...")
    brain.wake_cycle()
    
    # Check calls
    print(f"Control Execute Calls: {brain.control.execute_action_code.call_count}")
    
    calls = brain.control.execute_action_code.call_args_list
    print(f"Actions Executed: {[c[0][0] for c in calls]}")
    
    if brain.control.execute_action_code.call_count >= 3:
        print("SUCCESS: Burst Mode triggered multiple actions.")
    else:
        print("FAILURE: Burst Mode did not trigger enough actions.")

if __name__ == "__main__":
    test_broca_dynamic_vocab()
    test_burst_mode()
