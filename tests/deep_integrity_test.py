import torch
import torch.nn as nn
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome
import os
import time
import numpy as np

def deep_test():
    print("=== Deep System Integrity Test (No Hidden Bugs) ===")
    
    # 1. Setup
    genome = Genome()
    brain = EvolutionaryBrain(genome)
    
    print("\n--- Testing 1: Action Space & Output Shapes ---")
    dummy_input = torch.randn(1, brain.input_size)
    
    # Test PyTorch Path
    brain.use_onnx = False
    action_py, logits_py = brain.decide(dummy_input)
    print(f"PyTorch Action: {action_py}, Logits Shape: {logits_py.shape}")
    
    # Test ONNX Path
    onnx_path = "integrity_test.onnx"
    brain.export_reflex_path(onnx_path)
    brain.use_onnx = True
    action_on, logits_on = brain.decide(dummy_input)
    print(f"ONNX Action: {action_on}, Logits Shape: {logits_on.shape}")
    
    print("\n--- Testing 2: State Persistence (ONNX) ---")
    # Run 5 steps and check if states are updated (not all zeros)
    for i in range(5):
        brain.decide(dummy_input)
    
    states_sum = sum([np.abs(s).sum() for s in brain.onnx_states])
    print(f"ONNX States Absolute Sum after 5 steps: {states_sum:.4f}")
    if states_sum > 0:
        print("SUCCESS: ONNX states are evolving.")
    else:
        print("FAILED: ONNX states are stuck at zero.")

    print("\n--- Testing 3: Hybrid Switching Logic ---")
    # Force low confidence by zeroing out reflex hidden state
    # Actually, we can just check the logic in decide()
    # We'll mock the confidence in the ONNX path to be 0
    print("Verifying System 1 -> System 2 Fallback...")
    # We'll temporarily set threshold to 2.0 to force fallback
    old_threshold = brain.genome.CONFIDENCE_THRESHOLD
    brain.genome.CONFIDENCE_THRESHOLD = 2.0 
    
    action_fallback, _ = brain.decide(dummy_input)
    print(f"Fallback System Used: {brain.last_used_system}")
    brain.genome.CONFIDENCE_THRESHOLD = old_threshold
    
    if brain.last_used_system == "System 2 (PyTorch)":
        print("SUCCESS: Fallback logic functional.")
    else:
        print("FAILED: Fallback logic did not trigger.")

    print("\n--- Testing 4: Dynamic Sprouting Stability ---")
    # Sprout 10 times and check if ONNX export still works
    for _ in range(10):
        brain.broca._check_growth(surprise=1.0)
    
    print(f"Experts after heavy sprouting: {brain.broca.num_experts}")
    success_reexport = brain.export_reflex_path("sprouted.onnx")
    if success_reexport:
        print("SUCCESS: ONNX export compatible with dynamic growth.")
    else:
        print("FAILED: Dynamic growth broke ONNX export.")

    # Cleanup
    for p in [onnx_path, onnx_path+".meta", "sprouted.onnx", "sprouted.onnx.meta"]:
        if os.path.exists(p): os.remove(p)

    print("\n=== Deep Test Complete ===")

if __name__ == "__main__":
    deep_test()
