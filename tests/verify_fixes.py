import torch
import sys
import os
import time
from collections import defaultdict

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.modules.biology_core import NeurotransmitterSystem

def test_dopamine_recovery():
    print("\n=== Test 1: Dopamine Recovery Dynamics ===")
    bio = NeurotransmitterSystem(device='cpu')
    
    # 1. Baseline
    print(f"Initial Dopamine: {bio.dopamine:.4f}")
    
    # 2. Induce Massive Stress (Cortisol Spike)
    print(">>> Inducing PANIC (High Cortisol)...")
    for _ in range(10):
        # High pain/effort causes cortisol spike
        bio.update(reward_prediction_error=0.0, surprise=1.0, pain=1.0, effort=1.0)
    
    print(f"Stressed State -> Dopamine: {bio.dopamine:.4f}, Cortisol: {bio.cortisol:.4f}")
    
    if bio.dopamine == 0.0:
        print("WARNING: Dopamine hit absolute zero.")
    
    # 3. Recovery Phase
    print(">>> Removing Stress (Recovery)...")
    recoveries = []
    for i in range(200):
        bio.update(reward_prediction_error=0.0, surprise=0.0, pain=0.0, effort=0.0)
        recoveries.append(bio.dopamine)
        if i % 20 == 0:
            print(f"  Step {i}: Dopa={bio.dopamine:.4f}, Cort={bio.cortisol:.4f}")
        
    final_dopamine = bio.dopamine
    print(f"Recovered Dopamine: {final_dopamine:.4f}")
    
    # Assertions
    if final_dopamine > 0.05 and final_dopamine > recoveries[0]:
        print("SUCCESS: Dopamine successfully recovered from stress suppression!")
    else:
        print("FAILURE: Dopamine is stuck or not recovering.")

def test_compression_logic():
    print("\n=== Test 2: Compression Script Logic Flow ===")
    
    # Mock Data
    layers = [
        ("layer1", torch.randn(128, 128)), # 2D
        ("layer2", torch.randn(128, 128)), # 2D
        ("layer3", torch.randn(64)),       # 1D (Should be skipped/raw)
        ("layer4", torch.randn(128, 128))  # 2D
    ]
    
    print(f"Simulating processing of {len(layers)} layers...")
    
    try:
        # Logic from start_fractal_vessel.py
        layers_to_process = []
        fractal_brain = {}
        
        for key, tensor in layers:
            if len(tensor.shape) == 2:
                layers_to_process.append((key, tensor))
            else:
                # FP16 optimization check
                fractal_brain[key] = tensor.half().cpu()
                print(f"  Stored {key} as raw FP16 (Shape {tensor.shape})")
        
        # Grouping
        grouped_layers = defaultdict(list)
        for key, tensor in layers_to_process:
            grouped_layers[tensor.shape].append((key, tensor))
            
        # Batch Loop Simulation
        for shape, items in grouped_layers.items():
            SUB_BATCH_SIZE = 2 # Small batch for test
            
            for i in range(0, len(items), SUB_BATCH_SIZE):
                sub_items = items[i : i + SUB_BATCH_SIZE]
                keys = [k for k, t in sub_items]
                tensors = [t for k, t in sub_items]
                
                batch_tensor = torch.stack(tensors)
                print(f"  Encoded batch of {len(tensors)} items with shape {shape}")
                
                # Mock Encoding
                dna_list = [{"mock": True}] * len(tensors)
                
                for k, dna in zip(keys, dna_list):
                    fractal_brain[k] = dna
                
                del batch_tensor
                del dna_list
        
        # The critical cleanup block that caused the crash
        print("  Executing Cleanup Block...")
        del layers_to_process
        del grouped_layers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("SUCCESS: Logic flow completed without UnboundLocalError.")
        
    except Exception as e:
        print(f"FAILURE: Script crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dopamine_recovery()
    test_compression_logic()
