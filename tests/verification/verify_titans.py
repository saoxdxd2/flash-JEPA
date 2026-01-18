import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.modules.neural_memory import TitansMemory

def verify_titans():
    print("🧠 VERIFYING TITANS MEMORY")
    print("==========================")
    
    input_dim = 128
    titans = TitansMemory(input_dim, input_dim)
    
    x = torch.randn(1, input_dim)
    out = titans(x)
    print(f"Initial Forward Pass: {out.shape} (Expected 1x{input_dim})")
    
    # 1. Test Growth
    print("\n--- Testing Growth (128 -> 256) ---")
    titans.resize(256, 256)
    x_new = torch.randn(1, 256)
    out_new = titans(x_new)
    print(f"Growth Forward Pass: {out_new.shape} (Expected 1x256)")
    
    # 2. Test Shrinking
    print("\n--- Testing Shrinking (256 -> 64) ---")
    try:
        titans.resize(64, 64)
        x_small = torch.randn(1, 64)
        out_small = titans(x_small)
        print(f"Shrink Forward Pass: {out_small.shape} (Expected 1x64)")
        print("✅ PASS: Shrinking worked.")
    except Exception as e:
        print(f"❌ FAIL: Shrinking failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_titans()
