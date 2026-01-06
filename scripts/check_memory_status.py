import torch
import os
import sys
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain

def check_memory():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    elite_path = os.path.join(project_root, "models", "saved", "gen_349_elite.pkl")
    unified_path = os.path.join(project_root, "brain", "n2n2_unified_checkpoint.pkl")
    
    print(f"--- Memory Inspection ---")
    
    # 1. Inspect Elite (Original)
    if os.path.exists(elite_path):
        print(f"\n[Elite Model] {elite_path}")
        try:
            elite = EvolutionaryBrain()
            elite.load_model(elite_path)
            
            # Check Hippocampus (Associative Memory)
            if hasattr(elite, 'memory'):
                mem_size = len(elite.memory.memory_bank) if hasattr(elite.memory, 'memory_bank') else "Unknown"
                print(f"  Hippocampus Size: {mem_size}")
            else:
                print("  Hippocampus: Not found")
                
            # Check Broca Titans Memory
            if hasattr(elite.broca, 'titans'):
                # Check if weights are non-zero/random
                w_norm = torch.norm(elite.broca.titans.memory_weights).item()
                print(f"  Titans Memory Norm: {w_norm:.4f} (Higher = More Learned/Random)")
            else:
                print("  Titans Memory: Not initialized")
                
        except Exception as e:
            print(f"  Error loading Elite: {e}")
    else:
        print(f"  Elite model not found.")

    # 2. Inspect Unified (New)
    if os.path.exists(unified_path):
        print(f"\n[Unified Checkpoint] {unified_path}")
        try:
            unified = EvolutionaryBrain()
            unified.load_model(unified_path)
            
            # Check Hippocampus
            if hasattr(unified, 'memory'):
                mem_size = len(unified.memory.memory_bank) if hasattr(unified.memory, 'memory_bank') else "Unknown"
                print(f"  Hippocampus Size: {mem_size}")
            else:
                print("  Hippocampus: Not found")
                
            # Check Broca Titans Memory
            if hasattr(unified.broca, 'titans'):
                w_norm = torch.norm(unified.broca.titans.memory_weights).item()
                print(f"  Titans Memory Norm: {w_norm:.4f}")
            else:
                print("  Titans Memory: Not initialized")
                
        except Exception as e:
            print(f"  Error loading Unified: {e}")
    else:
        print(f"  Unified checkpoint not found.")

if __name__ == "__main__":
    check_memory()
