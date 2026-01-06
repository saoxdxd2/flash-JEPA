
import torch
import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def test_memory_persistence():
    print("=== Verifying Memory Persistence ===")
    
    # 1. Create Brain and Add Memory
    print("Initializing Brain...")
    genome = Genome()
    brain = EvolutionaryBrain(genome)
    
    # Add a dummy memory
    state = torch.randn(brain.input_size)
    action = 5
    reward = 1.0
    
    print(f"Storing Memory: Action={action}, Reward={reward}")
    brain.memory.store(state, action, reward)
    
    # Verify it's in memory
    memories, metadata, _ = brain.memory.recall(state, k=1)
    print(f"Immediate Recall: Action={metadata[0][0].item()}, Reward={metadata[0][1].item()}")
    
    # 2. Save Model
    os.makedirs("models/test", exist_ok=True)
    save_path = "models/test/memory_test.pkl"
    print(f"Saving model to {save_path}...")
    brain.save_model(save_path)
    
    # 3. Load into NEW Brain
    print("Loading into NEW Brain...")
    new_brain = EvolutionaryBrain(genome)
    new_brain.load_model(save_path)
    
    # 4. Verify Memory in New Brain
    print("Verifying Memory in New Brain...")
    memories, metadata, _ = new_brain.memory.recall(state, k=1)
    
    if metadata is not None:
        loaded_action = int(metadata[0][0].item())
        loaded_reward = metadata[0][1].item()
        print(f"Loaded Recall: Action={loaded_action}, Reward={loaded_reward}")
        
        if loaded_action == action and loaded_reward == reward:
            print("SUCCESS: Memory persisted correctly!")
        else:
            print("FAILURE: Memory content mismatch.")
    else:
        print("FAILURE: No memory recalled from loaded brain.")
        
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)

if __name__ == "__main__":
    test_memory_persistence()
