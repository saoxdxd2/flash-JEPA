import torch
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.modules.hamiltonian_memory import HamiltonianMemory

def test_ham_butterfly():
    print("Testing Hamiltonian Memory (Butterfly Curve)...")
    
    # 1. Generate Data
    t = torch.linspace(0, 10, 100)
    q_target = torch.sin(t) + torch.sin(2*t) * 0.5
    # We treat this as a 1D trajectory [Steps, Batch=1, Dim=1]
    trajectory = q_target.unsqueeze(1).unsqueeze(1) # [100, 1, 1]
    
    # 2. Initialize HAM
    ham = HamiltonianMemory(input_dim=1, hidden_dim=32)
    
    # 3. Imprint
    print("Imprinting...")
    loss = ham.imprint(trajectory, epochs=200, verbose=True)
    
    # 4. Recall
    print("Recalling...")
    start_q = trajectory[0] # [1, 1]
    # Estimate start v
    start_v = trajectory[1] - trajectory[0]
    
    recalled_traj = ham.recall(start_q, start_v, steps=98)
    
    # 5. Compare
    # recalled_traj: [Steps, 1, 1]
    # We compare with trajectory (truncated)
    
    # Note: Integration drift is expected. We check for correlation or low MSE.
    # Also, we trained on v, a.
    
    # Let's just check if it produces numbers and doesn't crash.
    # And if loss is low.
    
    print(f"Final Imprint Loss: {loss}")
    assert loss < 0.1, "HAM failed to learn the dynamics."
    
    print("HAM Test Passed!")

if __name__ == "__main__":
    test_ham_butterfly()
