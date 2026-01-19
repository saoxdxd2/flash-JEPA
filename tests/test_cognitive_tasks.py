import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.neuromodulated_holographic import NeuromodulatedHolographicBrain

def train_copy_task(brain, sequence_length=5, vector_dim=8, iterations=100):
    """
    Trains the brain to copy a sequence of vectors.
    Input: [v1, v2, v3, ..., 0, 0, ...]
    Target: [0, 0, 0, ..., v1, v2, v3, ...]
    """
    print(f"\n--- Training Copy Task (SeqLen={sequence_length}) ---")
    optimizer = optim.Adam(brain.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Brain Input/Output Size must match vector_dim
    # But brain has fixed input/output size.
    # We assume brain input/output size >= vector_dim.
    
    max_seq_len = 6
    current_seq_len = 1
    
    for i in range(1000): # Increased iterations for curriculum
        optimizer.zero_grad()
        
        # Generate Data (Variable Length)
        seq_len = current_seq_len
        seq = torch.randn(1, seq_len, vector_dim)
        zeros = torch.zeros(1, seq_len, vector_dim)
        
        # Input: [Seq, Zeros]
        inputs = torch.cat([seq, zeros], dim=1) # [1, 2*Seq, In]
        # Target: [Zeros, Seq]
        targets = torch.cat([zeros, seq], dim=1)
        
        loss = 0
        
        # Reset Brain State
        brain.reflex.memory_state = None
        brain.concept.memory_state = None
        brain.strategy.memory_state = None
        # Also reset hidden states if they persist (they do in Brain)
        brain.h_reflex = torch.zeros(1, brain.reflex.vm.max_npus, brain.reflex.vm.npu_register_size)
        brain.h_concept = torch.zeros(1, brain.concept.vm.max_npus, brain.concept.vm.npu_register_size)
        brain.h_strategy = torch.zeros(1, brain.strategy.vm.max_npus, brain.strategy.vm.npu_register_size)
        
        # Forward Pass
        for t in range(inputs.shape[1]):
            input_t = inputs[:, t, :]
            
            # Pad input
            if brain.input_projection.in_features > vector_dim:
                padding = torch.zeros(1, brain.input_projection.in_features - vector_dim)
                input_t = torch.cat([input_t, padding], dim=1)
            
            actions, _, _, _ = brain(input_t)
            
            output_t = actions[:, :vector_dim] # Slice to input size
            target_t = targets[:, t, :]
            loss += criterion(output_t, target_t)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Iter {i}: Loss = {loss.item():.4f} (SeqLen {current_seq_len})")
            
        # Curriculum Update
        if loss.item() < 0.05 * current_seq_len:
            if current_seq_len < max_seq_len:
                current_seq_len += 1
                print(f"Promoted to SeqLen {current_seq_len}!")
            elif loss.item() < 0.01:
                print("Converged!")
                break
                
    print(f"Final Loss: {loss.item():.4f}")
    return loss.item()

def test_associative_recall(brain, num_pairs=3, vector_dim=8, iterations=100):
    """
    Trains the brain to recall a value given a key.
    Input: [k1, v1, k2, v2, ..., query_k]
    Target: [..., target_v]
    """
    print(f"\n--- Training Associative Recall (Pairs={num_pairs}) ---")
    optimizer = optim.Adam(brain.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Generate Pairs
        keys = [torch.randn(1, vector_dim) for _ in range(num_pairs)]
        vals = [torch.randn(1, vector_dim) for _ in range(num_pairs)]
        
        # Query one random key
        query_idx = random.randint(0, num_pairs-1)
        query_k = keys[query_idx]
        target_v = vals[query_idx]
        
        # Construct Input Sequence
        # k1, v1, k2, v2, ..., query_k
        input_seq = []
        for k, v in zip(keys, vals):
            input_seq.append(k)
            input_seq.append(v)
        input_seq.append(query_k)
        
        loss = 0
        
        # Reset Brain State
        brain.h_reflex = None
        brain.h_concept = None
        brain.h_strategy = None
        
        # Forward Pass
        for t, input_t in enumerate(input_seq):
             # Pad input
            if brain.input_projection.in_features > vector_dim:
                padding = torch.zeros(1, brain.input_projection.in_features - vector_dim)
                input_t = torch.cat([input_t, padding], dim=1)
                
            actions, _, _, _ = brain(input_t)
            
            # Only calculate loss on the final step (Recall)
            if t == len(input_seq) - 1:
                output_t = actions[:, :vector_dim]
                loss = criterion(output_t, target_v)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        # Reset Block Memory States
        brain.reflex.memory_state = None
        brain.concept.memory_state = None
        brain.strategy.memory_state = None
        
        if i % 20 == 0:
            print(f"Iter {i}: Loss = {loss.item():.4f}")
            
    print(f"Final Loss: {loss.item():.4f}")
    return loss.item()

if __name__ == "__main__":
    # Setup Brain
    input_size = 16
    hidden_size = 64
    output_size = 16
    
    brain = NeuromodulatedHolographicBrain(input_size, hidden_size, output_size)
    
    # Run Tests
    copy_loss = train_copy_task(brain, sequence_length=3, vector_dim=8, iterations=100)
    recall_loss = test_associative_recall(brain, num_pairs=2, vector_dim=8, iterations=100)
    
    if copy_loss < 0.5 and recall_loss < 0.5:
        print("\nSUCCESS: Brain demonstrated learning capability!")
    else:
        print("\nWARNING: Brain struggled to learn tasks. Needs hyperparameter tuning.")
