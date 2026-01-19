import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.modules.neural_vm import NeuralVirtualizationLayer

def test_vm_copy_task():
    print("\n--- Testing Neural VM Directly on Copy Task ---")
    input_size = 8
    hidden_size = 32
    seq_len = 3
    
    # Create VM
    vm = NeuralVirtualizationLayer(input_size, hidden_size, max_npus=8, memory_size=16)
    
    # Projector for output (Hidden -> Input)
    output_proj = nn.Linear(hidden_size, input_size)
    
    optimizer = optim.Adam(list(vm.parameters()) + list(output_proj.parameters()), lr=0.005)
    criterion = nn.MSELoss()
    
    # Curriculum Learning
    max_seq_len = 4
    current_seq_len = 1
    
    # Initialize weights
    for name, param in vm.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:
                nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
            
    # Bias gates to encourage exploration
    # MMU output: [Reg] -> [Ram_Q, Ram_K, Ram_V, Ram_G, SSD_K, SSD_V, SSD_G]
    # Ram_G is index 3 in the first split.
    # We want Ram_G to be positive (sigmoid > 0.5) initially.
    # The MMU is a Linear layer. We can bias the output bias vector.
    # mmu.bias is [Out_Dim].
    # We need to find the index of Ram_G.
    # Splits: Ram(D, D, D, 1), SSD(D, D, 3)
    # Ram_G index = D*3
    ram_g_idx = vm.vectorized_npu.memory_dim * 3
    with torch.no_grad():
        vm.vectorized_npu.mmu.bias[ram_g_idx] = 1.0 # Sigmoid(1.0) ~= 0.73
    
    print(f"Starting Curriculum: SeqLen {current_seq_len} -> {max_seq_len}")
    
    for i in range(1000):
        optimizer.zero_grad()
        
        # Generate Data
        seq = torch.randn(1, current_seq_len, input_size)
        zeros = torch.zeros(1, current_seq_len, input_size)
        inputs = torch.cat([seq, zeros], dim=1) 
        targets = torch.cat([zeros, seq], dim=1)
        
        loss = 0
        hidden_state = None
        memory_state = None
        
        for t in range(inputs.shape[1]):
            input_t = inputs[:, t, :]
            
            # VM Forward
            vm_out, hidden_state, memory_state = vm(input_t, hidden_state, memory_state)
            
            # Prediction
            pred = output_proj(vm_out)
            
            target_t = targets[:, t, :]
            loss += criterion(pred, target_t)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vm.parameters(), 1.0)
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Iter {i}: Loss = {loss.item():.4f} (SeqLen {current_seq_len})")
            
        # Curriculum Update
        if loss.item() < 0.1 * current_seq_len: # Threshold scaled by length
            if current_seq_len < max_seq_len:
                current_seq_len += 1
                print(f"Promoted to SeqLen {current_seq_len}!")
            elif loss.item() < 0.05:
                print("Converged!")
                break
            
    print(f"Final VM Loss: {loss.item():.4f}")
    if loss.item() < 0.5:
        print("VM Passed Copy Task!")
    else:
        print("VM Failed Copy Task.")

if __name__ == "__main__":
    test_vm_copy_task()
