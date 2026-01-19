import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class VectorizedNPU(nn.Module):
    """
    A Vectorized implementation of the Neural Processing Unit.
    Instead of a list of modules, this processes all NPUs in parallel using tensor operations.
    Shape: [Batch, Num_NPUs, Dim]
    """
    def __init__(self, input_size, register_size, memory_dim, max_npus=1024):
        super().__init__()
        self.input_size = input_size
        self.register_size = register_size
        self.memory_dim = memory_dim
        self.max_npus = max_npus
        
        # Shared Weights for all NPUs (Homogeneous Cores)
        # We use Linear layers that will be applied to [Batch, Num_NPUs, Features]
        
        # Core (ALU): [Input + Mem_Read + Registers] -> [New_Registers]
        # We use a GRU Cell for better state retention
        input_dim = input_size + memory_dim
        self.gru_cell = nn.GRUCell(input_dim, register_size)
        
        # We need to handle the batch * active_npus dimension.
        # GRUCell expects [Batch, Input]. We have [Batch, Active, Input].
        # We can flatten: [Batch * Active, Input] -> GRU -> [Batch * Active, Reg] -> Reshape
        
        # Memory Interface (MMU): [Registers] -> [Query, Key, Value, Gate]
        # We also add SSD Interface: [SSD_Key, SSD_Value, SSD_Op]
        # SSD_Op: 0=NoOp, 1=Read, 2=Write (Continuous gate)
        # Let's add 3 outputs for SSD: Key(Dim), Value(Dim), Gate(3: Read, Write, Nothing)
        self.mmu = nn.Linear(register_size, memory_dim * 3 + 1 + memory_dim * 2 + 3)
        
        # Output Interface: [Registers] -> [Output]
        self.output_interface = nn.Linear(register_size, input_size) 
        
    def forward(self, input_vector, registers, memory_context, active_npus):
        """
        Args:
            input_vector: [Batch, Input_Size] - Broadcasted instruction
            registers: [Batch, Max_NPUs, Register_Size] - State of all NPUs
            memory_context: [Batch, Memory_Size, Memory_Dim] - Shared RAM
            active_npus: int - Number of currently active NPUs (for masking)
        """
        batch_size = input_vector.shape[0]
        device = input_vector.device
        
        # Slice registers to active NPUs
        # [Batch, Active_NPUs, Reg_Size]
        curr_regs = registers[:, :active_npus, :]
        
        # 1. MMU: Generate Memory & SSD Access Intent
        mmu_out = self.mmu(curr_regs)
        
        # Split:
        # RAM: Query(D), Key(D), Value(D), Gate(1)
        # SSD: Key(D), Value(D), Gate(3)
        # Total: D*3 + 1 + D*2 + 3 = D*5 + 4
        
        ram_splits = [self.memory_dim, self.memory_dim, self.memory_dim, 1]
        ssd_splits = [self.memory_dim, self.memory_dim, 3]
        
        splits = torch.split(mmu_out, ram_splits + ssd_splits, dim=2)
        
        query, w_key, w_val, w_gate = splits[0], splits[1], splits[2], splits[3]
        ssd_key, ssd_val, ssd_gate = splits[4], splits[5], splits[6]
        
        w_gate = torch.sigmoid(w_gate)
        ssd_gate = F.softmax(ssd_gate, dim=-1) # [Read, Write, NoOp]
        
        # 2. Read from Memory (Attention)
        # Query: [Batch, Active, Mem_Dim]
        # Memory: [Batch, Mem_Size, Mem_Dim]
        # Scores: [Batch, Active, Mem_Size]
        scores = torch.bmm(query, memory_context.transpose(1, 2))
        scores = scores / math.sqrt(self.memory_dim)
        attn = F.softmax(scores, dim=-1)
        
        # Read: [Batch, Active, Mem_Size] @ [Batch, Mem_Size, Mem_Dim] -> [Batch, Active, Mem_Dim]
        read_data = torch.bmm(attn, memory_context)
        
        # 3. Core Execution
        # Broadcast Input: [Batch, Input] -> [Batch, Active, Input]
        input_expanded = input_vector.unsqueeze(1).expand(-1, active_npus, -1)
        
        # Concat: [Batch, Active, Input + Mem]
        # Note: GRU takes input and hidden separately.
        gru_input = torch.cat([input_expanded, read_data], dim=2)
        
        # Flatten for GRUCell
        # Input: [Batch * Active, Input_Dim]
        # Hidden: [Batch * Active, Reg_Size]
        gru_input_flat = gru_input.reshape(-1, gru_input.shape[-1])
        curr_regs_flat = curr_regs.reshape(-1, curr_regs.shape[-1])
        
        new_regs_flat = self.gru_cell(gru_input_flat, curr_regs_flat)
        
        # Reshape back
        new_regs_active = new_regs_flat.view(batch_size, active_npus, self.register_size)
        
        # 4. Write Intent
        write_intent = {
            'key': w_key,
            'value': w_val,
            'gate': w_gate
        }
        
        ssd_intent = {
            'key': ssd_key,
            'value': ssd_val,
            'gate': ssd_gate
        }
        
        # 5. Output
        output_active = self.output_interface(new_regs_active)
        
        return output_active, new_regs_active, write_intent, ssd_intent

from brain.modules.neural_ssd import NeuralSSD

class NeuralVirtualizationLayer(nn.Module):
    """
    The 'Hypervisor' with Neural V-Sync.
    Manages a Vectorized NPU pool and dynamically adjusts active cores.
    """
    def __init__(self, input_size, hidden_size, max_npus=64, memory_size=128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_npus = max_npus
        self.memory_size = memory_size
        
        # V-Sync Settings
        self.target_load = 0.90 # 90% stability target
        self.active_npus = 4    # Start small
        self.perf_history = []  # Rolling window of execution times
        
        # NPU Settings
        self.npu_register_size = hidden_size // 4 # Arbitrary ratio
        self.memory_dim = self.npu_register_size
        
        # Vectorized NPU Core
        self.vectorized_npu = VectorizedNPU(input_size, self.npu_register_size, self.memory_dim, max_npus)
        
        # Neural SSD (Long-Term Memory)
        # Shared across all NPUs
        self.ssd = NeuralSSD(self.memory_dim, self.memory_dim)
        
        # Aggregator: [Batch, Active, Input] -> [Batch, Hidden]
        # We use Attention aggregation to handle variable number of NPUs
        self.aggregator_query = nn.Parameter(torch.randn(1, hidden_size))
        self.aggregator_proj = nn.Linear(input_size, hidden_size)
        self.aggregator_norm = nn.LayerNorm(hidden_size) # Fix: Invariant Aggregation
        
    def forward(self, x, hidden_state=None, memory_state=None):
        start_time = time.time()
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize States
        if hidden_state is None or hidden_state.shape[1] != self.max_npus:
            # [Batch, Max_NPUs, Reg_Size]
            hidden_state = torch.zeros(batch_size, self.max_npus, self.npu_register_size, device=device)
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_size, self.memory_dim, device=device)
            
        # --- Neural V-Sync Logic ---
        # Adjust active_npus based on previous performance?
        # Or just track performance for now and adjust periodically.
        # For this step, we use self.active_npus.
        
        # Run Vectorized NPU
        npu_out, new_regs_active, w_intent, ssd_intent = self.vectorized_npu(x, hidden_state, memory_state, self.active_npus)
        
        # Update Hidden State (Only active slots)
        # We need to clone to avoid in-place modification errors in autograd if we were modifying a leaf, 
        # but hidden_state is usually passed in.
        # Safer to create new tensor.
        new_hidden_state = hidden_state.clone()
        new_hidden_state[:, :self.active_npus, :] = new_regs_active
        
        # Initialize new_memory_state
        new_memory_state = memory_state.clone()
        
        # --- SSD Operations ---
        # Aggregate SSD Intents
        # ssd_gate: [Batch, Active, 3] (Read, Write, NoOp)
        # We take the max gate across NPUs? Or average?
        # Let's just process the strongest intent.
        
        # For simplicity in this prototype:
        # We average the keys/values weighted by the "Write" gate for writing.
        # We average the keys weighted by the "Read" gate for reading.
        
        ssd_keys = ssd_intent['key']
        ssd_vals = ssd_intent['value']
        ssd_gates = ssd_intent['gate'] # [Batch, Active, 3]
        
        read_gate = ssd_gates[:, :, 0]
        write_gate = ssd_gates[:, :, 1]
        
        # Write to SSD
        # If mean write gate > 0.5, we write.
        avg_write_gate = write_gate.mean(dim=1) # [Batch]
        if avg_write_gate.mean() > 0.5:
            # Aggregate Key/Value
            # Weighted average by write_gate
            # [Batch, Active, 1] * [Batch, Active, Dim] -> Sum -> [Batch, Dim]
            w_sum = write_gate.sum(dim=1, keepdim=True) + 1e-6
            agg_key = (ssd_keys * write_gate.unsqueeze(2)).sum(dim=1) / w_sum
            agg_val = (ssd_vals * write_gate.unsqueeze(2)).sum(dim=1) / w_sum
            
            self.ssd.write(agg_key, agg_val)
            
        # Read from SSD
        # If mean read gate > 0.5, we read.
        avg_read_gate = read_gate.mean(dim=1)
        if avg_read_gate.mean() > 0.5:
             w_sum = read_gate.sum(dim=1, keepdim=True) + 1e-6
             # ssd_keys: [Batch, Active, Dim]
             # read_gate: [Batch, Active] -> [Batch, Active, 1]
             # sum(dim=1) -> [Batch, Dim]
             # w_sum: [Batch, 1]
             agg_key = (ssd_keys * read_gate.unsqueeze(2)).sum(dim=1) / w_sum
             
             read_vals, _ = self.ssd.read(agg_key)
             # read_vals: [Batch, 1, Dim] -> [Batch, Dim]
             read_val = read_vals.squeeze(1)
             
             # DMA Write to Memory Slot 0
             # We modify new_memory_state in place (clone first)
             new_memory_state = new_memory_state.clone()
             new_memory_state[:, 0, :] = read_val
        
        # --- Memory Update (Write Back) ---
        # Aggregate writes from active NPUs
        # [Batch, Active, Mem_Dim]
        w_keys = w_intent['key']
        w_vals = w_intent['value']
        w_gates = w_intent['gate']
        
        # Write Attention: [Batch, Active, Mem_Size]
        write_scores = torch.bmm(w_keys, memory_state.transpose(1, 2)) / math.sqrt(self.memory_dim)
        write_attn = F.softmax(write_scores, dim=-1)
        
        # Weighted Sum of Writes
        # Contribution: Gate * Attn
        # [Batch, Active, Mem_Size] -> [Batch, Mem_Size, Active]
        contributions = (w_gates * write_attn).transpose(1, 2)
        total_weight = contributions.sum(dim=2, keepdim=True).clamp(0.0, 1.0)
        
        # [Batch, Mem_Size, Active] @ [Batch, Active, Mem_Dim] -> [Batch, Mem_Size, Mem_Dim]
        combined_val = torch.bmm(contributions, w_vals)
        
        # Update new_memory_state (which might have SSD data)
        new_memory_state = new_memory_state * (1.0 - total_weight) + combined_val
        
        # Aggregation (Variable Length)
        # npu_out: [Batch, Active, Input_Size]
        # Proj: [Batch, Active, Hidden]
        proj_out = self.aggregator_proj(npu_out)
        
        # Attention Aggregation
        # Query: [1, Hidden] -> [Batch, 1, Hidden]
        q = self.aggregator_query.expand(batch_size, -1, -1)
        # Keys/Vals: proj_out
        # Scores: [Batch, 1, Active]
        agg_scores = torch.bmm(q, proj_out.transpose(1, 2)) / math.sqrt(self.hidden_size)
        agg_attn = F.softmax(agg_scores, dim=-1)
        
        # [Batch, 1, Active] @ [Batch, Active, Hidden] -> [Batch, 1, Hidden]
        aggregated_output = torch.bmm(agg_attn, proj_out).squeeze(1)
        
        # Fix: Invariant Aggregation (V-Sync Jitter)
        aggregated_output = self.aggregator_norm(aggregated_output)
        
        # --- V-Sync Update ---
        end_time = time.time()
        duration = end_time - start_time
        self.update_vsync(duration)
        
        return aggregated_output, new_hidden_state, new_memory_state

    def update_vsync(self, duration):
        """
        Adjusts active_npus to maintain stable performance (Neural V-Sync).
        Target: 90% of max stable throughput.
        """
        self.perf_history.append(duration)
        if len(self.perf_history) > 100: self.perf_history.pop(0)
        
        avg_duration = sum(self.perf_history) / len(self.perf_history)
        
        # Simple Logic: If fast, grow. If slow, shrink.
        # Thresholds need to be tuned or relative.
        # Let's assume a target latency (e.g., 10ms).
        target_latency = 0.01 # 10ms
        
        # If we are faster than target (load < 90%), we can grow.
        if avg_duration < target_latency * 0.9:
            if self.active_npus < self.max_npus:
                self.active_npus += 1
        elif avg_duration > target_latency:
            if self.active_npus > 1:
                self.active_npus -= 1
                
    def resize(self, new_num_npus=None):
        # Manual resize override
        if new_num_npus:
            self.active_npus = min(new_num_npus, self.max_npus)
