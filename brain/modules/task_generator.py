import torch
import random
import math

class TaskGenerator:
    """
    Generates diverse cognitive tasks for the Neural Computer.
    Tasks:
    1. Copy Task (Memory)
    2. Associative Recall (Memory + Binding)
    3. Arithmetic (Logic/Math) - Simple addition/subtraction
    4. Pattern Completion (Logic) - 1, 2, 3, ?
    """
    def __init__(self, input_size=16, output_size=16):
        self.input_size = input_size
        self.output_size = output_size
        self.difficulty = 1 # 1 to 10
        
    def set_difficulty(self, level):
        self.difficulty = max(1, min(10, level))
        
    def generate_task(self, task_type=None):
        """
        Returns:
            inputs: [1, Seq_Len, Input_Size]
            targets: [1, Seq_Len, Output_Size]
            metadata: dict (description, type)
        """
        if task_type is None:
            # Randomly select task based on difficulty
            # Higher difficulty unlocks more complex tasks
            options = ['copy']
            if self.difficulty >= 2: options.append('recall')
            if self.difficulty >= 3: options.append('pattern')
            if self.difficulty >= 4: options.append('math')
            task_type = random.choice(options)
            
        if task_type == 'copy':
            return self._generate_copy_task()
        elif task_type == 'recall':
            return self._generate_recall_task()
        elif task_type == 'pattern':
            return self._generate_pattern_task()
        elif task_type == 'math':
            return self._generate_math_task()
        else:
            return self._generate_copy_task()
            
    def _generate_copy_task(self):
        seq_len = self.difficulty + 1 # Length scales with difficulty
        vector_dim = min(8, self.input_size)
        
        seq = torch.randn(1, seq_len, vector_dim)
        zeros = torch.zeros(1, seq_len, vector_dim)
        
        # Pad to input_size
        if self.input_size > vector_dim:
            padding = torch.zeros(1, seq_len, self.input_size - vector_dim)
            seq = torch.cat([seq, padding], dim=2)
            zeros = torch.cat([zeros, padding], dim=2)
            
        inputs = torch.cat([seq, zeros], dim=1)
        targets = torch.cat([zeros, seq], dim=1)
        
        return inputs, targets, {'type': 'copy', 'desc': f'Copy Sequence Len {seq_len}'}

    def _generate_recall_task(self):
        num_pairs = max(2, self.difficulty)
        vector_dim = min(8, self.input_size)
        
        keys = [torch.randn(1, 1, vector_dim) for _ in range(num_pairs)]
        vals = [torch.randn(1, 1, vector_dim) for _ in range(num_pairs)]
        
        # Query
        idx = random.randint(0, num_pairs-1)
        query_k = keys[idx]
        target_v = vals[idx]
        
        # Construct Sequence: K1, V1, K2, V2, ..., Query
        seq_items = []
        for k, v in zip(keys, vals):
            seq_items.append(k)
            seq_items.append(v)
        seq_items.append(query_k)
        
        # Pad
        if self.input_size > vector_dim:
            padding = torch.zeros(1, 1, self.input_size - vector_dim)
            seq_items = [torch.cat([item, padding], dim=2) for item in seq_items]
            target_v = torch.cat([target_v, padding], dim=2)
            
        inputs = torch.cat(seq_items, dim=1)
        
        # Target is zeros until the end
        targets = torch.zeros_like(inputs)
        targets[:, -1, :] = target_v
        
        return inputs, targets, {'type': 'recall', 'desc': f'Recall {num_pairs} Pairs'}

    def _generate_pattern_task(self):
        # Linear pattern: v, v+d, v+2d, ...
        seq_len = max(3, self.difficulty + 1)
        vector_dim = min(8, self.input_size)
        
        start = torch.randn(1, 1, vector_dim)
        delta = torch.randn(1, 1, vector_dim) * 0.5
        
        seq = []
        for i in range(seq_len):
            seq.append(start + delta * i)
            
        # Input: v1, v2, v3
        # Target: v2, v3, v4 (Next token prediction)
        
        inputs = torch.cat(seq[:-1], dim=1)
        targets = torch.cat(seq[1:], dim=1)
        
        # Pad
        if self.input_size > vector_dim:
            padding = torch.zeros(1, inputs.shape[1], self.input_size - vector_dim)
            inputs = torch.cat([inputs, padding], dim=2)
            targets = torch.cat([targets, padding], dim=2)
            
        return inputs, targets, {'type': 'pattern', 'desc': f'Linear Pattern Len {seq_len}'}
        
    def _generate_math_task(self):
        # Simple Vector Addition: A + B = C
        vector_dim = min(8, self.input_size)
        
        a = torch.randn(1, 1, vector_dim)
        b = torch.randn(1, 1, vector_dim)
        c = a + b
        
        # Input: A, B
        # Target: 0, C
        
        inputs = torch.cat([a, b], dim=1)
        targets = torch.cat([torch.zeros_like(a), c], dim=1)
        
        # Pad
        if self.input_size > vector_dim:
            padding = torch.zeros(1, inputs.shape[1], self.input_size - vector_dim)
            inputs = torch.cat([inputs, padding], dim=2)
            targets = torch.cat([targets, padding], dim=2)
            
        return inputs, targets, {'type': 'math', 'desc': 'Vector Addition'}
