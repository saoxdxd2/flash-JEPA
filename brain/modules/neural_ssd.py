import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class NeuralSSD(nn.Module):
    """
    Neural Solid State Drive (N-SSD).
    A persistent, content-addressable memory store.
    
    Features:
    - Vector Storage: Stores (Key, Value) pairs.
    - Content Addressing: Retrieval via Cosine Similarity.
    - Persistence: Saves/Loads from disk.
    """
    def __init__(self, key_dim, value_dim, capacity=10000, storage_path="neural_ssd.pt"):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        self.storage_path = storage_path
        
        # Memory Buffers (Persistent)
        # We use buffers so they are part of the state_dict but not trained by SGD directly (usually).
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class NeuralSSD(nn.Module):
    """
    Neural Solid State Drive (N-SSD).
    A persistent, content-addressable memory store.
    
    Features:
    - Vector Storage: Stores (Key, Value) pairs.
    - Content Addressing: Retrieval via Cosine Similarity.
    - Persistence: Saves/Loads from disk.
    """
    def __init__(self, key_dim, value_dim, capacity=10000, storage_path="neural_ssd.pt"):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        self.storage_path = storage_path
        
        # Memory Buffers (Persistent)
        # We use buffers so they are part of the state_dict but not trained by SGD directly (usually).
        # Actually, we want them to be updated explicitly, not by optimizer.
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.register_buffer("usage", torch.zeros(capacity)) # For LRU replacement
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        
        # Object Store (for non-tensor data like Fractal DNA)
        self.object_store = {} # {index: object}
        
        # Load if exists
        self.load()
        
    def write(self, keys, values, objects=None):
        """
        Writes batch of (key, value) pairs to SSD.
        Args:
            keys: [Batch, Key_Dim]
            values: [Batch, Value_Dim] (Optional if objects provided)
            objects: List of objects (Optional)
        """
        batch_size = keys.shape[0]
        device = keys.device
        
        # Simple Append / Circular Buffer for now
        # In future: Use Usage/LRU to replace least useful memories.
        
        curr_pos = self.count.item()
        
        for i in range(batch_size):
            idx = (curr_pos + i) % self.capacity
            self.keys[idx] = keys[i].detach() # Detach to stop gradient flowing into storage (unless we want DNC-like gradients?)
            # Usually SSD is "hard" storage, so no gradients.
            
            if values is not None:
                self.values[idx] = values[i].detach()
                
            if objects is not None and i < len(objects):
                self.object_store[int(idx)] = objects[i]
                
            self.usage[idx] = 1.0 # Mark as used
            
        self.count = (self.count + batch_size) % self.capacity
        
    def read(self, query, k=1):
        """
        Reads top-k nearest neighbors.
        Returns values (tensors) and objects (if available).
        """
        # Cosine Similarity
        # Normalize Query
        q_norm = F.normalize(query, p=2, dim=1)
        
        # Normalize Keys (Only used slots)
        # Optimization: Maintain normalized keys?
        # For prototype, just normalize on the fly.
        
        # We only search up to 'capacity' or 'count' if not full?
        # If circular, we search all.
        # Let's assume full capacity is available (zeros if empty).
        
        k_norm = F.normalize(self.keys, p=2, dim=1)
        
        # Similarity: [Batch, Capacity]
        scores = torch.mm(q_norm, k_norm.t())
        
        # Top-K
        top_scores, top_indices = torch.topk(scores, k, dim=1)
        
        # Retrieve Values
        # [Batch, k, Value_Dim]
        # Gather is tricky with multiple dims.
        # top_indices: [Batch, k]
        
        batch_size = query.shape[0]
        retrieved_values = torch.stack([self.values[top_indices[i]] for i in range(batch_size)])
        
        # Retrieve Objects
        retrieved_objects = []
        for i in range(batch_size):
            batch_objs = []
            for idx in top_indices[i]:
                idx_val = int(idx.item())
                batch_objs.append(self.object_store.get(idx_val, None))
            retrieved_objects.append(batch_objs)
        
        return retrieved_values, top_scores, retrieved_objects

    def save(self):
        # Save tensors
        torch.save(self.state_dict(), self.storage_path)
        # Save objects (pickle/json) - for now simplified to just torch save of dict if possible
        # But torch.save handles dicts of objects usually.
        # We'll save object_store separately to avoid buffer issues?
        # Actually, state_dict doesn't include python attributes.
        torch.save(self.object_store, self.storage_path + ".objs")
        
    def load(self):
        if os.path.exists(self.storage_path):
            try:
                state = torch.load(self.storage_path)
                self.load_state_dict(state)
                print(f"NeuralSSD loaded from {self.storage_path}")
            except Exception as e:
                print(f"Failed to load NeuralSSD Tensors: {e}")
                
        if os.path.exists(self.storage_path + ".objs"):
            try:
                self.object_store = torch.load(self.storage_path + ".objs")
                print(f"NeuralSSD Objects loaded.")
            except Exception as e:
                print(f"Failed to load NeuralSSD Objects: {e}")
