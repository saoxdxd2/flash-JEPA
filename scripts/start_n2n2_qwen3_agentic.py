import torch
import numpy as np
import os
import sys
import json
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n2 import HyperTransfer

class Qwen3Teacher:
    """
    Manages the Qwen-3 235B Model (Embeddings) for N2N2 Transfer.
    Allows step-by-step training.
    """
    def __init__(self, brain):
        self.brain = brain
        self.n2n2 = HyperTransfer(brain)
        self.source_data = {}
        self.vocab_size = 0
        self.loaded = False
        self.gate_weights = []
        
    def setup(self):
        """Downloads and loads the Qwen-3 Embeddings."""
        print("--- Qwen3Teacher: Setup ---")
        print("Locating Qwen-3-235B Embeddings...")
        
        try:
            from huggingface_hub import snapshot_download
            from safetensors import safe_open
            
            # Using the official Qwen-3 VL Model ID
            model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct" 
            
            print(f"Fetching index for {model_id}...")
            # 1. Download ONLY the index first
            try:
                index_path = snapshot_download(repo_id=model_id, allow_patterns=["*.index.json"])
            except Exception as e:
                print(f"Index download failed: {e}")
                return False

            # Find the index json
            index_file = None
            for root, dirs, files in os.walk(index_path):
                for file in files:
                    if file.endswith(".index.json"):
                        index_file = os.path.join(root, file)
                        break
            
            if not index_file:
                print("No index file found.")
                return False

            # 2. Parse Index to find the specific shard
            param_name = "model.embed_tokens.weight"
            target_filename = None
            
            with open(index_file, 'r') as f:
                index_data = json.load(f)
                
            # Search for embedding layer
            found_param = None
            for key in index_data["weight_map"].keys():
                if "embed_tokens" in key and "weight" in key:
                    found_param = key
                    break
                if "wte" in key and "weight" in key:
                    found_param = key
                    break
                    
            if found_param:
                param_name = found_param
                target_filename = index_data["weight_map"][param_name]
                print(f"Embeddings found in shard: {target_filename}")
            elif param_name in index_data["weight_map"]:
                target_filename = index_data["weight_map"][param_name]
            else:
                print(f"Could not find embedding layer in index.")
                return False

            # 3. Download ONLY that specific shard
            print(f"Downloading specific shard: {target_filename}...")
            shard_path = snapshot_download(repo_id=model_id, allow_patterns=[target_filename])
            
            # Locate the file
            target_file = None
            for root, dirs, files in os.walk(shard_path):
                if target_filename in files:
                    target_file = os.path.join(root, target_filename)
                    break

            if not target_file:
                print("No safetensors file found.")
                return False

            # 3. Load ONLY that file
            print(f"Loading tensor from {target_file}...")
            
            with safe_open(target_file, framework="pt", device="cpu") as f:
                if param_name in f.keys():
                    embeddings = f.get_tensor(param_name)
                    print(f"Loaded Embedding Matrix: {embeddings.shape}")
                    
                    self.vocab_size = embeddings.shape[0]
                    
                    # 4. Extract
                    print("Extracting Weights...")
                    # Convert to numpy (fits in RAM: ~1-3GB)
                    embeddings_np = embeddings.to(torch.float32).numpy()
                    
                    for i in range(self.vocab_size):
                        self.source_data[i] = embeddings_np[i]
                        if i % 10000 == 0:
                            print(f"Extracted {i}/{self.vocab_size}...", end='\r')
                else:
                    print(f"Parameter {param_name} not found.")
                    return False
                        
        except Exception as e:
            print(f"Partial Loading Failed: {e}")
            return False

        print(f"\nExtracted {len(self.source_data)} embeddings.")
        
        # Load into N2N2
        # Target Dim: Dynamic (Matches BrocaModule)
        target_dim = self.brain.broca.embedding_dim
        print(f"N2N2: Dynamic Target Dimension: {target_dim}")
        self.n2n2.load_source_weights(self.source_data, target_dim=target_dim, max_concepts=None)
        # 5. Optional: Load MoE Gates (The "Logic" of Qwen-3)
        # We ONLY load gates that are in the SAME shard as the embeddings to avoid massive downloads.
        print("Searching for MoE Gates in the current shard...")
        gate_params = [k for k in index_data["weight_map"].keys() if "mlp.gate.weight" in k]
        
        if gate_params:
            loaded_gates = 0
            for p_name in gate_params:
                shard_for_gate = index_data["weight_map"][p_name]
                
                # Only load if it's in the shard we already have
                if shard_for_gate == target_filename:
                    print(f"Loading Gate from current shard: {p_name}...")
                    try:
                        with safe_open(target_file, framework="pt", device="cpu") as f:
                            if p_name in f.keys():
                                gate = f.get_tensor(p_name).to(torch.float32).numpy()
                                self.gate_weights.append(gate)
                                loaded_gates += 1
                                print(f"  > Loaded Gate: {gate.shape}")
                    except Exception as e:
                        print(f"  > Failed to load gate {p_name}: {e}")
                
                if loaded_gates >= 3: break # Limit to 3 gates
            
            if loaded_gates == 0:
                print("No MoE gates found in the embedding shard. SSI will use fallback logic.")
        
        self.loaded = True
        return True

    def generate_logic_trajectory(self, sequence_length=5):
        """
        Generates a reasoning trajectory based on Qwen-3's MoE Routing Logic.
        """
        if not self.loaded or not self.source_data: return None
        
        import random
        start_key = random.choice(list(self.source_data.keys()))
        current_embedding = torch.tensor(self.source_data[start_key])
        
        input_seq = []
        visual_h_seq = []
        bus_seq = []
        motor_h_seq = []
        
        L = self.brain.genome.latent_dim
        bus_size = self.brain.trm.bus_size
        visual_hidden = self.brain.trm.visual_hidden
        motor_hidden = self.brain.trm.motor_hidden
        
        # Use a gate to simulate "Routing"
        gate = self.gate_weights[0] if self.gate_weights else np.random.randn(current_embedding.shape[0], 64)
        gate_t = torch.tensor(gate)
        
        for _ in range(sequence_length):
            # 1. Input (Projected Embedding)
            proj_input = torch.matmul(current_embedding, self.n2n2.projection_matrix)
            proj_input = torch.tanh(proj_input)
            
            # 2. Routing Logic (Simulated via Gate)
            # current_embedding is [4096], gate_t is [128, 4096]
            # We want [128]
            routing = torch.matmul(gate_t, current_embedding)
            expert_idx = torch.argmax(routing)
            
            expert_proj = torch.randn(proj_input.shape[0], bus_size) / np.sqrt(bus_size)
            bus_vector = torch.tanh(torch.matmul(proj_input, expert_proj))
            
            # 3. Visual Hidden State (Simulated)
            visual_h = torch.tanh(torch.matmul(proj_input, torch.randn(proj_input.shape[0], visual_hidden) / np.sqrt(visual_hidden)))
            
            # 4. Motor State (Simulated)
            motor_h = torch.tanh(torch.matmul(bus_vector, torch.randn(bus_size, motor_hidden) / np.sqrt(motor_hidden)))
            
            # 5. Construct Full Input (Pad to input_size)
            full_input = torch.zeros(self.brain.input_size)
            # Place semantic vector in the correct slot (2*L to 3*L)
            full_input[2*L : 3*L] = proj_input
            
            input_seq.append(full_input)
            bus_seq.append(bus_vector)
            visual_h_seq.append(visual_h)
            motor_h_seq.append(motor_h)
            
            # 5. Next Thought (Transition)
            current_embedding = current_embedding + torch.randn_like(current_embedding) * 0.05
            
        return torch.stack(input_seq).unsqueeze(1), torch.stack(visual_h_seq).unsqueeze(1), torch.stack(motor_h_seq).unsqueeze(1), torch.stack(bus_seq).unsqueeze(1)

    def train_step(self, steps=100):
        """Runs a batch of N2N2 imprinting."""
        if not self.loaded:
            print("Teacher not loaded. Call setup() first.")
            return
            
        # Use sequential sampling to help TitansMemory find patterns
        if not hasattr(self, 'last_key_index'):
            self.last_key_index = 0
            
        keys = sorted(list(self.source_data.keys()))
        
        # Select a contiguous block of keys
        start_idx = self.last_key_index
        end_idx = (start_idx + steps) % len(keys)
        
        if end_idx > start_idx:
            batch_keys = keys[start_idx:end_idx]
        else:
            batch_keys = keys[start_idx:] + keys[:end_idx]
            
        self.last_key_index = end_idx
        
        # Check for Dynamic Compression (Variable Latent Dim)
        target_dim = self.brain.broca.embedding_dim
        if self.n2n2.projection_matrix is not None:
            if self.n2n2.projection_matrix.shape[1] != target_dim:
                print(f"N2N2: Updating Compression Dimension to {target_dim}...")
                source_dim = self.n2n2.projection_matrix.shape[0]
                self.n2n2.projection_matrix = torch.randn(source_dim, target_dim) / np.sqrt(target_dim)
        
        broca = self.brain.broca
        total_loss = 0.0
        count = 0
        
        for key in batch_keys:
            target_embedding = self.source_data[key]
            
            # 1. Prepare Input Spike
            vector = torch.tensor(target_embedding, dtype=torch.float32)
            if vector.dim() > 1:
                vector = vector.flatten()
                
            # Apply Projection
            if self.n2n2.projection_matrix is not None:
                vector = torch.matmul(vector, self.n2n2.projection_matrix)
                
            # Normalize
            vector = torch.tanh(vector)
            
            # 2. Forward Pass & Learn
            loss_val = 0.1 # Default
            reward = 1.0
            if hasattr(broca, 'process_text_embedding'):
                _, surprise = broca.process_text_embedding(vector, reward=reward)
                loss_val = surprise
            
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            total_loss += loss_val
            count += 1
            
        return total_loss / max(1, count)

def start_qwen3_n2n2():
    print("--- N2N2: Qwen-3 235B (MoE) Knowledge Transfer ---")
    brain = EvolutionaryBrain()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "models", "saved")
    
    # Dynamic Checkpoint Loading
    latest_checkpoint = EvolutionaryBrain.find_latest_checkpoint(models_dir)
    
    if latest_checkpoint:
        checkpoint_path = latest_checkpoint
        print(f"Resuming N2N2 from {checkpoint_path}...")
        try:
            brain.load_model(checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            checkpoint_path = os.path.join(models_dir, "gen_349_elite.pt") # Fallback
    else:
        checkpoint_path = os.path.join(models_dir, "gen_349_elite.pt")
        print(f"Starting fresh. Will save to {checkpoint_path}")
        
    teacher = Qwen3Teacher(brain)
    if teacher.setup():
        print("Imprinting into Broca...")
        teacher.n2n2.imprint_knowledge(save_interval=5000, checkpoint_path=checkpoint_path, resume=True)
        brain.save_model(checkpoint_path)
        print(f"Brain saved to {checkpoint_path}")

if __name__ == "__main__":
    start_qwen3_n2n2()
