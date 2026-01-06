import torch
import numpy as np
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n2 import HyperTransfer

def run_agentic_n2n2():
    print("--- N2N2: Agentic Transfer (UI-TARS 72B) ---")
    
    # 1. Initialize Brain
    brain = EvolutionaryBrain()
    n2n2 = HyperTransfer(brain)
    print("Brain initialized.")
    
    # 2. Load UI-TARS Embeddings (Partial Loading)
    # We want the 72B model's embeddings, but we can't load 144GB.
    # We use `safetensors` to read ONLY the embedding layer from disk.
    print("Locating UI-TARS-72B Embeddings...")
    
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        
        model_id = "ByteDance-Seed/UI-TARS-72B-DPO"
        
        print(f"Fetching index for {model_id}...")
        # 1. Download ONLY the index first
        try:
            index_path = snapshot_download(repo_id=model_id, allow_patterns=["*.index.json"])
        except Exception as e:
            print(f"Index download failed: {e}")
            return

        # Find the index json
        index_file = None
        for root, dirs, files in os.walk(index_path):
            for file in files:
                if file.endswith(".index.json"):
                    index_file = os.path.join(root, file)
                    break
        
        if not index_file:
            print("No index file found. Is this model sharded?")
            return

        # 2. Parse Index to find the specific shard
        param_name = "model.embed_tokens.weight"
        target_filename = None
        
        with open(index_file, 'r') as f:
            index_data = json.load(f)
            
        if param_name in index_data["weight_map"]:
            target_filename = index_data["weight_map"][param_name]
            print(f"Embeddings found in shard: {target_filename}")
        else:
            # Try generic name
            param_name = "transformer.wte.weight" # GPT style
            if param_name in index_data["weight_map"]:
                target_filename = index_data["weight_map"][param_name]
                print(f"Embeddings found in shard (legacy name): {target_filename}")
            else:
                print(f"Could not find '{param_name}' in index.")
                return

        # 3. Download ONLY that specific shard
        print(f"Downloading specific shard: {target_filename}...")
        shard_path = snapshot_download(repo_id=model_id, allow_patterns=[target_filename])
        
        # Locate the file in the downloaded folder
        target_file = None
        for root, dirs, files in os.walk(shard_path):
            if target_filename in files:
                target_file = os.path.join(root, target_filename)
                break

        if not target_file:
            print("No safetensors file found.")
            return

        # 3. Load ONLY that file
        print(f"Loading tensor from {target_file}...")
        
        source_data = {}
        
        with safe_open(target_file, framework="pt", device="cpu") as f:
            if param_name in f.keys():
                embeddings = f.get_tensor(param_name)
                print(f"Loaded Embedding Matrix: {embeddings.shape}")
                
                vocab_size = embeddings.shape[0]
                
                # 4. Extract
                print("Extracting Weights...")
                # Convert to numpy (fits in RAM)
                # Must cast to float32 first because numpy doesn't support bfloat16
                embeddings_np = embeddings.to(torch.float32).numpy()
                
                for i in range(vocab_size):
                    source_data[i] = embeddings_np[i]
                    if i % 10000 == 0:
                        print(f"Extracted {i}/{vocab_size}...", end='\r')
            else:
                print(f"Parameter {param_name} not found.")
                return
                    
    except Exception as e:
        print(f"Partial Loading Failed: {e}")
        print("Ensure `safetensors` and `huggingface_hub` are installed.")
        return

    print(f"\nExtracted {len(source_data)} embeddings.")
    
    # 5. Imprint into Brain
    print("Imprinting into Broca...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Dynamic Checkpoint Loading
    latest_checkpoint = EvolutionaryBrain.find_latest_checkpoint(os.path.join(project_root, "models", "saved"))
    
    if latest_checkpoint:
        checkpoint_path = latest_checkpoint
        print(f"Resuming N2N2 from {checkpoint_path}...")
        try:
            brain.load_model(checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            checkpoint_path = os.path.join(project_root, "models", "saved", "gen_349_elite.pt") # Fallback
    else:
        checkpoint_path = os.path.join(project_root, "models", "saved", "gen_349_elite.pt")
        print(f"Starting fresh. Will save to {checkpoint_path}")
            
    # n2n2 will handle projection (HiddenDim -> 64) and synaptic training
    n2n2.load_source_weights(source_data, target_dim=64, max_concepts=None) 
    n2n2.imprint_knowledge(save_interval=5000, checkpoint_path=checkpoint_path, resume=True)
    
    # Final Save
    brain.save_model(checkpoint_path)
    print(f"Brain saved to {checkpoint_path}")
    
    print("--- Agentic Transfer Complete ---")

if __name__ == "__main__":
    run_agentic_n2n2()
