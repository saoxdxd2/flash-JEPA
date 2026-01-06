import torch
import numpy as np
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n2 import HyperTransfer

def run_qwen3_vision_n2n2():
    print("--- N2N2: Visual Transfer (Qwen3-VL-235B) ---")
    
    # 1. Initialize Brain
    brain = EvolutionaryBrain()
    n2n2 = HyperTransfer(brain)
    print("Brain initialized.")
    
    # 2. Load Qwen3-VL Visual Weights (Partial Loading)
    print("Locating Qwen3-VL Visual Weights...")
    
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        
        model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
        
        # 1. Download/Locate the Index File
        print(f"Fetching index for {model_id}...")
        # Download ONLY the index first
        folder_path = snapshot_download(repo_id=model_id, allow_patterns=["*.index.json"])
        
        # Find the index json
        index_file = None
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".index.json"):
                    index_file = os.path.join(root, file)
                    break
        
        target_file = None
        # Qwen-VL usually uses 'visual.patch_embed.proj.weight' or similar
        # We'll try to find it dynamically or fallback to standard names
        param_name = "visual.patch_embed.proj.weight" 
        
        if index_file:
            print(f"Index file found: {index_file}")
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Try to find the visual projection layer
            found_param = None
            for key in index_data["weight_map"].keys():
                if "visual" in key and "patch_embed" in key and "weight" in key:
                    found_param = key
                    break
            
            if found_param:
                param_name = found_param
                print(f"Identified Visual Layer: {param_name}")
            else:
                print(f"Could not find visual patch embedding in index. Trying default '{param_name}'...")
            
            if param_name in index_data["weight_map"]:
                target_filename = index_data["weight_map"][param_name]
                print(f"Visual Weights found in shard: {target_filename}")
                
                # Download ONLY the specific shard
                print(f"Downloading specific shard: {target_filename}...")
                shard_path = snapshot_download(repo_id=model_id, allow_patterns=[target_filename])
                
                target_file = os.path.join(shard_path, target_filename)
                # If snapshot_download returns folder, we might need to find it
                if not os.path.exists(target_file):
                     for root, dirs, files in os.walk(shard_path):
                        if target_filename in files:
                            target_file = os.path.join(root, target_filename)
                            break
            else:
                print("Visual weights not found in index.")
                return
            
        else:
            print("No index file found. Checking for single safetensors file...")
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".safetensors"):
                        target_file = os.path.join(root, file)
                        break
            if not target_file:
                print("No safetensors file found.")
                return

        # 3. Load ONLY that file
        print(f"Loading tensor from {target_file}...")
        
        source_data = {}
        
        with safe_open(target_file, framework="pt", device="cpu") as f:
            if param_name in f.keys():
                # Shape: [OutChannels, InChannels, Kernel, Kernel]
                # e.g. [1280, 3, 14, 14]
                filters = f.get_tensor(param_name)
                print(f"Loaded Visual Filters: {filters.shape}")
                
                num_filters = filters.shape[0]
                
                print("Converting to Numpy...")
                # Must cast to float32 first because numpy doesn't support bfloat16
                filters_np = filters.to(torch.float32).numpy()
                
                # We treat each filter as a "Visual Concept"
                for i in range(num_filters):
                    # Store the filter weights
                    # Qwen3-VL uses 3D Conv: [Out, In, Time, H, W] -> [3, 2, 16, 16]
                    # We need 2D Conv: [Out, In, H, W] -> [3, 16, 16]
                    # We average over the temporal dimension (Time=2)
                    source_data[i] = filters_np[i].mean(axis=1) 
                    
            else:
                print(f"Parameter {param_name} not found in {target_file} keys.")
                return
                    
    except Exception as e:
        print(f"Partial Loading Failed: {e}")
        return

    print(f"\nExtracted {len(source_data)} visual concepts.")
    
    # 5. Imprint into Broca (Visual Knowledge)
    print("Imprinting into Broca (Visual Knowledge)...")
    
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
            
    n2n2.load_source_weights(source_data, target_dim=64, max_concepts=None) 
    n2n2.imprint_knowledge(save_interval=5000, checkpoint_path=checkpoint_path, resume=True)
    
    # Final Save
    brain.save_model(checkpoint_path)
    print(f"Brain saved to {checkpoint_path}")
    
    print("--- Visual Transfer Complete ---")

if __name__ == "__main__":
    run_qwen3_vision_n2n2()
