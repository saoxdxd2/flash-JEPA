import torch
import os
import sys
import json
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.fnd_encoder import FractalEncoder

def compress_qwen_to_fractal_dna():
    print("=== Fractal Neural DNA: Qwen-3 Compression Engine ===")
    
    # 1. Setup & Requirements
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError:
        print("Installing dependencies...")
        os.system("pip install huggingface_hub safetensors")
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("WARNING: Running on CPU. This will be extremely slow. GPU recommended.")

    # 2. Download Qwen-3-VL-235B (Metadata Only First)
    model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    print(f"Fetching model info for {model_id}...")
    
    # We download the index to map layers to shards
    try:
        index_path = snapshot_download(repo_id=model_id, allow_patterns=["*.index.json"])
    except Exception as e:
        print(f"Failed to download index: {e}")
        return

    # Find index file
    index_file = None
    for root, dirs, files in os.walk(index_path):
        for file in files:
            if file.endswith(".index.json"):
                index_file = os.path.join(root, file)
                break
                
    if not index_file:
        print("No index file found.")
        return

    with open(index_file, 'r') as f:
        index_data = json.load(f)
        
    weight_map = index_data["weight_map"]
    
    # 3. Compression Loop
    # We process shard by shard to avoid OOM
    unique_shards = sorted(list(set(weight_map.values())))
    print(f"Found {len(unique_shards)} shards to process.")
    
    fractal_brain = {} # The final DNA store
    encoder = FractalEncoder(device=device)
    
    total_params = 0
    compressed_params = 0
    
    for shard_file in tqdm(unique_shards, desc="Processing Shards"):
        print(f"\nDownloading shard: {shard_file}...")
        try:
            shard_path = snapshot_download(repo_id=model_id, allow_patterns=[shard_file])
        except Exception as e:
            print(f"Failed to download shard {shard_file}: {e}")
            continue
            
        # Locate the actual file
        full_shard_path = None
        for root, dirs, files in os.walk(shard_path):
            if shard_file in files:
                full_shard_path = os.path.join(root, shard_file)
                break
                
    # 4. Save Final DNA
    print("\n=== Compression Complete ===")
    print(f"Total Original Parameters: {total_params:,}")
    print(f"Total Compressed Parameters (DNA): {compressed_params:,}")
    print(f"Compression Ratio: {total_params/compressed_params:.2f}:1")
    
    output_path = "fractal_brain.pt"
    print(f"Saving Fractal Brain to {output_path}...")
    torch.save(fractal_brain, output_path)
    print("Done! Download 'fractal_brain.pt' and load it into the Vessel.")

if __name__ == "__main__":
    compress_qwen_to_fractal_dna()
