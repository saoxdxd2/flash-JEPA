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
    # We process shards in batches to optimize download/compute overlap and disk usage
    unique_shards = sorted(list(set(weight_map.values())))
    print(f"Found {len(unique_shards)} shards to process.")
    
    fractal_brain = {} # The final DNA store
    encoder = FractalEncoder(device=device)
    
    total_params = 0
    compressed_params = 0
    
    BATCH_SIZE = 8
    temp_dir = "./temp_shards"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process in batches
    for i in range(0, len(unique_shards), BATCH_SIZE):
        batch_shards = unique_shards[i : i + BATCH_SIZE]
        print(f"\n=== Processing Batch {i//BATCH_SIZE + 1}/{(len(unique_shards)+BATCH_SIZE-1)//BATCH_SIZE} ===")
        print(f"Shards: {batch_shards}")
        
        # 1. Download Batch
        batch_files = []
        for shard_file in tqdm(batch_shards, desc="Downloading Batch"):
            try:
                # Download to specific temp dir
                shard_path = snapshot_download(
                    repo_id=model_id, 
                    allow_patterns=[shard_file],
                    local_dir=temp_dir,
                    local_dir_use_symlinks=False # Ensure actual files for easy deletion
                )
                full_path = os.path.join(temp_dir, shard_file)
                batch_files.append(full_path)
            except Exception as e:
                print(f"Failed to download {shard_file}: {e}")
        
        # 2. Collect Layers from Batch
        layers_to_process = []
        for full_shard_path in batch_files:
            if not os.path.exists(full_shard_path): continue
            
            try:
                with safe_open(full_shard_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        layers_to_process.append((key, tensor))
            except Exception as e:
                print(f"Error reading {full_shard_path}: {e}")

        # 3. Compress Batch in Parallel
        from concurrent.futures import ThreadPoolExecutor
        
        def process_layer(item):
            key, tensor = item
            if len(tensor.shape) < 2:
                return key, tensor.cpu(), tensor.nelement(), tensor.nelement()
            
            # Compress
            dna = encoder.encode(tensor, num_transforms=16, iterations=200)
            
            orig_count = tensor.nelement()
            comp_count = 16 * 7
            return key, dna.to_json(), orig_count, comp_count

        print(f"Compressing {len(layers_to_process)} layers in parallel...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(process_layer, layers_to_process), total=len(layers_to_process), desc="Compressing"))
            
        # Store results
        for key, data, orig, comp in results:
            fractal_brain[key] = data
            total_params += orig
            compressed_params += comp
            
        # 4. Cleanup Batch
        print("Cleaning up batch files...")
        for fpath in batch_files:
            if os.path.exists(fpath):
                os.remove(fpath)
        
        # Clear RAM
        del layers_to_process
        del results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
