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
    
    BATCH_SIZE = 1 # Process 1 shard at a time to save RAM (5GB vs 40GB)
    temp_dir = "./temp_shards"
    os.makedirs(temp_dir, exist_ok=True)
    
    start_time_total = time.time()
    processed_shards = 0
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
        print("Loading layers into RAM...")
        layers_to_process = []
        for full_shard_path in tqdm(batch_files, desc="Loading Shards"):
            if not os.path.exists(full_shard_path): continue
            
            try:
                with safe_open(full_shard_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        if len(tensor.shape) == 2:
                            layers_to_process.append((key, tensor))
                        else:
                             # Store 1D, 3D, 4D tensors raw
                             # Optimization: Store as FP16 to save space
                             fractal_brain[key] = tensor.half().cpu()
                             total_params += tensor.nelement()
                             compressed_params += tensor.nelement()
            except Exception as e:
                print(f"Error reading {full_shard_path}: {e}")

        # 3. Compress Batch using GPU Parallelism (Batched Optimization)
        # Group by shape
        from collections import defaultdict
        grouped_layers = defaultdict(list)
        for key, tensor in layers_to_process:
            grouped_layers[tensor.shape].append((key, tensor))
            
        print(f"Grouped {len(layers_to_process)} layers into {len(grouped_layers)} shape groups.")
        
        for shape, items in grouped_layers.items():
            # items is list of (key, tensor)
            
            # Dynamic Batch Sizing
            # Calculate memory per layer (approx)
            # Tensor: H*W*4 bytes
            # Gradients/Optimizer: ~3-4x params
            # Monte Carlo overhead: Fixed small buffer
            H, W = shape
            bytes_per_layer = H * W * 4
            estimated_mem_per_layer = bytes_per_layer * 6 # Safety factor for optimizer states + gradients
            
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                target_mem = free_mem * 0.9 # Use up to 90% of free memory
                
                # Ensure we don't divide by zero or get 0 batch
                calc_batch_size = int(target_mem / (estimated_mem_per_layer + 1e-6))
                SUB_BATCH_SIZE = max(1, min(calc_batch_size, 64)) # Cap at 64 for stability
                print(f"  Shape {shape}: Free VRAM {free_mem/1e9:.2f}GB. Est. Layer Mem {estimated_mem_per_layer/1e6:.2f}MB. Dynamic Batch: {SUB_BATCH_SIZE}")
            else:
                SUB_BATCH_SIZE = 8
            
            for i in range(0, len(items), SUB_BATCH_SIZE):
                sub_items = items[i : i + SUB_BATCH_SIZE]
                keys = [k for k, t in sub_items]
                tensors = [t for k, t in sub_items]
                
                # Stack tensors: [B, H, W]
                batch_tensor = torch.stack(tensors)
                
                # Encode Batch
                # print(f"  Encoding batch of {len(tensors)} layers with shape {shape}...")
                dna_list = encoder.encode_batch(batch_tensor, num_transforms=16, iterations=200)
                
                # Store results
                for k, dna, t in zip(keys, dna_list, tensors):
                    fractal_brain[k] = dna.to_json()
                    total_params += t.nelement()
                    compressed_params += 16 * 7
                
                # Cleanup
                del batch_tensor
                del dna_list
                
        # Clear RAM
        del layers_to_process
        del grouped_layers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 4. Cleanup Batch
        print("Cleaning up batch files...")
        for fpath in batch_files:
            if os.path.exists(fpath):
                os.remove(fpath)
        
        # Update Progress & ETA
        processed_shards += len(batch_shards)
        elapsed = time.time() - start_time_total
        avg_time = elapsed / processed_shards
        remaining = len(unique_shards) - processed_shards
        eta_seconds = avg_time * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        print(f"Batch Complete. Progress: {processed_shards}/{len(unique_shards)}")
        print(f"Avg Time per Shard: {avg_time:.1f}s. Estimated Time Remaining: {eta_str}")

        import gc
        gc.collect()
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
