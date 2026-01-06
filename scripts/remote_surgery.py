# REMOTE SURGERY SCRIPT (Run on Google Colab / Kaggle)
# =====================================================
# MAX-RESOLUTION "GOD-TIER" TRANSPLANT
# =====================================================
# Output: brain_transplant.pt (~1.0 GB)
# =====================================================

import os
import sys
import json
import torch
import numpy as np
import gc
import psutil
import shutil

# 1. Install Dependencies
print("Installing dependencies...")
os.system("pip install -q torch safetensors huggingface_hub psutil")

from huggingface_hub import snapshot_download
from safetensors import safe_open

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

TARGET_DIM = 64        # Liquid Input Dim (Matches Char Embedding)
HIDDEN_DIM = 4096      # MAX RESOLUTION (Upgraded from 2048)
NUM_EXPERTS = 32       # MAX DENSITY (Upgraded from 16)
SEED = 42              # CRITICAL: Must match local seed

# Hardware Selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

print(f"Target Model: {MODEL_ID}")
print(f"Compression: -> {TARGET_DIM}x{HIDDEN_DIM} (Experts: {NUM_EXPERTS})")

def check_disk_space():
    total, used, free = shutil.disk_usage("/")
    print(f"Disk Free: {free // (2**30)} GB")
    return free

def get_projection_matrix(source_dim, target_dim, device=DEVICE):
    """Deterministic Random Projection."""
    g = torch.Generator(device=device)
    g.manual_seed(SEED)
    return torch.randn(source_dim, target_dim, generator=g, device=device) / np.sqrt(target_dim)

def perform_surgery():
    print("\n--- STARTING MAX-RESOLUTION REMOTE SURGERY ---")
    
    # 1. Download Index
    print("Fetching Model Index...")
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    
    index_dir = "model_index_cache"
    index_file = None
    
    if os.path.exists(index_dir):
        print("    > Using cached index.")
        for root, dirs, files in os.walk(index_dir):
            for file in files:
                if file.endswith(".index.json"):
                    index_file = os.path.join(root, file)
                    break
    
    if not index_file:
        try:
            index_path = snapshot_download(repo_id=MODEL_ID, allow_patterns=["*.index.json"], local_dir=index_dir, local_dir_use_symlinks=False)
            for root, dirs, files in os.walk(index_path):
                for file in files:
                    if file.endswith(".index.json"):
                        index_file = os.path.join(root, file)
                        break
        except Exception as e:
            print(f"Failed to fetch index: {e}")
            return

    if not index_file:
        print("Error: No index file found.")
        return
            
    with open(index_file, 'r') as f:
        index_data = json.load(f)

    # 2. Identify Layers
    num_experts = NUM_EXPERTS
    total_layers = 80
    layers_per_expert = total_layers // num_experts # 80 / 32 = 2.5 (We'll handle rounding)
    
    # RESUME LOGIC
    output_file = "brain_transplant.pt"
    if os.path.exists(output_file):
        print(f"Found existing transplant package: {output_file}. Resuming...")
        transplant_data = torch.load(output_file)
        start_expert = len(transplant_data.get("experts", []))
        print(f"    > Resuming from Expert {start_expert}")
    else:
        transplant_data = {
            "meta": {
                "model": MODEL_ID,
                "experts": num_experts,
                "source_layers": total_layers,
                "hidden_dim": HIDDEN_DIM,
                "target_dim": TARGET_DIM
            },
            "experts": []
        }
        start_expert = 0
    
    for expert_idx in range(start_expert, num_experts):
        l_start = int(expert_idx * (total_layers / num_experts))
        l_end = int((expert_idx + 1) * (total_layers / num_experts)) - 1
        print(f"\nProcessing Expert {expert_idx} (Layers {l_start} - {l_end})...")
        
        expert_weights = {
            "w_in": torch.zeros(HIDDEN_DIM, TARGET_DIM),
            "w_rec": torch.zeros(HIDDEN_DIM, HIDDEN_DIM)
        }
        
        # We'll sample the start and end layer of this block
        target_layers = [l_start, l_end]
        
        for layer_idx in target_layers:
            print(f"  > Harvesting Layer {layer_idx}...")
            
            # Dynamic Key Detection
            candidates = [
                f"model.layers.{layer_idx}.mlp.down_proj.weight",
                f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
                f"model.layers.{layer_idx}.mlp.experts.0.down_proj.weight",
                f"model.layers.{layer_idx}.block_sparse_moe.experts.0.down_proj.weight",
                f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj",
                f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight"
            ]
            
            target_key = None
            key_type = "dense"
            for c in candidates:
                if c in index_data["weight_map"]:
                    target_key = c
                    if "experts" in c and "0" not in c: key_type = "moe_merged"
                    break
            
            if not target_key:
                print(f"    Warning: Could not find down_proj for layer {layer_idx}.")
                continue
                
            shard_file = index_data["weight_map"][target_key]
            print(f"    > Located {key_type} in shard: {shard_file}")
            
            # Download Shard to a SPECIFIC local directory
            temp_shard_dir = f"temp_shard_{layer_idx}"
            if os.path.exists(temp_shard_dir):
                shutil.rmtree(temp_shard_dir)
            os.makedirs(temp_shard_dir, exist_ok=True)
            
            try:
                print(f"    > Downloading shard to {temp_shard_dir}...")
                shard_path = snapshot_download(
                    repo_id=MODEL_ID, 
                    allow_patterns=[shard_file], 
                    local_dir=temp_shard_dir,
                    local_dir_use_symlinks=False
                )
                full_shard_path = os.path.join(shard_path, shard_file)
                
                # Load Weights
                with safe_open(full_shard_path, framework="pt", device="cpu") as f:
                    if ".weight" in target_key:
                        prefix = target_key.replace(".down_proj.weight", "")
                        down_key = f"{prefix}.down_proj.weight"
                        up_key = f"{prefix}.up_proj.weight"
                        if up_key not in f.keys() and f"{prefix}.gate_up_proj.weight" in f.keys():
                            up_key = f"{prefix}.gate_up_proj.weight"
                    else:
                        prefix = target_key.replace(".down_proj", "")
                        down_key = f"{prefix}.down_proj"
                        up_key = f"{prefix}.gate_up_proj"
                        if up_key not in f.keys():
                             up_key = f"{prefix}.up_proj"
                    
                    if down_key not in f.keys() or up_key not in f.keys():
                        print(f"    Missing keys in shard. Found: {list(f.keys())[:5]}")
                        continue

                    # --- SAFE LOADING & PROJECTION ---
                    def get_free_gpu():
                        if torch.cuda.is_available():
                            t = torch.cuda.get_device_properties(0).total_memory
                            a = torch.cuda.memory_allocated(0)
                            return t - a
                        return 0

                    def safe_project_tensor(tensor_key, f_handle, p_in_dim, p_out_dim, name="Tensor"):
                        mem = psutil.virtual_memory()
                        if mem.percent > 90.0:
                            gc.collect()
                        
                        t_cpu = f_handle.get_tensor(tensor_key)
                        if t_cpu.dim() == 3:
                            t_cpu = t_cpu.mean(dim=0)
                            
                        tensor_size = t_cpu.element_size() * t_cpu.nelement()
                        gpu_free = get_free_gpu()
                        
                        use_gpu = torch.cuda.is_available() and (tensor_size * 2 < gpu_free * 0.9)
                        device = DEVICE if use_gpu else "cpu"
                        print(f"    > Processing {name} on {device}...")
                        
                        t_proc = t_cpu.to(device, dtype=torch.float32)
                        del t_cpu
                        
                        # P_left: [Target, In]
                        # P_right: [Out, Target]
                        p_left = get_projection_matrix(p_in_dim, t_proc.shape[0], device=device)
                        p_right = get_projection_matrix(t_proc.shape[1], p_out_dim, device=device)
                            
                        res = torch.matmul(torch.matmul(p_left, t_proc), p_right)
                        del t_proc, p_left, p_right
                        if use_gpu: torch.cuda.empty_cache()
                        return res.cpu()

                    expert_weights["w_rec"] += safe_project_tensor(down_key, f, HIDDEN_DIM, HIDDEN_DIM, "Down")
                    expert_weights["w_in"] += safe_project_tensor(up_key, f, HIDDEN_DIM, TARGET_DIM, "Up")
                    print(f"    > Extracted & Compressed.")
                    
                # CLEANUP SHARD
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                print(f"    > Deleting {temp_shard_dir}...")
                shutil.rmtree(temp_shard_dir)
                
            except Exception as e:
                print(f"    > Error processing layer: {e}")
                if os.path.exists(temp_shard_dir):
                    shutil.rmtree(temp_shard_dir)
                    
            check_disk_space()
            
        # SAVE PROGRESS AFTER EVERY EXPERT
        transplant_data["experts"].append(expert_weights)
        torch.save(transplant_data, output_file)
        print(f"--- PROGRESS SAVED: Expert {expert_idx} complete ---")

    # 4. Extract Embeddings (Lexical Grounding)
    print("\n--- HARVESTING LEXICAL EMBEDDINGS ---")
    embed_key = "model.embed_tokens.weight"
    if "model.language_model.embed_tokens.weight" in index_data["weight_map"]:
        embed_key = "model.language_model.embed_tokens.weight"
        
    if embed_key in index_data["weight_map"]:
        shard_file = index_data["weight_map"][embed_key]
        print(f"    > Located Embeddings in shard: {shard_file}")
        
        temp_dir = "temp_embed"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            shard_path = snapshot_download(repo_id=MODEL_ID, allow_patterns=[shard_file], local_dir=temp_dir, local_dir_use_symlinks=False)
            full_shard_path = os.path.join(shard_path, shard_file)
            
            with safe_open(full_shard_path, framework="pt", device="cpu") as f:
                embed_weight = f.get_tensor(embed_key) # [Vocab, Dim]
                print(f"    > Source Embeddings: {embed_weight.shape}")
                
                # Project Vocab -> Latent Space (256)
                # We want to keep the Vocab size but squash the Dim
                # P: [SourceDim, 256]
                p = get_projection_matrix(embed_weight.shape[1], 256, device="cpu")
                projected_embeds = torch.matmul(embed_weight.to(torch.float32), p)
                
                transplant_data["embeddings"] = projected_embeds
                print(f"    > Projected Embeddings: {projected_embeds.shape}")
                
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"    > Error harvesting embeddings: {e}")
    else:
        print("    > Warning: Could not find embedding key.")

    # 5. Finalize
    print("\nFINALIZING SURGERY...")
    torch.save(transplant_data, output_file)
    print(f"SUCCESS: {output_file} ({os.path.getsize(output_file) / 1024 / 1024:.2f} MB) is ready.")
    print("Please download this file and run 'perform_transplant.py' locally.")

if __name__ == "__main__":
    perform_surgery()
