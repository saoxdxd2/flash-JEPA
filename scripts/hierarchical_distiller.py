import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
from safetensors import safe_open
from huggingface_hub import snapshot_download

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from models.neuromodulated_holographic import SparseLinear

class HierarchicalDistiller:
    """
    Distills knowledge from Qwen-3 235B into the Flash-JEPA hierarchy.
    Maps 80 layers to Reflex/Concept/Strategy levels.
    """
    def __init__(self, brain):
        self.brain = brain
        self.model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
        self.hierarchy_map = {
            'reflex': range(0, 20),
            'concept': range(20, 60),
            'strategy': range(60, 80)
        }
        
    def setup_teacher(self):
        print(f"Distiller: Fetching index for {self.model_id}...")
        try:
            index_path = snapshot_download(repo_id=self.model_id, allow_patterns=["*.index.json"])
            index_file = None
            for root, dirs, files in os.walk(index_path):
                for file in files:
                    if file.endswith(".index.json"):
                        index_file = os.path.join(root, file)
                        break
            
            with open(index_file, 'r') as f:
                self.index_data = json.load(f)
            return True
        except Exception as e:
            print(f"Distiller: Setup failed: {e}")
            return False

    def distill_layer_block(self, level_name):
        """
        Distills a block of Qwen-3 layers into a specific hierarchical level.
        """
        layers = self.hierarchy_map[level_name]
        print(f"Distiller: Distilling layers {layers.start}-{layers.stop-1} into {level_name}...")
        
        # Target Sparse Layers
        if level_name == 'reflex':
            target_w = self.brain.trm.visual_cortex.W_reflex
            target_r = self.brain.trm.visual_cortex.R_reflex
        elif level_name == 'concept':
            target_w = self.brain.trm.visual_cortex.W_concept
            target_r = self.brain.trm.visual_cortex.R_concept
        else:
            target_w = self.brain.trm.visual_cortex.W_strategy
            target_r = self.brain.trm.visual_cortex.R_strategy
            
        # 1. Harvest and Project Weights
        # We sample the middle layer of the block for representative weights
        mid_layer = layers.start + len(layers) // 2
        down_key = f"model.layers.{mid_layer}.mlp.down_proj.weight"
        
        if down_key not in self.index_data["weight_map"]:
            # Try alternative keys
            down_key = f"model.layers.{mid_layer}.mlp.shared_expert.down_proj.weight"
            
        if down_key in self.index_data["weight_map"]:
            shard_file = self.index_data["weight_map"][down_key]
            shard_path = snapshot_download(repo_id=self.model_id, allow_patterns=[shard_file])
            
            full_shard_path = None
            for root, dirs, files in os.walk(shard_path):
                if shard_file in files:
                    full_shard_path = os.path.join(root, shard_file)
                    break
            
            if full_shard_path:
                with safe_open(full_shard_path, framework="pt", device="cpu") as f:
                    w_teacher = f.get_tensor(down_key)
                    print(f"  > Teacher Weight: {w_teacher.shape}")
                    
                    # 2. Sparse Imprinting
                    # We take the top-k most significant weights and map them to our sparse indices
                    self._imprint_sparse(target_w, w_teacher)
                    print(f"  > Imprinted into {level_name} SparseLinear.")
        
    def _imprint_sparse(self, sparse_layer, teacher_weight):
        """
        Imprints significant teacher weights into a sparse layer.
        """
        with torch.no_grad():
            # Project teacher weight to match sparse dimensions if needed
            # For simplicity, we'll use random projection alignment
            in_dim, out_dim = sparse_layer.in_features, sparse_layer.out_features
            
            # Sample significant weights (top 10%)
            t_flat = teacher_weight.flatten()
            threshold = torch.quantile(t_flat.abs(), 0.9)
            mask = t_flat.abs() > threshold
            
            # Map to sparse values
            num_to_copy = min(mask.sum().item(), sparse_layer.values.shape[0])
            sparse_layer.values.data[:num_to_copy] = t_flat[mask][:num_to_copy] * 0.1 # Blend factor

    def run_distillation(self):
        if not self.setup_teacher(): return
        
        for level in ['reflex', 'concept', 'strategy']:
            self.distill_layer_block(level)
            
        # Save distilled brain
        save_path = "models/saved/gen_350_distilled.pt"
        self.brain.save_model(save_path)
        print(f"Distiller: Distillation complete. Saved to {save_path}")

if __name__ == "__main__":
    brain = EvolutionaryBrain()
    latest = brain.find_latest_checkpoint("models/saved")
    if latest:
        brain.load_model(latest)
        
    distiller = HierarchicalDistiller(brain)
    distiller.run_distillation()
