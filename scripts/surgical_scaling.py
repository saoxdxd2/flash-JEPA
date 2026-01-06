import torch
import torch.nn as nn
import numpy as np
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def surgical_scale(checkpoint_path, output_path, target_hidden=16384, target_latent=1024, target_fovea=224):
    print(f"--- Surgical Scaling: {checkpoint_path} -> {output_path} ---")
    
    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    old_genome_dict = checkpoint['genome']
    
    # 2. Create Target Genome & Brain
    new_genome = Genome()
    # Update with old genome values first
    for k, v in old_genome_dict.items():
        if hasattr(new_genome, k):
            setattr(new_genome, k, v)
            
    # Apply Scaling Targets
    new_genome.hidden_size = target_hidden
    new_genome.latent_dim = target_latent
    new_genome.PERIPHERAL_RESOLUTION = target_fovea # Use for both peripheral and fovea targets
    
    print(f"Target Scale: Hidden={target_hidden}, Latent={target_latent}, Fovea={target_fovea}")
    
    new_brain = EvolutionaryBrain(genome=new_genome)
    
    # 3. Migrate Weights
    print("\nMigrating Weights...")
    
    # A. TRM (Cortex)
    trm_sd = checkpoint['trm_state']
    new_trm_sd = new_brain.trm.state_dict()
    
    migrated_count = 0
    for k, v in trm_sd.items():
        if k in new_trm_sd:
            old_v = trm_sd[k]
            new_v = new_trm_sd[k]
            
            if old_v.shape == new_v.shape:
                new_trm_sd[k] = old_v
                migrated_count += 1
            elif "indices" in k:
                # Sparse indices: They are valid as long as they are within the new bounds
                # We can just copy them.
                new_trm_sd[k][:, :old_v.shape[1]] = old_v
                migrated_count += 1
            elif "values" in k:
                # Sparse values: Copy to the beginning
                new_trm_sd[k][:old_v.shape[0]] = old_v
                migrated_count += 1
            elif "bias" in k:
                # Bias: Copy to the beginning
                min_size = min(old_v.shape[0], new_v.shape[0])
                new_trm_sd[k][:min_size] = old_v[:min_size]
                migrated_count += 1
            else:
                # Dense weights (e.g., meta_controller, decoder)
                # Resize and copy
                if old_v.dim() == 2:
                    min_0 = min(old_v.shape[0], new_v.shape[0])
                    min_1 = min(old_v.shape[1], new_v.shape[1])
                    new_trm_sd[k][:min_0, :min_1] = old_v[:min_0, :min_1]
                    migrated_count += 1
                elif old_v.dim() == 1:
                    min_0 = min(old_v.shape[0], new_v.shape[0])
                    new_trm_sd[k][:min_0] = old_v[:min_0]
                    migrated_count += 1

    new_brain.trm.load_state_dict(new_trm_sd)
    print(f"Migrated {migrated_count} TRM parameters.")
    
    # B. Broca
    broca_sd = checkpoint['broca_state']
    new_broca_sd = new_brain.broca.state_dict()
    for k, v in broca_sd.items():
        if k in new_broca_sd:
            old_v = broca_sd[k]
            new_v = new_broca_sd[k]
            if old_v.shape == new_v.shape:
                new_broca_sd[k] = old_v
            else:
                # Resize embedding or linear
                if old_v.dim() == 2:
                    min_0 = min(old_v.shape[0], new_v.shape[0])
                    min_1 = min(old_v.shape[1], new_v.shape[1])
                    new_broca_sd[k][:min_0, :min_1] = old_v[:min_0, :min_1]
    new_brain.broca.load_state_dict(new_broca_sd)
    print("Migrated Broca parameters.")
    
    # C. Retina
    # We don't migrate retina weights because the architecture changed to FastViT.
    # It will use pretrained weights.
    print("Retina: Using fresh FastViT weights (Architecture mismatch with checkpoint).")
    
    # 4. Save Scaled Brain
    save_dict = {
        'version': '3.0-scaled',
        'genome': new_brain.genome.__dict__,
        'trm_state': new_brain.trm.state_dict(),
        'retina_state': new_brain.retina.state_dict(),
        'broca_state': new_brain.broca.state_dict(),
        'latent_dim': new_brain.latent_dim,
        'hidden_size': new_brain.hidden_size
    }
    torch.save(save_dict, output_path)
    print(f"\nSuccessfully saved SCALED brain to {output_path}")

if __name__ == "__main__":
    checkpoint_path = r"c:\Users\sao\Documents\model\models\saved\gen_350_transplanted.pt"
    output_path = r"c:\Users\sao\Documents\model\models\saved\gen_350_scaled_16k.pt"
    surgical_scale(checkpoint_path, output_path)
