import torch
import sys

def inspect_checkpoint(path):
    print(f"--- Inspecting Checkpoint: {path} ---")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint Keys: {list(checkpoint.keys())}")
            
            if 'trm_state' in checkpoint:
                print("\n--- TRM State Keys ---")
                trm_sd = checkpoint['trm_state']
                for k in trm_sd.keys():
                    print(f"  {k}: {trm_sd[k].shape if torch.is_tensor(trm_sd[k]) else type(trm_sd[k])}")
            
            if 'genome' in checkpoint:
                print("\n--- Genome Sample ---")
                g = checkpoint['genome']
                if isinstance(g, dict):
                    for k, v in list(g.items())[:10]:
                        print(f"  {k}: {v}")
                else:
                    print(f"  Genome is {type(g)}")
        else:
            print(f"Checkpoint is of type: {type(checkpoint)}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    path = r"c:\Users\sao\Documents\model\models\saved\gen_350_scaled_16k.pt"
    inspect_checkpoint(path)
