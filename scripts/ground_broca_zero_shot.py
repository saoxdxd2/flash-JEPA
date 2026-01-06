import torch
import numpy as np
import os
import sys
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Add root to path
sys.path.append(os.getcwd())

from brain.evolutionary_brain import EvolutionaryBrain

def ground_broca():
    print("=== Broca Zero-Shot Grounding ===")
    brain = EvolutionaryBrain()
    latest = brain.find_latest_checkpoint()
    if latest:
        print(f"Loading brain from {latest}...")
        brain.load_model(latest)
    else:
        print("No checkpoint found.")
        return

    model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return
    
    print("Locating Embedding Shard...")
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files(model_id)
        target_file = None
        param_name = "model.language_model.embed_tokens.weight"
        for f in files:
            if f.endswith(".safetensors") and "model-00001" in f:
                target_file = hf_hub_download(repo_id=model_id, filename=f)
                break
                
        if not target_file:
            print("Could not find embedding shard.")
            return

        print(f"Loading embeddings from {os.path.basename(target_file)}...")
        with safe_open(target_file, framework="pt", device="cpu") as f:
            embeddings = f.get_tensor(param_name).to(torch.float32)
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return

    # Deterministic Projection (Matches N2N2)
    source_dim = embeddings.shape[1]
    target_dim = brain.broca.embedding_dim
    print(f"Creating Projection {source_dim} -> {target_dim}...")
    g = torch.Generator()
    g.manual_seed(42)
    projection_matrix = torch.randn(source_dim, target_dim, generator=g) / np.sqrt(target_dim)

    print("Grounding ASCII characters...")
    count = 0
    brain.broca.eval() # Set to eval for grounding
    with torch.no_grad():
        for i in range(128):
            char = chr(i)
            # Find token ID for this character
            tokens = tokenizer.encode(char, add_special_tokens=False)
            if tokens:
                token_id = tokens[0]
                if token_id < embeddings.shape[0]:
                    qwen_vec = embeddings[token_id]
                    # Project to Concept Space
                    concept_vec = torch.matmul(qwen_vec, projection_matrix)
                    # Project to Signal Space (64) via Broca's own N2N2 projection
                    signal_vec = brain.broca.n2n2_projection(concept_vec.unsqueeze(0)).squeeze(0)
                    # Ground
                    brain.broca.ground_character(i, signal_vec)
                    count += 1
    
    print(f"Successfully grounded {count}/128 characters.")
    brain.save_model(latest)
    print(f"Brain saved with grounded Broca to {latest}")

if __name__ == "__main__":
    ground_broca()
