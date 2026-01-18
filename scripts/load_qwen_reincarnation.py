"""
Qwen Reincarnation Loader

Loads the gen_0_qwen_reincarnation.pt created by colab.ipynb
and converts it to a working Flash-JEPA EvolutionaryBrain.

This module handles the knowledge transfer from a pretrained Qwen model
to the Flash-JEPA architecture, mapping:
- Embeddings → Broca's semantic memory
- Transformer layers → TRM liquid neurons
- LM head → action decoder
"""
import torch
import os
import sys
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome


# === CONFIGURATION CONSTANTS ===
# These control the knowledge transfer from Qwen to Flash-JEPA

# Maximum number of embedding vectors to transfer to Broca's semantic memory
# (Note: Flash-JEPA uses continuous latent representations, not discrete tokens)
MAX_EMBEDDING_VECTORS = 10000

# Default hidden size when not specified in checkpoint
DEFAULT_HIDDEN_SIZE = 4096

# Maximum hidden size for CPU-efficient operation
MAX_CPU_HIDDEN_SIZE = 4096

# Maximum latent dimension for CPU efficiency
MAX_CPU_LATENT_DIM = 1024

# Latent divisor for computing latent_dim from hidden_size
LATENT_DIVISOR = 4

# Projection normalization factor (1/sqrt(n) initialization)
PROJECTION_NORM_FACTOR = 0.5

# Layer logging interval for progress updates
LAYER_LOG_INTERVAL = 10


def load_qwen_reincarnation(
    reincarnation_path: str, 
    output_path: Optional[str] = None
) -> EvolutionaryBrain:
    """
    Loads a Qwen reincarnation checkpoint and creates a Flash-JEPA brain.
    
    Args:
        reincarnation_path: Path to gen_0_qwen_reincarnation.pt
        output_path: Where to save the converted brain (optional)
        
    Returns:
        EvolutionaryBrain initialized with Qwen's knowledge
    """
    # Validate input path exists
    if not os.path.exists(reincarnation_path):
        raise FileNotFoundError(
            f"Reincarnation checkpoint not found: {reincarnation_path}"
        )
    
    print(f"Loading Qwen reincarnation from {reincarnation_path}...")
    
    # Load the reincarnation checkpoint
    try:
        qwen_state: Dict[str, Any] = torch.load(reincarnation_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e
    
    version = qwen_state.get('version', 'unknown')
    config = qwen_state.get('config', {})
    
    print(f"Reincarnation version: {version}")
    print(f"Config: {config}")
    
    # Extract architecture parameters with sensible defaults
    vocab_size = config.get('vocab_size', 151936)  # Qwen-2.5 vocab size
    hidden_size = config.get('hidden_size', DEFAULT_HIDDEN_SIZE)
    num_layers = qwen_state.get('num_layers', 94)
    
    # Create genome based on Qwen's architecture
    genome = Genome()
    # Scale genome to match Qwen's capacity (capped for CPU efficiency)
    genome.hidden_size = min(hidden_size, MAX_CPU_HIDDEN_SIZE)
    genome.latent_dim = min(hidden_size // LATENT_DIVISOR, MAX_CPU_LATENT_DIM)
    
    print(f"Creating brain with hidden={genome.hidden_size}, latent={genome.latent_dim}")
    
    # Create brain
    brain = EvolutionaryBrain(genome)
    
    # === INJECT QWEN KNOWLEDGE INTO NEURONS ===
    
    # 1. Embeddings → Broca's semantic memory
    if qwen_state.get('embeddings') is not None:
        embeddings = qwen_state['embeddings']
        print(f"Injecting embeddings {embeddings.shape} into Broca...")
        
        # Project to Broca's dimension if needed
        # Limit embedding vectors to transfer for efficiency
        embeddings_to_transfer = embeddings[:MAX_EMBEDDING_VECTORS]
        
        if embeddings.shape[1] > brain.broca.embedding_dim:
            # Create projection matrix with proper initialization
            proj = torch.randn(embeddings.shape[1], brain.broca.embedding_dim)
            proj = proj / (embeddings.shape[1] ** PROJECTION_NORM_FACTOR)  # Xavier-like normalization
            projected = torch.mm(embeddings_to_transfer, proj)
            brain.broca.seed_knowledge(projected)
        else:
            brain.broca.seed_knowledge(embeddings_to_transfer)
    
    # 2. Layers → TRM liquid neurons
    layers = qwen_state.get('layers', [])
    print(f"Injecting {len(layers)} transformer layers into TRM...")
    
    trm_state = brain.trm.state_dict()
    
    for i, layer in enumerate(layers):
        if layer is None:
            continue
            
        # Map Qwen layer to Flash-JEPA components
        # W_input → visual_cortex (perception)
        # W_recurrent → motor_cortex (action)
        # tau → time constants
        
        if layer.get('W_input') is not None:
            # Project to TRM's input size
            w = layer['W_input']
            target_key = None
            
            # Find matching parameter in TRM
            for key in trm_state:
                if 'W_reflex' in key and 'values' in key:
                    target_key = key
                    break
            
            if target_key and target_key in trm_state:
                target_shape = trm_state[target_key].shape
                # Reshape/project to match
                if w.numel() >= target_shape.numel():
                    trm_state[target_key] = w.flatten()[:target_shape.numel()].reshape(target_shape)
        
        if layer.get('tau') is not None:
            # Inject time constants
            tau = layer['tau']
            for key in trm_state:
                if 'tau_' in key:
                    target_shape = trm_state[key].shape
                    if tau.numel() >= target_shape.numel():
                        trm_state[key] = tau.flatten()[:target_shape.numel()].reshape(target_shape)
                    break
        
        if i % LAYER_LOG_INTERVAL == 0:
            print(f"  Processed layer {i}/{len(layers)}")
    
    # Load modified state
    brain.trm.load_state_dict(trm_state, strict=False)
    
    # 3. LM Head → action decoder
    if qwen_state.get('lm_head') is not None:
        lm_head = qwen_state['lm_head']
        print(f"Injecting LM head {lm_head.shape} into action decoder...")
        
        # The LM head maps hidden → vocab
        # We need hidden → actions
        # Use first action_size rows
        
        decoder_state = brain.trm.core.decoder.state_dict()
        for key in decoder_state:
            if 'weight' in key:
                target_shape = decoder_state[key].shape
                # Sample from LM head
                if lm_head.shape[0] >= target_shape[0] and lm_head.shape[1] >= target_shape[1]:
                    decoder_state[key] = lm_head[:target_shape[0], :target_shape[1]]
                break
        
        brain.trm.core.decoder.load_state_dict(decoder_state, strict=False)
    
    print("Qwen knowledge injection complete!")
    
    # Save if requested
    if output_path:
        brain.save_model(output_path)
        print(f"Saved reincarnated brain to {output_path}")
    
    return brain


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load Qwen Reincarnation")
    parser.add_argument('--input', type=str, default='models/saved/gen_0_qwen_reincarnation.pt',
                        help='Path to reincarnation checkpoint')
    parser.add_argument('--output', type=str, default='models/saved/gen_0_flash_jepa.pt',
                        help='Path to save converted brain')
    
    args = parser.parse_args()
    
    brain = load_qwen_reincarnation(args.input, args.output)
    
    # Quick test
    print("\nTesting reincarnated brain...")
    import torch
    test_input = torch.randn(1, brain.input_size)
    brain.trm.eval()
    with torch.no_grad():
        output = brain.trm(test_input)
    print(f"Output shape: {output[0].shape}")
    print("Brain is alive! Ready for autonomous_life.py")


if __name__ == "__main__":
    main()
