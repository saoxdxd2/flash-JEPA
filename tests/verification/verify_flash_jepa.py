import torch
import numpy as np
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome
from models.ecg import ModularBrain
from brain.modules.predictive_retina import PredictiveRetina

def test_flash_jepa(checkpoint_path=None):
    print(f"--- Testing Unified Flash-JEPA (Scale: {'16k' if checkpoint_path else 'Fresh'}) ---")
    
    # 1. Initialize Brain
    genome = Genome()
    brain = EvolutionaryBrain(genome=genome)
    
    if checkpoint_path:
        print(f"Loading scaled checkpoint: {checkpoint_path}")
        brain.load_model(checkpoint_path)
        print("Checkpoint loaded successfully via load_model().")
    
    # 2. Test Forward Pass (Decide)
    L = brain.latent_dim
    input_size = brain.input_size
    dummy_input = torch.randn(1, input_size)
    
    print(f"Running decide() with input size {input_size}...")
    action, logits = brain.decide(dummy_input)
    
    print(f"Action selected: {action}")
    print(f"Logits shape: {logits.shape}")
    print(f"Used System: {brain.last_used_system}")
    
    assert action >= 0 and action < brain.action_size
    assert logits.shape == (1, brain.action_size)
    
    # 3. Test Training Step
    print("\nFilling replay buffer for training test...")
    for _ in range(40):
        s = np.random.randn(input_size)
        a = np.random.randint(0, brain.action_size)
        r = np.random.randn()
        ns = np.random.randn(input_size)
        d = False
        brain.replay_buffer.add(s, a, r, ns, d)
        
    print(f"Running train_step()...")
    loss = brain.train_step(batch_size=32)
    print(f"Training loss: {loss:.4f}")
    print(f"Last Surprise (JEPA): {brain.last_surprise:.4f}")
    
    assert loss > 0
    assert brain.last_surprise >= 0
    
    print("\n--- Unified Flash-JEPA Verification PASSED ---")

if __name__ == "__main__":
    checkpoint_path = r"c:\Users\sao\Documents\model\models\saved\gen_350_scaled_16k.pt"
    try:
        test_flash_jepa(checkpoint_path)
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
