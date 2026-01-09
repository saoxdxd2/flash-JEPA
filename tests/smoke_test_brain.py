import torch
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def test_brain_init():
    print("Testing Brain Initialization...")
    genome = Genome()
    brain = EvolutionaryBrain(genome=genome)
    print(f"Brain initialized with latent_dim={brain.latent_dim}, hidden_size={brain.hidden_size}")
    assert brain.latent_dim == genome.latent_dim
    assert brain.hidden_size == genome.hidden_size
    print("Initialization test passed!")

def test_brain_decide():
    print("Testing Brain Decision Step...")
    genome = Genome()
    brain = EvolutionaryBrain(genome=genome)
    
    # Create a dummy input tensor
    # input_size = (3 * latent_dim) + NON_VISUAL_INPUT_SIZE
    input_size = (3 * genome.latent_dim) + genome.NON_VISUAL_INPUT_SIZE
    dummy_input = torch.randn(1, input_size)
    
    action, logits = brain.decide(dummy_input)
    print(f"Brain decided on action: {action}")
    assert isinstance(action, int)
    assert logits.shape == (1, brain.action_size)
    print("Decision test passed!")

if __name__ == "__main__":
    try:
        test_brain_init()
        test_brain_decide()
        print("\nAll smoke tests passed successfully!")
    except Exception as e:
        print(f"\nSmoke tests failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
