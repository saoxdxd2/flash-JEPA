import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.genome import Genome
from models.ecg import ModularBrain

def test_meta_genome():
    print("Testing Meta-Genome Implementation...")
    
    # 1. Initialize Genome
    genome = Genome()
    print("Genome Initialized.")
    print(f"Plasticity Genes: H={genome.plasticity_hidden_size}, L={genome.plasticity_layers}, Act={genome.plasticity_activation}")
    
    # 2. Initialize ModularBrain with Genome
    input_size = 10
    hidden_size = 100
    output_size = 5
    brain = ModularBrain(input_size, hidden_size, output_size, genome=genome)
    print("ModularBrain Initialized with Genome.")
    
    # 3. Verify Visual Cortex Plasticity Net
    v_net = brain.visual_cortex.plasticity_net
    print(f"Visual Cortex Plasticity Net: {v_net}")
    assert v_net.hidden_size == genome.plasticity_hidden_size
    assert v_net.num_layers == genome.plasticity_layers
    assert v_net.activation_type == genome.plasticity_activation
    
    # 4. Verify Motor Cortex Plasticity Net
    m_net = brain.motor_cortex.plasticity_net
    print(f"Motor Cortex Plasticity Net: {m_net}")
    assert m_net.hidden_size == genome.plasticity_hidden_size
    assert m_net.num_layers == genome.plasticity_layers
    assert m_net.activation_type == genome.plasticity_activation
    
    # 5. Test Forward Pass of Plasticity Net
    pre = torch.randn(5)
    post = torch.randn(5)
    weight = torch.randn(5)
    reward = torch.randn(5)
    delta_w = v_net(pre, post, weight, reward)
    print(f"Plasticity Net Output Shape: {delta_w.shape}")
    assert delta_w.shape == (5,)
    
    print("Meta-Genome Verification Passed!")

if __name__ == "__main__":
    test_meta_genome()
