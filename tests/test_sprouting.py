import torch
import torch.nn as nn
from brain.modules.broca import BrocaModule
from brain.genome import Genome

def test_sprouting():
    print("Starting Broca Sprouting Test...")
    
    # 1. Setup Genome with low sprouting threshold for testing
    genome = Genome()
    genome.sprouting_threshold = 0.05 # Very low to trigger easily
    genome.max_experts = 4
    
    # 2. Initialize Broca with 1 expert
    broca = BrocaModule(num_experts=1, genome=genome)
    print(f"Initial experts: {broca.num_experts}")
    
    # 3. Feed "Surprising" data
    # We feed random vectors that TitansMemory won't be able to predict well
    device = torch.device("cpu")
    broca.to(device)
    
    for i in range(150): # Need at least 50 steps to trigger check
        # Random embedding [1, 256]
        surprise_data = torch.randn(1, 256)
        output, surprise = broca.process_text_embedding(surprise_data)
        
        if (i + 1) % 50 == 0:
            print(f"Step {i+1}: Experts = {broca.num_experts}, Avg Surprise = {surprise:.4f}")
            
    # 4. Final Check
    print(f"Final experts: {broca.num_experts}")
    if broca.num_experts > 1:
        print("SUCCESS: Broca sprouted new experts!")
    else:
        print("FAILURE: Broca did not sprout.")

if __name__ == "__main__":
    test_sprouting()
