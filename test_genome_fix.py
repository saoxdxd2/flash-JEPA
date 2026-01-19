import sys
import os
sys.path.append(os.getcwd())

from brain.genome import Genome
import pickle

def test_genome():
    print("Creating Genome...")
    g = Genome()
    print("Genome created.")
    print(f"Repr: {g}")
    
    # Simulate unpickling old state (remove _gene_lookup)
    state = g.__dict__.copy()
    del state['_gene_lookup']
    
    print("Simulating old state unpickling...")
    g2 = Genome.__new__(Genome)
    g2.__setstate__(state)
    
    print("Checking _gene_lookup...")
    if hasattr(g2, '_gene_lookup'):
        print("PASS: _gene_lookup restored.")
    else:
        print("FAIL: _gene_lookup missing.")
        
    print(f"Repr after restore: {g2}")

if __name__ == "__main__":
    test_genome()
