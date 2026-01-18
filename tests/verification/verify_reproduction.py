import sys
import os
import random
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.genome import Genome, BrainGene

def verify_reproduction():
    print("🧬 VERIFYING GENOME REPRODUCTION")
    print("================================")
    
    # 1. Create Parent 1 (Mom) - All Alleles = 0.0
    mom = Genome()
    for chrom in mom.chromosomes.values():
        for gene in chrom.genes.values():
            gene.allele_1 = 0.0
            gene.allele_2 = 0.0
            
    # 2. Create Parent 2 (Dad) - All Alleles = 1.0
    dad = Genome()
    for chrom in dad.chromosomes.values():
        for gene in chrom.genes.values():
            gene.allele_1 = 1.0
            gene.allele_2 = 1.0
            
    print(f"Mom: {mom.chromosomes['plasticity'].genes['bdnf'].allele_1}")
    print(f"Dad: {dad.chromosomes['plasticity'].genes['bdnf'].allele_1}")
    
    # 3. Crossover
    print("\n--- Crossing Over ---")
    child = mom.crossover(dad)
    
    # 4. Analyze Child
    # If correct sexual reproduction, child should have one 0.0 and one 1.0 for EVERY gene
    # (Since Mom gives 0.0 and Dad gives 1.0)
    
    # If cloning Mom, child has 0.0, 0.0
    # If cloning Dad, child has 1.0, 1.0 (Impossible in current logic)
    
    counts = {"Mom-Mom (0,0)": 0, "Mom-Dad (0,1)": 0, "Dad-Dad (1,1)": 0, "Other": 0}
    
    total_genes = 0
    for chrom_name, chrom in child.chromosomes.items():
        print(f"\nChromosome: {chrom_name}")
        for gene_name, gene in chrom.genes.items():
            a1 = gene.allele_1
            a2 = gene.allele_2
            pair = tuple(sorted((a1, a2)))
            
            if pair == (0.0, 0.0):
                counts["Mom-Mom (0,0)"] += 1
                # print(f"  {gene_name}: Mom-Mom")
            elif pair == (0.0, 1.0):
                counts["Mom-Dad (0,1)"] += 1
                # print(f"  {gene_name}: Mom-Dad")
            elif pair == (1.0, 1.0):
                counts["Dad-Dad (1,1)"] += 1
                # print(f"  {gene_name}: Dad-Dad")
            else:
                counts["Other"] += 1
                print(f"  {gene_name}: {pair}")
            
            total_genes += 1
            
    print("\n--- Results ---")
    print(f"Total Genes: {total_genes}")
    for k, v in counts.items():
        print(f"{k}: {v} ({v/total_genes*100:.1f}%)")
        
    # Check for bias
    if counts["Mom-Mom (0,0)"] > 0:
        print("\n❌ FAIL: Child has Mom-Mom clones! Crossover logic is biased.")
    elif counts["Dad-Dad (1,1)"] > 0:
        print("\n❌ FAIL: Child has Dad-Dad clones! (Unexpected)")
    elif counts["Mom-Dad (0,1)"] == total_genes:
        print("\n✅ PASS: Perfect Sexual Reproduction (Mom-Dad for all genes).")
    else:
        print("\n⚠️ WARNING: Mixed results.")

    # 5. Mutate
    print("\n--- Mutating Child ---")
    child.mutate(stress=1.0, fitness=0.0) # High stress = high mutation
    
    mutated_count = 0
    for chrom in child.chromosomes.values():
        for gene in chrom.genes.values():
            if gene.allele_1 != 0.0 and gene.allele_1 != 1.0:
                mutated_count += 1
            if gene.allele_2 != 0.0 and gene.allele_2 != 1.0:
                mutated_count += 1
                
    print(f"Mutated Alleles: {mutated_count}")
    if mutated_count > 0:
        print("✅ PASS: Mutation working.")
    else:
        print("❌ FAIL: No mutations occurred.")

if __name__ == "__main__":
    verify_reproduction()
