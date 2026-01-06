import torch
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from models.liquid_vectorized import VectorizedLiquidGraph

def perform_transplant(transplant_file="brain_transplant.pt"):
    print("--- BRAIN SURGERY: LOCAL GRAFTING ---")
    
    if not os.path.exists(transplant_file):
        print(f"Error: {transplant_file} not found. Please download it from Colab/Kaggle.")
        return
        
    print(f"Loading Transplant Package: {transplant_file}...")
    transplant_data = torch.load(transplant_file, map_location='cpu')
    
    print(f"Source Model: {transplant_data['meta']['model']}")
    print(f"Experts: {transplant_data['meta']['experts']}")
    
    # Load Brain
    brain = EvolutionaryBrain()
    latest = brain.find_latest_checkpoint("models/saved")
    if latest:
        brain.load_model(latest)
    else:
        print("Starting with fresh brain.")
        
    broca = brain.broca
    
    # --- UPGRADE: Resize Latent Dim to match Transplant ---
    target_latent_dim = transplant_data['meta'].get('hidden_dim', 4096)
    if brain.latent_dim != target_latent_dim:
        print(f"Upgrading Brain Latent Dim: {brain.latent_dim} -> {target_latent_dim}")
        brain.resize_latent(target_latent_dim)
    
    # Ensure Broca has enough experts
    required_experts = len(transplant_data['experts'])
    if len(broca.experts) < required_experts:
        print(f"Expanding Broca from {len(broca.experts)} to {required_experts} experts...")
        for i in range(len(broca.experts), required_experts):
            # Create new expert with same dimensions as first one (will be resized later)
            new_expert = VectorizedLiquidGraph(
                input_size=broca.experts[0].input_size,
                hidden_size=broca.experts[0].hidden_size,
                output_size=broca.experts[0].output_size
            )
            broca.experts.append(new_expert)
        broca.num_experts = required_experts
            
        # Update gate to handle more experts
        old_gate = broca.gate
        broca.gate = torch.nn.Linear(old_gate.in_features, required_experts)
        with torch.no_grad():
            # Initialize new gate weights randomly but preserve old ones
            nn.init.xavier_uniform_(broca.gate.weight)
            broca.gate.weight[:old_gate.out_features, :] = old_gate.weight
            broca.gate.bias[:old_gate.out_features] = old_gate.bias
        
    print("Grafting Weights...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brain.broca.to(device)
    # Move TRM and other modules if needed, but Broca is the focus here
    if hasattr(brain, 'trm'): brain.trm.to(device)
    
    for i, expert_data in enumerate(transplant_data['experts']):
        print(f"  > Grafting into Expert {i}...")
        expert = broca.experts[i]
        
        # 1. Resize Expert if needed
        target_hidden = expert_data['w_rec'].shape[0]
        target_input = expert_data['w_in'].shape[1]
        
        if expert.hidden_size != target_hidden:
            print(f"    Resizing Hidden: {expert.hidden_size} -> {target_hidden}")
            expert.resize_hidden(target_hidden)
            
        if expert.input_size != target_input:
            print(f"    Resizing Input: {expert.input_size} -> {target_input}")
            expert.resize_input(target_input)
        
        # 2. Graft W_in
        w_in_graft = expert_data['w_in'].to(device)
        with torch.no_grad():
            expert.w_in.data += w_in_graft * 0.1 # Blend factor
            
        # 3. Graft W_rec
        w_rec_graft = expert_data['w_rec'].to(device)
        with torch.no_grad():
            expert.w_rec.data += w_rec_graft * 0.1
            
    # 4. Graft Embeddings (Lexical Grounding)
    if "embeddings" in transplant_data:
        print("Grafting Lexical Embeddings (Dictionary Transplant)...")
        # Use register_buffer to ensure it's saved in the state_dict
        brain.broca.register_buffer('lexical_knowledge', transplant_data["embeddings"].to(device))
        print(f"  > {brain.broca.lexical_knowledge.shape[0]} concepts grounded.")
            
    # Enable Sequential Mode for the Transplanted Brain
    print("Enabling Sequential Mode (Deep Layer Processing)...")
    broca.sequential_mode = True
            
    # Reset State to clear transient buffers (x, y) before saving
    print("Cleaning up transient states...")
    brain.broca.reset_state()
    if hasattr(brain, 'trm') and hasattr(brain.trm, 'reset_state'):
        brain.trm.reset_state()
            
    # Save
    save_path = "models/saved/gen_350_transplanted.pt"
    brain.save_model(save_path)
    print(f"Surgery Complete. Brain saved to {save_path}")

if __name__ == "__main__":
    perform_transplant()
