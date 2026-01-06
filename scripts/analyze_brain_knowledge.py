import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain

def analyze_knowledge():
    print("--- Brain Knowledge Analysis ---")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    elite_path = os.path.join(project_root, "models", "saved", "gen_349_elite.pkl")
    
    if not os.path.exists(elite_path):
        print(f"Error: Model not found at {elite_path}")
        return

    # 1. Load Brain
    print(f"Loading Brain from {elite_path}...")
    brain = EvolutionaryBrain()
    brain.load_model(elite_path)
    
    broca = brain.broca
    print("\n[Broca Area Analysis]")
    
    # 2. Analyze Experts (Liquid Neural Networks)
    # These hold the "concepts" (Text/Agentic knowledge)
    print(f"  Total Experts: {len(broca.experts)}")
    print(f"  Active Experts per Token: {broca.active_experts}")
    
    expert_norms = []
    for i, expert in enumerate(broca.experts):
        # We check the recurrent weights (hidden-to-hidden) as a proxy for learned dynamics
        # LiquidGraph structure varies, but usually has an internal net
        # We'll just check the state_dict norm sum
        param_sum = 0
        for name, param in expert.named_parameters():
            param_sum += torch.norm(param).item()
        expert_norms.append(param_sum)
        
    avg_norm = np.mean(expert_norms)
    std_norm = np.std(expert_norms)
    print(f"  Expert Weight Norms: Avg={avg_norm:.4f}, Std={std_norm:.4f}")
    print(f"  (Higher Std Dev implies experts are specializing in different things)")
    
    # 3. Analyze Titans Memory (Long-Term Surprise)
    if hasattr(broca, 'titans'):
        mem_norm = torch.norm(broca.titans.memory_weights).item()
        print(f"  Titans Memory Strength: {mem_norm:.4f}")
        print(f"  (Non-zero means it has started learning sequences)")
    else:
        print("  Titans Memory: NOT FOUND")

    # 4. Analyze Visual Projection (Visual Knowledge)
    vis_norm = torch.norm(broca.visual_projection.weight).item()
    print(f"  Visual Projection Strength: {vis_norm:.4f}")
    print(f"  (Non-zero means visual pathway is active)")
    
    # 5. Stimulus Test (Attractor Check)
    print("\n[Stimulus Response Test]")
    print("  Injecting random thought vector...")
    
    # Generate random input
    seed = torch.randn(64)
    
    # Process
    with torch.no_grad():
        output = broca.process_text_embedding(seed)
        
    response_mag = torch.norm(output).item()
    print(f"  Response Magnitude: {response_mag:.4f}")
    
    if response_mag > 0.1:
        print("  Result: The brain produced a strong response. (Good)")
        print("  Interpretation: The Liquid Networks have formed stable attractors.")
    else:
        print("  Result: Weak response.")
        print("  Interpretation: The networks might be silent or untrained.")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    analyze_knowledge()
