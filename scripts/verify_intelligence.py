import torch
import numpy as np
import sys
import os

# Enable Anomaly Detection to debug inplace operations
torch.autograd.set_detect_anomaly(True)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.n2n import KnowledgeLoader

def verify_intelligence():
    print("=== Intelligence Verification ===")
    
    # 1. Initialize Brain
    brain = EvolutionaryBrain()
    brain.start()
    
    # 2. Inject Knowledge (Distillation)
    loader = KnowledgeLoader(brain)
    loader.inject_knowledge()
    
    # Set Epsilon to 0.0 for verification (Exploitation only)
    brain.ddqn.epsilon = 0.0
    brain.use_reflex_bias = False # Disable bias to test TRM directly
    brain.use_greedy_decide = True # Enable greedy selection for verification
    
    # 3. Test OCR (Communication Foundation)
    print("\n--- Testing OCR (Visual to Action) ---")
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
        
    correct_ocr = 0
    test_chars = "ANTIGRAVITY"
    latents = []
    for char in test_chars:
        brain.reset_memory() # Clear temporal context for pure OCR test
        img = Image.new('RGB', (64, 64), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((16, 8), char, fill=(0, 0, 0), font=font)
        
        # Process via Retina
        foveal_latent = brain.retina.process_image(img)
        latents.append(foveal_latent.numpy())
        
        # Construct State
        full_state = torch.zeros(brain.input_size)
        full_state[:brain.latent_dim] = foveal_latent
        
        # Decide (Disable internal RL training during verification)
        action, logits = brain.decide(full_state, train_internal_rl=False)
        
        # Action 15 is 'A', 16 is 'B', etc.
        expected_action = 15 + (ord(char) - ord('A'))
        if action == expected_action:
            correct_ocr += 1
            print(f"OCR: '{char}' -> Action {action} (CORRECT)")
        else:
            # Print top 3 logits for debugging
            top_vals, top_idxs = torch.topk(logits, 3)
            top_str = ", ".join([f"{idx}:{val:.2f}" for idx, val in zip(top_idxs, top_vals)])
            print(f"OCR: '{char}' -> Action {action} (EXPECTED {expected_action}) | Top Logits: {top_str}")
            
    # Latent Variance Check
    latents_np = np.array(latents)
    latent_var = np.var(latents_np, axis=0).mean()
    print(f"\nLatent Vector Variance: {latent_var:.6f}")
            
    print(f"OCR Accuracy: {correct_ocr}/{len(test_chars)}")
    
    # 4. Test Calculation (1+1)
    print("\n--- Testing Calculation (1+1) ---")
    # "WHAT IS 1+1" -> "2 "
    # We test if the brain predicts '2' (Action 44) after seeing '?' (Action 56)
    # or after the semantic vector for '1+1'
    
    # Test via Broca
    question = "1+1"
    semantic_vector = brain.broca.process_text(question)
    
    full_state = torch.zeros(brain.input_size)
    L = brain.latent_dim
    full_state[2*L : 2*L + 256] = semantic_vector # Semantic Part
    
    action, _ = brain.decide(full_state, train_internal_rl=False)
    # Action 42-51 are 0-9
    # '2' is Action 44
    if action == 44:
        print("Calculation: '1+1' -> Action 44 ('2') (CORRECT)")
    else:
        print(f"Calculation: '1+1' -> Action {action} (EXPECTED 44)")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_intelligence()
