import torch
import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain

def probe_capabilities():
    print("================================================================")
    print("CAPABILITY PROBE: models/saved/gen_349_elite_v2.pt")
    print("================================================================")
    
    brain = EvolutionaryBrain()
    checkpoint_path = "models/saved/gen_349_elite_v2.pt"
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        return
        
    brain.load_model(checkpoint_path)
    broca = brain.broca
    
    # 1. Language Understanding Probe
    print("\n[1] LANGUAGE PROBE")
    test_words = ["hello", "apple", "code", "python", "3+3"]
    vectors = []
    
    for word in test_words:
        vec = broca.process_text(word)
        vectors.append(vec)
        norm = torch.norm(vec).item()
        print(f"  > '{word}': Norm={norm:.4f}")
        
    # Check for differentiation (Cosine Similarity)
    def cosine_sim(a, b):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
        
    sim_hello_apple = cosine_sim(vectors[0], vectors[1]).item()
    sim_code_python = cosine_sim(vectors[2], vectors[3]).item()
    sim_hello_code = cosine_sim(vectors[0], vectors[2]).item()
    
    print(f"  > Similarity 'hello' vs 'apple': {sim_hello_apple:.4f}")
    print(f"  > Similarity 'code' vs 'python': {sim_code_python:.4f}")
    print(f"  > Similarity 'hello' vs 'code': {sim_hello_code:.4f}")
    
    if sim_hello_apple < 0.99 and sim_code_python > sim_hello_code:
        print("  > RESULT: DIFFERENTIATION DETECTED (Language embeddings are distinct and structured).")
    else:
        print("  > RESULT: WEAK/RANDOM (Embeddings are too similar or unstructured).")

    # 2. OCR Probe
    print("\n[2] OCR PROBE")
    # Create an image with text "A"
    img = Image.new('RGB', (64, 64), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    # Draw simple shape resembling 'A' since we might not have fonts
    d.line((32, 10, 10, 60), fill=(255, 255, 255), width=2)
    d.line((32, 10, 54, 60), fill=(255, 255, 255), width=2)
    d.line((20, 40, 44, 40), fill=(255, 255, 255), width=2)
    
    # Convert to tensor [1, 3, 64, 64] (RGB)
    img_np = np.array(img).transpose(2, 0, 1) / 255.0 # [3, 64, 64]
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
    
    # Process
    ocr_vec = broca.process_visual(img_tensor)
    ocr_norm = torch.norm(ocr_vec).item()
    print(f"  > OCR Output Norm: {ocr_norm:.4f}")
    
    # Compare with text "A"
    text_a_vec = broca.process_text("A")
    sim_ocr_text = cosine_sim(ocr_vec, text_a_vec).item()
    print(f"  > Similarity OCR('A') vs Text('A'): {sim_ocr_text:.4f}")
    
    if ocr_norm > 0.1:
        print("  > RESULT: SIGNAL DETECTED (Visual cortex is processing input).")
    else:
        print("  > RESULT: SILENT (No meaningful output from visual projection).")

    # 3. Math/Reasoning Probe (Action Output)
    print("\n[3] MATH/REASONING PROBE (Action Reflex)")
    # Feed "3+3" to the brain and check action logits
    # We construct a full input tensor
    full_input = torch.zeros(brain.input_size)
    
    # Inject "3+3" semantic vector into semantic slots (indices 512-768 approx, need to check evolutionary_brain.py)
    # In evolutionary_brain.py:
    # Input = Foveal(256) + Peripheral(256) + Semantic(256) ...
    # So Semantic is at index 512.
    
    math_vec = broca.process_text("3+3")
    full_input[512:512+256] = math_vec
    
    action, logits = brain.decide(full_input)
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    top_3 = np.argsort(probs)[-3:][::-1]
    
    print(f"  > Input: '3+3'")
    print(f"  > Top 3 Actions: {top_3} (Probs: {probs[top_3]})")
    
    # Check if any digit action (50-59) is triggered
    # Digits 0-9 are usually mapped to actions. 
    # In evolutionary_brain.py: 
    # 15-40: Typing A-Z
    # 50-59: Extended Actions (Digits?)
    
    is_math_response = any(50 <= a <= 59 for a in top_3)
    if is_math_response:
        print("  > RESULT: MATH REFLEX DETECTED (Brain attempted to type a digit).")
    else:
        print("  > RESULT: NO SPECIFIC REFLEX (Brain did not prioritize digits).")

    print("\n================================================================")

if __name__ == "__main__":
    probe_capabilities()
