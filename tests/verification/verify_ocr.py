import sys
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def generate_letter_image(char):
    """Generates a 64x64 image of a letter."""
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    d.text((16, 8), char, fill=(0, 0, 0), font=font)
    return img

def test_sentence(brain, text):
    print(f"\n--- Testing Sentence: \"{text}\" ---")
    typed_output = ""
    
    for char in text:
        if char == " ":
            # Handle Space (Action 41? Need to check mapping)
            # For now, let's assume Space is Action 41 or just skip/print space
            # Let's check ControlInterface mapping.
            # Assuming 15=A... 40=Z. 
            # If char is space, we might not have trained it yet.
            # Let's stick to A-Z for now, or handle space if trained.
            # TeacherDistillation only did A-Z.
            # So we will strip spaces for this test or skip them.
            typed_output += " "
            continue
            
        char = char.upper()
        if not ('A' <= char <= 'Z'):
            continue
            
        # 1. Generate Image
        img = generate_letter_image(char)
        
        # 2. Process via Retina
        foveal_latent = brain.retina.process_image(img)
        
        # 3. Construct State
        full_state = torch.zeros(brain.input_size)
        full_state[:brain.latent_dim] = foveal_latent
        
        # 4. Query Memory
        memories, metadata, similarities = brain.memory.recall(full_state)
        
        if metadata is not None:
            action = int(metadata[0][0].item())
            confidence = similarities[0].item()
            
            # Decode Action
            if 15 <= action <= 40:
                decoded_char = chr(action - 15 + 65)
                typed_output += decoded_char
            else:
                typed_output += "?"
        else:
            typed_output += "?"
            
    print(f"Original: {text}")
    print(f"Typed:    {typed_output}")
    
    # Calculate Accuracy (Levenshtein would be better, but simple match for now)
    clean_text = "".join([c.upper() for c in text if 'A' <= c.upper() <= 'Z' or c == ' '])
    if typed_output.strip() == clean_text.strip():
        print("  [SUCCESS] Perfect Match!")
        return True
    else:
        print("  [FAILURE] Mismatch.")
        return False

def verify_ocr():
    print("=== OCR Verification ===")
    
    # 1. Initialize Brain
    print("Initializing Brain...")
    genome = Genome()
    brain = EvolutionaryBrain(genome)
    
    # Force N2N Injection
    from brain.n2n import KnowledgeLoader
    loader = KnowledgeLoader(brain)
    loader.inject_knowledge()
    
    # 2. Test Single Letters
    print("\n--- Phase 1: Single Letters ---")
    correct = 0
    total = 0
    test_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for char in test_chars:
        img = generate_letter_image(char)
        foveal_latent = brain.retina.process_image(img)
        full_state = torch.zeros(brain.input_size)
        full_state[:brain.latent_dim] = foveal_latent
        memories, metadata, similarities = brain.memory.recall(full_state)
        
        if metadata is not None:
            action = int(metadata[0][0].item())
            expected_action = 15 + (ord(char) - 65)
            if action == expected_action:
                correct += 1
        total += 1
        
    print(f"Letter Accuracy: {(correct/total)*100:.1f}%")

    # 3. Test Sentences
    print("\n--- Phase 2: Sentences (Paragraphs) ---")
    sentences = [
        "HELLO WORLD",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "AGENTIC AI IS THE FUTURE"
    ]
    
    all_passed = True
    for s in sentences:
        if not test_sentence(brain, s):
            all_passed = False
            
    if all_passed and correct == total:
        print("\nOCR Verification PASSED (Letters & Sentences).")
    else:
        print("\nOCR Verification FAILED.")

if __name__ == "__main__":
    verify_ocr()
