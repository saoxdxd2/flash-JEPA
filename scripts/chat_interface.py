import torch
import numpy as np
import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

from brain.evolutionary_brain import EvolutionaryBrain
from brain.modules.cradle import Cradle

def chat_interface():
    print("=== Antigravity Chat Interface (Pure SNN Mode) ===")
    print("Initializing Brain...")
    
    brain = EvolutionaryBrain()
    # Load latest checkpoint
    checkpoint_dir = os.path.join(os.getcwd(), "models", "saved")
    latest = brain.find_latest_checkpoint(checkpoint_dir)
    if latest:
        print(f"Loading checkpoint: {latest}")
        brain.load_model(latest)
    else:
        print("No checkpoint found. Starting with fresh brain.")
    
    brain.start()
    cradle = Cradle()
    
    print("\nModel is ready. Type 'exit' to quit.")
    print("This interface uses pure character-level signals (No Tokenizers).")
    
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if not user_input.strip():
                continue
                
            # 1. Process Text Input via Broca (Character-based)
            print("Thinking...", end="\r")
            
            # Use Broca's native character-to-signal processing
            semantic_vector = brain.broca.process_text(user_input)
            
            # Create a dummy visual input (zeros)
            L = brain.genome.latent_dim if hasattr(brain.genome, 'latent_dim') else 256
            foveal_latent = np.zeros(L)
            peripheral_latent = np.zeros(L)
            
            # 2. Construct Input Tensor (Standardized)
            foveal_latent = torch.zeros(L)
            peripheral_latent = torch.zeros(L)
            
            padded_input = brain.get_input_vector(
                foveal_latent,
                peripheral_latent,
                semantic_vector
            )
            
            # 2. Decide Action (with Burst Mode support)
            response_chars = []
            actions_taken = []
            
            # CHAT BIAS: Encourage typing (15-41)
            chat_bias = torch.zeros(brain.action_size)
            chat_bias[15:42] = 2.0 # Lowered to allow "Stop" or silence
            
            max_burst = 40 # Allow longer sentences
            total_confidence = 0
            for step in range(max_burst):
                # Get raw logits
                _, action_logits = brain.decide(padded_input)
                
                # Apply Bias and select best typing action
                biased_logits = action_logits + chat_bias
                action = torch.argmax(biased_logits).item()
                
                # Confidence metric (Max logit vs Mean)
                confidence = torch.max(action_logits).item() - torch.mean(action_logits).item()
                total_confidence += confidence
                
                actions_taken.append(action)
                
                # Interpret Action
                if 15 <= action <= 40:
                    char = chr(action - 15 + ord('a'))
                    response_chars.append(char)
                    
                    # Update padded_input for autoregressive feedback
                    # Action feedback starts at 3*L + 6
                    auto_start = 3 * L + 6
                    if auto_start + 100 <= len(padded_input):
                        padded_input[auto_start:auto_start+100] = 0.0
                        padded_input[auto_start + action] = 5.0
                    
                    # Stop if confidence drops too low (raw confidence)
                    if brain.last_confidence < 0.2: 
                        break
                elif action == 41: # Space
                    response_chars.append(" ")
                    auto_start = 3 * L + 6
                    if auto_start + 100 <= len(padded_input):
                        padded_input[auto_start:auto_start+100] = 0.0
                        padded_input[auto_start + action] = 5.0
                else:
                    # Non-typing action ends the burst
                    break
            
            # 3. Display Response
            if response_chars:
                word = "".join(response_chars)
                print(f"Agent: {word}")
            elif actions_taken:
                last_action = actions_taken[-1]
                if last_action < 10:
                    print(f"Agent: [Action {last_action}]")
                else:
                    print(f"Agent: [Action {last_action}]")
                
            avg_confidence = total_confidence / max(1, len(actions_taken))
            print(f"Confidence: {avg_confidence:.2f} | Dopamine: {brain.chemistry.dopamine:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        brain.stop()
        print("Brain stopped.")

if __name__ == "__main__":
    chat_interface()
