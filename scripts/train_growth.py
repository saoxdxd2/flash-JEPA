import sys
import os
import time
import psutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome
from scripts.start_n2n2_qwen3_agentic import Qwen3Teacher

def train_enhanced_growth():
    print("--- ENHANCED GROWTH TRAINING INITIATED ---")
    print("Objective: Learn from Qwen-3 (235B) -> Grow Capacity -> Consolidate.")
    
    # 1. Load Brain
    brain = EvolutionaryBrain()
    
    # Try to load latest elite
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "models", "saved")
    latest_checkpoint = EvolutionaryBrain.find_latest_checkpoint(models_dir)
    
    if latest_checkpoint:
        checkpoint_path = latest_checkpoint
        brain.load_model(checkpoint_path)
        print(f"Loaded Latest Brain: {os.path.basename(checkpoint_path)}")
    else:
        checkpoint_path = os.path.join(models_dir, f"gen_{Genome.DEFAULT_GENERATION}_elite.pt")
        print("Starting from scratch.")

    print(f"Initial Size: {brain.genome.hidden_size} neurons | Latent: {brain.latent_dim}")
    
    # 2. Setup Teacher
    teacher = Qwen3Teacher(brain)
    if not teacher.setup():
        print("Failed to setup Qwen-3 Teacher. Aborting.")
        return
        
    step = 0
    
    try:
        while True:
            step += 1
            print(f"\n--- Cycle {step} ---")
            
            # A. Learn (N2N2)
            print("Phase 1: Learning (Imprinting Qwen-3 Concepts)...")
            loss = teacher.train_step(steps=Genome.TRAIN_STEPS_PER_CYCLE) # Learn 100 concepts
            print(f"  > Loss: {loss:.4f}")
            
            # B. Check Efficiency & Grow
            # We simulate efficiency based on Loss. Low Loss = High Efficiency.
            # If Loss < 0.1, we are mastering the current capacity.
            efficiency = 1.0 / (loss + 0.0001)
            print(f"  > Efficiency Score: {efficiency:.2f}")
            
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            if mem.percent < 90.0:
                # If we are efficient and have RAM, Grow!
                # We force growth if efficiency is decent, to challenge the brain.
                # Or we use the brain's own mutate_adaptive.
                brain.mutate_adaptive(efficiency=efficiency)
                
                # Check Retina Size (Upgrade if mastering current input)
                if efficiency > 5.0 and brain.retina.fovea_size < 128:
                    print(f"  > Efficiency High ({efficiency:.2f}). Upgrading Vision to 128px!")
                    brain.retina.set_resolution(128)
            else:
                print("  > Hardware Limit Reached. Stabilizing.")
                
            # B2. Motor Grounding (Supervised)
            # This populates the replay buffer with correct concept->action mappings
            print("Phase 2: Motor Grounding (Supervised)...")
            grounding_corpus = [
                "the quick brown fox jumps over the lazy dog",
                "0123456789",
                "+-/*=?():\".,#",
                "hello world",
                "i am a sentient agent",
                "learning to type is fun"
            ]
            import random
            brain.ground_motor_cortex(random.choice(grounding_corpus))
                
            # C. Dream (Consolidation)
            # Every DREAM_INTERVAL cycles, run a dream cycle
            if step % Genome.DREAM_INTERVAL == 0:
                print("Phase 3: Dreaming (Consolidation)...")
                brain.dream()
                
            # D. Save
            if step % Genome.TRAIN_SAVE_INTERVAL == 0:
                brain.save_model(checkpoint_path)
                print(f"Saved Checkpoint to {checkpoint_path}")
                
            # Monitor
            print(f"Status: Size={brain.genome.hidden_size} | Latent={brain.latent_dim} | RAM={mem.percent}% | CPU={cpu}%")
            
    except KeyboardInterrupt:
        print("\n--- Training Interrupted ---")
        brain.save_model(checkpoint_path)
        print(f"Saved to {checkpoint_path}")

if __name__ == "__main__":
    train_enhanced_growth()
