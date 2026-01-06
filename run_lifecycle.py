import sys
import os
import time
import argparse
import psutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.lifecycle import LifecycleManager
from brain.genome import Genome

def main():
    parser = argparse.ArgumentParser(description="Agent Lifecycle Runner")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles to run (0 for infinite)")
    parser.add_argument("--imprint-steps", type=int, default=100, help="Steps for language imprinting")
    parser.add_argument("--meditation-steps", type=int, default=200, help="Steps for stabilization")
    parser.add_argument("--dream-steps", type=int, default=20, help="Steps for dreaming")
    parser.add_argument("--school-episodes", type=int, default=10, help="Episodes for schooling")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "meditation", "growth"], help="Training mode")
    args = parser.parse_args()

    print("==========================================")
    print("      AGENT LIFECYCLE PIPELINE v1.0       ")
    print("==========================================")
    
    manager = LifecycleManager()
    
    cycle = 0
    try:
        while True:
            cycle += 1
            if args.cycles > 0 and cycle > args.cycles:
                break
                
            start_time = time.time()
            print(f"\n>>> STARTING CYCLE {cycle} <<<")
            
            # 1. Load
            manager.load()
            
            loss = 0.0
            reward = 0.0
            
            if args.mode in ["full", "growth"]:
                # 2. Imprint
                loss = manager.phase_imprint_language(steps=args.imprint_steps)
                
                # 2c. Logic Transfer (SSI)
                manager.phase_logic_transfer(steps=max(1, args.imprint_steps // 10))
            
            if args.mode in ["full", "meditation"]:
                # 3. Stabilize
                manager.phase_stabilize(steps=args.meditation_steps)
            
            if args.mode == "full":
                # 4. Ground
                manager.phase_grounding()
                
                # 5. School
                reward = manager.phase_schooling(episodes=args.school_episodes)
            
            # 6. Consolidate
            manager.phase_consolidation(steps=args.dream_steps)
            
            # 7. Evolve
            if args.mode in ["full", "growth"]:
                efficiency = (1.0 / (loss + 0.1)) + (reward * 0.5)
                manager.phase_evolution(efficiency)
            
            # 8. Save
            manager.save()
            
            duration = time.time() - start_time
            mem = psutil.virtual_memory()
            print(f"\n>>> CYCLE {cycle} COMPLETE ({duration:.1f}s) <<<")
            print(f"Metrics: Loss={loss:.4f} | Reward={reward:.1f} | Size={manager.brain.genome.hidden_size} | RAM={mem.percent}%")
            
            # Cooldown to prevent CPU thermal throttling if needed
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nLifecycle: Interrupted by user. Saving state...")
        manager.save()
        print("Lifecycle: Shutdown complete.")

if __name__ == "__main__":
    main()
