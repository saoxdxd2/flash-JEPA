from brain.evolutionary_brain import EvolutionaryBrain
from brain.population import PopulationManager
from brain.n2n import KnowledgeLoader
import time
import sys
import os

def main():
    print("=== Biological Agent Initialization (Population Mode) ===", flush=True)
    
    # 1. Initialize Population
    # Try to load existing population to share progress with Math School
    pop_manager = PopulationManager.load()
    if len(pop_manager.population) == 0:
        from brain.genome import Genome
        pop_manager = PopulationManager(population_size=Genome.DEFAULT_POPULATION_SIZE)
    
    print(f"Population initialized with {pop_manager.population_size} genomes. Generation: {pop_manager.generation}", flush=True)
    
    generation = pop_manager.generation
    
    try:
        while True:
            # 2. Select Best Genome for this Life Cycle
            # In a real loop, we'd iterate through the population
            # For now, we pick index 0 (the "Champion") to run on screen
            population = pop_manager.population
            for i, individual in enumerate(population):
                genome = individual['genome']
                
                print(f"\n=== Generation {generation} Started ===")
                print(f"Genome: {genome}")
                
                # 3. Birth of the Agent
                brain = EvolutionaryBrain(genome)
                
                # 4. Load Learned Weights (N2N2)
                transplanted_path = "models/saved/gen_350_transplanted.pt"
                model_path = individual.get('model_path')
                
                # Ensure model_path is valid for saving later
                if not model_path:
                    os.makedirs("models/saved", exist_ok=True)
                    model_path = f"models/saved/gen_{generation}_ind_{i}.pt"
                    individual['model_path'] = model_path

                if os.path.exists(transplanted_path):
                    print(f"Loading TRANSPLANTED model from: {transplanted_path}")
                    try:
                        brain.load_model(transplanted_path)
                        # Update both local and individual path
                        model_path = transplanted_path
                        individual['model_path'] = transplanted_path
                    except Exception as e:
                        print(f"WARNING: Failed to load transplanted model: {e}")
                        print("Starting fresh instead.")
                elif model_path and os.path.exists(model_path):
                    print(f"Loading model from: {model_path}")
                    try:
                        brain.load_model(model_path)
                    except Exception as e:
                        print(f"WARNING: Failed to load model {model_path}: {e}")
                        print("Starting fresh instead.")
                else:
                    # Inject Base Knowledge if new
                    print("Starting fresh with base knowledge...")
                    print("No existing model found. Injecting Base Knowledge...")
                    n2n = KnowledgeLoader(brain)
                    n2n.inject_knowledge()
                    # Ensure we have a path for future saves
                    os.makedirs("models/saved", exist_ok=True)
                    model_path = f"models/saved/gen_{generation}_elite.pt"
                    individual['model_path'] = model_path
                
                brain.start()
                
                # 5. Live Life (Wake Cycle)
                alive = True
                collapsed = False
                
                while alive:
                    result = brain.wake_cycle()
                    
                    if result == "COLLAPSED":
                        print(f"\n[Age: {brain.age}] COLLAPSED FROM EXHAUSTION!")
                        alive = False
                        collapsed = True
                    elif result is not None:
                         if time.time() % 2 < 0.1:
                            chem = brain.chemistry
                            print(f"Gen: {generation} | Age: {brain.age} | E: {chem.energy:.1f} | Cost: {brain.last_energy_cost:.2f} | Act: {result}", end='\r', flush=True)
                            pass
                    
                    if result == "WAITING_FOR_RETINA":
                        if time.time() % 5 < 0.1:
                            print("Waiting for Retina...", end='\r')
                    
                    time.sleep(0.01)
                    
                
                    # Force sleep if age > MAX_AGE (for evolution speed)
                    if brain.age > brain.genome.MAX_AGE:
                        print(f"\n[Age: {brain.age}] Decided to sleep (Old Age).")
                        alive = False
                    
                    # Periodic Save (for long lives)
                    if brain.age > 0 and brain.age % brain.genome.SAVE_INTERVAL == 0:
                        brain.save_model(model_path)
                        print(f" [Auto-Save at Age {brain.age}]", end='')
                
                brain.stop()
                
                # 6. Sleep & Evolve
                if collapsed:
                    print("Recovering from collapse... (Penalty)")
                    fitness = 0.0
                else:
                    print(f"\n[Generation {generation}] Sleeping & Evolving...")
                    
                    # Calculate Fitness
                    # Dual Evolution Strategy
                    if brain.genome.species == 'efficient':
                        # Efficiency: Reward per Energy
                        fitness = brain.accumulated_reward / (brain.accumulated_energy + 1e-6)
                    else:
                        # Power: Total Reward
                        fitness = brain.accumulated_reward
                    
                    print(f"Species: {brain.genome.species} | Fitness: {fitness:.2f} | R: {brain.accumulated_reward:.2f} | E: {brain.accumulated_energy:.2f}")
                    
                    # 7. Dreaming & Consolidation (Phase 3/5)
                    # This handles System 2 -> System 1 distillation and Structural Plasticity (Growth)
                    brain.dream(steps=20)
                    
                    # 8. Adaptive Evolution (N2N2 Growth)
                    brain.mutate_adaptive(fitness / brain.genome.FITNESS_SCALING)
                
                # Save Lifetime Learning (Weights)
                if not model_path:
                    os.makedirs("models/saved", exist_ok=True)
                    model_path = f"models/saved/gen_{generation}_elite.pt"
                    individual['model_path'] = model_path
                    
                brain.save_model(model_path)
                
                # Save Modified Genome
                individual['genome'] = brain.genome
                pop_manager.set_fitness(i, fitness)
                
                # Only run one individual per generation for the main loop demo
                break
            
            # Evolve Population (Crossover & Selection)
            best_genome = pop_manager.evolve()
            pop_manager.save()
            
            generation += 1
            time.sleep(brain.genome.EVOLUTION_DELAY)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pass

if __name__ == "__main__":
    main()
