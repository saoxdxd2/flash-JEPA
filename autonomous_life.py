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
            individual = pop_manager.population[0]
            genome = individual['genome']
            
            print(f"\n=== Generation {generation} Started ===")
            print(f"Genome: {genome}")
            
            # 3. Birth of the Agent
            brain = EvolutionaryBrain(genome)
            
            # 4. Load Learned Weights (N2N2)
            model_path = individual.get('model_path')
            if model_path and os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                brain.load_model(model_path)
            else:
                # Inject Base Knowledge if new
                n2n = KnowledgeLoader(brain)
                n2n.inject_knowledge()
            
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
                if brain.age % brain.genome.SAVE_INTERVAL == 0:
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
            pop_manager.set_fitness(0, fitness)
            
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
