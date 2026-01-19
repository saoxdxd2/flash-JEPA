from brain.evolutionary_brain import EvolutionaryBrain
from brain.population import PopulationManager
from brain.training.knowledge_transfer import KnowledgeLoader
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
                
                # Task Generator
                from brain.modules.task_generator import TaskGenerator
                from brain.modules.visualization import VisualizationLogger
                
                task_gen = TaskGenerator(input_size=16, output_size=16)
                task_gen.set_difficulty(1)
                
                logger = VisualizationLogger()
                logger.log_event("SYSTEM", "Autonomous Life Started")
                
                tasks_completed = 0
                avg_loss = 1.0
                
                while alive:
                    # A. Cognitive Training (The "School" of Life)
                    # Generate a task
                    inputs, targets, meta = task_gen.generate_task()
                    
                    # Train
                    loss = brain.train_cognitive_task(inputs, targets)
                    
                    # Update Stats
                    avg_loss = 0.95 * avg_loss + 0.05 * loss
                    tasks_completed += 1
                    
                    # Curriculum Adjustment
                    if avg_loss < 0.1:
                        task_gen.set_difficulty(task_gen.difficulty + 1)
                        # Reset avg_loss to avoid instant promotion
                        avg_loss = 0.5
                        logger.log_event("LEVEL_UP", f"Difficulty -> {task_gen.difficulty}")
                        print(f"\n[Level Up] Difficulty -> {task_gen.difficulty}")
                        
                    # B. Biological Cycle (Metabolism)
                    brain.chemistry.energy -= 0.1 # Thinking cost
                    brain.age += 0.01
                    
                    # Update Chemistry with Task Feedback
                    brain.chemistry.update(
                        reward_prediction_error=0.0, # No external reward yet
                        surprise=loss, # Task Loss = Surprise
                        pain=0.0,
                        effort=0.1,
                        fear=0.0,
                        aggression=0.0
                    )
                    
                    # Log State
                    task_info = {
                        'desc': meta['desc'],
                        'difficulty': task_gen.difficulty,
                        'loss': loss,
                        'avg_loss': avg_loss,
                        'confidence': getattr(brain, 'last_confidence', 0.0)
                    }
                    logger.log_step(brain, task_info)
                    
                    if brain.chemistry.energy <= 0:
                        logger.log_event("DEATH", "Collapsed from exhaustion")
                        print(f"\n[Age: {brain.age:.1f}] COLLAPSED FROM EXHAUSTION!")
                        alive = False
                        collapsed = True
                        
                    # Sleep / Eat
                    if brain.chemistry.energy < 20:
                        # Auto-Eat
                        brain.chemistry.energy += 50
                        logger.log_event("METABOLISM", "Eating (Energy Restored)")
                        
                    if tasks_completed % 10 == 0:
                        print(f"Gen: {generation} | Age: {brain.age:.1f} | Diff: {task_gen.difficulty} | Loss: {avg_loss:.4f} | Task: {meta['desc']}", end='\r', flush=True)
                    
                    # Periodic Save
                    if tasks_completed % 500 == 0:
                        brain.save_model(model_path)
                        logger.log_event("SYSTEM", "Model Saved")
                        
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
