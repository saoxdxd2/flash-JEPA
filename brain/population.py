import random
import pickle
from brain.genome import Genome

class PopulationManager:
    """
    Manages a population of Genomes.
    Handles selection, reproduction, and evolution.
    """
    def __init__(self, population_size=None):
        if population_size is None:
            population_size = Genome.DEFAULT_POPULATION_SIZE
        self.population_size = population_size
        self.genomes = [Genome() for _ in range(population_size)]
        self.generation = 0
        self.best_genome = None
        self.population = [{'genome': g, 'fitness': 0.0} for g in self.genomes]

    def evolve(self):
        """
        Runs one generation of evolution.
        Selection -> Crossover (N2N2) -> Mutation.
        """
        print(f"Population: Evolving Generation {self.generation}...")
        
        # Ensure save directory exists
        import os
        os.makedirs("models/saved", exist_ok=True)
        
        # 1. Selection (Top SURVIVAL_RATE)
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        survivors = self.population[:max(2, int(self.population_size * Genome.SURVIVAL_RATE))]
        
        best_genome = survivors[0]['genome']
        print(f"  Best Genome: {best_genome} | Score: {survivors[0]['fitness']:.4f}")
        
        new_population = []
        
        # 2. Elitism (Keep best unchanged)
        # We must copy the model file to the new generation ID
        best_model_path = survivors[0].get('model_path')
        if best_model_path and os.path.exists(best_model_path):
            # Preserve extension
            ext = os.path.splitext(best_model_path)[1]
            if not ext: ext = ".pt" # Default to v2.0
            new_path = f"models/saved/gen_{self.generation+1}_elite{ext}"
            import shutil
            shutil.copy(best_model_path, new_path)
            print(f"Population: Copied Elite Model from {best_model_path} to {new_path}")
            new_population.append({'genome': best_genome, 'fitness': 0.0, 'model_path': new_path, 'school_level': survivors[0].get('school_level', 0)})
        else:
            print(f"Population: WARNING - Elite has no model path or file missing: {best_model_path}")
            new_population.append({'genome': best_genome, 'fitness': 0.0, 'model_path': None, 'school_level': survivors[0].get('school_level', 0)})
        
        # 3. Reproduction (N2N2 Crossover + Mutation)
        from brain.evolutionary_brain import EvolutionaryBrain
        
        child_idx = 0
        while len(new_population) < self.population_size:
            parent1_data = random.choice(survivors)
            parent2_data = random.choice(survivors)
            
            # Load Parent Brains (if they exist)
            p1_brain = EvolutionaryBrain(parent1_data['genome'])
            if parent1_data.get('model_path'): p1_brain.load_model(parent1_data['model_path'])
            
            p2_brain = EvolutionaryBrain(parent2_data['genome'])
            if parent2_data.get('model_path'): p2_brain.load_model(parent2_data['model_path'])
            
            # Perform Crossover
            child_brain = p1_brain.crossover(p2_brain)
            child_brain.genome.mutate()
            
            # Save Child Model
            child_path = f"models/saved/gen_{self.generation+1}_child_{child_idx}.pt"
            child_brain.save_model(child_path)
            
            new_population.append({'genome': child_brain.genome, 'fitness': 0.0, 'model_path': child_path})
            child_idx += 1
            
        self.population = new_population
        self.population = new_population
        self.generation += 1
        
        # 4. Garbage Collection
        self.cleanup_old_models()
        
        return best_genome

    def cleanup_old_models(self, keep_generations=None):
        """
        Deletes model files from generations older than (current - keep).
        """
        if keep_generations is None:
            keep_generations = Genome.KEEP_GENERATIONS
        import os
        import re
        
        saved_dir = "models/saved"
        if not os.path.exists(saved_dir):
            return
            
        # Regex to parse gen_X_...
        pattern = re.compile(r"gen_(\d+)_.*\.(pt|pkl)")
        
        threshold = self.generation - keep_generations
        
        if threshold < 0:
            return

        print(f"Population: Cleaning up models older than Gen {threshold}...")
        
        count = 0
        for filename in os.listdir(saved_dir):
            match = pattern.match(filename)
            if match:
                gen_num = int(match.group(1))
                if gen_num < threshold:
                    file_path = os.path.join(saved_dir, filename)
                    try:
                        os.remove(file_path)
                        count += 1
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")
                        
        if count > 0:
            print(f"Population: Garbage Collection removed {count} old model files.")

    def get_genome(self, index):
        return self.population[index]['genome']
        
    def set_fitness(self, index, fitness):
        self.population[index]['fitness'] = fitness

    def save(self, filepath="population.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath="population.pkl"):
        try:
            with open(filepath, 'rb') as f:
                manager = pickle.load(f)
                
            # Sync with Disk (Self-Healing)
            import os
            import re
            saved_dir = "models/saved"
            if os.path.exists(saved_dir):
                max_gen = -1
                best_file = None
                pattern = re.compile(r"gen_(\d+)_elite\.(pt|pkl)")
                
                for filename in os.listdir(saved_dir):
                    match = pattern.match(filename)
                    if match:
                        gen = int(match.group(1))
                        ext = match.group(2)
                        
                        # Priority: Higher Gen, then .pt over .pkl if same gen
                        if gen > max_gen:
                            max_gen = gen
                            best_file = os.path.join(saved_dir, filename)
                        elif gen == max_gen and ext == "pt":
                            best_file = os.path.join(saved_dir, filename)
                
                # If disk has a newer generation, update the manager
                if max_gen > manager.generation:
                    print(f"Population: Found newer generation {max_gen} on disk (vs {manager.generation}). Updating...")
                    manager.generation = max_gen
                    # Update the champion (index 0) to point to this new file
                    if len(manager.population) > 0:
                        manager.population[0]['model_path'] = best_file
                        # We accept that the genome object might be slightly out of sync
                        # but loading the weights is the most important part.
                        
            return manager
        except FileNotFoundError:
            return PopulationManager()
