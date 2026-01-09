import random
import copy

class Genome:
    """
    Defines the genetic blueprint of an agent.
    Hyperparameters evolve over time to optimize for Intelligence vs Efficiency.
    """
    # --- Project-wide Constants ---
    NON_VISUAL_INPUT_SIZE = 256
    DEFAULT_POPULATION_SIZE = 5
    MAX_AGE = 50
    SAVE_INTERVAL = 100
    PERIPHERAL_RESOLUTION = 64
    
    # --- Brain Hyperparameters ---
    INPUT_BOOST_FACTOR = 20.0 # Default/Legacy
    SURPRISE_THRESHOLD = 0.5
    REWARD_THRESHOLD = 100.0
    GROWTH_RATE_MULTIPLIER = 1.1
    MAX_HIDDEN_SIZE = 32768
    REPLAY_BUFFER_CAPACITY = 10000
    ACTION_COST_BASE = 0.01
    DISTILLATION_TEMPERATURE = 2.0
    DISTILLATION_LOSS_WEIGHT = 0.5
    JEPA_LOSS_WEIGHT = 0.1
    FISHER_SAMPLE_SIZE = 100 # Default/Legacy
    CONFIDENCE_THRESHOLD = 0.7 # Threshold for System 1 (ONNX) vs System 2 (PyTorch)
    
    # --- Evolution Constants ---
    SURVIVAL_RATE = 0.2
    KEEP_GENERATIONS = 2
    FITNESS_SCALING = 100.0
    EVOLUTION_DELAY = 1.0
    
    # --- Training Constants ---
    DEFAULT_GENERATION = 349
    TRAIN_STEPS_PER_CYCLE = 100
    DREAM_INTERVAL = 5
    TRAIN_SAVE_INTERVAL = 2
    
    def __init__(self):
        # 0. Species (Strategy)
        # 'efficient': Optimized for Reward/Energy
        # 'powerful': Optimized for Total Reward
        self.species = random.choice(['efficient', 'powerful'])

        # 1. Structural Genes (Hardware Efficiency)
        self.hidden_size = random.randint(16, 32) * 16 # RAM vs Intelligence (256-512, divisible by 16)
        self.target_hidden_size = 1000000 # 1 Million Neurons (Ambitious Goal)
        self.growth_rate = 32 # Neurons added per growth spurt
        self.sparsity = random.uniform(0.9, 0.99)   # High Sparsity for Large Scale
        self.rank = 256 # Factorized Rank (Resolution/Complexity)
        self.target_rank = 2048 # Evolutionary Goal for Rank
        self.latent_dim = 256 # Compression Dimension (Retina/Broca)
        self.target_latent_dim = 4096 # Evolutionary Goal for Latent Dim (Matches Qwen-3)
        self.usage_bias = 0.5 # Dynamic growth based on usage (0.0 to 1.0)
        
        # Resource Constraints
        self.max_ram_mb = 8192 # Default 8GB, can be updated dynamically
        
        # 2. Plasticity Genes (Meta-Learning Architecture)
        self.plasticity_hidden_size = random.choice([8, 16, 32, 64])
        self.plasticity_layers = random.choice([1, 2, 3])
        self.plasticity_activation = random.choice(['tanh', 'relu', 'sigmoid'])
        
        # 3. Learning Genes (Adaptability)
        self.mutation_rate = random.uniform(0.01, 0.05)
        self.code_mutation_rate = random.uniform(0.05, 0.1) # Rate of code changes in ELUs
        self.max_instructions = random.randint(4, 8) # Complexity of each ELU
        self.learning_rate = random.uniform(0.0001, 0.01)
        # Evolved Learning Rule: A, B, C, D coefficients (Legacy/Init)
        self.plasticity_coefficients = [random.uniform(-1, 1) for _ in range(4)]
        
        # 4. Biological Genes (Survival)
        self.stamina_efficiency = random.uniform(0.8, 1.2) # How fast energy drains
        self.resilience = random.uniform(0.1, 0.5)         # Resistance to Suffering
        
        # 5. Evolved Heuristics (Previously Hardcoded)
        self.input_boost_factor = random.uniform(10.0, 30.0)
        self.action_size = 72 # Start with standard, but can mutate
        self.latent_adapter_dim = 4096 # Matches Qwen-3/Llama default
        self.fisher_sample_size = random.randint(50, 200)
        
        # 6. Phase 3: Dynamic Expert Sprouting
        self.sprouting_threshold = random.uniform(0.3, 0.7)
        self.max_experts = 16
        
        # 6. Meta-Data
        self.generation = 0
        self.parent_id = None

    def mutate(self):
        """
        Self-adaptive mutation.
        The genome modifies itself based on its own mutation rate.
        """
        if random.random() < self.mutation_rate:
            # Mutate Hidden Size (RAM usage)
            change = random.choice([-32, -16, 16, 32])
            self.hidden_size = max(128, min(self.target_hidden_size, self.hidden_size + change))
            self.hidden_size = (self.hidden_size // 16) * 16 # Enforce alignment
            
        if random.random() < self.mutation_rate:
            # Mutate Sparsity (CPU usage)
            self.sparsity = max(0.5, min(0.99, self.sparsity + random.uniform(-0.05, 0.05)))
            
        if random.random() < self.mutation_rate:
            # Mutate Rank (Resolution)
            # Rank scales linearly with cost, but quadratically with capability?
            change = random.choice([-16, 16, 32])
            self.rank = max(64, min(self.target_rank, self.rank + change))
            
        if random.random() < self.mutation_rate:
            # Mutate Latent Dim (Compression)
            change = random.choice([-32, 32, 64])
            self.latent_dim = max(128, min(self.target_latent_dim, self.latent_dim + change))

        if random.random() < self.mutation_rate:
            # Mutate Mutation Rate (Meta-evolution)
            self.mutation_rate = max(0.001, min(0.2, self.mutation_rate * random.uniform(0.8, 1.2)))
            self.code_mutation_rate = max(0.01, min(0.5, self.code_mutation_rate * random.uniform(0.8, 1.2)))
            
        if random.random() < self.mutation_rate:
            # Mutate Complexity
            self.max_instructions = max(2, min(16, self.max_instructions + random.choice([-1, 1])))
            
        if random.random() < self.mutation_rate:
            # Mutate Learning Rule
            idx = random.randint(0, 3)
            self.plasticity_coefficients[idx] += random.uniform(-0.1, 0.1)
            
        if random.random() < self.mutation_rate:
            # Mutate Plasticity Architecture
            self.plasticity_hidden_size = random.choice([8, 16, 32, 64])
            self.plasticity_layers = random.choice([1, 2, 3])
            self.plasticity_activation = random.choice(['tanh', 'relu', 'sigmoid'])

        if random.random() < self.mutation_rate:
            # Mutate Evolved Heuristics
            self.input_boost_factor = max(1.0, self.input_boost_factor + random.uniform(-2.0, 2.0))
            self.fisher_sample_size = max(10, self.fisher_sample_size + random.randint(-10, 10))
            # Action size mutation is rare and structural
            if random.random() < 0.1:
                self.action_size = max(10, self.action_size + random.choice([-8, 8]))
                
        if random.random() < self.mutation_rate:
            # Mutate Sprouting Threshold
            self.sprouting_threshold = max(0.1, min(1.0, self.sprouting_threshold + random.uniform(-0.1, 0.1)))

    def crossover(self, other_genome):
        """
        Creates a child by combining genes from self and other_genome.
        """
        child = Genome()
        child.hidden_size = random.choice([self.hidden_size, other_genome.hidden_size])
        child.sparsity = random.choice([self.sparsity, other_genome.sparsity])
        child.rank = random.choice([self.rank, other_genome.rank])
        child.latent_dim = random.choice([self.latent_dim, other_genome.latent_dim])
        child.mutation_rate = (self.mutation_rate + other_genome.mutation_rate) / 2
        child.stamina_efficiency = random.choice([self.stamina_efficiency, other_genome.stamina_efficiency])
        child.plasticity_coefficients = [random.choice([a, b]) for a, b in zip(self.plasticity_coefficients, other_genome.plasticity_coefficients)]
        child.generation = max(self.generation, other_genome.generation) + 1
        child.species = random.choice([self.species, other_genome.species])
        
        # Crossover Plasticity Genes
        child.plasticity_hidden_size = random.choice([self.plasticity_hidden_size, other_genome.plasticity_hidden_size])
        child.plasticity_layers = random.choice([self.plasticity_layers, other_genome.plasticity_layers])
        child.plasticity_activation = random.choice([self.plasticity_activation, other_genome.plasticity_activation])
        
        # Crossover Sprouting Genes
        child.sprouting_threshold = random.choice([self.sprouting_threshold, other_genome.sprouting_threshold])
        child.max_experts = random.choice([self.max_experts, other_genome.max_experts])
        
        return child

    def __repr__(self):
        return (f"Genome(Species={self.species}, H={self.hidden_size}, PlasH={self.plasticity_hidden_size}, "
                f"PlasL={self.plasticity_layers}, Act={self.plasticity_activation})")
