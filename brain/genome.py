"""
Brain-Specific DNA Genome System

This module implements a biologically-accurate genome based on real neuroscience genes.
Each gene has two alleles (diploid inheritance), epigenetic state, and activity-dependent
expression - just like real brain genes.

Key Gene Systems:
- Neurotransmitter: DRD2, COMT, 5-HTTLPR, DAT1 (dopamine/serotonin)
- Plasticity: BDNF, CREB, Arc, DARPP-32 (learning/memory)
- Excitability: NMDA, AMPA, GABA (ion channels)
- Stress: FKBP5, NR3C1, CRH (HPA axis)

References:
- COMT Val158Met: Egan et al. (2001) PNAS
- 5-HTTLPR: Caspi et al. (2003) Science
- BDNF and LTP: Lu et al. (2008) Nat Rev Neurosci
- FKBP5 and stress: Binder (2009) Psychoneuroendocrinology
"""

import random
import copy
from typing import Dict, List, Optional, Tuple


# =============================================================================
# BRAIN GENE CLASS
# =============================================================================

class BrainGene:
    """
    Represents a single brain gene with realistic biological properties.
    
    Features:
    - Two alleles (diploid inheritance like real organisms)
    - Epigenetic state (methylation silences, acetylation activates)
    - Activity-dependent expression (learning triggers expression)
    - Realistic mutation with small perturbations
    """
    
    def __init__(
        self, 
        name: str, 
        gene_type: str,
        baseline: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        mutation_delta: float = 0.05
    ):
        self.name = name
        self.gene_type = gene_type  # 'receptor', 'enzyme', 'transcription_factor', 'channel', 'hormone'
        
        # DNA sequence (alleles) - two copies like diploid organisms
        # Random variation around baseline (±10%)
        self.allele_1 = self._clamp(baseline + random.uniform(-0.1, 0.1) * baseline, min_val, max_val)
        self.allele_2 = self._clamp(baseline + random.uniform(-0.1, 0.1) * baseline, min_val, max_val)
        
        # Epigenetic state (can change without DNA sequence change)
        self.methylation = 0.0           # 0=unmethylated (active), 1=fully methylated (silenced)
        self.histone_acetylation = 1.0   # 0=tight chromatin (silenced), 1=open (active)
        
        # Bounds for clamping
        self.min_val = min_val
        self.max_val = max_val
        self.mutation_delta = mutation_delta
        
        # Activity-dependent expression boost (temporary)
        self._activity_boost = 0.0
        self._boost_decay = 0.95
        
    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to valid range."""
        return max(min_val, min(max_val, value))
    
    @property
    def expression(self) -> float:
        """
        Calculate gene expression level (phenotype).
        Expression = (average of alleles) * epigenetic_modifiers + activity_boost
        """
        # Base expression from alleles (diploid average)
        base = (self.allele_1 + self.allele_2) / 2.0
        
        # Epigenetic modulation
        # Methylation silences, acetylation opens chromatin
        epigenetic_factor = (1.0 - self.methylation) * self.histone_acetylation
        
        # Activity-dependent boost
        total = base * epigenetic_factor + self._activity_boost
        
        return self._clamp(total, self.min_val, self.max_val)
    
    def boost_expression(self, amount: float):
        """Temporarily boost expression (activity-dependent)."""
        self._activity_boost = min(0.3, self._activity_boost + amount)
    
    def decay_activity_boost(self):
        """Decay the activity-dependent boost over time."""
        self._activity_boost *= self._boost_decay
    
    def methylate(self, amount: float = 0.1):
        """Increase methylation (silence gene)."""
        self.methylation = min(1.0, self.methylation + amount)
    
    def demethylate(self, amount: float = 0.1):
        """Decrease methylation (activate gene)."""
        self.methylation = max(0.0, self.methylation - amount)
    
    def mutate(self, mutation_rate: float) -> bool:
        """
        Potentially mutate this gene's alleles.
        Returns True if mutation occurred.
        """
        mutated = False
        
        # Mutate allele 1
        if random.random() < mutation_rate:
            delta = random.uniform(-self.mutation_delta, self.mutation_delta)
            self.allele_1 = self._clamp(self.allele_1 + delta, self.min_val, self.max_val)
            mutated = True
            
        # Mutate allele 2
        if random.random() < mutation_rate:
            delta = random.uniform(-self.mutation_delta, self.mutation_delta)
            self.allele_2 = self._clamp(self.allele_2 + delta, self.min_val, self.max_val)
            mutated = True
            
        return mutated
    
    def inherit_from(self, parent1: 'BrainGene', parent2: 'BrainGene'):
        """
        Inherit alleles from parents (Mendelian inheritance).
        Child gets one allele from each parent.
        """
        # Get one allele from each parent (random choice)
        self.allele_1 = random.choice([parent1.allele_1, parent1.allele_2])
        self.allele_2 = random.choice([parent2.allele_1, parent2.allele_2])
        
        # Epigenetic state partially inherited (transgenerational)
        self.methylation = (parent1.methylation + parent2.methylation) / 2.0 * 0.5  # Partial reset
        self.histone_acetylation = 1.0  # Reset to active
    
    def copy(self) -> 'BrainGene':
        """Create a deep copy of this gene."""
        new_gene = BrainGene(
            self.name, self.gene_type, 
            (self.allele_1 + self.allele_2) / 2,
            self.min_val, self.max_val, self.mutation_delta
        )
        new_gene.allele_1 = self.allele_1
        new_gene.allele_2 = self.allele_2
        new_gene.methylation = self.methylation
        new_gene.histone_acetylation = self.histone_acetylation
        new_gene._activity_boost = self._activity_boost
        return new_gene
    
    def __repr__(self) -> str:
        return f"BrainGene({self.name}: expr={self.expression:.3f}, a1={self.allele_1:.3f}, a2={self.allele_2:.3f})"


# =============================================================================
# MICRORNA CLASS (Post-Transcriptional Regulation)
# =============================================================================

class MicroRNA:
    """
    MicroRNA (miRNA) - small non-coding RNA that silences genes post-transcriptionally.
    
    In real biology:
    - ~60% of genes are regulated by miRNAs
    - miRNAs bind to mRNA and prevent translation
    - Fast and reversible regulation (unlike DNA methylation)
    - Activity-dependent (can be induced by neural activity)
    
    This provides a faster, more dynamic layer of gene control than epigenetics.
    """
    
    def __init__(self, name: str, target_genes: List[str], baseline_activity: float = 0.5):
        self.name = name
        self.target_genes = target_genes  # List of gene names this miRNA silences
        self.activity = baseline_activity  # 0 = inactive, 1 = fully active (silencing)
        self.decay_rate = 0.1  # How fast activity decays
    
    def silence_targets(self, gene_lookup: Dict[str, BrainGene], strength: float = None):
        """
        Apply silencing effect to target genes.
        Reduces their effective expression temporarily.
        """
        silence_strength = strength if strength else self.activity
        for gene_name in self.target_genes:
            if gene_name in gene_lookup:
                gene = gene_lookup[gene_name]
                # miRNA reduces the activity boost (fast, reversible)
                gene._activity_boost -= silence_strength * 0.1
                gene._activity_boost = max(-0.2, gene._activity_boost)
    
    def induce(self, amount: float = 0.2):
        """Increase miRNA activity (e.g., from neural activity)."""
        self.activity = min(1.0, self.activity + amount)
    
    def decay(self):
        """Natural decay of miRNA activity."""
        self.activity = max(0.0, self.activity - self.decay_rate)


# =============================================================================
# GENE REGULATORY NETWORK (GRN)
# =============================================================================

class GeneRegulatoryNetwork:
    """
    Gene Regulatory Network (GRN) - genes that regulate other genes.
    
    In real biology:
    - Transcription factors bind to promoters and activate/repress genes
    - Creates complex cascades and feedback loops
    - Enables homeostasis and oscillations
    
    This allows emergent, self-regulating gene expression patterns.
    """
    
    def __init__(self):
        # Regulatory connections: regulator_gene -> [(target_gene, effect)]
        # effect > 0 = activation, < 0 = repression
        self.regulations: Dict[str, List[Tuple[str, float]]] = {}
        
        # Define brain-specific regulatory network
        self._init_brain_grn()
    
    def _init_brain_grn(self):
        """Initialize brain-specific gene regulatory network."""
        self.regulations = {
            # CREB is a master transcription factor that activates BDNF, Arc
            'creb': [('bdnf', 0.3), ('arc', 0.4), ('npy', 0.2)],
            
            # BDNF feeds back to boost CREB (positive feedback for learning)
            'bdnf': [('creb', 0.2), ('gria1', 0.1)],
            
            # Arc regulates AMPA receptor trafficking
            'arc': [('gria1', -0.1), ('gria2', 0.1)],  # Arc internalizes GluA1
            
            # Stress genes interact
            'crh': [('fkbp5', 0.2), ('nr3c1', -0.1)],  # CRH induces FKBP5
            'fkbp5': [('nr3c1', -0.2)],  # FKBP5 reduces GR sensitivity
            
            # DARPP-32 affects dopamine signaling
            'darpp32': [('drd2', 0.1), ('comt', -0.05)],
            
            # Dopamine system cross-talk
            'drd2': [('dat1', 0.1)],
            
            # Inhibitory/excitatory balance
            'gabra1': [('scn1a', -0.1)],  # GABA inhibits excitability
            'grin2b': [('camk2', 0.2)],   # NMDA activates CaMKII
        }
    
    def apply_regulation(self, gene_lookup: Dict[str, BrainGene]):
        """
        Apply all regulatory effects based on current gene expression.
        This creates cascading effects where genes influence each other.
        """
        # Collect all effects first (to avoid order-dependent issues)
        effects: Dict[str, float] = {}
        
        for regulator, targets in self.regulations.items():
            if regulator in gene_lookup:
                regulator_expr = gene_lookup[regulator].expression
                
                for target, effect in targets:
                    if target in gene_lookup:
                        # Effect scaled by regulator expression
                        delta = effect * (regulator_expr - 0.5)  # Centered around 0.5
                        effects[target] = effects.get(target, 0) + delta
        
        # Apply all effects
        for gene_name, delta in effects.items():
            if gene_name in gene_lookup:
                gene = gene_lookup[gene_name]
                gene._activity_boost += delta * 0.1  # Scale down for stability
                gene._activity_boost = max(-0.3, min(0.3, gene._activity_boost))
    
    def add_regulation(self, regulator: str, target: str, effect: float):
        """Add a new regulatory connection."""
        if regulator not in self.regulations:
            self.regulations[regulator] = []
        self.regulations[regulator].append((target, effect))


# =============================================================================
# TELOMERE SYSTEM (Cellular Aging)
# =============================================================================

class TelomereSystem:
    """
    Telomeres - protective caps at chromosome ends that shorten with each division.
    
    In real biology:
    - Telomeres shorten with each cell division
    - When too short, cell enters senescence or dies
    - Telomerase can extend telomeres (active in stem cells, cancer)
    
    This provides natural aging and death pressure.
    """
    
    def __init__(self, initial_length: int = 10000):
        self.length = initial_length  # Base pairs (10,000 is normal for human cells)
        self.initial_length = initial_length
        self.shortening_per_tick = 1  # bp lost per tick
        self.critical_length = 2000   # Below this = senescence
        self.telomerase_active = False
    
    @property
    def health(self) -> float:
        """Telomere health as fraction (1.0 = full, 0.0 = critical)."""
        return max(0.0, (self.length - self.critical_length) / 
                   (self.initial_length - self.critical_length))
    
    @property
    def is_senescent(self) -> bool:
        """True if telomeres are critically short."""
        return self.length <= self.critical_length
    
    def tick(self):
        """Called each tick - shortens telomeres."""
        if not self.telomerase_active:
            self.length = max(0, self.length - self.shortening_per_tick)
    
    def extend(self, amount: int = 100):
        """Telomerase extends telomeres."""
        self.length = min(self.initial_length, self.length + amount)
    
    def inherit_from(self, parent1: 'TelomereSystem', parent2: 'TelomereSystem'):
        """Child inherits average telomere length from parents (with some reset)."""
        avg = (parent1.length + parent2.length) / 2
        # Offspring get longer telomeres than parents (partial reset)
        self.length = int(avg * 0.8 + self.initial_length * 0.2)


# =============================================================================
# DNA REPAIR SYSTEM
# =============================================================================

class DNARepairSystem:
    """
    DNA Repair - mechanisms that fix mutations.
    
    In real biology:
    - MMR (Mismatch Repair) fixes replication errors
    - BER (Base Excision Repair) fixes chemical damage
    - Proofreading during replication
    
    This provides genome stability and can reverse some mutations.
    """
    
    def __init__(self, efficiency: float = 0.8):
        self.efficiency = efficiency  # 0-1, how good at repair
        self.error_log: List[Tuple[str, float]] = []  # (gene_name, original_value)
    
    def attempt_repair(self, gene: BrainGene, mutation_rate: float) -> bool:
        """
        Attempt to repair a mutation before it becomes permanent.
        
        Returns True if repair was successful (mutation prevented).
        """
        # Repair chance depends on efficiency
        if random.random() < self.efficiency * 0.5:  # Repair can prevent mutation
            return True  # Mutation was repaired
        return False  # Mutation persists
    
    def log_error(self, gene_name: str, original_value: float):
        """Log an error for potential future repair."""
        self.error_log.append((gene_name, original_value))
        # Keep only last 100 errors
        if len(self.error_log) > 100:
            self.error_log.pop(0)
    
    def stress_damage(self, stress_level: float):
        """High stress damages DNA repair machinery."""
        damage = stress_level * 0.1
        self.efficiency = max(0.3, self.efficiency - damage)
    
    def recover(self, amount: float = 0.01):
        """Natural recovery of repair efficiency."""
        self.efficiency = min(1.0, self.efficiency + amount)


# =============================================================================
# TRANSPOSON (Jumping Genes)
# =============================================================================

class Transposon:
    """
    Transposons - mobile genetic elements that can move/duplicate.
    
    In real biology:
    - Can insert into new locations
    - Can duplicate genes
    - Source of genetic innovation (also instability)
    - Active in neurons! (L1 retrotransposons)
    
    This enables sudden large mutations and gene birth.
    """
    
    def __init__(self, activity: float = 0.01):
        self.activity = activity  # Probability of jumping per tick
        self.jump_count = 0
    
    def attempt_jump(self, gene_lookup: Dict[str, BrainGene]) -> Optional[str]:
        """
        Attempt to have a transposon jump event.
        
        Returns the name of the affected gene if jump occurred.
        """
        if random.random() > self.activity:
            return None
        
        # Pick a random gene to affect
        gene_names = list(gene_lookup.keys())
        if not gene_names:
            return None
        
        target = random.choice(gene_names)
        gene = gene_lookup[target]
        
        # Transposon insertion can:
        # 1. Disrupt gene (reduce expression)
        # 2. Enhance gene (if it lands in promoter)
        effect = random.choice([-0.2, -0.1, 0.1, 0.2])
        gene.allele_1 = gene._clamp(gene.allele_1 + effect, gene.min_val, gene.max_val)
        
        self.jump_count += 1
        return target
    
    def stress_activation(self, stress: float):
        """Stress activates transposons (real biology!)."""
        self.activity = min(0.1, self.activity + stress * 0.02)
    
    def calm(self):
        """Reduce transposon activity."""
        self.activity = max(0.001, self.activity * 0.9)


# =============================================================================
# ALTERNATIVE SPLICING
# =============================================================================

class AlternativeSplicing:
    """
    Alternative Splicing - one gene produces multiple protein isoforms.
    
    In real biology:
    - One gene can produce many different proteins
    - Tissue-specific isoforms (brain vs liver)
    - Activity-dependent splicing in neurons
    - BDNF has 11 different splice variants!
    
    This provides more phenotypic diversity from the same genes.
    """
    
    def __init__(self):
        # Gene -> list of available isoforms with relative weights
        self.splice_variants: Dict[str, List[Tuple[str, float]]] = {}
        self._init_brain_splicing()
    
    def _init_brain_splicing(self):
        """Initialize brain-specific splice variants."""
        self.splice_variants = {
            # BDNF has multiple isoforms affecting different brain regions
            'bdnf': [
                ('bdnf_I', 0.3),    # Hippocampus-enriched
                ('bdnf_IV', 0.4),   # Activity-dependent (main form)
                ('bdnf_VI', 0.2),   # Cortex-enriched
                ('bdnf_IX', 0.1),   # Constitutive
            ],
            # GRIN2 (NMDA) has developmental variants
            'grin2a': [
                ('grin2a_long', 0.6),   # Mature form
                ('grin2a_short', 0.4),  # Alternative C-terminus
            ],
            # DRD2 has long and short forms
            'drd2': [
                ('drd2_long', 0.5),    # Postsynaptic (inhibits neuron)
                ('drd2_short', 0.5),   # Presynaptic (autoreceptor)
            ],
        }
        
        # Currently active isoform per gene
        self.active_isoforms: Dict[str, str] = {}
        for gene, isoforms in self.splice_variants.items():
            # Start with most abundant
            self.active_isoforms[gene] = max(isoforms, key=lambda x: x[1])[0]
    
    def switch_isoform(self, gene: str, trigger: str = 'activity'):
        """Switch to a different isoform based on trigger."""
        if gene not in self.splice_variants:
            return
        
        isoforms = self.splice_variants[gene]
        
        if trigger == 'activity':
            # Activity favors certain isoforms (e.g., BDNF-IV)
            weights = [w * (1.2 if 'IV' in name or 'activity' in name.lower() else 1.0) 
                       for name, w in isoforms]
        elif trigger == 'stress':
            # Stress can shift isoform balance
            weights = [w * (0.8 if 'long' in name else 1.2) for name, w in isoforms]
        else:
            weights = [w for _, w in isoforms]
        
        # Weighted random choice
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for i, (name, _) in enumerate(isoforms):
            cumsum += weights[i]
            if r <= cumsum:
                self.active_isoforms[gene] = name
                break
    
    def get_isoform_modifier(self, gene: str) -> float:
        """Get expression modifier based on current isoform."""
        if gene not in self.active_isoforms:
            return 1.0
        
        isoform = self.active_isoforms[gene]
        
        # Different isoforms have different functional effects
        if 'short' in isoform:
            return 0.8  # Short isoforms often less active
        elif 'long' in isoform:
            return 1.2  # Long isoforms often more active
        elif 'IV' in isoform:
            return 1.3  # Activity-dependent forms highly active
        
        return 1.0


# =============================================================================
# GENOMIC IMPRINTING (Parent-of-Origin Effects)
# =============================================================================

class GenomicImprinting:
    """
    Genomic Imprinting - alleles silenced based on parent of origin.
    
    In real biology:
    - ~1% of genes are imprinted
    - Paternal or maternal allele is silenced
    - Examples: IGF2 (paternal), H19 (maternal)
    - Creates parent-of-origin effects
    
    This enables sexual dimorphism and parental conflict dynamics.
    """
    
    def __init__(self):
        # Imprinted genes: gene_name -> which parent's allele is silenced
        self.imprinted_genes: Dict[str, str] = {}
        self._init_brain_imprinting()
    
    def _init_brain_imprinting(self):
        """Initialize brain-relevant imprinted genes."""
        self.imprinted_genes = {
            # Paternally expressed (maternal silenced) - growth promoters
            'growth_factor': 'maternal',  # Like IGF2
            
            # Maternally expressed (paternal silenced) - growth inhibitors
            # Note: These are conceptual mappings to our existing genes
            
            # BDNF has some imprinting effects in real biology
            # 'bdnf' is biased toward paternal expression in some contexts
        }
    
    def apply_imprinting(self, gene: BrainGene, parent_origin: str = 'maternal') -> float:
        """
        Apply imprinting effect to gene expression.
        
        Args:
            gene: The gene to check
            parent_origin: Which parent's allele to check ('maternal' or 'paternal')
        
        Returns:
            Modified expression value
        """
        gene_name = gene.name.lower()
        
        if gene_name in self.imprinted_genes:
            silenced_parent = self.imprinted_genes[gene_name]
            
            # If this is the silenced parent's allele, reduce expression
            if parent_origin == silenced_parent:
                # Only use non-silenced allele
                if silenced_parent == 'maternal':
                    return gene.allele_1  # Use paternal
                else:
                    return gene.allele_2  # Use maternal
        
        # Normal diploid expression
        return (gene.allele_1 + gene.allele_2) / 2
    
    def is_imprinted(self, gene_name: str) -> bool:
        """Check if a gene is imprinted."""
        return gene_name.lower() in self.imprinted_genes


# =============================================================================
# CHROMOSOME CLASS (Linked Genes)
# =============================================================================

class Chromosome:
    """
    Collection of linked genes that tend to be inherited together.
    Crossover can occur within chromosomes at random points.
    """
    
    def __init__(self, name: str, genes: Dict[str, BrainGene]):
        self.name = name
        self.genes = genes  # Dict[gene_name, BrainGene]
    
    def crossover_with(self, other: 'Chromosome', crossover_rate: float = 0.5) -> 'Chromosome':
        """
        Create a child chromosome by crossing over with another.
        """
        child_genes = {}
        gene_names = list(self.genes.keys())
        
        # In this simplified model, we assume independent assortment for each gene
        # (or that crossover happens between every gene).
        # This ensures full mixing of alleles.
        
        for name in gene_names:
            child_gene = self.genes[name].copy()
            # Always inherit from both parents (Sexual Reproduction)
            child_gene.inherit_from(self.genes[name], other.genes[name])
            child_genes[name] = child_gene
        
        return Chromosome(self.name, child_genes)
    
    def mutate(self, mutation_rate: float):
        """Mutate all genes in this chromosome."""
        for gene in self.genes.values():
            gene.mutate(mutation_rate)
    
    def __repr__(self) -> str:
        return f"Chromosome({self.name}: {len(self.genes)} genes)"


# =============================================================================
# GENOME CLASS (Brain-Specific DNA)
# =============================================================================

class Genome:
    """
    Brain-Specific DNA Genome with real neuroscience genes.
    
    Organized into 4 chromosomes:
    1. Neurotransmitter genes (dopamine, serotonin)
    2. Plasticity genes (learning, memory)
    3. Excitability genes (ion channels)
    4. Stress genes (HPA axis)
    
    Also maintains structural genes for backward compatibility.
    """
    
    # --- Project-wide Constants (unchanged) ---
    NON_VISUAL_INPUT_SIZE = 256
    DEFAULT_POPULATION_SIZE = 5
    MAX_AGE = 50
    SAVE_INTERVAL = 100
    PERIPHERAL_RESOLUTION = 64
    
    # --- Brain Constants ---
    INPUT_BOOST_FACTOR = 20.0
    SURPRISE_THRESHOLD = 0.5
    REWARD_THRESHOLD = 100.0
    GROWTH_RATE_MULTIPLIER = 1.1
    MAX_HIDDEN_SIZE = 32768
    REPLAY_BUFFER_CAPACITY = 10000
    ACTION_COST_BASE = 0.01
    DISTILLATION_TEMPERATURE = 2.0
    DISTILLATION_LOSS_WEIGHT = 0.5
    JEPA_LOSS_WEIGHT = 0.1
    FISHER_SAMPLE_SIZE = 100
    CONFIDENCE_THRESHOLD = 0.7
    
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
        # =====================================================================
        # STRUCTURAL GENES (Backward Compatible)
        # =====================================================================
        self.species = random.choice(['efficient', 'powerful'])
        self.hidden_size = random.randint(16, 32) * 16
        self.target_hidden_size = 1000000
        self.growth_rate = 32
        self.sparsity = random.uniform(0.9, 0.99)
        self.rank = 256
        self.target_rank = 2048
        self.latent_dim = 256
        self.target_latent_dim = 4096
        self.usage_bias = 0.5
        self.max_ram_mb = 8192
        
        self.plasticity_hidden_size = random.choice([8, 16, 32, 64])
        self.plasticity_layers = random.choice([1, 2, 3])
        self.plasticity_activation = random.choice(['tanh', 'relu', 'sigmoid'])
        
        self.mutation_rate = random.uniform(0.01, 0.05)
        self.code_mutation_rate = random.uniform(0.05, 0.1)
        self.max_instructions = random.randint(4, 8)
        self.learning_rate = random.uniform(0.0001, 0.01)
        self.plasticity_coefficients = [random.uniform(-1, 1) for _ in range(4)]
        
        self.stamina_efficiency = random.uniform(0.8, 1.2)
        self.resilience = random.uniform(0.1, 0.5)
        
        self.input_boost_factor = random.uniform(10.0, 30.0)
        self.action_size = 72
        self.latent_adapter_dim = 4096
        self.fisher_sample_size = random.randint(50, 200)
        
        self.sprouting_threshold = random.uniform(0.3, 0.7)
        self.max_experts = 16
        
        # Meta-Data
        self.generation = 0
        self.parent_id = None
        
        # =====================================================================
        # BRAIN GENES (Neuroscience-Based)
        # =====================================================================
        self._init_brain_chromosomes()
    
    def _init_brain_chromosomes(self):
        """Initialize the 4 brain chromosomes with real neuroscience genes."""
        
        # --- Chromosome 1: Neurotransmitter Genes ---
        neurotransmitter_genes = {
            # Dopamine system
            'drd2': BrainGene('DRD2', 'receptor', 0.5, 0.1, 1.0, 0.05),      # D2 receptor density
            'drd4': BrainGene('DRD4', 'receptor', 0.5, 0.1, 1.0, 0.05),      # D4 receptor (novelty)
            'comt': BrainGene('COMT', 'enzyme', 0.5, 0.1, 1.0, 0.05),        # Dopamine breakdown (Val/Met)
            'dat1': BrainGene('DAT1', 'transporter', 0.5, 0.1, 1.0, 0.05),   # Dopamine transporter
            'th': BrainGene('TH', 'enzyme', 0.5, 0.2, 1.0, 0.03),            # Tyrosine hydroxylase
            
            # Serotonin system
            'sert': BrainGene('5-HTTLPR', 'transporter', 0.5, 0.1, 1.0, 0.05),  # Serotonin transporter
            'tph2': BrainGene('TPH2', 'enzyme', 0.5, 0.2, 1.0, 0.03),           # Tryptophan hydroxylase
            
            # Norepinephrine system  
            'dbh': BrainGene('DBH', 'enzyme', 0.5, 0.2, 1.0, 0.03),          # Dopamine beta-hydroxylase
            'adra2a': BrainGene('ADRA2A', 'receptor', 0.5, 0.2, 1.0, 0.03),  # Alpha-2 adrenergic
        }
        
        # --- Chromosome 2: Plasticity & Learning Genes ---
        plasticity_genes = {
            'bdnf': BrainGene('BDNF', 'neurotrophin', 0.6, 0.2, 1.0, 0.05),       # Brain-derived neurotrophic factor
            'creb': BrainGene('CREB', 'transcription_factor', 0.5, 0.2, 1.0, 0.03), # Memory consolidation
            'arc': BrainGene('Arc', 'immediate_early', 0.5, 0.1, 1.0, 0.05),       # Synaptic plasticity
            'darpp32': BrainGene('DARPP-32', 'signaling', 0.5, 0.2, 1.0, 0.03),    # Reward learning switch
            'camk2': BrainGene('CaMKII', 'kinase', 0.5, 0.2, 1.0, 0.03),           # LTP induction
            'ppp1r1b': BrainGene('PPP1R1B', 'phosphatase_reg', 0.5, 0.2, 1.0, 0.03), # PP-1 regulation
        }
        
        # --- Chromosome 3: Ion Channel / Excitability Genes ---
        excitability_genes = {
            # Glutamate (excitatory)
            'gria1': BrainGene('GluA1', 'channel', 0.5, 0.2, 1.0, 0.03),      # AMPA receptor subunit 1
            'gria2': BrainGene('GluA2', 'channel', 0.6, 0.3, 1.0, 0.03),      # AMPA receptor subunit 2 (Ca2+ permeability)
            'grin2a': BrainGene('GluN2A', 'channel', 0.5, 0.2, 1.0, 0.03),    # NMDA receptor subunit 2A
            'grin2b': BrainGene('GluN2B', 'channel', 0.5, 0.2, 1.0, 0.03),    # NMDA receptor subunit 2B
            
            # GABA (inhibitory)
            'gabra1': BrainGene('GABRA1', 'channel', 0.5, 0.2, 1.0, 0.03),    # GABA-A alpha-1
            'gabrb2': BrainGene('GABRB2', 'channel', 0.5, 0.2, 1.0, 0.03),    # GABA-A beta-2
            'gabbr1': BrainGene('GABBR1', 'receptor', 0.5, 0.2, 1.0, 0.03),   # GABA-B receptor
            
            # Voltage-gated
            'scn1a': BrainGene('SCN1A', 'channel', 0.5, 0.3, 0.8, 0.02),      # Sodium channel (excitability)
            'kcnq2': BrainGene('KCNQ2', 'channel', 0.5, 0.2, 1.0, 0.02),      # Potassium channel (repolarization)
        }
        
        # --- Chromosome 4: Stress Response Genes (HPA Axis) ---
        stress_genes = {
            'fkbp5': BrainGene('FKBP5', 'chaperone', 0.4, 0.1, 1.0, 0.05),    # GR sensitivity, PTSD risk
            'nr3c1': BrainGene('NR3C1', 'receptor', 0.5, 0.2, 1.0, 0.03),     # Glucocorticoid receptor
            'crh': BrainGene('CRH', 'hormone', 0.4, 0.1, 0.8, 0.03),          # Corticotropin-releasing hormone
            'npy': BrainGene('NPY', 'neuropeptide', 0.5, 0.2, 1.0, 0.03),     # Neuropeptide Y (stress resilience)
            'oxtr': BrainGene('OXTR', 'receptor', 0.5, 0.2, 1.0, 0.03),       # Oxytocin receptor (social bonding)
            'avpr1a': BrainGene('AVPR1A', 'receptor', 0.5, 0.2, 1.0, 0.03),   # Vasopressin receptor
        }
        
        # Create chromosomes
        self.chromosomes = {
            'neurotransmitter': Chromosome('neurotransmitter', neurotransmitter_genes),
            'plasticity': Chromosome('plasticity', plasticity_genes),
            'excitability': Chromosome('excitability', excitability_genes),
            'stress': Chromosome('stress', stress_genes),
        }
        
        # Create flat gene lookup for convenience
        self._gene_lookup: Dict[str, BrainGene] = {}
        for chrom in self.chromosomes.values():
            self._gene_lookup.update(chrom.genes)
        
        # =====================================================================
        # ADVANCED BIOLOGY SYSTEMS
        # =====================================================================
        
        # Gene Regulatory Network (genes regulate each other)
        self.grn = GeneRegulatoryNetwork()
        
        # MicroRNAs (fast, reversible gene silencing)
        self.mirnas = {
            'miR-132': MicroRNA('miR-132', ['creb', 'arc'], 0.3),           # Activity-induced, regulates plasticity
            'miR-134': MicroRNA('miR-134', ['bdnf', 'camk2'], 0.2),         # Dendritic plasticity
            'miR-124': MicroRNA('miR-124', ['comt', 'drd2'], 0.3),          # Brain-specific, dopamine
            'miR-29': MicroRNA('miR-29', ['fkbp5', 'nr3c1'], 0.2),          # Stress response
        }
        
        # Telomere system (cellular aging)
        self.telomeres = TelomereSystem(initial_length=10000)
        
        # DNA Repair (prevents some mutations)
        self.dna_repair = DNARepairSystem(efficiency=0.8)
        
        # Transposons (jumping genes - source of innovation)
        self.transposon = Transposon(activity=0.01)
        
        # Alternative Splicing (one gene -> multiple proteins)
        self.splicing = AlternativeSplicing()
        
        # Genomic Imprinting (parent-of-origin effects)
        self.imprinting = GenomicImprinting()
    
    # =========================================================================
    # GENE ACCESS (for modules to read)
    # =========================================================================
    
    def get_gene(self, name: str) -> Optional[BrainGene]:
        """Get a gene by name."""
        return self._gene_lookup.get(name)
    
    def get_expression(self, name: str, default: float = 0.5) -> float:
        """Get the expression level of a gene."""
        gene = self._gene_lookup.get(name)
        return gene.expression if gene else default
    
    # =========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # These map old gene names to new brain gene expressions
    # =========================================================================
    
    @property
    def dopamine_rpe_sensitivity(self) -> float:
        """COMT affects dopamine levels in prefrontal cortex."""
        return 1.0 - self.get_expression('comt', 0.5)  # Low COMT = high DA = high sensitivity
    
    @property
    def baseline_dopamine(self) -> float:
        """DRD2 and DAT1 affect baseline dopamine."""
        drd2 = self.get_expression('drd2', 0.5)
        dat1 = self.get_expression('dat1', 0.5)
        return 0.15 + (drd2 * 0.1) + ((1.0 - dat1) * 0.1)  # 0.15-0.35 range
    
    @property
    def serotonin_reuptake_efficiency(self) -> float:
        """5-HTTLPR (SERT) determines serotonin transporter efficiency."""
        return self.get_expression('sert', 0.5)
    
    @property
    def metabolic_rate(self) -> float:
        """Derived from overall gene expression activity."""
        return 0.03 + (self.get_expression('th', 0.5) * 0.04)  # 0.03-0.07
    
    @property
    def starvation_threshold(self) -> float:
        """NPY affects hunger/satiety signals."""
        npy = self.get_expression('npy', 0.5)
        return 15.0 + (npy * 10.0)  # 15-25 range
    
    @property
    def hijack_threshold(self) -> float:
        """FKBP5 and amygdala genes affect fear threshold."""
        fkbp5 = self.get_expression('fkbp5', 0.4)
        oxtr = self.get_expression('oxtr', 0.5)
        return 0.7 + (oxtr * 0.1) - (fkbp5 * 0.1)  # High oxytocin = calmer
    
    @property
    def fear_pain_weight(self) -> float:
        """CRH affects stress reactivity."""
        crh = self.get_expression('crh', 0.4)
        return 0.3 + (crh * 0.3)  # 0.3-0.6 range
    
    @property  
    def learning_rate_multiplier(self) -> float:
        """BDNF is critical for learning/LTP."""
        bdnf = self.get_expression('bdnf', 0.6)
        return 0.5 + (bdnf * 1.0)  # 0.5-1.5 multiplier
    
    @property
    def nmda_threshold(self) -> float:
        """NMDA receptor subunits affect learning threshold."""
        grin2a = self.get_expression('grin2a', 0.5)
        grin2b = self.get_expression('grin2b', 0.5)
        return 0.3 + ((grin2a + grin2b) / 2.0 * 0.4)  # 0.3-0.7
    
    @property
    def gaba_inhibition(self) -> float:
        """GABA genes affect inhibition strength."""
        gabra1 = self.get_expression('gabra1', 0.5)
        gabrb2 = self.get_expression('gabrb2', 0.5)
        return (gabra1 + gabrb2) / 2.0
    
    @property
    def cortisol_decay(self) -> float:
        """FKBP5 affects cortisol clearance - high FKBP5 = slow decay."""
        fkbp5 = self.get_expression('fkbp5', 0.4)
        return 0.99 - (fkbp5 * 0.05)  # 0.94-0.99 range
    
    @property
    def dopamine_go_modulation(self) -> float:
        """DRD2 affects Go pathway in basal ganglia."""
        drd2 = self.get_expression('drd2', 0.5)
        return 0.8 + (drd2 * 0.4)  # 0.8-1.2
    
    @property
    def system_1_base_threshold(self) -> float:
        """DARPP-32 affects automatic vs deliberate processing."""
        darpp32 = self.get_expression('darpp32', 0.5)
        return 0.6 + (darpp32 * 0.2)  # 0.6-0.8
    
    @property
    def temperature_min(self) -> float:
        """DRD4 affects exploration tendency."""
        drd4 = self.get_expression('drd4', 0.5)
        return 0.05 + ((1.0 - drd4) * 0.1)  # High DRD4 = lower temp (more exploration)
    
    @property
    def priority_alpha(self) -> float:
        """Arc affects experience replay prioritization."""
        arc = self.get_expression('arc', 0.5)
        return 0.5 + (arc * 0.2)  # 0.5-0.7
    
    # =========================================================================
    # ACTIVITY-DEPENDENT EXPRESSION
    # =========================================================================
    
    def on_reward(self, reward_magnitude: float):
        """Reward triggers BDNF, CREB, and dopamine-related gene expression."""
        if reward_magnitude > 0:
            self.get_gene('bdnf').boost_expression(reward_magnitude * 0.1)
            self.get_gene('creb').boost_expression(reward_magnitude * 0.05)
            self.get_gene('arc').boost_expression(reward_magnitude * 0.1)
            self.get_gene('darpp32').boost_expression(reward_magnitude * 0.05)
    
    def on_stress(self, stress_magnitude: float):
        """Stress triggers HPA axis genes and may silence plasticity genes."""
        if stress_magnitude > 0.5:
            self.get_gene('fkbp5').boost_expression(stress_magnitude * 0.15)
            self.get_gene('crh').boost_expression(stress_magnitude * 0.1)
            # Chronic stress can methylate (silence) BDNF
            if stress_magnitude > 0.8:
                self.get_gene('bdnf').methylate(0.01)
    
    def on_learning(self):
        """Learning event triggers immediate early genes."""
        self.get_gene('arc').boost_expression(0.1)
        self.get_gene('creb').boost_expression(0.05)
        self.get_gene('bdnf').boost_expression(0.05)
    
    def on_social_interaction(self, positive: bool = True):
        """Social interactions affect oxytocin/vasopressin genes."""
        if positive:
            self.get_gene('oxtr').boost_expression(0.1)
        else:
            self.get_gene('crh').boost_expression(0.05)
    
    def tick(self):
        """
        Update all biology systems for one timestep.
        
        Call this every tick to:
        1. Apply Gene Regulatory Network effects
        2. Apply miRNA silencing
        3. Age telomeres
        4. Recover DNA repair
        5. Check for transposon jumps
        6. Decay activity boosts
        """
        # 1. Gene Regulatory Network - genes affect each other
        self.grn.apply_regulation(self._gene_lookup)
        
        # 2. MicroRNAs - fast gene silencing
        for mirna in self.mirnas.values():
            mirna.silence_targets(self._gene_lookup)
            mirna.decay()
        
        # 3. Telomere aging
        self.telomeres.tick()
        
        # 4. DNA repair recovery
        self.dna_repair.recover()
        
        # 5. Transposon activity (rare)
        self.transposon.attempt_jump(self._gene_lookup)
        self.transposon.calm()  # Calm down over time
        
        # 6. Decay activity boosts on all genes
        self.decay_activity_boosts()
    
    @property
    def telomere_health(self) -> float:
        """Get telomere health (1.0 = young, 0.0 = senescent)."""
        return self.telomeres.health
    
    @property
    def is_senescent(self) -> bool:
        """True if telomeres are critically short (old age)."""
        return self.telomeres.is_senescent
    
    @property
    def biological_age(self) -> float:
        """Biological age as inferred from telomere length (0-1)."""
        return 1.0 - self.telomere_health
    
    def decay_activity_boosts(self):
        """Decay activity-dependent expression boosts on all genes."""
        for gene in self._gene_lookup.values():
            gene.decay_activity_boost()
    
    # =========================================================================
    # STRESS-INDUCED MUTAGENESIS
    # =========================================================================
    
    def calculate_mutation_rate(self, stress: float = 0.0, fitness: float = 1.0) -> float:
        """
        Calculate adaptive mutation rate based on stress and fitness.
        
        Biological basis:
        - Stress increases mutation rate (DNA repair impaired)
        - Low fitness triggers exploration (adaptive mutation)
        - FKBP5 genotype modulates stress response
        """
        fkbp5 = self.get_expression('fkbp5', 0.4)
        base_rate = self.mutation_rate
        
        # High FKBP5 = prolonged stress = more mutations
        stress_multiplier = 1.0 + (stress * fkbp5 * 2.0)
        
        # Low fitness = explore more (adaptive mutation)
        fitness_multiplier = 1.0 + max(0.0, (0.5 - fitness))
        
        return min(0.5, base_rate * stress_multiplier * fitness_multiplier)
    
    # =========================================================================
    # MUTATION
    # =========================================================================
    
    def mutate(self, stress: float = 0.0, fitness: float = 1.0):
        """
        Self-adaptive mutation with stress modulation.
        """
        effective_rate = self.calculate_mutation_rate(stress, fitness)
        
        # Mutate brain genes (chromosomes)
        for chromosome in self.chromosomes.values():
            chromosome.mutate(effective_rate)
        
        # Mutate structural genes (backward compat)
        self._mutate_structural_genes(effective_rate)
    
    def _mutate_structural_genes(self, mutation_rate: float):
        """Mutate the structural genes for backward compatibility."""
        if random.random() < mutation_rate:
            change = random.choice([-32, -16, 16, 32])
            self.hidden_size = max(128, min(self.target_hidden_size, self.hidden_size + change))
            self.hidden_size = (self.hidden_size // 16) * 16
            
        if random.random() < mutation_rate:
            self.sparsity = max(0.5, min(0.99, self.sparsity + random.uniform(-0.05, 0.05)))
            
        if random.random() < mutation_rate:
            change = random.choice([-16, 16, 32])
            self.rank = max(64, min(self.target_rank, self.rank + change))
            
        if random.random() < mutation_rate:
            change = random.choice([-32, 32, 64])
            self.latent_dim = max(128, min(self.target_latent_dim, self.latent_dim + change))

        if random.random() < mutation_rate:
            self.mutation_rate = max(0.001, min(0.2, self.mutation_rate * random.uniform(0.8, 1.2)))
            
        if random.random() < mutation_rate:
            self.plasticity_hidden_size = random.choice([8, 16, 32, 64])
            self.plasticity_layers = random.choice([1, 2, 3])
            
        if random.random() < mutation_rate:
            self.input_boost_factor = max(1.0, self.input_boost_factor + random.uniform(-2.0, 2.0))
            
        if random.random() < mutation_rate:
            self.sprouting_threshold = max(0.1, min(1.0, self.sprouting_threshold + random.uniform(-0.1, 0.1)))
    
    # =========================================================================
    # CROSSOVER
    # =========================================================================
    
    def crossover(self, other: 'Genome') -> 'Genome':
        """
        Create child genome through sexual reproduction with chromosome crossover.
        """
        child = Genome()
        
        # Crossover brain chromosomes (real genetics!)
        for chrom_name, chrom in self.chromosomes.items():
            other_chrom = other.chromosomes[chrom_name]
            child.chromosomes[chrom_name] = chrom.crossover_with(other_chrom)
        
        # Rebuild gene lookup
        child._gene_lookup = {}
        for chrom in child.chromosomes.values():
            child._gene_lookup.update(chrom.genes)
        
        # Crossover structural genes (backward compat)
        child.hidden_size = random.choice([self.hidden_size, other.hidden_size])
        child.sparsity = random.choice([self.sparsity, other.sparsity])
        child.rank = random.choice([self.rank, other.rank])
        child.latent_dim = random.choice([self.latent_dim, other.latent_dim])
        child.mutation_rate = (self.mutation_rate + other.mutation_rate) / 2
        child.stamina_efficiency = random.choice([self.stamina_efficiency, other.stamina_efficiency])
        child.plasticity_coefficients = [random.choice([a, b]) for a, b in 
                                          zip(self.plasticity_coefficients, other.plasticity_coefficients)]
        child.species = random.choice([self.species, other.species])
        child.plasticity_hidden_size = random.choice([self.plasticity_hidden_size, other.plasticity_hidden_size])
        child.plasticity_layers = random.choice([self.plasticity_layers, other.plasticity_layers])
        child.plasticity_activation = random.choice([self.plasticity_activation, other.plasticity_activation])
        child.sprouting_threshold = random.choice([self.sprouting_threshold, other.sprouting_threshold])
        child.max_experts = random.choice([self.max_experts, other.max_experts])
        
        child.generation = max(self.generation, other.generation) + 1
        
        return child
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def get_gene_summary(self) -> Dict[str, float]:
        """Get a summary of all gene expressions for logging."""
        return {name: gene.expression for name, gene in self._gene_lookup.items()}
    
    def __repr__(self) -> str:
        bdnf = self.get_expression('bdnf', 0.5)
        comt = self.get_expression('comt', 0.5)
        fkbp5 = self.get_expression('fkbp5', 0.4)
        return (f"Genome(Gen={self.generation}, Species={self.species}, H={self.hidden_size}, "
                f"BDNF={bdnf:.2f}, COMT={comt:.2f}, FKBP5={fkbp5:.2f})")
