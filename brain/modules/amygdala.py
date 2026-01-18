"""
Amygdala: Fear Processing and Threat Detection Module

The amygdala is the brain's fear center, responsible for:
- Rapid threat assessment from sensory data
- Fight/Flight/Freeze responses (amygdala hijack)
- Emotional learning and fear conditioning

When the amygdala detects danger, it can bypass the cortex entirely,
triggering immediate survival responses before conscious processing.

Reference: LeDoux, J. (1996). The Emotional Brain
"""
import numpy as np
import random


# === AMYGDALA CONSTANTS ===
# These control threat detection and hijack behavior

# Fear Calculation Weights
FEAR_PAIN_WEIGHT = 0.5          # Contribution of pain to fear
FEAR_SURPRISE_WEIGHT = 0.3      # Contribution of surprise to fear  
FEAR_CORTISOL_WEIGHT = 0.2      # Contribution of cortisol to fear
FEAR_NOREPINEPHRINE_WEIGHT = 0.1  # Contribution of arousal to fear

# Aggression Calculation Weights
AGGRESSION_DOPAMINE_DEFICIT_WEIGHT = 0.4  # Low dopamine → frustration
AGGRESSION_CORTISOL_WEIGHT = 0.6          # High stress → aggression

# Hijack Thresholds
HIJACK_THRESHOLD = 0.8          # Fear level that triggers amygdala override
EXTREME_TERROR_THRESHOLD = 0.95 # Fear level that triggers FREEZE
FRUSTRATION_AGGRESSION_THRESHOLD = 0.9  # Aggression level for attack
FRUSTRATION_AROUSAL_THRESHOLD = 0.7     # Norepinephrine needed for attack

# Reflex Action IDs (should match the action space)
ACTION_FREEZE = 9       # Wait/do nothing (freeze response)
ACTION_FLIGHT = 10      # Escape/back away (flight response)
FIGHT_ACTIONS = [1, 8, 9]  # Random clicking/typing (fight response)


class Amygdala:
    """
    The Fear Center.
    Processes raw sensory data for immediate threat detection.
    Can trigger a "Hijack" (Fight/Flight/Freeze) that overrides the Cortex.
    
    When a genome is provided, fear/aggression thresholds are controlled by genes.
    """
    def __init__(self, genome=None):
        self.genome = genome
        self.fear_level = 0.0
        self.aggression_level = 0.0
        # Use genome hijack_threshold if available
        self.hijack_threshold = getattr(genome, 'hijack_threshold', HIJACK_THRESHOLD) if genome else HIJACK_THRESHOLD
    
    def _get_gene(self, gene_name, default):
        """Helper to get a gene value from genome or fall back to default."""
        if self.genome is not None:
            return getattr(self.genome, gene_name, default)
        return default
        
    def process(self, surprise, pain, chemicals):
        """
        Evaluates threat level and triggers hijacks.
        
        Args:
            surprise (float): Visual chaos.
            pain (float): Error signals.
            chemicals (np.array): [Dopamine, Serotonin, Norepinephrine, Cortisol]
            
        Returns:
            hijack (bool): True if the Amygdala is taking control.
            reflex_action (int or None): The action to force if hijacked.
        """
        dopamine, serotonin, norepinephrine, cortisol = chemicals
        
        # Get evolvable genes
        fear_pain_weight = self._get_gene('fear_pain_weight', FEAR_PAIN_WEIGHT)
        fear_surprise_weight = self._get_gene('fear_surprise_weight', FEAR_SURPRISE_WEIGHT)
        fear_cortisol_weight = self._get_gene('fear_cortisol_weight', FEAR_CORTISOL_WEIGHT)
        fear_norepinephrine_weight = self._get_gene('fear_norepinephrine_weight', FEAR_NOREPINEPHRINE_WEIGHT)
        aggression_dopamine_deficit_weight = self._get_gene('aggression_dopamine_deficit_weight', AGGRESSION_DOPAMINE_DEFICIT_WEIGHT)
        aggression_cortisol_weight = self._get_gene('aggression_cortisol_weight', AGGRESSION_CORTISOL_WEIGHT)
        extreme_terror_threshold = self._get_gene('extreme_terror_threshold', EXTREME_TERROR_THRESHOLD)
        frustration_aggression_threshold = self._get_gene('frustration_aggression_threshold', FRUSTRATION_AGGRESSION_THRESHOLD)
        frustration_arousal_threshold = self._get_gene('frustration_arousal_threshold', FRUSTRATION_AROUSAL_THRESHOLD)
        
        # Fear is driven by Pain and Surprise, amplified by Cortisol and Norepinephrine
        self.fear_level = (
            (pain * fear_pain_weight) + 
            (surprise * fear_surprise_weight) + 
            (cortisol * fear_cortisol_weight) + 
            (norepinephrine * fear_norepinephrine_weight)
        )
        
        # Aggression is driven by Frustration (Low Dopamine, High Cortisol)
        self.aggression_level = (
            (1.0 - dopamine) * aggression_dopamine_deficit_weight + 
            cortisol * aggression_cortisol_weight
        )
        
        self.fear_level = np.clip(self.fear_level, 0.0, 1.0)
        self.aggression_level = np.clip(self.aggression_level, 0.0, 1.0)
        
        # Hijack Logic
        if self.fear_level > self.hijack_threshold:
            if self.fear_level > extreme_terror_threshold:
                # EXTREME TERROR: FREEZE
                return True, ACTION_FREEZE
            else:
                # HIGH FEAR: FLIGHT (Avoidance)
                return True, ACTION_FLIGHT
                
        if self.aggression_level > frustration_aggression_threshold and norepinephrine > frustration_arousal_threshold:
            # FRUSTRATION ATTACK: FIGHT
            return True, random.choice(FIGHT_ACTIONS)
            
        return False, None

