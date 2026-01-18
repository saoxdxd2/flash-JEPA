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
import torch
import torch.nn as nn
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


class Amygdala(nn.Module):
    """
    The Fear Center.
    Processes raw sensory data for immediate threat detection.
    Can trigger a "Hijack" (Fight/Flight/Freeze) that overrides the Cortex.
    
    When a genome is provided, fear/aggression thresholds are controlled by genes.
    """
    def __init__(self, genome=None, device='cpu'):
        super().__init__()
        self.genome = genome
        self.device = device
        
        # Initialize state as tensors
        self.register_buffer('fear_level', torch.tensor(0.0, device=device))
        self.register_buffer('aggression_level', torch.tensor(0.0, device=device))
        
        # Pre-fetch genes into tensors for fast access
        self._init_parameters()
        
    def _init_parameters(self):
        """Load genes into tensor buffers."""
        def g(name, default):
            val = getattr(self.genome, name, default) if self.genome else default
            return float(val)

        self.register_buffer('hijack_threshold', torch.tensor(g('hijack_threshold', HIJACK_THRESHOLD), device=self.device))
        
        # Weights
        self.register_buffer('w_fear_pain', torch.tensor(g('fear_pain_weight', FEAR_PAIN_WEIGHT), device=self.device))
        self.register_buffer('w_fear_surprise', torch.tensor(g('fear_surprise_weight', FEAR_SURPRISE_WEIGHT), device=self.device))
        self.register_buffer('w_fear_cortisol', torch.tensor(g('fear_cortisol_weight', FEAR_CORTISOL_WEIGHT), device=self.device))
        self.register_buffer('w_fear_norepinephrine', torch.tensor(g('fear_norepinephrine_weight', FEAR_NOREPINEPHRINE_WEIGHT), device=self.device))
        
        self.register_buffer('w_agg_dopa_deficit', torch.tensor(g('aggression_dopamine_deficit_weight', AGGRESSION_DOPAMINE_DEFICIT_WEIGHT), device=self.device))
        self.register_buffer('w_agg_cortisol', torch.tensor(g('aggression_cortisol_weight', AGGRESSION_CORTISOL_WEIGHT), device=self.device))
        
        # Thresholds
        self.register_buffer('th_terror', torch.tensor(g('extreme_terror_threshold', EXTREME_TERROR_THRESHOLD), device=self.device))
        self.register_buffer('th_frust_agg', torch.tensor(g('frustration_aggression_threshold', FRUSTRATION_AGGRESSION_THRESHOLD), device=self.device))
        self.register_buffer('th_frust_arousal', torch.tensor(g('frustration_arousal_threshold', FRUSTRATION_AROUSAL_THRESHOLD), device=self.device))

    def process(self, surprise, pain, chemicals):
        """
        Evaluates threat level and triggers hijacks.
        
        Args:
            surprise (Tensor): Visual chaos [Batch] or scalar.
            pain (Tensor): Error signals [Batch] or scalar.
            chemicals (Tensor): [Batch, 4] -> [Dopamine, Serotonin, Norepinephrine, Cortisol]
            
        Returns:
            hijack (Tensor): Boolean tensor [Batch], True if hijacked.
            reflex_action (Tensor): Int tensor [Batch], action ID or -1 if no hijack.
        """
        # Ensure inputs are tensors
        if not torch.is_tensor(surprise):
            surprise = torch.tensor(surprise, device=self.device)
        if not torch.is_tensor(pain):
            pain = torch.tensor(pain, device=self.device)
            
        # Handle batching
        if chemicals.dim() == 1:
            chemicals = chemicals.unsqueeze(0) # [1, 4]
        
        batch_size = chemicals.shape[0]
        
        # Extract chemicals
        dopamine = chemicals[:, 0]
        serotonin = chemicals[:, 1]
        norepinephrine = chemicals[:, 2]
        cortisol = chemicals[:, 3]
        
        # 1. Calculate Fear
        # Fear = Pain*w + Surprise*w + Cortisol*w + Norepinephrine*w
        self.fear_level = (
            (pain * self.w_fear_pain) + 
            (surprise * self.w_fear_surprise) + 
            (cortisol * self.w_fear_cortisol) + 
            (norepinephrine * self.w_fear_norepinephrine)
        )
        self.fear_level = torch.clamp(self.fear_level, 0.0, 1.0)
        
        # 2. Calculate Aggression
        # Aggression = (1 - Dopamine)*w + Cortisol*w
        self.aggression_level = (
            (1.0 - dopamine) * self.w_agg_dopa_deficit + 
            cortisol * self.w_agg_cortisol
        )
        self.aggression_level = torch.clamp(self.aggression_level, 0.0, 1.0)
        
        # 3. Hijack Logic (Vectorized)
        hijack_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        reflex_actions = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        
        # Condition A: Fear Hijack
        fear_hijack = self.fear_level > self.hijack_threshold
        
        # Sub-condition: Terror (Freeze) vs Flight
        terror_mask = fear_hijack & (self.fear_level > self.th_terror)
        flight_mask = fear_hijack & (~terror_mask)
        
        reflex_actions[terror_mask] = ACTION_FREEZE
        reflex_actions[flight_mask] = ACTION_FLIGHT
        hijack_mask |= fear_hijack
        
        # Condition B: Aggression Hijack (Frustration)
        # Only if not already hijacked by fear (Fear takes precedence)
        agg_hijack = (self.aggression_level > self.th_frust_agg) & (norepinephrine > self.th_frust_arousal)
        agg_hijack = agg_hijack & (~hijack_mask)
        
        if agg_hijack.any():
            # Random fight action (vectorized random choice is tricky, using simple approach)
            # Since FIGHT_ACTIONS is small, we can just pick one randomly for the whole batch
            # or generate random indices.
            random_fight = torch.tensor(random.choice(FIGHT_ACTIONS), device=self.device)
            reflex_actions[agg_hijack] = random_fight
            hijack_mask |= agg_hijack
            
        return hijack_mask, reflex_actions

