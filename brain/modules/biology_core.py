"""
NeurotransmitterSystem: Biologically-Inspired Neurochemistry Simulation (Optimized)

Simulates the dynamic interaction of four key neurotransmitter systems:
- Dopamine: Reward prediction, motivation, wanting
- Serotonin: Mood regulation, impulse control
- Norepinephrine: Arousal, attention, alertness
- Cortisol: Stress response, survival instincts

Reference: Doya, K. (2002). Metalearning and neuromodulation
"""
import torch
import numpy as np

# === NEUROCHEMICAL CONSTANTS ===
# All values are biologically-inspired but tuned for real-time simulation

# Initial Chemical Levels (0.0 to 1.0 scale)
INITIAL_DOPAMINE = 0.5      # Neutral motivation
INITIAL_SEROTONIN = 0.5     # Neutral mood
INITIAL_NOREPINEPHRINE = 0.2  # Low baseline arousal
INITIAL_CORTISOL = 0.0      # No stress

# Baseline and Decay
BASELINE_DOPAMINE = 0.2     # Resting dopamine level
DECAY_RATE = 0.01           # General homeostatic decay

# Energy/Metabolism
INITIAL_ENERGY = 100.0      # Full energy
MAX_ENERGY = 100.0          # Energy cap
METABOLIC_RATE = 0.05       # Passive energy drain per step
EFFORT_ENERGY_COST = 0.1    # Energy cost scaling for effort
STARVATION_THRESHOLD = 20.0 # Energy level that triggers stress

# Dopamine Dynamics
DOPAMINE_RPE_SENSITIVITY = 0.5     # How much RPE affects dopamine
DOPAMINE_CORTISOL_SUPPRESSION = 0.3  # Cortisol dampens dopamine
SEROTONIN_BRAKE_THRESHOLD = 0.5    # Serotonin level that starts braking
SEROTONIN_BRAKE_STRENGTH = 0.5     # How much serotonin limits dopamine spikes

# Norepinephrine Dynamics
NOREPINEPHRINE_SURPRISE_WEIGHT = 0.5
NOREPINEPHRINE_PAIN_WEIGHT = 0.5
NOREPINEPHRINE_STARVATION_WEIGHT = 0.2
NOREPINEPHRINE_DECAY = 0.90  # Faster decay for attention spikes

# Cortisol Dynamics
CORTISOL_PAIN_WEIGHT = 0.1
CORTISOL_EFFORT_WEIGHT = 0.01
CORTISOL_STARVATION_WEIGHT = 0.1
CORTISOL_AROUSAL_AMPLIFICATION = 0.05
CORTISOL_DOPAMINE_RELIEF = 0.02
CORTISOL_DECAY = 0.98  # Slower decay for chronic stress

# Serotonin Dynamics
SEROTONIN_STRESS_THRESHOLD = 0.4   # Cortisol level that depletes serotonin
SEROTONIN_DEPLETION_RATE = 0.02
SEROTONIN_SAFETY_THRESHOLD = 0.2   # Cortisol level for recovery
SEROTONIN_RECOVERY_RATE = 0.01
SURPRISE_SAFETY_THRESHOLD = 0.1    # Low surprise enables serotonin recovery

# Energy Factor
MIN_ENERGY_FACTOR = 0.1  # Minimum chemical activity at zero energy


class NeurotransmitterSystem:
    """
    Manages the agent's neurochemistry (Optimized Tensor Implementation).
    Mimics the interaction of Dopamine, Serotonin, Norepinephrine, and Cortisol.
    """
    def __init__(self, genome=None, device='cpu'):
        self.device = device
        self.genome = genome
        
        # State Tensor: [Dopamine, Serotonin, Norepinephrine, Cortisol]
        self.state = torch.tensor([
            INITIAL_DOPAMINE,
            INITIAL_SEROTONIN,
            INITIAL_NOREPINEPHRINE,
            INITIAL_CORTISOL
        ], device=device, dtype=torch.float32)
        
        # Energy (Scalar)
        self.energy = INITIAL_ENERGY
        
        # Pre-fetch Genes / Constants into Tensors for fast computation
        self._init_parameters()
        
    def _init_parameters(self):
        """Pre-load genes into tensors to avoid getattr overhead in loop."""
        def g(name, default):
            val = getattr(self.genome, name, default) if self.genome else default
            return float(val)

        # 1. Metabolic
        self.metabolic_rate = g('metabolic_rate', METABOLIC_RATE)
        self.effort_energy_cost = g('effort_energy_cost', EFFORT_ENERGY_COST)
        self.starvation_threshold = g('starvation_threshold', STARVATION_THRESHOLD)
        
        # 2. Dopamine
        self.baseline_dopamine = g('baseline_dopamine', BASELINE_DOPAMINE)
        self.decay_rate = DECAY_RATE
        self.dopamine_params = torch.tensor([
            g('dopamine_rpe_sensitivity', DOPAMINE_RPE_SENSITIVITY),
            g('dopamine_cortisol_suppression', DOPAMINE_CORTISOL_SUPPRESSION),
            g('serotonin_brake_threshold', SEROTONIN_BRAKE_THRESHOLD),
            g('serotonin_brake_strength', SEROTONIN_BRAKE_STRENGTH)
        ], device=self.device)
        
        # 3. Norepinephrine
        self.norepinephrine_params = torch.tensor([
            g('norepinephrine_decay', NOREPINEPHRINE_DECAY),
            g('norepinephrine_surprise_weight', NOREPINEPHRINE_SURPRISE_WEIGHT),
            g('norepinephrine_pain_weight', NOREPINEPHRINE_PAIN_WEIGHT),
            g('norepinephrine_starvation_weight', NOREPINEPHRINE_STARVATION_WEIGHT)
        ], device=self.device)
        
        # 4. Cortisol
        self.cortisol_params = torch.tensor([
            g('cortisol_pain_weight', CORTISOL_PAIN_WEIGHT),
            g('cortisol_effort_weight', CORTISOL_EFFORT_WEIGHT),
            g('cortisol_starvation_weight', CORTISOL_STARVATION_WEIGHT),
            g('cortisol_arousal_amplification', CORTISOL_AROUSAL_AMPLIFICATION),
            g('cortisol_dopamine_relief', CORTISOL_DOPAMINE_RELIEF),
            g('cortisol_decay', CORTISOL_DECAY)
        ], device=self.device)
        
        # 5. Serotonin
        self.serotonin_params = torch.tensor([
            g('serotonin_stress_threshold', SEROTONIN_STRESS_THRESHOLD),
            g('serotonin_depletion_rate', SEROTONIN_DEPLETION_RATE),
            g('serotonin_safety_threshold', SEROTONIN_SAFETY_THRESHOLD),
            g('serotonin_recovery_rate', SEROTONIN_RECOVERY_RATE)
        ], device=self.device)

    @property
    def dopamine(self): return self.state[0].item()
    @dopamine.setter
    def dopamine(self, v): self.state[0] = v
    
    @property
    def serotonin(self): return self.state[1].item()
    @serotonin.setter
    def serotonin(self, v): self.state[1] = v
    
    @property
    def norepinephrine(self): return self.state[2].item()
    @norepinephrine.setter
    def norepinephrine(self, v): self.state[2] = v
    
    @property
    def cortisol(self): return self.state[3].item()
    @cortisol.setter
    def cortisol(self, v): self.state[3] = v

    def update(self, reward_prediction_error, surprise, pain, effort, fear=0.0, aggression=0.0):
        """
        Updates chemical levels based on experience (Vectorized).
        """
        # Ensure inputs are tensors or floats
        if isinstance(reward_prediction_error, torch.Tensor): reward_prediction_error = reward_prediction_error.item()
        
        # --- Metabolic Cost ---
        energy_drain = self.metabolic_rate + (effort * self.effort_energy_cost)
        self.energy = max(0.0, self.energy - energy_drain)
        
        # Starvation Stress
        starvation_stress = 0.0
        if self.energy < self.starvation_threshold:
            starvation_stress = (self.starvation_threshold - self.energy) / self.starvation_threshold
            
        # Unpack State for readability (views)
        dopamine = self.state[0]
        serotonin = self.state[1]
        norepinephrine = self.state[2]
        cortisol = self.state[3]
        
        # --- Dopamine Dynamics ---
        # params: [rpe_sens, cort_suppress, sero_thresh, sero_strength]
        dopamine_delta = reward_prediction_error * self.dopamine_params[0]
        serotonin_brake = torch.relu(serotonin - self.dopamine_params[2]) * self.dopamine_params[3]
        
        dopamine_new = dopamine + (dopamine_delta - (cortisol * self.dopamine_params[1]) - serotonin_brake)
        dopamine_new += (self.baseline_dopamine - dopamine_new) * self.decay_rate
        
        # --- Norepinephrine Dynamics ---
        # params: [decay, surprise_w, pain_w, starve_w]
        arousal_spike = (surprise * self.norepinephrine_params[1]) + \
                       (pain * self.norepinephrine_params[2]) + \
                       (starvation_stress * self.norepinephrine_params[3]) + \
                       (fear * 0.2) + (aggression * 0.3)
                       
        norepinephrine_new = (norepinephrine + arousal_spike) * self.norepinephrine_params[0]
        
        # --- Cortisol Dynamics ---
        # params: [pain_w, effort_w, starve_w, arousal_amp, dopa_relief, decay]
        stress_accumulation = (pain * self.cortisol_params[0]) + \
                             (effort * self.cortisol_params[1]) + \
                             (starvation_stress * self.cortisol_params[2]) + \
                             (fear * 0.3)
                             
        arousal_stress = norepinephrine_new * self.cortisol_params[3]
        reward_relief = dopamine_new * self.cortisol_params[4]
        
        cortisol_new = (cortisol + stress_accumulation + arousal_stress - reward_relief) * self.cortisol_params[5]
        
        # --- Serotonin Dynamics ---
        # params: [stress_thresh, deplet_rate, safety_thresh, recov_rate]
        # Vectorized conditional update using torch.where
        
        # Condition 1: High Cortisol -> Deplete Serotonin
        # mask_high_stress = cortisol_new > self.serotonin_params[0]
        # serotonin_new = torch.where(mask_high_stress, serotonin_new - self.serotonin_params[1] * cortisol_new, serotonin_new)
        
        # Condition 2: Low Cortisol & Low Surprise -> Recover Serotonin
        # mask_safety = (cortisol_new < self.serotonin_params[2]) & (surprise < SURPRISE_SAFETY_THRESHOLD)
        # serotonin_new = torch.where(mask_safety, serotonin_new + self.serotonin_params[3] * (1.0 - cortisol_new), serotonin_new)
        
        # Since we are doing in-place updates on a cloned tensor, we can use standard logic if we are careful about shapes
        # But torch.where is safer for gradients and batching
        
        stress_thresh = self.serotonin_params[0]
        deplet_rate = self.serotonin_params[1]
        safety_thresh = self.serotonin_params[2]
        recov_rate = self.serotonin_params[3]
        
        # Ensure surprise is a tensor for comparison
        if not torch.is_tensor(surprise):
             surprise = torch.tensor(surprise, device=self.device)
             
        # High Stress Depletion
        delta_deplete = deplet_rate * cortisol_new
        serotonin_new = torch.where(cortisol_new > stress_thresh, serotonin - delta_deplete, serotonin)
        
        # Safety Recovery
        delta_recover = recov_rate * (1.0 - cortisol_new)
        mask_safety = (cortisol_new < safety_thresh) & (surprise < SURPRISE_SAFETY_THRESHOLD)
        serotonin_new = torch.where(mask_safety, serotonin_new + delta_recover, serotonin_new)
            
        # --- Metabolic Constraint ---
        energy_factor = max(MIN_ENERGY_FACTOR, self.energy / MAX_ENERGY)
        dopamine_new *= energy_factor
        norepinephrine_new *= energy_factor
        
        # Update State Tensor
        self.state[0] = dopamine_new
        self.state[1] = serotonin_new
        self.state[2] = norepinephrine_new
        self.state[3] = cortisol_new
        
        # Clamp
        self.state.clamp_(0.0, 1.0)
        
    def feed(self, amount):
        """Replenishes energy."""
        self.energy = min(MAX_ENERGY, self.energy + amount)
        
    def get_state_vector(self):
        """Returns the 4-dim chemical state tensor."""
        return self.state.clone()
