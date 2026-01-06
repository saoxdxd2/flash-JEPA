import numpy as np

class NeurotransmitterSystem:
    """
    Manages the agent's neurochemistry.
    Mimics the interaction of Dopamine, Serotonin, Norepinephrine, and Cortisol.
    """
    def __init__(self):
        # 1. Dopamine (The Driver) - Reward Prediction Error / Wanting
        # Range: 0.0 (Lethargy) to 1.0 (Mania)
        self.dopamine = 0.5 
        
        # 2. Serotonin (The Stabilizer) - Mood / Impulse Control
        # Range: 0.0 (Impulsive/Depressed) to 1.0 (Calm/Satiated)
        self.serotonin = 0.5
        
        # 3. Norepinephrine (The Volume Knob) - Arousal / Attention
        # Range: 0.0 (Drowsy) to 1.0 (Panic/Hyper-focus)
        self.norepinephrine = 0.2
        
        # 4. Cortisol (The Stressor) - Stress / Anxiety
        # Range: 0.0 (Relaxed) to 1.0 (Meltdown)
        self.cortisol = 0.0
        
        # Dynamics
        self.decay_rate = 0.01
        self.baseline_dopamine = 0.2
        
        # 5. Energy (ATP/Glucose)
        # Range: 0.0 (Collapse) to 100.0 (Full)
        self.energy = 100.0
        self.metabolic_rate = 0.05 # Constant energy drain per step
        
    def update(self, reward_prediction_error, surprise, pain, effort):
        """
        Updates chemical levels based on experience.
        
        Args:
            reward_prediction_error (float): Actual Reward - Predicted Value.
            surprise (float): 0.0 to 1.0 (Visual/Cognitive Chaos).
            pain (float): 0.0 to 1.0 (Error signals, damage).
            effort (float): 0.0 to 1.0 (Mental/Physical exertion).
        """
        # --- Metabolic Cost (The Cost of Living) ---
        # Energy drains constantly (Basal Metabolic Rate) + Effort
        energy_drain = self.metabolic_rate + (effort * 0.1)
        self.energy -= energy_drain
        self.energy = max(0.0, self.energy)
        
        # --- Starvation Stress ---
        # If energy is low, Cortisol spikes (Survival Instinct)
        starvation_stress = 0.0
        if self.energy < 20.0:
            starvation_stress = (20.0 - self.energy) / 20.0 # 0.0 to 1.0
            
        # --- Dopamine Dynamics ---
        # Spikes with positive RPE, dips with negative.
        # Suppressed by Cortisol.
        dopamine_delta = reward_prediction_error * 0.5
        suppression = self.cortisol * 0.5
        self.dopamine += dopamine_delta - suppression
        # Decay towards baseline
        self.dopamine += (self.baseline_dopamine - self.dopamine) * self.decay_rate
        
        # --- Norepinephrine Dynamics ---
        # Spikes with Surprise and Pain.
        # Decays rapidly.
        arousal_spike = (surprise * 0.5) + (pain * 0.5) + (starvation_stress * 0.2)
        self.norepinephrine += arousal_spike
        self.norepinephrine *= 0.95 # Fast decay
        
        # --- Cortisol Dynamics ---
        # Accumulates with Pain, Sustained Effort, and Starvation.
        # Decays slowly.
        stress_accumulation = (pain * 0.1) + (effort * 0.01) + (starvation_stress * 0.1)
        self.cortisol += stress_accumulation
        self.cortisol *= 0.99 # Slow decay
        
        # --- Serotonin Dynamics ---
        # Depleted by Stress (Cortisol).
        # Recovered by "Safety" (Low Surprise, Low Pain).
        if self.cortisol > 0.5:
            self.serotonin -= 0.01
        elif self.cortisol < 0.1 and surprise < 0.1:
            self.serotonin += 0.005
            
        # Clamp values
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        self.norepinephrine = np.clip(self.norepinephrine, 0.0, 1.0)
        self.cortisol = np.clip(self.cortisol, 0.0, 1.0)
        
    def feed(self, amount):
        """Replenishes energy."""
        self.energy += amount
        self.energy = min(100.0, self.energy)
        
    def get_state_vector(self):
        """Returns the 4-dim chemical state for the Brain."""
        return np.array([self.dopamine, self.serotonin, self.norepinephrine, self.cortisol])
