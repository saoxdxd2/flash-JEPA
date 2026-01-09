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
        # Suppressed by Cortisol, modulated by Serotonin.
        dopamine_delta = reward_prediction_error * 0.5
        
        # Serotonin "Brake": High serotonin prevents dopamine spikes from becoming mania
        serotonin_brake = max(0.0, self.serotonin - 0.5) * 0.5
        
        self.dopamine += (dopamine_delta - (self.cortisol * 0.3) - serotonin_brake)
        # Decay towards baseline
        self.dopamine += (self.baseline_dopamine - self.dopamine) * self.decay_rate
        
        # --- Norepinephrine Dynamics ---
        # Spikes with Surprise and Pain.
        # Decays rapidly.
        arousal_spike = (surprise * 0.5) + (pain * 0.5) + (starvation_stress * 0.2)
        self.norepinephrine += arousal_spike
        self.norepinephrine *= 0.90 # Faster decay for attention spikes
        
        # --- Cortisol Dynamics ---
        # Accumulates with Pain, Sustained Effort, and Starvation.
        # Norepinephrine (Arousal) amplifies Cortisol.
        # Dopamine (Reward) suppresses Cortisol.
        stress_accumulation = (pain * 0.1) + (effort * 0.01) + (starvation_stress * 0.1)
        arousal_stress = self.norepinephrine * 0.05
        reward_relief = self.dopamine * 0.02
        
        self.cortisol += stress_accumulation + arousal_stress - reward_relief
        self.cortisol *= 0.98 # Slower decay for chronic stress
        
        # --- Serotonin Dynamics ---
        # Depleted by Stress (Cortisol).
        # Recovered by "Safety" (Low Surprise, Low Pain).
        if self.cortisol > 0.4:
            self.serotonin -= 0.02 * self.cortisol
        elif self.cortisol < 0.2 and surprise < 0.1:
            self.serotonin += 0.01 * (1.0 - self.cortisol)
            
        # --- Metabolic Constraint ---
        # Low energy reduces all chemical activity
        energy_factor = max(0.1, self.energy / 100.0)
        self.dopamine *= energy_factor
        self.norepinephrine *= energy_factor
        
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
