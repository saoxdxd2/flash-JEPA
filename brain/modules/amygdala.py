import numpy as np

class Amygdala:
    """
    The Fear Center.
    Processes raw sensory data for immediate threat detection.
    Can trigger a "Hijack" (Fight/Flight/Freeze) that overrides the Cortex.
    """
    def __init__(self):
        self.fear_level = 0.0
        self.aggression_level = 0.0
        self.hijack_threshold = 0.8
        
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
        
        # Fear is driven by Pain and Surprise, amplified by Cortisol and Norepinephrine
        self.fear_level = (pain * 0.5) + (surprise * 0.3) + (cortisol * 0.2) + (norepinephrine * 0.1)
        
        # Aggression is driven by Frustration (Low Dopamine, High Cortisol)
        self.aggression_level = (1.0 - dopamine) * 0.4 + cortisol * 0.6
        
        self.fear_level = np.clip(self.fear_level, 0.0, 1.0)
        self.aggression_level = np.clip(self.aggression_level, 0.0, 1.0)
        
        # Hijack Logic
        if self.fear_level > self.hijack_threshold:
            if self.fear_level > 0.95:
                # EXTREME TERROR: FREEZE
                return True, 9 # Force WAIT/FREEZE
            else:
                # HIGH FEAR: FLIGHT (Avoidance)
                # In this context, maybe "Back" or "Esc"
                return True, 10 # Assuming 10 is a safe/reset action
                
        if self.aggression_level > 0.9 and norepinephrine > 0.7:
            # FRUSTRATION ATTACK: FIGHT
            # Random clicking or typing
            import random
            return True, random.choice([1, 8, 9]) # Click, Enter, or Backspace
            
        return False, None
