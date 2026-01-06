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
        
    def process(self, surprise, pain, cortisol):
        """
        Evaluates threat level.
        
        Args:
            surprise (float): Visual chaos.
            pain (float): Error signals.
            cortisol (float): Current stress level.
            
        Returns:
            hijack (bool): True if the Amygdala is taking control.
            reflex_action (int or None): The action to force if hijacked.
        """
        # Fear is driven by Pain and Surprise, amplified by Cortisol
        self.fear_level = (pain * 0.6) + (surprise * 0.4) + (cortisol * 0.2)
        self.fear_level = np.clip(self.fear_level, 0.0, 1.0)
        
        # Hijack Logic
        if self.fear_level > self.hijack_threshold:
            # PANIC RESPONSE
            # 10 = Sleep/Freeze, 9 = Wait/Hesitate
            # In a real desktop, this might be ALT+F4 or ESC
            return True, 9 # Force WAIT/FREEZE
            
        return False, None
