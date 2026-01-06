import numpy as np
import torch

class BasalGanglia:
    """
    The Action Gatekeeper.
    Selects actions based on Cortical proposals and Dopaminergic drive.
    Implements Metacognitive Gating (System 1 vs System 2).
    """
    def __init__(self, action_size):
        self.action_size = action_size
        
    def gate_action(self, action_logits, dopamine, flash_info=None, surprise=0.0, greedy=False):
        """
        Decides whether to trust System 1 (Flash) or System 2 (Deep).
        
        Args:
            action_logits (torch.Tensor): Proposed actions from System 2 (Full Hierarchy).
            dopamine (float): Current motivation level.
            flash_info (tuple): (flash_actions, confidence) from System 1.
            surprise (float): JEPA prediction error.
            greedy (bool): If True, pick the argmax.
            
        Returns:
            selected_action (int): The chosen action index.
            used_system (int): 1 for Flash, 2 for Deep.
        """
        # 1. Extract Flash Info
        flash_actions, flash_confidence = None, 0.0
        if flash_info is not None:
            flash_actions, flash_confidence = flash_info
            if hasattr(flash_confidence, 'item'):
                flash_confidence = flash_confidence.item()

        # 2. Metacognitive Gating (The "If-Else" Logic)
        # We trust System 1 if confidence is high AND surprise is low.
        # High dopamine (excitement) favors System 1.
        # High surprise (confusion) forces System 2.
        
        system_1_threshold = 0.8 - (dopamine * 0.2) # Lower threshold if excited
        
        if flash_actions is not None and flash_confidence > system_1_threshold and surprise < 0.2:
            # System 1 (Flash) Path
            final_logits = flash_actions
            used_system = 1
        else:
            # System 2 (Deep) Path
            final_logits = action_logits
            used_system = 2

        # 3. Action Selection
        if hasattr(final_logits, 'detach'):
            final_logits = final_logits.detach().cpu().numpy()
        
        if final_logits.ndim > 1:
            final_logits = final_logits.flatten()
            
        if greedy:
            return np.argmax(final_logits), used_system
            
        temperature = max(0.1, 1.0 - dopamine)
        exp_logits = np.exp((final_logits - np.max(final_logits)) / temperature)
        probs = exp_logits / np.sum(exp_logits)
        
        action = np.random.choice(self.action_size, p=probs)
        return action, used_system
