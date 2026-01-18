"""
VM Agent

Connects EvolutionaryBrain to VM Session.
Maps brain decisions to mouse/keyboard actions.
"""

import numpy as np
import torch
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome


@dataclass
class VMAction:
    """An action that can be executed in the VM."""
    name: str
    action_type: str  # 'mouse_move', 'click', 'type', 'key', 'scroll'
    params: Dict[str, Any] = field(default_factory=dict)


# Action mapping: brain action index -> VM action
# Based on the 72 action space from Genome.action_size
ACTION_MAP = {
    # Mouse movement (actions 0-35: grid of 6x6 positions)
    **{i: VMAction(f"move_{i}", "mouse_move", 
                   {"x": (i % 6) * 170 + 85,  # 1024/6 = ~170
                    "y": (i // 6) * 128 + 64})  # 768/6 = ~128
       for i in range(36)},
    
    # Mouse clicks (actions 36-38)
    36: VMAction("left_click", "click", {"button": "left"}),
    37: VMAction("right_click", "click", {"button": "right"}),
    38: VMAction("double_click", "double_click", {}),
    
    # Scroll (actions 39-40)
    39: VMAction("scroll_up", "scroll", {"clicks": 3}),
    40: VMAction("scroll_down", "scroll", {"clicks": -3}),
    
    # Keyboard keys (actions 41-55)
    41: VMAction("enter", "key", {"key": "enter"}),
    42: VMAction("tab", "key", {"key": "tab"}),
    43: VMAction("escape", "key", {"key": "escape"}),
    44: VMAction("backspace", "key", {"key": "backspace"}),
    45: VMAction("space", "key", {"key": "space"}),
    46: VMAction("up", "key", {"key": "up"}),
    47: VMAction("down", "key", {"key": "down"}),
    48: VMAction("left", "key", {"key": "left"}),
    49: VMAction("right", "key", {"key": "right"}),
    50: VMAction("ctrl_c", "hotkey", {"keys": ["ctrl", "c"]}),
    51: VMAction("ctrl_v", "hotkey", {"keys": ["ctrl", "v"]}),
    52: VMAction("ctrl_a", "hotkey", {"keys": ["ctrl", "a"]}),
    53: VMAction("ctrl_s", "hotkey", {"keys": ["ctrl", "s"]}),
    54: VMAction("alt_tab", "hotkey", {"keys": ["alt", "tab"]}),
    55: VMAction("ctrl_t", "hotkey", {"keys": ["ctrl", "t"]}),  # New tab
    
    # Common characters (actions 56-71)
    **{56 + i: VMAction(f"type_{chr(ord('a') + i)}", "type", {"text": chr(ord('a') + i)})
       for i in range(16)},
}


class VMAgent:
    """
    Wraps EvolutionaryBrain for VM interaction.
    
    Responsibilities:
    - Capture screen → brain input
    - Brain decision → VM action
    - Energy/reward management
    - Gene expression updates
    """
    
    def __init__(
        self, 
        agent_id: int,
        name: str,
        brain: EvolutionaryBrain,
        session,  # AgentSession from session_manager
    ):
        self.agent_id = agent_id
        self.name = name
        self.brain = brain
        self.session = session
        self.genome = brain.genome
        
        # State tracking
        self.last_action = None
        self.last_action_time = None
        self.action_count = 0
        self.reward_total = 0.0
        
        # Screen processing
        self.screen_size = (session.screen_width, session.screen_height)
    
    def process_screen(self, screen: np.ndarray) -> torch.Tensor:
        """
        Convert screen capture to brain input tensor.
        
        Args:
            screen: numpy array (H, W, 3) RGB image
        
        Returns:
            Input tensor for brain.decide()
        """
        # Resize to expected input size
        from PIL import Image
        
        # Convert to PIL, resize, back to numpy
        img = Image.fromarray(screen)
        
        # Create foveal (center) and peripheral (full) views
        cx, cy = self.screen_size[0] // 2, self.screen_size[1] // 2
        foveal = img.crop((cx - 64, cy - 64, cx + 64, cy + 64))
        peripheral = img.resize((64, 64))
        
        # Process through brain's visual cortex
        foveal_tensor = torch.from_numpy(np.array(foveal)).permute(2, 0, 1).float() / 255.0
        peripheral_tensor = torch.from_numpy(np.array(peripheral)).permute(2, 0, 1).float() / 255.0
        
        # Get latent representations
        foveal_latent = self.brain.retina.encode_foveal(foveal_tensor.unsqueeze(0))
        peripheral_latent = self.brain.retina.encode_peripheral(peripheral_tensor.unsqueeze(0))
        
        # Empty semantic for now (could be from OCR or other sources)
        semantic_latent = torch.zeros(1, self.genome.latent_dim, device=self.brain.device)
        
        # Build full input
        input_tensor = self.brain.get_input_vector(
            foveal_latent, 
            peripheral_latent, 
            semantic_latent
        )
        
        return input_tensor
    
    def decide(self, screen: np.ndarray) -> VMAction:
        """
        Get brain decision and return VM action.
        
        Args:
            screen: Current screen capture
        
        Returns:
            VMAction to execute
        """
        # Convert screen to input tensor
        input_tensor = self.process_screen(screen)
        
        # Get brain decision
        action_idx, action_probs = self.brain.decide(input_tensor)
        
        # Map to VM action
        action = ACTION_MAP.get(action_idx, ACTION_MAP[0])  # Default to move
        
        # Track
        self.last_action = action
        self.last_action_time = time.time()
        self.action_count += 1
        
        return action
    
    def execute_action(self, action: VMAction):
        """
        Execute a VM action through the session.
        """
        try:
            if action.action_type == "mouse_move":
                self.session.move_mouse(action.params["x"], action.params["y"])
            
            elif action.action_type == "click":
                x, y = self.session.screen_width // 2, self.session.screen_height // 2
                self.session.click(x, y, action.params.get("button", "left"))
            
            elif action.action_type == "double_click":
                x, y = self.session.screen_width // 2, self.session.screen_height // 2
                self.session.double_click(x, y)
            
            elif action.action_type == "type":
                self.session.type_text(action.params["text"])
            
            elif action.action_type == "key":
                self.session.press_key(action.params["key"])
            
            elif action.action_type == "hotkey":
                self.session.hotkey(*action.params["keys"])
            
            elif action.action_type == "scroll":
                self.session.scroll(action.params["clicks"])
        
        except Exception as e:
            print(f"Action execution error: {e}")
    
    def tick(self) -> Tuple[VMAction, float]:
        """
        Run one agent tick: sense → decide → act.
        
        Returns:
            (action_taken, energy_cost)
        """
        # 1. Capture screen
        screen = self.session.get_screen()
        
        # 2. Decide action
        action = self.decide(screen)
        
        # 3. Execute action
        self.execute_action(action)
        
        # 4. Calculate energy cost (different actions cost differently)
        energy_cost = self._calculate_energy_cost(action)
        
        # 5. Update gene expression based on state
        self._update_gene_expression()
        
        return action, energy_cost
    
    def _calculate_energy_cost(self, action: VMAction) -> float:
        """Calculate energy cost for an action."""
        base_cost = 0.01
        
        # Movement is cheap
        if action.action_type == "mouse_move":
            return base_cost * 0.5
        
        # Clicking is medium
        if action.action_type in ("click", "double_click"):
            return base_cost * 1.0
        
        # Typing is expensive (thinking required)
        if action.action_type == "type":
            return base_cost * 2.0
        
        return base_cost
    
    def _update_gene_expression(self):
        """Update gene expression based on current state."""
        # Decay activity boosts
        self.genome.decay_activity_boosts()
        
        # If we got a recent reward, trigger reward genes
        # (This would be called from external reward signals)
    
    def receive_reward(self, reward: float):
        """Receive reward from environment."""
        self.reward_total += reward
        
        if reward > 0:
            # Trigger reward-related gene expression
            self.genome.on_reward(reward)
        elif reward < -0.5:
            # Negative reward = stress
            self.genome.on_stress(abs(reward))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "id": self.agent_id,
            "name": self.name,
            "action_count": self.action_count,
            "reward_total": self.reward_total,
            "bdnf": self.genome.get_expression("bdnf"),
            "stress_level": self.genome.get_expression("fkbp5"),
        }
