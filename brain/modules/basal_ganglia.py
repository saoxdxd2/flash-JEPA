"""
NeuralBasalGanglia: Biologically-Inspired Action Selection Module

Implements the Direct/Indirect pathway model of the basal ganglia:
- Striatum (D1/D2): Learns state-action associations
- Direct Pathway (Go): D1 receptors → Facilitates rewarded actions
- Indirect Pathway (NoGo): D2 receptors → Suppresses unrewarded actions
- Dopamine TD-Learning: Modulates striatal plasticity

Reference: Frank, M.J. (2005). Dynamic dopamine modulation in the basal ganglia
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === CONFIGURATION CONSTANTS ===
# These parameters control the dopamine-modulated action selection

# Go/NoGo Pathway Modulation
DOPAMINE_GO_MODULATION = 1.0      # Base modulation for D1 (Go) pathway
DOPAMINE_NOGO_SUPPRESSION = 0.5  # How much dopamine suppresses D2 (NoGo)

# Basal Ganglia Weighting
BG_WEIGHT_MIN = 0.1              # Minimum influence of BG on final decision
BG_WEIGHT_MAX = 0.9              # Maximum influence of BG on final decision
BG_WEIGHT_DOPAMINE_SCALE = 0.5   # How much dopamine affects BG vs cortex balance
BG_WEIGHT_BASELINE = 0.25        # Baseline BG influence

# Temperature Parameters for Action Selection
TEMPERATURE_MIN = 0.1            # Minimum temperature (exploitation)
TEMPERATURE_DOPAMINE_SCALE = 0.5 # How much dopamine reduces temperature

# System 1 vs System 2 Thresholds
SYSTEM_1_BASE_THRESHOLD = 0.7     # Base confidence threshold for System 1
SYSTEM_1_DOPAMINE_REDUCTION = 0.2 # How much dopamine lowers threshold
SYSTEM_1_CONFIDENCE_REDUCTION = 0.1  # How much internal confidence lowers threshold
SURPRISE_THRESHOLD_FOR_SYSTEM1 = 0.15  # Maximum surprise for System 1 usage

# Legacy System 1 Threshold
LEGACY_SYSTEM_1_THRESHOLD = 0.8
LEGACY_SURPRISE_THRESHOLD = 0.2


class NeuralBasalGanglia(nn.Module):
    """
    Neural Basal Ganglia with Go/NoGo Pathways.
    
    Implements dopamine-modulated action selection with:
    - Striatal D1 (Direct/Go) pathway
    - Striatal D2 (Indirect/NoGo) pathway  
    - TD-error based learning
    - Value-guided exploration
    """
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # === STRIATUM (Input Nucleus) ===
        # Learns state representations that map to action preferences
        self.striatum_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # === DIRECT PATHWAY (Go - D1 Receptors) === 
        # High dopamine → *excites* D1 neurons → disinhibits thalamus → GO!
        # These neurons learn WHICH actions to take
        self.d1_go = nn.Linear(hidden_size, action_size)
        
        # === INDIRECT PATHWAY (NoGo - D2 Receptors) ===
        # Low dopamine → *excites* D2 neurons → inhibits thalamus → STOP!
        # These neurons learn WHICH actions to avoid
        self.d2_nogo = nn.Linear(hidden_size, action_size)
        
        # === SUBTHALAMIC NUCLEUS (STN) ===
        # Global inhibition / "pause and think" signal
        # Activated by high uncertainty/surprise
        self.stn = nn.Linear(hidden_size, 1)
        
        # === GPi/SNr (Output Nucleus) ===
        # Combines Go/NoGo signals into final gating
        self.gpi_gate = nn.Linear(action_size * 2, action_size)
        
        # === DOPAMINE PREDICTION (Internal Critic) ===
        # Predicts expected value for TD-error computation
        self.value_head = nn.Linear(hidden_size, 1)
        
        # === METACOGNITIVE CONFIDENCE ===
        # Decides System 1 vs System 2 usage
        self.confidence_head = nn.Linear(hidden_size + 4, 1)  # +4 for chemicals
        
        # Learning parameters
        self.td_error = 0.0  # Store last TD error for logging
        self.go_trace = None  # Eligibility trace for Go pathway
        self.nogo_trace = None  # Eligibility trace for NoGo pathway
        
    def forward(self, state, action_logits, dopamine, flash_info=None, 
                surprise=0.0, value_estimate=None, chemicals=None, greedy=False):
        """
        Performs action selection with Go/NoGo pathway modulation.
        
        Args:
            state: Current state representation (from TRM hidden or full input)
            action_logits: Proposed actions from System 2 (cortex/TRM)
            dopamine: Current dopamine level (0-1)
            flash_info: (flash_actions, confidence) from System 1 (reflex)
            surprise: JEPA prediction error (exploration signal)
            value_estimate: Critic value from TRM
            chemicals: Full chemical state [dopamine, serotonin, norepinephrine, cortisol]
            greedy: If True, pick argmax
            
        Returns:
            action: Selected action index
            meta: Dict with pathway activations and used_system
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        
        # Handle input size mismatch with projection
        if state.shape[1] != self.state_size:
            # Resize striatum encoder if needed (happens during brain growth)
            if not hasattr(self, '_resized') or self._resized != state.shape[1]:
                self._resize_input(state.shape[1])
                
        # === 1. STRIATAL ENCODING ===
        striatum = self.striatum_encoder(state)
        
        # === 2. GO PATHWAY (D1) ===
        # Dopamine *modulates* D1 directly (high DA → more Go)
        go_signal = self.d1_go(striatum) * (DOPAMINE_GO_MODULATION + dopamine)
        
        # === 3. NOGO PATHWAY (D2) ===  
        # Dopamine *inhibits* D2 (high DA → less NoGo)
        nogo_signal = self.d2_nogo(striatum) * (1.0 - dopamine * DOPAMINE_NOGO_SUPPRESSION)
        
        # === 4. STN (GLOBAL PAUSE) ===
        # Surprise triggers global inhibition ("hold on, let me think")
        stn_inhibition = torch.sigmoid(self.stn(striatum)) * surprise
        
        # === 5. GPi GATING ===
        # Combine Go and NoGo signals
        combined = torch.cat([go_signal, nogo_signal], dim=-1)
        gpi_output = self.gpi_gate(combined)
        
        # Apply STN inhibition (reduces all action preferences)
        gpi_output = gpi_output * (1.0 - stn_inhibition)
        
        # === 6. THALAMIC OUTPUT ===
        # Blend basal ganglia output with cortical proposals
        # High dopamine → trust BG, Low dopamine → trust cortex
        bg_weight = torch.clamp(
            torch.tensor(dopamine * BG_WEIGHT_DOPAMINE_SCALE + BG_WEIGHT_BASELINE), 
            BG_WEIGHT_MIN, 
            BG_WEIGHT_MAX
        )
        
        if action_logits.dim() == 1:
            action_logits = action_logits.unsqueeze(0)
            
        # Match shapes if needed
        if action_logits.shape[1] != self.action_size:
            action_logits = action_logits[:, :self.action_size]
            
        final_logits = (bg_weight * gpi_output) + ((1 - bg_weight) * action_logits)
        
        # === 7. VALUE PREDICTION ===
        predicted_value = self.value_head(striatum)
        
        # === 8. METACOGNITIVE GATING (System 1 vs 2) ===
        if chemicals is None:
            chemicals = torch.tensor([dopamine, 0.5, 0.2, 0.0], device=state.device)
        if chemicals.dim() == 1:
            chemicals = chemicals.unsqueeze(0)
            
        meta_input = torch.cat([striatum, chemicals], dim=-1)
        confidence = torch.sigmoid(self.confidence_head(meta_input))
        
        # Check Flash path (System 1)
        flash_actions, flash_confidence = None, 0.0
        if flash_info is not None:
            flash_actions, flash_confidence = flash_info
            if hasattr(flash_confidence, 'item'):
                flash_confidence = flash_confidence.mean().item()
                
        # System 1 threshold based on internal confidence
        system_1_threshold = (
            SYSTEM_1_BASE_THRESHOLD - 
            (dopamine * SYSTEM_1_DOPAMINE_REDUCTION) - 
            (confidence.item() * SYSTEM_1_CONFIDENCE_REDUCTION)
        )
        
        if flash_actions is not None and flash_confidence > system_1_threshold and surprise < SURPRISE_THRESHOLD_FOR_SYSTEM1:
            used_system = 1
            output_logits = flash_actions if flash_actions.dim() > 1 else flash_actions.unsqueeze(0)
        else:
            used_system = 2
            output_logits = final_logits
            
        # === 9. ACTION SELECTION ===
        logits_np = output_logits.detach().cpu().numpy()
        if logits_np.ndim > 1:
            logits_np = logits_np.flatten()
            
        if greedy:
            action = int(np.argmax(logits_np))
        else:
            # Temperature based on dopamine (high DA → more exploitation)
            temperature = max(TEMPERATURE_MIN, 1.0 - dopamine * TEMPERATURE_DOPAMINE_SCALE)
            exp_logits = np.exp((logits_np - np.max(logits_np)) / temperature)
            probs = exp_logits / (np.sum(exp_logits) + 1e-8)
            
            if np.isnan(probs).any():
                probs = np.ones(self.action_size) / self.action_size
                
            action = int(np.random.choice(self.action_size, p=probs))
            
        # Store eligibility traces for learning
        self.go_trace = go_signal.detach()
        self.nogo_trace = nogo_signal.detach()
        
        meta = {
            'used_system': used_system,
            'go_signal': go_signal.mean().item(),
            'nogo_signal': nogo_signal.mean().item(),
            'stn_inhibition': stn_inhibition.mean().item(),
            'confidence': confidence.item(),
            'predicted_value': predicted_value.mean().item()
        }
        
        return action, meta
    
    def learn(self, reward, next_value=None, gamma=0.99):
        """
        Updates Go/NoGo pathways using TD-error (dopamine burst/dip).
        
        This implements the dopamine learning rule:
        - Positive TD error → strengthen Go, weaken NoGo
        - Negative TD error → weaken Go, strengthen NoGo
        
        Args:
            reward: Actual reward received
            next_value: Predicted value of next state (for TD target)
            gamma: Discount factor
        """
        if self.go_trace is None or self.nogo_trace is None:
            return 0.0
            
        # TD Error = R + γ * V(s') - V(s)
        if next_value is None:
            next_value = 0.0
        elif hasattr(next_value, 'item'):
            next_value = next_value.item()
            
        # Use stored predicted value
        td_target = reward + gamma * next_value
        td_error = td_target  # Simplified: just use reward signal
        
        self.td_error = td_error
        
        # The TD error IS the dopamine signal
        # Positive → dopamine burst → strengthen Go
        # Negative → dopamine dip → strengthen NoGo
        
        # Learning happens via standard backprop in train_step
        # But we return the TD error for logging
        return td_error
    
    def _resize_input(self, new_state_size):
        """Resize input layer for brain growth compatibility."""
        print(f"NeuralBasalGanglia: Resizing input {self.state_size} -> {new_state_size}")
        device = next(self.parameters()).device
        
        # Create new encoder
        self.striatum_encoder = nn.Sequential(
            nn.Linear(new_state_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU()
        ).to(device)
        
        self.state_size = new_state_size
        self._resized = new_state_size
        
    def get_go_nogo_balance(self):
        """Returns the current Go/NoGo balance for monitoring."""
        if self.go_trace is None or self.nogo_trace is None:
            return 0.0
        return (self.go_trace.mean() - self.nogo_trace.mean()).item()


# === LEGACY WRAPPER ===
# Keep backward compatibility with existing code
class BasalGanglia:
    """
    Legacy wrapper for backward compatibility.
    Delegates to NeuralBasalGanglia when available.
    """
    def __init__(self, action_size, state_size=None):
        self.action_size = action_size
        self._neural_bg = None
        self._state_size = state_size
        
    def _ensure_neural(self, state_size):
        """Lazy initialization of neural BG."""
        if self._neural_bg is None:
            self._neural_bg = NeuralBasalGanglia(state_size, self.action_size)
            
    def gate_action(self, action_logits, dopamine, flash_info=None, surprise=0.0, greedy=False):
        """
        Legacy interface - falls back to simple gating if neural BG not initialized.
        """
        if self._neural_bg is not None:
            # Use neural pathway (requires state input)
            # But legacy interface doesn't provide state, so we fall back
            pass
            
        # Original simple gating logic
        flash_actions, flash_confidence = None, 0.0
        if flash_info is not None:
            flash_actions, flash_confidence = flash_info
            if hasattr(flash_confidence, 'item'):
                flash_confidence = flash_confidence.item()

        system_1_threshold = LEGACY_SYSTEM_1_THRESHOLD - (dopamine * SYSTEM_1_DOPAMINE_REDUCTION)
        
        if flash_actions is not None and flash_confidence > system_1_threshold and surprise < LEGACY_SURPRISE_THRESHOLD:
            final_logits = flash_actions
            used_system = 1
        else:
            final_logits = action_logits
            used_system = 2

        if hasattr(final_logits, 'detach'):
            final_logits = final_logits.detach().cpu().numpy()
        
        if final_logits.ndim > 1:
            final_logits = final_logits.flatten()
            
        if greedy:
            return np.argmax(final_logits), used_system
            
        temperature = max(TEMPERATURE_MIN, 1.0 - dopamine)
        exp_logits = np.exp((final_logits - np.max(final_logits)) / temperature)
        probs = exp_logits / np.sum(exp_logits)
        
        if np.isnan(probs).any():
            probs = np.ones(self.action_size) / self.action_size
            
        action = np.random.choice(self.action_size, p=probs)
        return action, used_system
