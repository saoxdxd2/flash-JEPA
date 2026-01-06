from models.liquid_vectorized import VectorizedLiquidGraph
from models.neuromodulated_holographic import NeuromodulatedHolographicBrain

import torch.nn as nn

class ModularBrain(nn.Module):
    """
    A Hierarchical Brain composed of specialized sub-modules.
    
    Structure:
    1. Visual Cortex (LiquidGraph): Retina Input -> Concept Vector (Bus)
    2. Motor Cortex (LiquidGraph): Concept Vector + Meta Inputs -> Actions
    
    Evolution determines the 'Language' of the Concept Vector.
    """
    def __init__(self, input_size, hidden_size, output_size, genome=None, use_neuromodulated=True, use_sparse=False):
        super().__init__()
        # Configuration
        self.input_size = input_size
        self.output_size = output_size
        self.use_neuromodulated = use_neuromodulated
        self.use_sparse = use_sparse
        
        # Architecture Hyperparameters
        # We split the hidden_size budget between the two cortices
        self.visual_hidden = int(hidden_size * 0.4)
        self.motor_hidden = int(hidden_size * 0.6)
        
        # Communication Bus Size (Bottleneck)
        self.bus_size = 256 
        
        if use_sparse:
            from models.liquid_sparse_vectorized import SparseVectorizedLiquidGraph
            GraphClass = SparseVectorizedLiquidGraph
        elif use_neuromodulated:
            GraphClass = NeuromodulatedHolographicBrain
        else:
            GraphClass = VectorizedLiquidGraph
            
        print(f"ModularBrain: Initialized with GraphClass: {GraphClass.__name__}")
        print(f"ModularBrain: use_neuromodulated={use_neuromodulated}")
        
        # 1. Visual Cortex
        # Input: Retina (input_size)
        # Output: Bus (bus_size)
        sparsity = getattr(genome, 'sparsity', 0.8) if genome else 0.8
        
        self.visual_cortex = GraphClass(
            input_size=input_size,
            hidden_size=self.visual_hidden,
            output_size=self.bus_size,
            genome=genome,
            sparsity=sparsity
        )
        
        # 2. Motor Cortex
        # Input: Bus (bus_size)
        # Output: Actions (output_size)
        self.motor_cortex = GraphClass(
            input_size=self.bus_size,
            hidden_size=self.motor_hidden,
            output_size=output_size,
            genome=genome,
            sparsity=sparsity
        )

    def forward(self, input_vector, dt=0.1, reward=0.0, chemicals=None, train_internal_rl=True):
        # 1. Visual Cortex
        # Returns: bus, value, energy, flash_info
        res_v = self.visual_cortex(input_vector, dt=dt, reward=reward, chemicals=chemicals, train_internal_rl=train_internal_rl)
        bus, _, v_energy, v_flash = res_v if len(res_v) == 4 else (*res_v, None)
        
        # 2. Motor Cortex
        # Returns: actions, value, energy, flash_info
        res_m = self.motor_cortex(bus, dt=dt, reward=reward, chemicals=chemicals, train_internal_rl=train_internal_rl)
        actions, params, m_energy, m_flash = res_m if len(res_m) == 4 else (*res_m, None)
        
        total_energy = v_energy + m_energy
        # We return the motor cortex's flash info as the primary one for action selection
        return actions, params, total_energy, m_flash

    def reset_state(self):
        """Resets the hidden state of the brain."""
        if hasattr(self.visual_cortex, 'reset_state'):
            self.visual_cortex.reset_state()
        if hasattr(self.motor_cortex, 'reset_state'):
            self.motor_cortex.reset_state()

    def detach_state(self):
        """Detaches the hidden state of the brain."""
        if hasattr(self.visual_cortex, 'detach_state'):
            self.visual_cortex.detach_state()
        if hasattr(self.motor_cortex, 'detach_state'):
            self.motor_cortex.detach_state()

    def learn_trajectory(self, input_sequence, target_h_visual_sequence=None, target_h_motor_sequence=None, target_bus_sequence=None, lr=0.001):
        """
        SSI for the whole Modular Brain.
        Trains the cortices to match teacher trajectories.
        
        input_sequence: [T, Batch, input_size]
        target_h_visual_sequence: [T, Batch, visual_hidden]
        target_h_motor_sequence: [T, Batch, motor_hidden]
        target_bus_sequence: [T, Batch, bus_size] (Used as input for motor SSI if provided)
        """
        total_loss = 0.0
        
        # 1. Train Visual Cortex (Retina -> Bus)
        if target_h_visual_sequence is not None:
            if hasattr(self.visual_cortex, 'learn_trajectory'):
                total_loss += self.visual_cortex.learn_trajectory(input_sequence, target_h_visual_sequence, lr=lr)
        
        # 2. Train Motor Cortex (Bus -> Action/Hidden)
        if target_h_motor_sequence is not None:
            if hasattr(self.motor_cortex, 'learn_trajectory'):
                # Motor cortex takes the bus as input. 
                # If target_bus_sequence is provided, use it as the "Teacher Bus".
                # Otherwise, we'd need to run the visual cortex to get the bus (slow).
                if target_bus_sequence is not None:
                    total_loss += self.motor_cortex.learn_trajectory(target_bus_sequence, target_h_motor_sequence, lr=lr)
                else:
                    # Fallback: If input matches motor input size, use it directly
                    if input_sequence.shape[-1] == self.bus_size:
                        total_loss += self.motor_cortex.learn_trajectory(input_sequence, target_h_motor_sequence, lr=lr)
                    else:
                        print("ModularBrain: Skipping Motor SSI - No bus trajectory provided and input size mismatch.")
                
        return total_loss

    def resize_hidden(self, new_hidden_size):
        self.visual_hidden = int(new_hidden_size * 0.4)
        self.motor_hidden = int(new_hidden_size * 0.6)
        
        if hasattr(self.visual_cortex, 'resize_hidden'):
            self.visual_cortex.resize_hidden(self.visual_hidden)
        if hasattr(self.motor_cortex, 'resize_hidden'):
            self.motor_cortex.resize_hidden(self.motor_hidden)

    def resize_input(self, new_input_size):
        """
        Resizes the input layer of the visual cortex.
        """
        self.input_size = new_input_size
        if hasattr(self.visual_cortex, 'resize_input'):
            self.visual_cortex.resize_input(new_input_size)

    def set_plasticity(self, plasticity_coefficients, learning_rate):
        if hasattr(self.visual_cortex, 'set_plasticity'):
            self.visual_cortex.set_plasticity(plasticity_coefficients, learning_rate)
        if hasattr(self.motor_cortex, 'set_plasticity'):
            self.motor_cortex.set_plasticity(plasticity_coefficients, learning_rate)

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
