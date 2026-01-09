from models.liquid_vectorized import VectorizedLiquidGraph
from models.neuromodulated_holographic import NeuromodulatedHolographicBrain

import torch.nn as nn
from brain.modules.device import get_best_device

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
        self.device = get_best_device()
        
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
        bus, _, v_energy, v_flash = res_v
        
        # 2. Motor Cortex
        # Returns: actions, value, energy, flash_info
        res_m = self.motor_cortex(bus, dt=dt, reward=reward, chemicals=chemicals, train_internal_rl=train_internal_rl)
        actions, params, m_energy, m_flash = res_m
        
        total_energy = v_energy + m_energy
        # We return the motor cortex's flash info as the primary one for action selection
        return actions, params, total_energy, m_flash

    def forward_onnx(self, input_vector, chemicals, h_v_r, h_v_c, h_v_s, h_m_r, h_m_c, h_m_s, dt=0.1):
        """
        Stateless forward pass for ONNX export.
        h_v_r, h_v_c, h_v_s: Visual Cortex states (Reflex, Concept, Strategy)
        h_m_r, h_m_c, h_m_s: Motor Cortex states
        """
        # 1. Visual Cortex
        if self.use_neuromodulated:
            bus, _, _, v_flash, next_h_v = self.visual_cortex.forward_onnx(
                input_vector, chemicals, h_v_r, h_v_c, h_v_s, dt
            )
            next_h_v_r, next_h_v_c, next_h_v_s = next_h_v
        else:
            # For non-neuromodulated, we only use first two states
            bus, _, _, next_h_v_r, next_h_v_c = self.visual_cortex.forward_onnx(
                input_vector, h_v_r, h_v_c, dt
            )
            next_h_v_s = h_v_s # Unused
        
        # 2. Motor Cortex
        if self.use_neuromodulated:
            actions, params, _, m_flash, next_h_m = self.motor_cortex.forward_onnx(
                bus, chemicals, h_m_r, h_m_c, h_m_s, dt
            )
            next_h_m_r, next_h_m_c, next_h_m_s = next_h_m
            confidence = m_flash[1]
        else:
            actions, params, _, next_h_m_r, next_h_m_c = self.motor_cortex.forward_onnx(
                bus, h_m_r, h_m_c, dt
            )
            next_h_m_s = h_m_s # Unused
            confidence = torch.ones(actions.shape[0], 1, device=actions.device)
            
        return actions, params, confidence, next_h_v_r, next_h_v_c, next_h_v_s, next_h_m_r, next_h_m_c, next_h_m_s

    def to_onnx(self, file_path):
        """
        Exports the "Reflex" path of the ModularBrain to ONNX.
        """
        import torch
        import os
        
        # 1. Prepare dummy inputs
        batch_size = 1
        dummy_input = torch.randn(batch_size, self.input_size)
        dummy_chemicals = torch.tensor([[0.5, 0.5, 0.2, 0.0]])
        
        # States depend on architecture
        if self.use_neuromodulated:
            # h_reflex, h_concept, h_strategy
            h_v_r = torch.zeros(batch_size, self.visual_cortex.W_reflex.out_features)
            h_v_c = torch.zeros(batch_size, self.visual_cortex.W_concept.out_features)
            h_v_s = torch.zeros(batch_size, self.visual_cortex.W_strategy.out_features)
            
            h_m_r = torch.zeros(batch_size, self.motor_cortex.W_reflex.out_features)
            h_m_c = torch.zeros(batch_size, self.motor_cortex.W_concept.out_features)
            h_m_s = torch.zeros(batch_size, self.motor_cortex.W_strategy.out_features)
            
            states = (h_v_r, h_v_c, h_v_s, h_m_r, h_m_c, h_m_s)
            input_names = ['input', 'chemicals', 'h_v_r', 'h_v_c', 'h_v_s', 'h_m_r', 'h_m_c', 'h_m_s']
            output_names = ['actions', 'params', 'confidence', 'next_h_v_r', 'next_h_v_c', 'next_h_v_s', 'next_h_m_r', 'next_h_m_c', 'next_h_m_s']
        else:
            h_v_x = torch.zeros(batch_size, self.visual_cortex.hidden_size)
            h_v_y = torch.zeros(batch_size, self.visual_cortex.hidden_size)
            h_v_z = torch.zeros(batch_size, 1) # Dummy
            h_m_x = torch.zeros(batch_size, self.motor_cortex.hidden_size)
            h_m_y = torch.zeros(batch_size, self.motor_cortex.hidden_size)
            h_m_z = torch.zeros(batch_size, 1) # Dummy
            
            states = (h_v_x, h_v_y, h_v_z, h_m_x, h_m_y, h_m_z)
            input_names = ['input', 'chemicals', 'h_v_x', 'h_v_y', 'h_v_z', 'h_m_x', 'h_m_y', 'h_m_z']
            output_names = ['actions', 'params', 'confidence', 'next_h_v_x', 'next_h_v_y', 'next_h_v_z', 'next_h_m_x', 'next_h_m_y', 'next_h_m_z']

        # 2. Create a wrapper model for export
        class ONNXWrapper(nn.Module):
            def __init__(self, brain):
                super().__init__()
                self.brain = brain
            def forward(self, *args):
                return self.brain.forward_onnx(*args)

        wrapper = ONNXWrapper(self).eval()
        
        # 3. Export
        print(f"ModularBrain: Exporting to ONNX -> {file_path}")
        torch.onnx.export(
            wrapper,
            (dummy_input, dummy_chemicals, *states),
            file_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names
        )
        return os.path.exists(file_path)

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
        self.motor_hidden = new_hidden_size - self.visual_hidden # Avoid rounding loss
        
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
