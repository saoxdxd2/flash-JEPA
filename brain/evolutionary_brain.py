import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import os

from brain.modules.predictive_retina import PredictiveRetina
from brain.modules.broca import BrocaModule
from brain.modules.biology_core import NeurotransmitterSystem
from brain.modules.amygdala import Amygdala
from brain.modules.basal_ganglia import BasalGanglia
from brain.utils import ResourceMonitor, get_best_device, get_memory_stats, check_ram_limit, ONNXEngine
from brain.modules.neural_memory import TitansMemory
from models.ecg import ModularBrain
from brain.modules.replay_buffer import PrioritizedReplayBuffer
from brain.modules.cradle import Cradle

class EvolutionaryBrain:
    """
    Biological Brain Controller.
    Driven by Neurochemistry and Structural Biology.
    """
    def __init__(self, genome=None):
        # Hyperparameters from Genome
        if genome is None:
            from brain.genome import Genome
            self.genome = Genome()
        else:
            self.genome = genome
            
        if not hasattr(self.genome, 'latent_dim'):
            self.genome.latent_dim = 256
            
        self.latent_dim = self.genome.latent_dim
        
        # Input Size: Visual (3*L) + Memory Prediction (3*L) + Bio/Action (Non-Visual)
        # We add a Hippocampus (Titans Memory) that predicts the next sensory state
        self.input_size = (6 * self.genome.latent_dim) + self.genome.NON_VISUAL_INPUT_SIZE
        
        self.hidden_size = self.genome.hidden_size 
        self.action_size = self.genome.action_size 
        
        self.device = get_best_device()
        
        # --- Biological Core ---
        self.chemistry = NeurotransmitterSystem(genome=self.genome)
        self.amygdala = Amygdala(genome=self.genome, device=self.device)
        # Distributed Motor Cortex: Basal Ganglia only selects the INTENT (0-9)
        self.intent_size = 10
        # Pass input_size as state_size for Basal Ganglia
        self.basal_ganglia = BasalGanglia(self.input_size, self.intent_size).to(self.device)
        
        # --- Replay Buffer (Unified System 1/2) ---
        self.replay_buffer = PrioritizedReplayBuffer(capacity=self.genome.REPLAY_BUFFER_CAPACITY)
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0.0
        
        # Meta-Cognition
        self.last_action_success = True
        self.last_confidence = 1.0
        self.last_energy_cost = 0.0
        
        # --- Cognitive Modules ---
        self.trm = ModularBrain(
            self.input_size, 
            self.hidden_size, 
            self.action_size, 
            genome=self.genome, 
            use_neuromodulated=True,
            memory_size=3 * self.genome.latent_dim # Pass Memory Size for Thalamic Gating
        )
        self.trm.set_plasticity(self.genome.plasticity_coefficients, self.genome.learning_rate)
        self.retina = PredictiveRetina(latent_size=self.genome.latent_dim, genome=self.genome)
        self.broca = BrocaModule(
            embedding_dim=self.genome.latent_dim, 
            visual_dim=self.genome.latent_dim,
            genome=self.genome
        ).to(self.device)
        
        # Hippocampus (Episodic/Working Memory)
        # Predicts next sensory state (Foveal + Peripheral + Semantic)
        self.hippocampus = TitansMemory(
            input_dim=3 * self.genome.latent_dim,
            hidden_dim=3 * self.genome.latent_dim
        ).to(self.device)
        
        self.memory = self.replay_buffer 
        
        # --- 8. Latent Adapter (N2N2 Alignment) ---
        # Maps teacher embeddings (e.g. 4096) to our latent_dim
        self.latent_adapter = nn.Sequential(
            nn.Linear(self.genome.latent_adapter_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_dim)
        ).to(self.device)
        
        # --- 9. Optimizers ---
        self.trm_optimizer = torch.optim.Adam(self.trm.parameters(), lr=self.genome.learning_rate * 2.0)
        
        # --- Interface (Cradle) ---
        self.cradle = Cradle()
        self.monitor = ResourceMonitor()
        
        # Set RAM Limit
        mem_stats = get_memory_stats()
        sys_ram = mem_stats["sys_total"]
        self.genome.max_ram_mb = min(12288, int(sys_ram * 0.75))
        print(f"EvolutionaryBrain: RAM Limit set to {self.genome.max_ram_mb}MB on {self.device}")
        
        # Metrics
        self.accumulated_reward = 0.0
        self.accumulated_energy = 0.0
        self.age = 0
        self.stamina = 1.0
        self.last_surprise = 0.0
        
        # --- ONNX Optimization ---
        self.use_onnx = False
        self.onnx_engine = None
        self.onnx_path = "reflex_path.onnx"
        self.onnx_states = None # Persistent states for ONNX

    def start(self):
        print("Brain: Waking up...")
        self.retina.start()

    def stop(self):
        print("Brain: Shutting down...")
        self.retina.stop()

    def reset_memory(self):
        self.prev_action = None
        if hasattr(self.trm, 'reset_state'):
            self.trm.reset_state()

    def decide(self, full_input_tensor, train_internal_rl=True, greedy=False, disable_reflex=False):
        # 1. Input Processing & Conscious Masking
        if full_input_tensor.dim() == 1: full_input_tensor = full_input_tensor.unsqueeze(0)
        
        L = self.latent_dim
        boosted_input = full_input_tensor.clone()
        boosted_input[:, :3*L] *= self.genome.INPUT_BOOST_FACTOR
        
        # Mask Biological State from Conscious Stream (Indices [6*L : 6*L + 8])
        boosted_input[:, 6*L : 6*L + 8] = 0.0
        
        chemicals = self.chemistry.get_state_vector() # [1, 4] Tensor
        
        # 2. Forward Pass (Hybrid ONNX/PyTorch)
        use_onnx = self.use_onnx and self.onnx_engine is not None
        
        if use_onnx:
            if self.onnx_states is None: self._init_onnx_states()
            outputs = self.onnx_engine.run({
                'input': boosted_input.cpu().numpy(),
                'chemicals': chemicals.cpu().numpy(),
                **{name.name: state for name, state in zip(self.onnx_engine.session.get_inputs()[2:], self.onnx_states)}
            })
            logits_np, params_np, conf_np = outputs[:3]
            self.onnx_states = outputs[3:]
            
            confidence = torch.from_numpy(conf_np).to(self.device)
            
            # Fallback to System 2 if confidence is low
            if confidence.mean().item() < self.genome.CONFIDENCE_THRESHOLD:
                use_onnx = False # Trigger PyTorch path
            else:
                logits = torch.from_numpy(logits_np).to(self.device)
                value = torch.from_numpy(params_np).to(self.device)
                energy = torch.tensor(0.0, device=self.device)
                flash_info = (logits, confidence, None, None, None, None, None, None)

        if not use_onnx:
            # PyTorch Forward Pass (Thalamic Gating enabled via memory_input)
            res = self.trm.forward(
                boosted_input, 
                chemicals=chemicals, 
                train_internal_rl=train_internal_rl, 
                memory_input=boosted_input[:, 3*L : 6*L]
            )
            logits, value, energy, flash_info = res
            
        self.last_value = value.mean().item() if torch.is_tensor(value) else value
        surprise = getattr(self, 'last_surprise', 0.0)
        
        # 3. Amygdala Hijack (Immediate Threat)
        if not disable_reflex:
            surprise_t = torch.tensor(surprise, device=self.device) if not torch.is_tensor(surprise) else surprise
            hijack_mask, reflex_actions = self.amygdala.process(surprise_t, torch.tensor(0.0, device=self.device), chemicals)
            
            if hijack_mask.any():
                self.last_used_system = 0
                reflex_action = reflex_actions[0].item()
                reflex_logits = torch.zeros_like(logits)
                reflex_logits[0, reflex_action] = 10.0
                return reflex_action, reflex_logits

        # 4. Basal Ganglia Gating (Intent Selection)
        intent_action, meta = self.basal_ganglia(
            state=boosted_input,
            action_logits=logits[:, :self.intent_size],
            dopamine=self.chemistry.dopamine,
            flash_info=(flash_info[0][:, :self.intent_size] if flash_info[0] is not None else None, flash_info[1]),
            surprise=surprise,
            chemicals=chemicals,
            greedy=greedy
        )
        
        self.last_used_system = meta['used_system']
        return intent_action, logits[:, self.intent_size:]

    def train_step(self, batch_size=32, distillation=False):
        if len(self.replay_buffer) < batch_size:
            return 0.0
            
        # Detach state to avoid backpropping through history (Recurrent Dynamics)
        if hasattr(self.trm, 'detach_state'):
            self.trm.detach_state()
            
        batch_size = int(batch_size)
        
        # Sample Batch (Zero-Copy Numpy Arrays)
        batch_data, indices, weights = self.replay_buffer.sample(batch_size)
        if batch_data is None: return 0.0
        
        b_states, b_actions, b_rewards, b_next_states, b_dones = batch_data
        
        # Convert to Tensors (Zero-Copy where possible)
        states = torch.from_numpy(b_states).float().to(self.device)
        actions = torch.from_numpy(b_actions).long().to(self.device)
        rewards = torch.from_numpy(b_rewards).float().to(self.device)
        next_states = torch.from_numpy(b_next_states).float().to(self.device)
        dones = torch.from_numpy(b_dones).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        mask = rewards.abs() > 0.01
        if mask.any():
            states = states[mask]
            actions = actions[mask]
            rewards = rewards[mask]
            next_states = next_states[mask]
            dones = dones[mask]
            weights = weights[mask]
        else:
            return 0.0
            
        L = self.latent_dim
        boosted_states = states.clone()
        boosted_states[:, :3*L] *= self.genome.INPUT_BOOST_FACTOR
        boosted_next_states = next_states.clone()
        boosted_next_states[:, :3*L] *= self.genome.INPUT_BOOST_FACTOR
        
        self.trm_optimizer.zero_grad()
        chemicals = torch.zeros(states.shape[0], 4, device=states.device)
        
        # Detach state to prevent graph growth and shape mismatches from previous steps
        self.trm.detach_state()
        
        res = self.trm.forward(boosted_states, chemicals=chemicals)
        logits, value, _, (flash_logits, flash_conf, p_reflex, p_concept, p_strategy, h_reflex, h_concept, h_strategy) = res
        
        with torch.no_grad():
            res_next = self.trm.forward(boosted_next_states, chemicals=chemicals)
            # We want the ACTUAL hidden states of the next step as targets for JEPA
            _, _, _, (_, _, _, _, _, t_reflex, t_concept, t_strategy) = res_next
        
        # 1. Standard RL Losses (A2C / PPO-lite)
        # Calculate Advantage
        # We use the actual return (reward + gamma * next_value) - value
        # But here we just use raw reward as a simple proxy for return in this step-based setup
        # Ideally we should use n-step returns, but for now:
        advantage = rewards - value.squeeze().detach()
        
        # Policy Loss: -log_prob * advantage
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantage).mean()
        
        # Flash Loss (System 1) - Also needs to be RL-based or Distilled
        # For Flash, we can use the same advantage
        flash_log_probs = F.log_softmax(flash_logits, dim=1)
        selected_flash_log_probs = flash_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        flash_loss = -(selected_flash_log_probs * advantage).mean()
        
        value_loss = F.mse_loss(value.squeeze(), rewards)
        
        try:
            # JEPA Loss: Predict the NEXT hidden state
            jepa_loss = F.mse_loss(p_reflex, t_reflex) + \
                        F.mse_loss(p_concept, t_concept) + \
                        F.mse_loss(p_strategy, t_strategy)
        except RuntimeError as e:
            print(f"DEBUG: Shape Mismatch in JEPA Loss!")
            print(f"DEBUG: p_reflex: {p_reflex.shape}, t_reflex: {t_reflex.shape}")
            print(f"DEBUG: p_concept: {p_concept.shape}, t_concept: {t_concept.shape}")
            print(f"DEBUG: p_strategy: {p_strategy.shape}, t_strategy: {t_strategy.shape}")
            raise e
            
        # 2. Distillation Loss (System 2 -> System 1)
        distill_loss = 0.0
        if distillation:
            T = self.genome.DISTILLATION_TEMPERATURE
            p_s1 = F.log_softmax(flash_logits / T, dim=1)
            p_s2 = F.softmax(logits / T, dim=1)
            distill_loss = F.kl_div(p_s1, p_s2, reduction='batchmean') * (T**2)
            
        total_loss = policy_loss + flash_loss + value_loss + (jepa_loss * self.genome.JEPA_LOSS_WEIGHT) + (distill_loss * self.genome.DISTILLATION_LOSS_WEIGHT)
        
        if torch.isnan(total_loss):
            print("DEBUG: NaN Loss detected!")
            return 0.0
            
        try:
            total_loss.backward()
        except RuntimeError as e:
            print(f"DEBUG: Backward Pass Failed!")
            print(f"DEBUG: Error: {e}")
            print(f"DEBUG: Parameter Shapes:")
            for name, param in self.trm.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.shape}")
            raise e
            
        self.trm_optimizer.step()
        
        self.last_surprise = jepa_loss.item()
        return total_loss.item()

    def get_input_vector(self, foveal_latent, peripheral_latent, semantic_latent, bio_state=None, prev_action=None, memory_latent=None):
        if bio_state is None:
            bio_state = torch.tensor([
                self.chemistry.dopamine, 
                self.chemistry.serotonin, 
                self.chemistry.norepinephrine, 
                self.stamina,
                0.0, # Default surprise
                0.0, # Default text density
                0.0, # Default RAM
                0.0  # Default CPU
            ], device=foveal_latent.device)
        elif not isinstance(bio_state, torch.Tensor):
            bio_state = torch.tensor(bio_state, dtype=torch.float32, device=foveal_latent.device)
            
        last_action_vec = torch.zeros(100, device=foveal_latent.device)
        if prev_action is not None:
            last_action_vec[prev_action % 100] = 1.0
        elif self.prev_action is not None:
            last_action_vec[self.prev_action % 100] = 1.0
            
        # Calculate padding based on actual bio_state size
        bio_size = bio_state.shape[0]
        padding_size = self.genome.NON_VISUAL_INPUT_SIZE - (bio_size + 100)
        
        # Memory Latent (Hippocampus Prediction)
        if memory_latent is None:
            # Default to zeros if not provided (e.g. first step)
            memory_latent = torch.zeros(3 * self.latent_dim, device=foveal_latent.device)
        
        full_input = torch.cat([
            foveal_latent.flatten(),
            peripheral_latent.flatten(),
            semantic_latent.flatten(),
            memory_latent.flatten(), # Added Memory Context
            bio_state,
            last_action_vec,
            torch.zeros(max(0, padding_size), device=foveal_latent.device)
        ]).unsqueeze(0)
        
        return full_input

    def wake_cycle(self, reward=0.0, pain=0.0):
        device = next(self.trm.parameters()).device
        
        # --- THALAMIC GATING SYSTEM (Structural) ---
        # "The Doors of Perception"
        # The Thalamus (in NeuromodulatedHolographicBrain) handles the mixing physically.
        # Here we just prepare the inputs: Reality (Sensory) and Expectation (Memory).
        
        # 1. Get Bottom-Up Signal (Reality)
        vision_data = self.retina.get_latest_input()
        
        if vision_data is not None:
            # Reality is available
            foveal_latent, peripheral_latent, surprise, text_density, fovea_tensor = vision_data
            
            # Convert to tensors
            if isinstance(foveal_latent, np.ndarray):
                foveal_latent = torch.from_numpy(foveal_latent).float().to(device)
            if isinstance(peripheral_latent, np.ndarray):
                peripheral_latent = torch.from_numpy(peripheral_latent).float().to(device)
                
            # Get Semantic Context from Broca
            semantic_latent = self.broca.get_current_context()
            
        else:
            # Reality is missing (Silence/Dreaming)
            # We provide ZEROS for the sensory input.
            # The Thalamus will detect this (and the chemical state) and switch to Memory.
            foveal_latent = torch.zeros(self.latent_dim, device=device)
            peripheral_latent = torch.zeros(self.latent_dim, device=device)
            semantic_latent = torch.zeros(self.latent_dim, device=device)
            surprise = 0.0
            text_density = 0.0
            
        # 7. Construct Bio State
        bio_state = torch.tensor([
            self.chemistry.dopamine, 
            self.chemistry.serotonin, 
            self.chemistry.norepinephrine, 
            self.stamina,
            surprise, 
            text_density, 
            0.0, # RAM (Placeholder)
            0.0  # CPU (Placeholder)
        ], device=device)
        # --- HIPPOCAMPUS INTEGRATION ---
        # 1. Construct Sensory State (What we are seeing/thinking NOW)
        sensory_state = torch.cat([
            foveal_latent.flatten(),
            peripheral_latent.flatten(),
            semantic_latent.flatten()
        ])
        
        # 2. Update Memory (Learn from Prediction Error)
        # "Was my previous prediction correct?"
        memory_surprise = self.hippocampus.observe(sensory_state)
        
        # 3. Predict Next State (Context for Decision)
        # "What do I expect to happen next?"
        memory_prediction = self.hippocampus(sensory_state)
        
        # Add memory surprise to global surprise
        surprise += memory_surprise
        
        full_input = self.get_input_vector(
            foveal_latent, 
            peripheral_latent, 
            semantic_latent, 
            bio_state=bio_state,
            memory_latent=memory_prediction
        )
        
        # 4. Decide Action
        # Returns: Intent Index (int), Parameter Logits (Tensor)
        intent_action, param_logits = self.decide(full_input)
        
        # 5. Execute Action (Distributed)
        # We pass the parameters (X, Y, etc.) to the cradle
        # Process Parameters for Execution
        # X, Y are at indices 0, 1 of param_logits (which corresponds to 10, 11 of original)
        execution_params = param_logits.clone().detach().squeeze()
        if execution_params.dim() == 0: execution_params = execution_params.unsqueeze(0)
        
        # Apply Sigmoid to X,Y (first 2 params) to ensure 0-1 range
        if execution_params.shape[0] >= 2:
            execution_params[0] = torch.sigmoid(execution_params[0])
            execution_params[1] = torch.sigmoid(execution_params[1])
            
        self.cradle.execute_distributed(intent_action, execution_params)
        
        # Store for Replay (We store the INTENT as the action)
        action = intent_action
        
        # 6. Update Replay Buffer (Experience Replay)
        if self.prev_state is not None:
            self.replay_buffer.add(self.prev_state, self.prev_action, reward, full_input, False)
            
        # 7. Update State for next cycle
        self.prev_state = full_input
        self.prev_action = action
        self.prev_reward = reward
        
        # 8. Update chemistry based on effort and sensory feedback
        # CPU load increases effort/metabolic cost
        norm_cpu = 0.0
        metabolic_cost = self.genome.ACTION_COST_BASE + (norm_cpu * 0.05)
        
        rpe = reward - getattr(self, 'last_value', 0.0)
        self.chemistry.update(
            reward_prediction_error=rpe, 
            surprise=surprise, 
            pain=pain, 
            effort=metabolic_cost,
            fear=self.amygdala.fear_level,
            aggression=self.amygdala.aggression_level
        )
        
        # 9. Increment Age
        self.age += 1
        
        return action
        
    def check_growth_triggers(self):
        """
        Monitors cognitive demand and triggers neurogenesis if needed.
        """
        # Triggers:
        # 1. High Surprise: Model is struggling to predict (needs more capacity)
        # 2. High Reward: Model has found a good strategy (needs to lock it in)
        surprise_threshold = self.genome.SURPRISE_THRESHOLD
        reward_threshold = self.genome.REWARD_THRESHOLD
        
        if self.last_surprise > surprise_threshold or self.accumulated_reward > reward_threshold:
            print(f"Brain: Growth Triggered! (Surprise: {self.last_surprise:.4f}, Reward: {self.accumulated_reward:.2f})")
            # Expand hidden size
            new_hidden = int(self.hidden_size * self.genome.GROWTH_RATE_MULTIPLIER)
            # Cap for CPU efficiency
            new_hidden = min(new_hidden, self.genome.MAX_HIDDEN_SIZE)
            
            if new_hidden > self.hidden_size:
                self.trm.resize_hidden(new_hidden)
                self.hidden_size = new_hidden
                # Re-init optimizer for new parameters
                self.trm_optimizer = torch.optim.Adam(self.trm.parameters(), lr=self.genome.learning_rate)
                # Reset reward to avoid immediate re-trigger
                self.accumulated_reward = 0.0

    def dream(self, steps=10):
        """Dreaming phase: Consolidation and System 2 -> System 1 distillation."""
        print(f"Brain: Dreaming (Distillation Mode) for {steps} steps...")
        for _ in range(steps):
            self.train_step(distillation=True)
            
        # --- Structural Plasticity: Check for Growth ---
        self.check_growth_triggers()
    
    def mutate_adaptive(self, fitness_ratio):
        """Adaptive mutation based on fitness performance."""
        # Only mutate if underperforming
        if fitness_ratio < 0.5:
            self.genome.mutate()
            print(f"Brain: Adaptive mutation triggered (fitness_ratio={fitness_ratio:.2f})")

    def crossover(self, other_brain):
        """
        N2N2-style crossover: Creates a child brain by averaging weights with another brain.
        
        Args:
            other_brain: Another EvolutionaryBrain to crossover with.
            
        Returns:
            child_brain: A new EvolutionaryBrain with mixed genetics.
        """
        # Create child with averaged genome
        child_genome = self.genome.crossover(other_brain.genome)
        child_brain = EvolutionaryBrain(child_genome)
        
        # Weight averaging (N2N2 crossover)
        my_state = self.trm.state_dict()
        other_state = other_brain.trm.state_dict()
        child_state = child_brain.trm.state_dict()
        
        for key in child_state.keys():
            if key in my_state and key in other_state:
                if my_state[key].shape == other_state[key].shape == child_state[key].shape:
                    # Average the weights
                    alpha = 0.5 + (random.random() - 0.5) * 0.2  # 0.4 to 0.6
                    child_state[key] = alpha * my_state[key] + (1 - alpha) * other_state[key]
                    
        child_brain.trm.load_state_dict(child_state, strict=False)
        return child_brain


    def export_reflex_path(self, path=None):
        """
        Freezes the current brain and exports the Reflex path to ONNX.
        """
        if path is None: path = self.onnx_path
        print(f"Brain: Exporting Reflex Path to {path}...")
        
        # Ensure model is in eval mode
        self.trm.eval()
        
        success = self.trm.to_onnx(path)
        if success:
            print("Brain: ONNX Export Successful.")
            self.onnx_engine = ONNXEngine(path)
            self.onnx_path = path
            self._init_onnx_states()
        else:
            print("Brain: ONNX Export Failed.")
        return success

    def _init_onnx_states(self):
        """Initializes persistent states for ONNX inference."""
        if self.onnx_engine is None: return
        
        self.onnx_states = []
        # Get state shapes from ONNX inputs (skipping input and chemicals)
        for input_meta in self.onnx_engine.session.get_inputs()[2:]:
            shape = input_meta.shape
            # Replace dynamic batch with 1
            fixed_shape = [1 if isinstance(s, str) or s is None else s for s in shape]
            self.onnx_states.append(np.zeros(fixed_shape, dtype=np.float32))
        print(f"Brain: Initialized {len(self.onnx_states)} ONNX states.")

    @staticmethod
    def find_latest_checkpoint(directory):
        if not os.path.exists(directory): return None
        files = [f for f in os.listdir(directory) if f.endswith(('.pt', '.pth'))]
        if not files: return None
        files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
        return os.path.join(directory, files[0])

    def save_model(self, filepath):
        checkpoint = {
            'version': '3.0-scaled',
            'genome': self.genome.__dict__,
            'trm_state': self.trm.state_dict(),
            'retina_state': self.retina.state_dict(),
            'broca_state': self.broca.save(),
            'retina_size': self.retina.fovea_size,
            'latent_dim': self.latent_dim,
            'hidden_size': self.hidden_size
        }
        torch.save(checkpoint, filepath)

    def load_model(self, filepath):
        print(f"Brain: Loading model from {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        version = str(checkpoint.get('version', '2.0'))
        
        if 'genome' in checkpoint:
            for k, v in checkpoint['genome'].items():
                setattr(self.genome, k, v)
        
        if 'retina_size' in checkpoint:
            self.retina.set_resolution(checkpoint['retina_size'])
        if 'genome' in checkpoint and 'PERIPHERAL_RESOLUTION' in checkpoint['genome']:
            self.retina.set_peripheral_resolution(checkpoint['genome']['PERIPHERAL_RESOLUTION'])
            
        target_hidden = checkpoint.get('hidden_size', self.hidden_size)
        target_latent = checkpoint.get('latent_dim', self.latent_dim)
            
        # Load states
        trm_state = checkpoint['trm_state']
        
        # Resize if needed
        if target_latent != self.latent_dim or target_hidden != self.hidden_size:
            print(f"Brain: Schema mismatch detected (Latent: {self.latent_dim}->{target_latent}, Hidden: {self.hidden_size}->{target_hidden})")
            print("Brain: Clearing replay buffer to avoid shape errors.")
            self.replay_buffer = PrioritizedReplayBuffer(capacity=self.genome.REPLAY_BUFFER_CAPACITY)
            
            if target_latent != self.latent_dim:
                self.resize_latent(target_latent)
            if target_hidden != self.hidden_size:
                self.resize_hidden(target_hidden)
                
            # Aggressively strip size-sensitive parameters on schema mismatch
            # These will be re-initialized with correct shapes during resize_hidden or kept at random if missing
            keys_to_remove = [k for k in trm_state.keys() if any(x in k for x in ['tau_', 'router_', 'flash_', 'decoder', 'critic', 'intent_gate'])]
            for k in keys_to_remove:
                if k in trm_state: del trm_state[k]
        
        # Smart Healing: Strip any parameters that don't match the current model's shapes
        # This prevents "Frankenstein" models with mismatched hierarchical layers
        current_model_dict = self.trm.state_dict()
        mismatched_keys = []
        for k, v in trm_state.items():
            if k in current_model_dict:
                if v.shape != current_model_dict[k].shape:
                    mismatched_keys.append(k)
            else:
                mismatched_keys.append(k) # Orphaned key
                
        if mismatched_keys:
            print(f"Brain: Stripping {len(mismatched_keys)} mismatched parameters from state dict to ensure shape stability.")
            for k in mismatched_keys:
                del trm_state[k]
        
        # Always strip stale activity buffers and CSR indices
        keys_to_remove = [k for k in trm_state.keys() if any(x in k for x in ['activity_in', 'activity_out', 'crow_indices', 'col_indices'])]
        for k in keys_to_remove:
            if k in trm_state: del trm_state[k]
            
        self.trm.load_state_dict(trm_state, strict=False)
        self.retina.load_state_dict(checkpoint['retina_state'], strict=False)
        if 'broca_state' in checkpoint:
            self.broca.load(checkpoint['broca_state'])
            
        # Re-init optimizer after loading to ensure it tracks the current parameters
        # and starts with fresh momentum/state for the new model
        self.trm_optimizer = torch.optim.Adam(self.trm.parameters(), lr=self.genome.learning_rate)
        
        print(f"Brain: Successfully loaded v{version} checkpoint.")

    def resize_latent(self, new_latent_size):
        if new_latent_size == self.latent_dim: return
        print(f"Brain: Resizing Latent Dim {self.latent_dim} -> {new_latent_size}")
        self.latent_dim = new_latent_size
        self.genome.latent_dim = new_latent_size
        self.input_size = (3 * new_latent_size) + self.genome.NON_VISUAL_INPUT_SIZE
        self.retina.resize_latent(new_latent_size)
        self.broca.resize_latent(new_latent_size)
        if hasattr(self.trm, 'resize_input'):
            self.trm.resize_input(self.input_size)
        # Re-init optimizer for new parameters
        self.trm_optimizer = torch.optim.Adam(self.trm.parameters(), lr=self.genome.learning_rate)

    def resize_hidden(self, new_hidden_size):
        if new_hidden_size == self.hidden_size: return
        print(f"Brain: Resizing Hidden Size {self.hidden_size} -> {new_hidden_size}")
        self.hidden_size = new_hidden_size
        self.genome.hidden_size = new_hidden_size
        if hasattr(self.trm, 'resize_hidden'):
            self.trm.resize_hidden(new_hidden_size)
        # Re-init optimizer for new parameters
        self.trm_optimizer = torch.optim.Adam(self.trm.parameters(), lr=self.genome.learning_rate)
