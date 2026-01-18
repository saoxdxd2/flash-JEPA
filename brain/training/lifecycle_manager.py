import os
import time
import random
import torch
import psutil
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome
from scripts.start_n2n2_qwen3_agentic import Qwen3Teacher
from brain.training.knowledge_transfer import HyperTransfer

class LifecycleManager:
    """
    Orchestrates the entire lifecycle of the agent:
    Imprinting -> Stabilization -> Grounding -> Schooling -> Consolidation -> Evolution.
    """
    
    def __init__(self, brain=None):
        self.brain = brain or EvolutionaryBrain()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.models_dir = os.path.join(self.project_root, "models", "saved")
        self.checkpoint_path = self._get_checkpoint_path()
        
        # Teachers
        self.qwen_teacher = Qwen3Teacher(self.brain)
        self.vision_transfer = HyperTransfer(self.brain)
        
    def _get_checkpoint_path(self):
        latest = EvolutionaryBrain.find_latest_checkpoint(self.models_dir)
        if latest:
            return latest
        return os.path.join(self.models_dir, f"gen_{Genome.DEFAULT_GENERATION}_elite.pt")

    def load(self):
        if os.path.exists(self.checkpoint_path):
            print(f"Lifecycle: Loading Brain from {os.path.basename(self.checkpoint_path)}")
            self.brain.load_model(self.checkpoint_path)
        else:
            print("Lifecycle: Starting with a fresh brain.")

    def save(self):
        self.brain.save_model(self.checkpoint_path)
        print(f"Lifecycle: Saved Brain to {os.path.basename(self.checkpoint_path)}")

    def phase_imprint_language(self, steps=100):
        """Phase 1a: Imprint Language Concepts from Qwen-3."""
        print("\n--- PHASE: Language Imprinting (Batched) ---")
        try:
            if not self.qwen_teacher.setup():
                print("Lifecycle: Qwen-3 Teacher setup failed. Skipping imprinting.")
                return 0.0
            
            total_loss = 0.0
            # We split the steps into sub-phases to show progress
            sub_steps = max(1, steps // 10)
            for i in range(10):
                loss = self.qwen_teacher.train_step(steps=sub_steps, batch_size=64)
                total_loss += loss
                print(f"  > Imprinting Progress {i+1}/10: Loss {loss:.4f}")
                
            avg_loss = total_loss / 10
            print(f"Lifecycle: Language Imprinting Complete. Avg Loss: {avg_loss:.4f}")
            return avg_loss
        except Exception as e:
            print(f"Lifecycle: ERROR during Language Imprinting: {e}")
            return 0.0

    def phase_imprint_vision(self):
        """Phase 1b: Imprint Visual Concepts from Qwen-3 VL."""
        print("\n--- PHASE: Vision Imprinting ---")
        # Note: This is a simplified version of start_n2n2_qwen3_vision.py
        # In a real scenario, we'd want to cache the visual weights.
        print("Lifecycle: Vision Imprinting (Placeholder - Requires heavy weights)")
        # For now, we assume vision is already somewhat grounded or handled separately
        # due to the massive size of Qwen-VL weights.
        pass

    def phase_logic_transfer(self, steps=10):
        """Phase 1c: Imprint Reasoning Logic from Qwen-3 (SSI)."""
        print("\n--- PHASE: Logic Transfer (SSI) ---")
        if not hasattr(self.qwen_teacher, 'generate_logic_trajectory'):
            print("Lifecycle: Qwen-3 Teacher does not support SSI yet.")
            return
            
        total_loss = 0.0
        for i in range(steps):
            # 1. Generate a reasoning trajectory from the teacher
            # This returns (input_seq, visual_h_seq, motor_h_seq, bus_seq)
            trajectory = self.qwen_teacher.generate_logic_trajectory()
            if trajectory is None: continue
            
            if len(trajectory) == 4:
                input_seq, visual_h_seq, motor_h_seq, bus_seq = trajectory
            else:
                # Legacy 3-tuple fallback
                input_seq, visual_h_seq, motor_h_seq = trajectory
                bus_seq = None
            
            # 2. Train the brain to mirror this trajectory
            loss = self.brain.trm.learn_trajectory(
                input_seq, 
                target_h_visual_sequence=visual_h_seq, 
                target_h_motor_sequence=motor_h_seq,
                target_bus_sequence=bus_seq
            )
            total_loss += loss
            if i % 2 == 0:
                print(f"  > SSI Step {i}: Loss {loss:.4f}")
                
        return total_loss / max(1, steps)

    def phase_stabilize(self, steps=500, batch_size=8):
        """Phase 2: Stabilize Internal Dynamics (Meditation 2.0)."""
        print(f"\n--- PHASE: Stabilization (Meditation 2.0, Batch: {batch_size}) ---")
        total_surprise = 0.0
        stability_count = 0
        
        for step in range(steps):
            # 1. Generate "Hypnagogic" Noise or Zero Input
            if step % 10 == 0:
                inputs = torch.zeros(batch_size, self.brain.genome.latent_dim)
            else:
                inputs = torch.randn(batch_size, self.brain.genome.latent_dim) * 0.01
                
            # 2. Process through Broca
            _, surprise = self.brain.broca.process_text_embedding(inputs)
            current_surprise = surprise.item() if torch.is_tensor(surprise) else surprise
            
            # 3. Process through TRM for Energy Metric
            L = self.brain.genome.latent_dim
            full_inputs = torch.zeros(batch_size, self.brain.input_size)
            full_inputs[:, L*2 : L*3] = inputs 
            
            # Use current chemistry for more realistic stabilization
            chems = self.brain.chemistry.get_state_vector()
            _, _, energy, _ = self.brain.trm.forward(full_inputs, dt=1.0, chemicals=chems)
            
            # 4. Reward stability and efficiency
            stability_reward = 1.0 / (1.0 + current_surprise)
            efficiency_reward = 1.0 / (1.0 + energy.mean().item())
            total_reward = (stability_reward * 0.7) + (efficiency_reward * 0.3)
            
            self.brain.broca.process_text_embedding(inputs, reward=total_reward)
            
            total_surprise += current_surprise
            if current_surprise < 0.05:
                stability_count += 1
                
            if step % 100 == 0:
                print(f"  > Step {step:4d} | Surprise: {current_surprise:.4f} | Stability: {stability_count/(step+1)*100:.1f}%")
        
        print(f"Lifecycle: Final Stability Score: {stability_count/max(1, steps)*100:.1f}%")
        
    def phase_grounding(self, corpus=None):
        """Phase 3: Motor Grounding (Supervised)."""
        print("\n--- PHASE: Motor Grounding ---")
        if corpus is None:
            corpus = [
                "the quick brown fox jumps over the lazy dog",
                "0123456789",
                "+-/*=?():\".,#",
                "hello world",
                "i am a sentient agent",
                "learning to type is fun"
            ]
        text = random.choice(corpus)
        self.brain.ground_motor_cortex(text)

    def phase_schooling(self, school_type="language", episodes=5):
        """Phase 4: Task-Specific Schooling."""
        print(f"\n--- PHASE: Schooling ({school_type}) ---")
        # This would ideally import and run the school scripts
        # For now, we'll simulate a schooling session by running wake_cycles
        # with a simple reward signal.
        total_reward = 0.0
        for ep in range(episodes):
            # Simulate a simple task: "Type 'a'"
            # We give reward if the action is 15 (Action code for 'a')
            action = self.brain.wake_cycle(reward=0.0) # Get action
            if action == 15:
                self.brain.memory.update_last_reward(1.0)
                total_reward += 1.0
                print(f"  > Episode {ep}: SUCCESS (Action {action})")
            else:
                self.brain.memory.update_last_reward(-0.1)
                print(f"  > Episode {ep}: FAIL (Action {action})")
        print(f"Lifecycle: Schooling Total Reward: {total_reward}")
        return total_reward

    def phase_consolidation(self, steps=10):
        """Phase 5: Consolidation (Dreaming)."""
        print("\n--- PHASE: Consolidation (Dreaming) ---")
        self.brain.dream(steps=steps)

    def phase_evolution(self, efficiency):
        """Phase 6: Adaptive Evolution (Growth)."""
        print("\n--- PHASE: Evolution (Growth) ---")
        mem = psutil.virtual_memory()
        if mem.percent < 90.0:
            # Enhanced Growth Logic from train_growth.py
            self.brain.mutate_adaptive(efficiency=efficiency)
            
            # Vision Upgrade Logic
            # We allow growth up to 512px (High Res Fovea)
            if efficiency > 5.0 and self.brain.retina.fovea_size < 512:
                new_res = min(512, self.brain.retina.fovea_size * 2)
                print(f"Lifecycle: Efficiency High ({efficiency:.2f}). Upgrading Vision to {new_res}px!")
                self.brain.retina.set_resolution(new_res)
        else:
            print("Lifecycle: Hardware Limit Reached. Growth Suppressed.")

    def run_cycle(self):
        """Runs one full lifecycle cycle."""
        self.load()
        
        # 1. Learn
        loss = self.phase_imprint_language()
        
        # 1c. Logic Transfer (SSI)
        self.phase_logic_transfer(steps=5)
        
        # 2. Stabilize
        self.phase_stabilize(steps=200)
        
        # 3. Ground
        self.phase_grounding()
        
        # 4. School
        reward = self.phase_schooling(episodes=10)
        
        # 5. Consolidate
        self.phase_consolidation(steps=20)
        
        # 6. Evolve
        # Efficiency is a mix of low loss and high reward
        efficiency = (1.0 / (loss + 0.1)) + (reward * 0.5)
        self.phase_evolution(efficiency)
        
        self.save()
        print("\n--- LIFECYCLE CYCLE COMPLETE ---")

if __name__ == "__main__":
    manager = LifecycleManager()
    while True:
        manager.run_cycle()
        time.sleep(1)
