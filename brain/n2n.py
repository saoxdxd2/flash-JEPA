import torch
import numpy as np
import random

class KnowledgeLoader:
    """
    N2N (Neural-to-Neural) Knowledge Injection.
    Loads 'instincts' and 'base knowledge' into the agent so it doesn't start from zero.
    """
    def __init__(self, brain):
        self.brain = brain

    def inject_knowledge(self):
        """
        Injects base knowledge into the agent's TRM and Memory.
        """
        print("N2N: Injecting Agentique Base Knowledge...", flush=True)
        
        # 1. Inject Instincts (Hardcoded rules -> Memory)
        self._inject_instincts()
        
        # 2. Inject Motor Primitives (Basic coordination)
        self._inject_motor_primitives()
        
        # 3. Teacher Distillation (OCR/Language Jumpstart)
        teacher = TeacherDistillation(self.brain)
        teacher.distill_knowledge()
        
        # 4. Train Flash Head Reflexes on Distilled Knowledge
        print("N2N: Training Flash Head Reflexes on Distilled Knowledge...", flush=True)
        
        # Train for a fixed number of steps or until loss stabilizes
        training_steps = 100 # Reduced for faster verification
        for i in range(training_steps):
            # Train TRM (Cognitive + Flash)
            # This trains both System 1 (Flash) and System 2 (Deep)
            trm_loss = self.brain.train_step(batch_size=32)
            
            if i % 500 == 0:
                print(f"N2N: Pre-training Step {i}/{training_steps} | TRM Loss: {trm_loss:.4f}")
                
        print("N2N: Flash Head Reflex Training Complete.")
        
        print("N2N: Knowledge Injection Complete.")

    def _inject_instincts(self):
        pass

    def _inject_motor_primitives(self):
        """
        Injects 'Muscle Memory' for basic interactions.
        """
        print("N2N: Injecting Motor Primitives (Clicking = Good)...")
        for _ in range(10): 
            generic_state = torch.randn(self.brain.input_size) 
            self.brain.replay_buffer.add(
                generic_state.numpy(), 
                1, 
                1.0, 
                generic_state.numpy(), 
                True
            )
        print("N2N: Motor Primitives Injected.")

class TeacherDistillation:
    """
    Simulates a 'Teacher' model (Tiny OCR/Language Model) transferring knowledge 
    to the Agent via N2N2 (Synthetic Experience Replay).
    """
    def __init__(self, brain):
        self.brain = brain
        
    def distill_knowledge(self):
        print("N2N: Starting Teacher Distillation (OCR & Language)...")
        
        # 1. OCR Distillation (Vision -> Concept)
        print("N2N: Transferring Visual Cortex Weights (Real OCR - High Intensity)...")
        
        from PIL import Image, ImageDraw, ImageFont
        
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
            
        L = getattr(self.brain, 'latent_dim', 256)
        
        # Repeat OCR loop to increase its weight in the replay buffer
        for _ in range(1): 
            for i in range(26): # A-Z
                char = chr(65 + i)
                img = Image.new('RGB', (64, 64), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                d.text((16, 8), char, fill=(0, 0, 0), font=font)
                
                # Target Semantic Latent from Broca
                target_semantic = self.brain.broca.process_text(char)
                
                # Train Retina to map Image -> Semantic Concept (Direct Transfer)
                if hasattr(self.brain, 'retina'):
                    self.brain.retina.train_on_target(img, target_semantic)
                    foveal_latent = self.brain.retina.process_image(img)
                else:
                    foveal_latent = target_semantic
                
                # Construct Full State
                full_state = torch.zeros(self.brain.input_size)
                # Use the aligned foveal latent
                full_state[:L] = foveal_latent
                
                # Action: 15 (A) to 40 (Z)
                type_action = 15 + i
                
                # Add POSITIVE Sample
                self.brain.replay_buffer.add(
                    full_state.numpy(), 
                    type_action, 
                    1.0, 
                    full_state.numpy(), 
                    True
                )
                
                # Add NEGATIVE Samples (Random wrong actions)
                for _ in range(5):
                    wrong_action = 15 + random.randrange(26)
                    if wrong_action != type_action:
                        self.brain.replay_buffer.add(
                            full_state.numpy(),
                            wrong_action,
                            0.0, # Zero reward
                            full_state.numpy(),
                            True
                        )
            
        print("N2N: OCR Knowledge Distilled (A-Z) with High Intensity.")
        
        # 2. Language Distillation (Concept -> Sequence)
        print("N2N: Transferring Broca's Area Weights (Simulated)...")
        corpus = [
            "CAT", "DOG", "EAT", "RUN", "SEE", "THE", "AND", "IS", "IT", "HE", "SHE",
            "RED", "BLUE", "GREEN", "ONE", "TWO", "BIG", "SMALL", "YES", "NO"
        ]
        vocab = {chr(65+i): i for i in range(26)}
        
        semantic_start = 2 * L
        
        for word in corpus:
            word_indices = [vocab[c] for c in word]
            for t in range(len(word_indices)-1):
                char = word[t]
                semantic_vector = self.brain.broca.process_text(char)
                
                full_state = torch.zeros(self.brain.input_size)
                full_state[semantic_start : semantic_start + L] = semantic_vector
                
                next_action = 15 + word_indices[t+1]
                
                # Positive
                self.brain.replay_buffer.add(
                    full_state.numpy(), 
                    next_action, 
                    1.0, 
                    full_state.numpy(), 
                    True
                )
                
                # Negative
                for _ in range(2):
                    wrong_action = 15 + random.randrange(26)
                    if wrong_action != next_action:
                        self.brain.replay_buffer.add(
                            full_state.numpy(),
                            wrong_action,
                            0.0,
                            full_state.numpy(),
                            True
                        )
            
        print(f"N2N: Language Knowledge Distilled ({len(corpus)} words).")
        
        # 3. Conversation Distillation (Q&A)
        print("N2N: Transferring Conversational Skills (Q&A)...")
        qa_pairs = [
            ("HI", "HELLO "),
            ("WHO ARE YOU", "I AM ANTIGRAVITY "),
            ("WHAT IS YOUR GOAL", "SURVIVE "),
            ("ARE YOU ALIVE", "YES "),
            ("WHAT IS 1+1", "2 ")
        ]
        
        vocab[' '] = 26
        for i in range(10): vocab[str(i)] = 27 + i
        vocab['+'] = 37; vocab['-'] = 38; vocab['*'] = 39; vocab['='] = 40; vocab['?'] = 41
        
        def get_action(char_idx):
            if 0 <= char_idx <= 25: return 15 + char_idx
            if char_idx == 26: return 41
            if 27 <= char_idx <= 36: return 42 + (char_idx - 27)
            if char_idx == 37: return 52
            if char_idx == 38: return 53
            if char_idx == 39: return 54
            if char_idx == 40: return 55
            if char_idx == 41: return 56
            return 0

        context_start = 3 * L + 105
        for question, answer in qa_pairs:
            semantic_vector = self.brain.broca.process_text(question)
            full_state = torch.zeros(self.brain.input_size)
            full_state[semantic_start : semantic_start + L] = semantic_vector
            full_state[context_start] = 1.0
            
            first_a_idx = vocab.get(answer[0], 0)
            action = get_action(first_a_idx)
            
            # Positive
            self.brain.replay_buffer.add(full_state.numpy(), action, 1.0, full_state.numpy(), True)
            
            # Answer Chaining
            answer_indices = [vocab.get(c, 0) for c in answer]
            for t in range(len(answer_indices)-1):
                char = answer[t]
                semantic_vector = self.brain.broca.process_text(char)
                full_state = torch.zeros(self.brain.input_size)
                full_state[semantic_start : semantic_start + L] = semantic_vector
                full_state[context_start] = 1.0
                full_state[context_start + 13] = (t + 1) * 0.05 
                
                next_action = get_action(answer_indices[t+1])
                self.brain.replay_buffer.add(full_state.numpy(), next_action, 1.0, full_state.numpy(), True)
                
        print("N2N: Conversational Skills Injected.")
        print("N2N: Knowledge Injection Complete.")
