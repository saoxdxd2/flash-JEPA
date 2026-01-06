import torch
import numpy as np
import time

class HyperTransfer:
    """
    N2N2 (Neural-to-Neural 2.0) - Hyper-Stimulation Transfer.
    
    Instead of Teacher-Student Distillation (running inference on a big model),
    we treat the Big Model's parameters (embeddings) as a dataset.
    
    We "spike" the Liquid Network with these parameter vectors directly,
    forcing it to form associations (Long Term Potentiation) without
    needing the Big Model to actually "think".
    """
    def __init__(self, brain):
        self.brain = brain
        
    def load_source_weights(self, source_data, target_dim=64, max_concepts=10000):
        """
        Loads and compresses source weights.
        
        Args:
            source_data: Dict {Word: Vector} (e.g., Llama embeddings)
            target_dim: Our brain's input dimension (64 for Broca)
            max_concepts: Limit to top K words to save RAM
        """
        print(f"N2N2: Loading source data ({len(source_data)} items)...")
        
        # 1. Filter Vocabulary (Don't take everything!)
        # In a real scenario, we'd sort by frequency. Here we just take the first N.
        if max_concepts is not None:
            filtered_items = list(source_data.items())[:max_concepts]
        else:
            filtered_items = list(source_data.items())
        
        # 2. Dimension Projection (4096 -> 256)
        # If the source vectors are huge, we project them down.
        # We use a Random Projection matrix (Johnson-Lindenstrauss lemma says this preserves distances)
        
        sample_vec = np.array(filtered_items[0][1])
        
        # Check if Visual Data (3D: Channels, H, W)
        self.is_visual = (len(sample_vec.shape) == 3)
        
        if not self.is_visual and len(sample_vec.shape) > 1:
            sample_vec = sample_vec.flatten()
            
        source_dim = sample_vec.shape[0] if not self.is_visual else sample_vec.shape[0] # Channels?
        # Actually, for visual, we don't project dimensions usually, we just pass the patch.
        # Unless we want to project channels? Qwen=3, Broca=3. No projection needed.
        
        if not self.is_visual and source_dim > target_dim:
            print(f"N2N2: Compressing dimensions {source_dim} -> {target_dim} (Deterministic)...")
            # Create a fixed projection matrix using a seed for consistency
            g = torch.Generator()
            g.manual_seed(42)
            self.projection_matrix = torch.randn(source_dim, target_dim, generator=g) / np.sqrt(target_dim)
        else:
            self.projection_matrix = None
            
        self.source_embeddings = filtered_items
        print(f"N2N2: Ready to imprint {len(filtered_items)} concepts.")
        
        # --- Intelligence Boost: Direct Seeding ---
        # If the brain's Broca module exists, we seed it with these concepts immediately.
        # This gives the agent "innate" knowledge before the imprinting phase even starts.
        if hasattr(self.brain, 'broca'):
            # Convert filtered vectors to a single tensor for seeding
            vectors = [torch.tensor(v, dtype=torch.float32) for _, v in filtered_items]
            concepts_tensor = torch.stack(vectors)
            
            # Apply projection if needed to match Broca's embedding_dim (256)
            if self.projection_matrix is not None:
                # Note: projection_matrix here is source_dim -> 64. 
                # But Broca's seed_knowledge expects 256 (embedding_dim).
                # We need a projection to 256.
                source_dim = concepts_tensor.shape[1]
                target_dim = 256
                g = torch.Generator()
                g.manual_seed(42)
                proj_256 = torch.randn(source_dim, target_dim, generator=g) / np.sqrt(target_dim)
                concepts_tensor = torch.matmul(concepts_tensor, proj_256)
                
            self.brain.broca.seed_knowledge(concepts_tensor)
        
    def calculate_fisher_information(self, sample_size=100):
        """
        Calculates the Fisher Information Matrix (FIM) to identify important weights.
        Used for Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
        """
        print("N2N2: Calculating Fisher Information (EWC)...")
        broca = self.brain.broca
        fisher = {}
        
        # Initialize Fisher dict for all parameters
        for name, param in broca.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
                
        # Sample from existing knowledge (or just random noise if no prior data)
        # In a perfect world, we'd replay old data. Here, we use the current state as a proxy.
        broca.eval()
        
        for i in range(sample_size):
            # Generate random input or sample from Titans Memory if available
            # Using random input to probe sensitivity
            input_vec = torch.randn(1, broca.experts[0].input_size)
            
            broca.zero_grad()
            output = broca.process_text_embedding(input_vec)
            
            # Use the output magnitude as a proxy for "importance" or "activity"
            # We want to preserve the mapping: Input -> Output
            loss = output.pow(2).mean()
            loss.backward()
            
            for name, param in broca.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2) / sample_size
                    
        self.fisher_information = fisher
        
        # Store current parameters as "optimal" for the previous task
        self.opt_params = {}
        for name, param in broca.named_parameters():
            if param.requires_grad:
                self.opt_params[name] = param.data.clone()
                
        print("N2N2: Fisher Information Calculated.")

    def ewc_loss(self, lambda_ewc=1000):
        """
        Calculates the EWC regularization loss.
        """
        loss = 0
        if not hasattr(self, 'fisher_information'):
            return 0
            
        for name, param in self.brain.broca.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                opt_param = self.opt_params[name]
                loss += (fisher * (param - opt_param).pow(2)).sum()
        return loss * (lambda_ewc / 2)

    def imprint_knowledge(self, save_interval=5000, checkpoint_path=None, resume=True, use_ewc=True):
        """
        The Core Loop: Hyper-Stimulation (Biological Mode).
        
        Instead of filling RAM with vectors, we use the Big Model's data 
        to TRAIN the Liquid Network's synapses directly.
        
        We 'show' the concept to the brain and reward it for activating,
        triggering Long-Term Potentiation (LTP) in the weights.
        """
        import os
        import json
        
        print("N2N2: Starting Biological Imprinting (Synaptic Transfer)...")
        
        # Calculate Fisher Information before starting new learning
        if use_ewc:
            self.calculate_fisher_information()
            
        count = 0
        start_index = 0
        
        # Resume Logic
        if resume and checkpoint_path:
            meta_path = checkpoint_path + ".meta"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        start_index = meta.get("count", 0)
                        print(f"N2N2: Resuming from concept index {start_index}...")
                except Exception as e:
                    print(f"N2N2: Failed to load resume metadata: {e}")
        
        # We need to access the Broca module to train it
        broca = self.brain.broca
        broca.train() # Ensure training mode
        
        optimizer = torch.optim.Adam(broca.parameters(), lr=0.001)
        
        batch_size = 512 # Massive speedup from batching
        
        for i in range(start_index, len(self.source_embeddings), batch_size):
            batch = self.source_embeddings[i:i+batch_size]
            
            # 1. Prepare Batched Input Spike
            batch_vectors = []
            for _, vector in batch:
                if not isinstance(vector, torch.Tensor):
                    vector = torch.tensor(vector, dtype=torch.float32)
                if not self.is_visual and vector.dim() > 1:
                    vector = vector.flatten()
                batch_vectors.append(vector)
            
            vectors_tensor = torch.stack(batch_vectors)
            
            # --- Dynamic Dimension Handling ---
            # Check if Broca's dimensions have changed (e.g., due to growth)
            target_dim = broca.embedding_dim
            if not self.is_visual:
                if self.projection_matrix is not None:
                    if self.projection_matrix.shape[1] != target_dim:
                        print(f"N2N2: Updating Projection Matrix to match Broca's latent dim {target_dim}...")
                        source_dim = self.projection_matrix.shape[0]
                        self.projection_matrix = torch.randn(source_dim, target_dim, device=vectors_tensor.device) / np.sqrt(target_dim)
                    
                    # Project to Broca's latent space (e.g., 256)
                    vectors_tensor = torch.matmul(vectors_tensor, self.projection_matrix)
                
                # Normalize
                vectors_tensor = torch.tanh(vectors_tensor)
            
            # 2. "Show" the batch to the Liquid Network
            optimizer.zero_grad()
            
            # Forward pass (activates neurons)
            if self.is_visual:
                output = broca.process_visual(vectors_tensor)
            else:
                # We use process_text_embedding which projects 256 -> 64 and then runs experts
                output = broca.process_text_embedding(vectors_tensor) 
            
            # 3. Trigger Plasticity (Learning)
            # Autoencoder objective: Output should reconstruct Input
            # target is the latent vector (e.g., 256), output is also in latent space (256)
            target = vectors_tensor
            
            # MSE against target
            reconstruction_loss = torch.nn.functional.mse_loss(output, target)
            
            # Add EWC Loss
            total_loss = reconstruction_loss
            if use_ewc:
                ewc = self.ewc_loss()
                total_loss += ewc
                
            total_loss.backward()
            optimizer.step()
            
            count += len(batch)
            
            # Checkpoint Logic
            if checkpoint_path and (i + batch_size) % save_interval < batch_size:
                print(f"N2N2: Checkpointing at index {i+batch_size}...")
                self.brain.save_model(checkpoint_path)
                with open(checkpoint_path + ".meta", 'w') as f:
                    json.dump({"count": i + batch_size}, f)
            
            if count % 100 < batch_size:
                print(f"N2N2: Imprinted {i+batch_size} concepts into synapses (Loss: {total_loss.item():.4f})...", end='\r')
                
        print(f"\nN2N2: Synaptic Transfer Complete. {count} new concepts learned.")

    def imprint_hierarchy(self, teacher_trajectories, lr=0.001):
        """
        Transfers hierarchical "Thought" logic (e.g., MoE routing) to the brain.
        teacher_trajectories: Dict {Level: Tensor [T, Batch, Dim]}
        """
        print("N2N2: Imprinting Hierarchical Logic (H-JEPA)...")
        
        # We use the brain's learn_trajectory if it supports hierarchy
        # Or we manually train the hierarchy levels
        
        # For now, we'll assume the brain's TRM is a ModularBrain 
        # which contains NeuromodulatedHolographicBrain cortices.
        
        trm = self.brain.trm
        if not hasattr(trm, 'visual_cortex') or not hasattr(trm, 'motor_cortex'):
            print("N2N2: Brain does not have a hierarchical cortex. Skipping.")
            return
            
        # Training loop for hierarchy
        optimizer = torch.optim.Adam(trm.parameters(), lr=lr)
        
        for level, target in teacher_trajectories.items():
            print(f"N2N2: Training Level '{level}' to match teacher...")
            # ... (Implementation of hierarchical MSE training) ...
            # This would involve running the brain and matching h_reflex, h_concept, etc.
            
        print("N2N2: Hierarchical Imprinting Complete.")
        
    def _word_to_actions(self, word):
        """
        Maps a word string to a list of typing actions.
        Assumes standard action mapping (A=15, B=16...)
        """
        actions = []
        if not isinstance(word, str):
            return actions
            
        for char in word.upper():
            if 'A' <= char <= 'Z':
                actions.append(ord(char) - ord('A') + 15)
            elif '0' <= char <= '9':
                actions.append(ord(char) - ord('0') + 42) # Assuming 0 is 42 based on n2n.py
            elif char == ' ':
                actions.append(41)
        return actions
