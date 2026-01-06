import numpy as np
import random
import pickle

class SparseLiquidTree:
    """
    Alternative SNN Architecture for CPU (i5-1035G1).
    
    Design Philosophy:
    - Avoid heavy matrix multiplications (O(N^2)).
    - Use conditional logic and sparse indexing (O(k*N)).
    - "Liquid" state that ripples through the tree.
    """
    def __init__(self, input_size, output_size, reservoir_size=256):
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        
        # State
        self.potentials = np.zeros(reservoir_size, dtype=np.float32)
        self.spikes = np.zeros(reservoir_size, dtype=np.float32)
        
        # Sparse Connectivity: Adjacency List approach
        # For each neuron, list of targets and weights
        # This is much faster on CPU for high sparsity than a dense matrix with zeros
        self.connections = [[] for _ in range(reservoir_size)]
        self.input_connections = [[] for _ in range(input_size)]
        
        self._init_topology()

    def _init_topology(self):
        # Random sparse initialization
        # Connect inputs to reservoir
        for i in range(self.input_size):
            # Connect to 5 random neurons
            targets = np.random.choice(self.reservoir_size, 5, replace=False)
            for t in targets:
                weight = np.random.uniform(0.1, 1.0)
                self.input_connections[i].append((t, weight))
                
        # Connect reservoir to itself
        for i in range(self.reservoir_size):
            # Connect to 3 random neighbors
            targets = np.random.choice(self.reservoir_size, 3, replace=False)
            for t in targets:
                if t != i:
                    weight = np.random.uniform(-0.5, 0.5)
                    self.connections[i].append((t, weight))

    def forward(self, input_vector):
        """
        Forward pass using sparse propagation (Event-Driven).
        """
        # 1. Input Spikes
        # Only process non-zero inputs (Sparse Input)
        active_inputs = np.where(input_vector > 0)[0]
        
        for i in active_inputs:
            val = input_vector[i]
            for target, weight in self.input_connections[i]:
                self.potentials[target] += val * weight

        # 2. Internal Spikes (from previous step)
        active_neurons = np.where(self.spikes > 0)[0]
        
        # Propagate spikes
        for i in active_neurons:
            for target, weight in self.connections[i]:
                self.potentials[target] += weight

        # 3. Fire & Reset
        # Vectorized threshold check
        fired = self.potentials >= 1.0
        
        self.spikes[:] = 0.0
        self.spikes[fired] = 1.0
        
        # Soft reset
        self.potentials[fired] -= 1.0
        self.potentials *= 0.9 # Decay
        
        # 4. Readout (Simple mean of last N neurons for now)
        # In a real tree, we'd have specific output nodes
        output = self.spikes[:self.output_size] # Mock readout
        
        # Calculate Energy Cost (Total Spikes)
        energy_cost = np.sum(self.spikes)
        
        return output, energy_cost

    def apply_plasticity(self, learning_rate=0.001):
        """
        Hebbian Learning: "Neurons that fire together, wire together."
        If Pre-Synaptic (i) and Post-Synaptic (Target) both fired, strengthen connection.
        """
        # Identify neurons that fired in this step
        fired_indices = np.where(self.spikes > 0)[0]
        fired_set = set(fired_indices)
        
        for i in fired_indices:
            # This neuron (i) is Pre-Synaptic
            new_connections = []
            for target, weight in self.connections[i]:
                if target in fired_set:
                    # Post-Synaptic (target) also fired -> Strengthen
                    new_weight = min(2.0, weight + learning_rate)
                else:
                    # Post-Synaptic didn't fire -> Weak decay (Homeostasis)
                    new_weight = weight * 0.999 
                
                # Pruning: Remove weak connections
                if abs(new_weight) > 0.01:
                    new_connections.append((target, new_weight))
            
            self.connections[i] = new_connections

    def mutate(self):
        """
        Evolves the sparse topology.
        """
        # Add connection
        src = random.randint(0, self.reservoir_size - 1)
        tgt = random.randint(0, self.reservoir_size - 1)
        w = random.uniform(-1, 1)
        self.connections[src].append((tgt, w))
        
        # Remove connection
        src = random.randint(0, self.reservoir_size - 1)
        if self.connections[src]:
            self.connections[src].pop(random.randint(0, len(self.connections[src])-1))

    def crossover(self, other_tree):
        """
        N2N2: Creates a child tree by combining this tree with another.
        Inherits connections from both parents.
        """
        # Create child with same dimensions (or average/max if we were evolving size here)
        child = SparseLiquidTree(self.input_size, self.output_size, self.reservoir_size)
        
        # Clear random init
        child.connections = [[] for _ in range(self.reservoir_size)]
        child.input_connections = [[] for _ in range(self.input_size)]
        
        # Inherit Input Connections
        # Handle size mismatch (if parents have different hidden sizes)
        min_input = min(self.input_size, other_tree.input_size)
        
        for i in range(self.input_size):
            if i < min_input:
                # Inherit from both
                conns = self.input_connections[i] + other_tree.input_connections[i]
                if len(conns) > 10:
                    conns = random.sample(conns, 10)
                child.input_connections[i] = conns
            else:
                # New inputs (if child is larger than parents) -> Random Init
                targets = np.random.choice(child.reservoir_size, 5, replace=False)
                for t in targets:
                    weight = np.random.uniform(0.1, 1.0)
                    child.input_connections[i].append((t, weight))
            
        # Inherit Internal Connections
        min_res = min(self.reservoir_size, other_tree.reservoir_size)
        
        for i in range(self.reservoir_size):
            if i < min_res:
                conns = self.connections[i] + other_tree.connections[i]
                # Filter out targets that are out of bounds for the new child
                conns = [(t, w) for t, w in conns if t < child.reservoir_size]
                
                if len(conns) > 10:
                    conns = random.sample(conns, 10)
                child.connections[i] = conns
            else:
                # New neurons -> Random Init
                targets = np.random.choice(child.reservoir_size, 3, replace=False)
                for t in targets:
                    if t != i:
                        weight = np.random.uniform(-0.5, 0.5)
                        child.connections[i].append((t, weight))
            
        return child

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
