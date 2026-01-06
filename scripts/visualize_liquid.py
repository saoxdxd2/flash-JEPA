import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.liquid import LiquidGraph

def visualize_dynamics():
    print("=== Liquid Graph Dynamics Visualization ===")
    
    # 1. Initialize Graph
    input_size = 10
    output_size = 5
    hidden_size = 20
    graph = LiquidGraph(input_size, hidden_size, output_size)
    
    print(f"Graph Initialized: {hidden_size} neurons")
    
    # 2. Simulation Loop
    duration = 200 # steps
    dt = 0.1
    
    inputs = np.zeros((duration, input_size))
    # Pulse input at t=10 to t=30
    inputs[10:30, :] = 1.0
    # Random noise input at t=100 to t=150
    inputs[100:150, :] = np.random.rand(50, input_size)
    
    history = {
        'outputs': [],
        'states': [],
        'energy': []
    }
    
    print("Simulating...")
    for t in range(duration):
        input_vec = inputs[t]
        outputs, params, energy = graph.forward(input_vec, dt)
        
        history['outputs'].append(outputs.numpy())
        current_states = np.array([node.x for node in graph.nodes])
        history['states'].append(current_states)
        history['energy'].append(energy)
        
    # 3. Analysis
    outputs = np.array(history['outputs'])
    states = np.array(history['states'])
    energy = np.array(history['energy'])
    
    print(f"Simulation Complete. Steps: {duration}")
    print(f"Max Energy Cost: {np.max(energy):.4f}")
    print(f"Mean Energy Cost: {np.mean(energy):.4f}")
    
    # Check for "Liquid" properties
    # 1. Fading Memory: Do states decay after input stops?
    post_pulse_states = states[30:50]
    decay = np.mean(np.abs(post_pulse_states[-1])) < np.mean(np.abs(post_pulse_states[0]))
    print(f"Fading Memory (Decay after pulse): {decay}")
    
    # 2. Activity: Do states react to input?
    pulse_response = np.mean(np.abs(states[20])) > np.mean(np.abs(states[0]))
    print(f"Input Sensitivity (Response to pulse): {pulse_response}")
    
    # 3. Stability: Do states explode?
    stable = np.max(np.abs(states)) < 100.0
    print(f"Stability (Max state < 100): {stable} (Max: {np.max(np.abs(states)):.2f})")

    # ASCII Plot of Energy
    print("\nEnergy Profile:")
    for t in range(0, duration, 5):
        val = energy[t]
        bars = int(val * 10)
        print(f"T={t:3d} | {'#' * bars} ({val:.2f})")

if __name__ == "__main__":
    visualize_dynamics()
