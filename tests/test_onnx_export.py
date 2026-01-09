import torch
import numpy as np
import os
import time
from brain.evolutionary_brain import EvolutionaryBrain
from brain.genome import Genome

def test_onnx_export():
    print("--- Testing ONNX Export and Inference ---")
    
    # 1. Setup Brain
    genome = Genome()
    genome.hidden_size = 128 # Small for fast test
    brain = EvolutionaryBrain(genome)
    brain.trm.eval() # Eval mode for export
    
    # 2. Export to ONNX
    onnx_path = "test_reflex.onnx"
    print(f"Exporting to {onnx_path}...")
    success = brain.export_reflex_path(onnx_path)
    
    if not success:
        print("FAILED: ONNX Export failed.")
        return
    
    print("SUCCESS: ONNX Export successful.")
    
    # 3. Compare Outputs
    # 3. Compare Outputs
    dummy_input = torch.randn(1, brain.input_size)
    chemicals = torch.tensor([[0.5, 0.5, 0.2, 0.0]])
    
    # PyTorch Output
    # Ensure states are initialized
    if brain.trm.visual_cortex.h_reflex is None:
        with torch.no_grad():
            _ = brain.trm.forward(dummy_input, chemicals=chemicals)
    
    with torch.no_grad():
        # Capture states BEFORE forward pass
        h_v_r = brain.trm.visual_cortex.h_reflex.clone()
        h_v_c = brain.trm.visual_cortex.h_concept.clone()
        h_v_s = brain.trm.visual_cortex.h_strategy.clone()
        h_m_r = brain.trm.motor_cortex.h_reflex.clone()
        h_m_c = brain.trm.motor_cortex.h_concept.clone()
        h_m_s = brain.trm.motor_cortex.h_strategy.clone()
        initial_states = (h_v_r, h_v_c, h_v_s, h_m_r, h_m_c, h_m_s)

        # Run standard (sparse) forward pass
        py_logits, py_params, py_energy, py_flash = brain.trm.forward(dummy_input, chemicals=chemicals)
        py_conf = py_flash[1]
    
    # ONNX Output
    brain.use_onnx = True
    brain._init_onnx_states()
    for i, s in enumerate(initial_states):
        brain.onnx_states[i] = s.cpu().numpy()
    
    # We'll call decide() but it does action selection, so we'll look at the internal logits
    # Actually, let's just run the engine directly for precise comparison
    onnx_inputs = {
        'input': dummy_input.cpu().numpy(),
        'chemicals': chemicals.cpu().numpy()
    }
    for i, name in enumerate(brain.onnx_engine.session.get_inputs()[2:]):
        onnx_inputs[name.name] = brain.onnx_states[i]
        
    onnx_outputs = brain.onnx_engine.run(onnx_inputs)
    onnx_logits = onnx_outputs[0]
    onnx_params = onnx_outputs[1]
    onnx_conf = onnx_outputs[2]
    
    # 4. Validate
    diff_logits = np.abs(py_logits.cpu().numpy() - onnx_logits).max()
    diff_params = np.abs(py_params.cpu().numpy() - onnx_params).max()
    diff_conf = np.abs(py_conf.cpu().numpy() - onnx_conf).max()
    
    print(f"Max Difference Logits: {diff_logits:.6f}")
    print(f"Max Difference Params: {diff_params:.6f}")
    print(f"Max Difference Conf: {diff_conf:.6f}")
    
    # Tolerance is slightly higher due to Sparse CSR vs Dense numerical differences
    if diff_logits < 5e-3 and diff_params < 5e-3 and diff_conf < 5e-3:
        print("SUCCESS: Outputs match (within tolerance)!")
    else:
        print("WARNING: Outputs differ significantly.")

    # 4.5 Test decide() with ONNX
    print("\n--- Testing EvolutionaryBrain.decide() with ONNX ---")
    brain.use_onnx = True
    action, logits = brain.decide(dummy_input)
    print(f"Decide Action: {action}")
    print("SUCCESS: decide() executed with ONNX path.")

    # 5. Benchmark
    print("\n--- Benchmarking Latency (100 iterations) ---")
    
    # PyTorch Benchmark
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = brain.trm.forward(dummy_input, chemicals=chemicals)
    py_time = (time.time() - start) / 100
    print(f"PyTorch Avg Latency: {py_time*1000:.3f}ms")
    
    # ONNX Benchmark
    start = time.time()
    for _ in range(100):
        _ = brain.onnx_engine.run(onnx_inputs)
    onnx_time = (time.time() - start) / 100
    print(f"ONNX Avg Latency: {onnx_time*1000:.3f}ms")
    
    speedup = py_time / onnx_time
    print(f"Speedup: {speedup:.2f}x")

    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

if __name__ == "__main__":
    test_onnx_export()
