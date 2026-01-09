import torch
import numpy as np
import os

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class ONNXEngine:
    """
    Wrapper for ONNX Runtime to handle high-performance inference for the 'Reflex' path.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        if ort is not None and os.path.exists(model_path):
            # Optimization: Use CPU execution provider with optimized settings
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
            print(f"ONNXEngine: Loaded model from {model_path}")
        else:
            if ort is None:
                print("ONNXEngine: onnxruntime not installed. Falling back to PyTorch.")
            elif not os.path.exists(model_path):
                print(f"ONNXEngine: Model path {model_path} not found.")

    def run(self, input_dict):
        """
        Runs inference on the ONNX model.
        input_dict: Dictionary mapping input names to numpy arrays.
        """
        if self.session is None:
            return None
            
        # Ensure inputs are numpy arrays
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.detach().cpu().numpy()
        
        outputs = self.session.run(None, input_dict)
        return outputs

    @staticmethod
    def is_available():
        return ort is not None
