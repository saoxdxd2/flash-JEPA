import torch
import psutil
import os

# === DEVICE MANAGEMENT ===

def get_best_device():
    """
    Returns the most capable torch device available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Future-proofing for MPS (Mac) or XPU (Intel)
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_memory_stats():
    """
    Returns system and GPU memory statistics in MB.
    """
    stats = {
        "sys_total": psutil.virtual_memory().total / (1024 * 1024),
        "sys_available": psutil.virtual_memory().available / (1024 * 1024),
        "sys_percent": psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        stats["gpu_total"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        stats["gpu_allocated"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
        stats["gpu_reserved"] = torch.cuda.memory_reserved(0) / (1024 * 1024)
        stats["gpu_free"] = stats["gpu_total"] - stats["gpu_allocated"]
        
    return stats

def auto_map_to_device(tensor_or_module):
    """
    Automatically maps a tensor or module to the best available device.
    """
    device = get_best_device()
    return tensor_or_module.to(device)

def check_ram_limit(max_ram_mb):
    """
    Checks if current process memory exceeds the limit.
    """
    process = psutil.Process(os.getpid())
    current_mem = process.memory_info().rss / (1024 * 1024)
    return current_mem >= max_ram_mb

# === RESOURCE MONITORING ===

class ResourceMonitor:
    """
    Monitors the physical resources used by the agent (Process).
    Used to calculate biological cost of computing.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_usage(self):
        """
        Returns (ram_mb, cpu_percent).
        """
        # RAM in MB
        ram_bytes = self.process.memory_info().rss
        ram_mb = ram_bytes / (1024 * 1024)
        
        # CPU Percent (interval=None is non-blocking)
        cpu_percent = self.process.cpu_percent(interval=None)
        
        return ram_mb, cpu_percent

# === INFERENCE ENGINE ===

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
