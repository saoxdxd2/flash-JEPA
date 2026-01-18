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
