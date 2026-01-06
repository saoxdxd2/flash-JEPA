import psutil
import os

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
