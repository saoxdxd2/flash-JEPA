import json
import time
import os
import torch

class VisualizationLogger:
    """
    Logs brain state for real-time visualization.
    Writes to a JSONL file that can be tailed by a dashboard.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "brain_state.jsonl")
        # Clear old log
        with open(self.log_file, 'w') as f:
            pass
            
    def log_step(self, brain, task_info=None):
        """
        Log a single step of the brain's life.
        """
        # 1. Chemical State
        chem = brain.chemistry
        chemicals = {
            'dopamine': float(chem.dopamine),
            'serotonin': float(chem.serotonin),
            'norepinephrine': float(chem.norepinephrine),
            'cortisol': float(chem.cortisol),
            'energy': float(chem.energy),
            'surprise': float(getattr(chem, 'surprise', 0.0)) # Add surprise
        }
        
        # 2. Genome State (Expression of key genes)
        genes = {}
        key_genes = ['bdnf', 'comt', 'drd2', 'fkbp5', 'arc']
        for g in key_genes:
            genes[g] = brain.genome.get_expression(g)
            
        # 3. VM State (If available)
        vm_stats = {}
        if hasattr(brain.trm, 'visual_cortex') and hasattr(brain.trm.visual_cortex, 'vm'):
            vm = brain.trm.visual_cortex.vm
            vm_stats['active_npus'] = int(vm.active_npus)
            vm_stats['load'] = float(vm.perf_history[-1]) if vm.perf_history else 0.0
            
        # 4. Task Info
        task_data = {}
        if task_info:
            task_data = task_info
            
        # Construct Record
        record = {
            'timestamp': time.time(),
            'age': float(brain.age),
            'chemicals': chemicals,
            'genes': genes,
            'vm': vm_stats,
            'task': task_data
        }
        
        # Write
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def log_event(self, event_type, message):
        """Log a significant event (e.g. Growth, Pruning, Level Up)."""
        record = {
            'timestamp': time.time(),
            'type': 'EVENT',
            'event_type': event_type,
            'message': message
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
