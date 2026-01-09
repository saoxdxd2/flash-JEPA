# Flash-JEPA (Project Overview)

Flash-JEPA is a Python/PyTorch research codebase for a biologically-inspired agentic AI system. It implements an EvolutionaryBrain with multiple brain modules (e.g., Broca, Basal Ganglia, Amygdala, Neural Memory), lifecycle training phases (imprinting, stabilization, grounding, schooling, consolidation/dreaming, evolution), and optional ONNX export for a reflex inference path.

## Primary entry points
- autonomous_life.py: runs a live agent loop using a population/genome.
- run_lifecycle.py: runs an automated lifecycle pipeline.

## Testing approach
The repository includes verification scripts under tests/ and scripts/verify_*.py (not primarily pytest-based). Key checks include ONNX export, hybrid switching logic, and dynamic growth/sprouting stability.
