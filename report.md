# Flash-JEPA: Research & Bug Report

This report summarizes the bugs, architectural inconsistencies, and "magic numbers" identified during the comprehensive technical documentation and research phase.

## 🐛 Identified Bugs & Inconsistencies

### 1. [FIXED] Missing `device.py` Module
- **Resolution**: Implemented `brain/modules/device.py` as a unified HAL.
- **Impact**: Centralized hardware and memory management.

### 2. [FIXED] Dimension Mismatch in `n2n2.py`
- **Resolution**: Replaced fixed random projection with a learnable `projection_adapter` and unified latent dimensions via the genome.
- **Impact**: High-fidelity knowledge transfer with adaptive compression.

### 3. Legacy `n2n.py` Redundancy
- **Issue**: `brain/n2n.py` remains in the codebase despite being superseded by the more advanced `brain/n2n2.py` (Hyper-Stimulation).
- **Impact**: Increased codebase noise and potential for accidental use of outdated transfer methods.

---

## 🏗️ Architectural Issues & "Magic Numbers"

### 1. [FIXED] Hardcoded Action Space
- **Resolution**: `action_size` is now an evolvable gene in `genome.py`.

### 2. [FIXED] Static Latent Adapter Dimensions
- **Resolution**: `latent_adapter_dim` is now an evolvable gene in `genome.py`.

### 3. [FIXED] Heuristic RAM Management
- **Resolution**: `max_ram_mb` is now managed via `device.py` and influenced by genomic parameters.

### 4. Arbitrary Learning Rate Multipliers
- **File**: `brain/evolutionary_brain.py` (Line 72)
- **Issue**: `lr=self.genome.learning_rate * 2.0` uses a magic multiplier of `2.0`.
- **Impact**: Unclear why the TRM optimizer requires exactly double the base learning rate; this should be a genomic hyperparameter.

### 5. Naive Concept Filtering in `n2n2.py`
- **File**: `brain/n2n2.py` (Line 33)
- **Issue**: `list(source_data.items())[:max_concepts]` takes the *first* N items from the teacher's vocabulary.
- **Impact**: This likely captures rare or irrelevant tokens instead of the most frequent/important concepts, leading to inefficient knowledge transfer.

---

## 📈 Performance & Scalability Observations

- **Sparse vs. Dense Transition**: The project is in the middle of a transition to `SparseVectorizedLiquidGraph`. Some modules still default to dense `VectorizedLiquidGraph`, which may limit scalability on massive neuron counts.
- **Random Projection Loss**: `n2n2.py` uses deterministic random projection (seed 42). While mathematically sound for distance preservation, it is a lossy compression that could be replaced with a learned autoencoder for higher fidelity.

---
### 6. [NEW] Static Expert Count in Broca
- **Issue**: `BrocaModule` starts with a fixed number of experts (`num_experts=8`).
- **Risk**: Stagnation in language acquisition if the complexity of the teacher's knowledge exceeds the capacity of the initial experts.
- **Recommendation**: Implement "Dynamic Expert Sprouting" based on `TitansMemory` surprise signals.
