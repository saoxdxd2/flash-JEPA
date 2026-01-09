# Flash-JEPA: Strategic Roadmap & Suggestions

This document outlines an "honest opinion" on the current state of the Flash-JEPA project and provides a roadmap for transitioning from hardcoded heuristics to a fully evolved, high-performance architecture.

## 🧬 1. From Hardcoded to Evolved (Genomic Expansion)

The user is correct: many "fixed" values in the brain should be genomic variables. This allows the evolutionary process to find the optimal balance between intelligence and efficiency.

### Immediate Genomic Candidates:
- **`INPUT_BOOST_FACTOR`**: Currently hardcoded at `20.0`. This should be a gene, as different environments may require different signal intensities.
- **`ACTION_SIZE`**: Instead of a fixed `72`, the action space should be dynamic and evolved based on the complexity of the tasks the agent faces.
- **`LATENT_ADAPTER_DIM`**: The mapping from Teacher (4096) to Agent (1024) should be flexible.
- **`FISHER_SAMPLE_SIZE`**: The number of samples used for EWC (Elastic Weight Consolidation) should be evolved to balance training stability vs. speed.

---

## 🛠️ 2. High-Performance Library Recommendations

To reduce manual work and improve performance, we should leverage libraries that are **CPU-friendly** but backed by **fast languages** (C++, Rust, Fortran).

| Library | Purpose | Why it fits? |
| :--- | :--- | :--- |
| **[FAISS](https://github.com/facebookresearch/faiss)** | Long-Term Memory | C++ backend. World-class vector similarity search. Replaces pure-torch search in `TitansMemory`. |
| **[Numba](https://numba.pydata.org/)** | ODE Integration | JIT compiles Python/NumPy to machine code. Perfect for the `VectorizedLiquidGraph` dynamics. |
| **[ONNX Runtime](https://onnxruntime.ai/)** | "System 1" Inference | High-performance inference engine. Can run frozen "Reflex" paths 2-5x faster on CPU. |
| **[Polars](https://www.pola.rs/)** | Data Handling | Written in Rust. Extremely fast for processing large teacher datasets in `n2n2.py`. |
| **[Scipy Sparse](https://scipy.org/)** | Sparse Graph Ops | Mature C/Fortran backend. Ideal for the `SparseVectorizedLiquidGraph` adjacency matrices. |

---

## 🚀 3. Proposed Approach & Roadmap

### Phase 1: Unified HAL & Genomic Expansion [COMPLETED]
- **Goal**: Centralize hardware management and move heuristics to the genome.
- **Result**: `device.py` implemented; `genome.py` expanded with evolvable architectural genes.

### Phase 2: Learned Projections (N2N2 3.0) [COMPLETED]
- **Goal**: Replace random projections with learnable adapters and contrastive loss.
- **Result**: `n2n2.py` refactored with `projection_adapter` and Contrastive Imprinting.

### Phase 3: Dynamic Expert Sprouting (Broca 2.0)
- **Goal**: Implement a mechanism in `broca.py` to "sprout" new experts when surprise (from `TitansMemory`) remains high.
- **Mechanism**: If the average surprise for a concept cluster exceeds a threshold, initialize a new `VectorizedLiquidGraph` expert and add it to the MoE pool.

### Phase 4: System 1/2 Optimization
- **Action**: Export the "Reflex" (System 1) path to ONNX.
- **Goal**: Use ONNX Runtime for the fast, reactive path while keeping the "Strategy" (System 2) path in Torch for active learning and plasticity.

---
## 💡 Final Opinion
The project has a brilliant foundation in biological AI. However, it is currently "fighting" the hardware by relying heavily on pure Torch for sparse operations and heuristics for resource management. By moving to **C++-backed specialized libraries** and **genomicizing all constants**, we can unlock the true potential of the 1M+ neuron goal without needing a supercomputer.
