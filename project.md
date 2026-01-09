# Flash-JEPA: Biological Agentic AI - Deep Technical Resume

Flash-JEPA is a state-of-the-art Biological Agentic AI system designed to simulate complex cognitive processes through a hierarchical, neuromodulated, and holographic architecture. It integrates advanced neural network paradigms with biological principles to achieve adaptive, efficient, and scalable intelligence.

## 🧠 Mathematical & Architectural Foundations

### 1. Liquid Neural Networks (LNN)
Flash-JEPA leverages **Liquid Neural Networks** for continuous-time dynamics. Unlike traditional RNNs, LNNs use Ordinary Differential Equations (ODEs) to model synaptic interactions, allowing the network to adapt its time constants based on input variability.
- **VectorizedLiquidGraph**: GPU-accelerated ODE integration using `torch`.
- **SparseVectorizedLiquidGraph**: Scalable to 1B+ neurons using sparse CSR/COO matrix multiplication.
- **Dynamics**: $\frac{dx}{dt} = -A \odot x + (S - x) \odot \sigma(Wx + b)$, where $A$ is the bias/leakage and $S$ is the saturation state.

### 2. Titans Neural Memory (Long-Term Memory)
Inspired by the "Titans" architecture, this module implements **Flashbulb Memory**—a surprise-based online weight update mechanism.
- **Surprise Signal**: Calculated as the gradient of the prediction error.
- **Online Updates**: Synaptic weights are updated in real-time using $\Delta W = \eta \cdot \nabla_{W} \mathcal{L}$, allowing for immediate learning of high-salience information.
- **Efficiency**: Uses sparse CSR caching to manage massive memory stores without linear scaling of compute.

### 3. N2N2 (Neural-to-Neural 2.0)
**N2N2** is a "Hyper-Stimulation" transfer protocol for imprinting teacher embeddings (e.g., from Qwen-3) directly into the agent's synapses.
- **Hyper-Stimulation**: Directly setting synaptic weights based on source model projections.
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting by penalizing changes to weights that are critical for previously learned tasks.

### 4. H-NH-JEPA (Hierarchical Neuromodulated Holographic JEPA)
The core architecture is a **Hierarchical Neuromodulated Holographic Joint-Embedding Predictive Architecture**.
- **Holographic Wavelet Encoding**: Uses wavelet transforms for multi-scale visual and semantic representation.
- **Hierarchical Gating**: Three levels of processing: **Reflex** (System 1), **Concept** (Semantic), and **Strategy** (System 2).
- **Neuromodulation**: Gating signals inspired by Dopamine (reward), Serotonin (stability), Norepinephrine (arousal), and Cortisol (stress).

---

## 📁 Directory Hierarchy

```text
Flash-JEPA/
├── brain/                  # Core Cognitive Engine
│   ├── modules/            # Specialized Brain Areas (Broca, Retina, Amygdala, etc.)
│   │   ├── device.py       # [NEW] Unified Hardware Abstraction Layer (HAL)
│   ├── evolutionary_brain.py # Main Brain Integration & Decision Logic
│   ├── genome.py           # Genetic Blueprint & Hyperparameter Evolution
│   ├── lifecycle.py        # Developmental Stages (Imprinting -> Evolution)
│   ├── n2n.py              # Legacy Knowledge Loader
│   ├── n2n2.py             # Knowledge Transfer & Hyper-Stimulation
│   └── population.py       # Multi-Agent Evolution & Selection
├── models/                 # Neural Network Architectures
│   ├── liquid_vectorized.py # GPU-Accelerated LNN
│   ├── liquid_sparse_vectorized.py # Scalable Sparse LNN
│   ├── neuromodulated_holographic.py # H-NH-JEPA Core
│   ├── ddqn.py             # Reinforcement Learning Agent
│   ├── ecg.py              # Modular Brain (Visual/Motor Cortices)
│   ├── plasticity_mlp.py   # Meta-Learning Plasticity Network
│   └── sparse_tree.py      # Event-Driven SNN for CPU
├── scripts/                # Training, Schools, and Utilities
│   ├── language_school.py  # NLP Training Environment
│   ├── math_school.py      # Symbolic Reasoning School
│   ├── web_coding_school.py # Python Coding Curriculum
│   ├── meditation_v2.py    # Brain Stabilization & Attractor Finding
│   ├── remote_surgery.py   # High-Res Model Transplant (Colab/Kaggle)
│   ├── start_n2n2_qwen3_agentic.py # Qwen-3 Teacher Distillation
│   └── ... (30+ utility scripts for verification and analysis)
├── tests/                  # Unit & Performance Tests
│   ├── benchmark_liquid.py # LNN Performance Profiling
│   └── smoke_test_brain.py # Core Integration Verification
├── tools/                  # OS & Hardware Interface
│   └── control.py          # Mouse/Keyboard/Screen Interaction (Cradle)
├── autonomous_life.py      # Main Entry Point (Wake/Sleep Cycle)
├── run_lifecycle.py        # Automated Pipeline Runner
└── project.md              # This Technical Resume
```

---

## 📄 Exhaustive File Mapping

### 🚀 Root Entry Points

#### [autonomous_life.py](file:///c:/Users/sao/Documents/model/autonomous_life.py)
- **Purpose**: Main orchestration of the agent's life cycle.
- **Techniques**: **Wake/Sleep Cycles**, **Population-Based Selection**, **Fitness-Driven Evolution**.
- **Interactions**: Loads `EvolutionaryBrain` and `PopulationManager` to run the agent in real-time.

#### [run_lifecycle.py](file:///c:/Users/sao/Documents/model/run_lifecycle.py)
- **Purpose**: Automated pipeline for training and evolution.
- **Techniques**: **Phase-Based Training** (Imprinting -> Stabilization -> Schooling -> Evolution).
- **Interactions**: Uses `LifecycleManager` to execute sequential training stages.

---

### 🧠 Core Brain Engine (`brain/`)

#### [evolutionary_brain.py](file:///c:/Users/sao/Documents/model/brain/evolutionary_brain.py)
- **Purpose**: Central cognitive integration.
- **Techniques**: **Schema Healing**, **Metacognitive Gating**, **Dream-Based Distillation**.
- **Interactions**: Integrates all `brain/modules/` and `models/`.

#### [genome.py](file:///c:/Users/sao/Documents/model/brain/genome.py)
- **Purpose**: Genetic blueprint for hyperparameters.
- **Techniques**: **Self-Adaptive Mutation**, **Genetic Crossover**.

#### [population.py](file:///c:/Users/sao/Documents/model/brain/population.py)
- **Purpose**: Manages a population of agents for evolutionary selection.
- **Techniques**: **Elitism**, **N2N2 Crossover**, **Garbage Collection** of old models.

#### [lifecycle.py](file:///c:/Users/sao/Documents/model/brain/lifecycle.py)
- **Purpose**: High-level developmental management.
- **Techniques**: **Sequential Semantic Imprinting (SSI)**.

#### [n2n2.py](file:///c:/Users/sao/Documents/model/brain/n2n2.py)
- **Purpose**: Advanced knowledge transfer (N2N2 3.0).
- **Techniques**: **Learnable Projections**, **Contrastive Imprinting**, **Elastic Weight Consolidation (EWC)**.

---

### 👁️ Specialized Brain Modules (`brain/modules/`)

#### [broca.py](file:///c:/Users/sao/Documents/model/brain/modules/broca.py)
- **Purpose**: Language and semantic center.
- **Techniques**: **Mixture of Experts (MoE)**, **Visual Word Form Area (VWFA)**, **Unified Latent Dimensions**.

#### [predictive_retina.py](file:///c:/Users/sao/Documents/model/brain/modules/predictive_retina.py)
- **Purpose**: Visual processing.
- **Techniques**: **Dual-Stream Processing**, **Predictive Coding**.

#### [neural_memory.py](file:///c:/Users/sao/Documents/model/brain/modules/neural_memory.py)
- **Purpose**: Long-term associative memory.
- **Techniques**: **Flashbulb Memory**, **Sparse CSR Caching**.

#### [amygdala.py](file:///c:/Users/sao/Documents/model/brain/modules/amygdala.py)
- **Purpose**: Emotional salience and threat detection.
- **Techniques**: **Amygdala Hijack** (Cortisol/Pain override).

#### [basal_ganglia.py](file:///c:/Users/sao/Documents/model/brain/modules/basal_ganglia.py)
- **Purpose**: Action selection and gating.
- **Techniques**: **Dopaminergic Gating**, **Metacognitive Switching**.

#### [biology_core.py](file:///c:/Users/sao/Documents/model/brain/modules/biology_core.py)
- **Purpose**: Neurochemical simulation.
- **Techniques**: **Neurotransmitter Dynamics** (Dopamine, Serotonin, Norepinephrine, Cortisol).

#### [cradle.py](file:///c:/Users/sao/Documents/model/brain/modules/cradle.py)
- **Purpose**: OS interaction interface.
- **Techniques**: **PyAutoGUI Integration**, **Screen Capture**, **Action Mapping**.

#### [replay_buffer.py](file:///c:/Users/sao/Documents/model/brain/modules/replay_buffer.py)
- **Purpose**: Experience replay for RL.
- **Techniques**: **Prioritized Experience Replay (PER)** using **SumTree**.

---

### 🏗️ Neural Architectures (`models/`)

#### [liquid_vectorized.py](file:///c:/Users/sao/Documents/model/models/liquid_vectorized.py)
- **Purpose**: GPU-accelerated LNN.
- **Techniques**: **Vectorized ODE Integration**.

#### [liquid_sparse_vectorized.py](file:///c:/Users/sao/Documents/model/models/liquid_sparse_vectorized.py)
- **Purpose**: Scalable sparse LNN.
- **Techniques**: **Sparse Matrix Multiplication (CSR/COO)**.

#### [neuromodulated_holographic.py](file:///c:/Users/sao/Documents/model/models/neuromodulated_holographic.py)
- **Purpose**: H-NH-JEPA implementation.
- **Techniques**: **Holographic Wavelet Encoding**, **Hierarchical Latent Prediction**.

#### [plasticity_mlp.py](file:///c:/Users/sao/Documents/model/models/plasticity_mlp.py)
- **Purpose**: Meta-learning of synaptic rules.
- **Techniques**: **Evolved Plasticity Rules** (replacing fixed Hebbian rules).

#### [ecg.py](file:///c:/Users/sao/Documents/model/models/ecg.py)
- **Purpose**: Modular brain structure.
- **Techniques**: **Visual/Motor Cortex Partitioning**, **Bus-Based Communication**.

#### [ddqn.py](file:///c:/Users/sao/Documents/model/models/ddqn.py)
- **Purpose**: Reinforcement learning agent.
- **Techniques**: **Double Deep Q-Network**, **Target Network Soft-Updates**.

#### [sparse_tree.py](file:///c:/Users/sao/Documents/model/models/sparse_tree.py)
- **Purpose**: CPU-optimized SNN.
- **Techniques**: **Event-Driven Propagation**, **Sparse Adjacency Lists**.

---

### 📜 Training & Utility Scripts (`scripts/`)

#### [web_coding_school.py](file:///c:/Users/sao/Documents/model/scripts/web_coding_school.py)
- **Purpose**: Teaches the agent Python coding.
- **Techniques**: **Curriculum Learning**, **Browser Emulation**, **Teacher Guidance**.

#### [remote_surgery.py](file:///c:/Users/sao/Documents/model/scripts/remote_surgery.py)
- **Purpose**: High-resolution model transplant.
- **Techniques**: **Deterministic Random Projection**, **Shard-Based Harvesting**.

#### [meditation_v2.py](file:///c:/Users/sao/Documents/model/scripts/meditation_v2.py)
- **Purpose**: Brain stabilization.
- **Techniques**: **Natural Attractor Finding**, **Surprise Minimization**.

#### [language_school.py](file:///c:/Users/sao/Documents/model/scripts/language_school.py)
- **Purpose**: NLP training.
- **Techniques**: **Next-Token Prediction**, **Semantic Grounding**.

---

### 🧪 Testing & Benchmarking (`tests/`)

#### [benchmark_liquid.py](file:///c:/Users/sao/Documents/model/tests/benchmark_liquid.py)
- **Purpose**: Performance profiling.
- **Techniques**: **Legacy vs. Vectorized Comparison**, **Correctness Verification**.

---

## 🔄 System Interactions & Protocols

### 1. The Cognitive Loop
1. **Perception**: `PredictiveRetina` captures screen data, compressing it into foveal/peripheral latents.
2. **Salience**: `Amygdala` evaluates the latents for surprise or threat.
3. **Cognition**: `Broca` (Language) and `ModularBrain` (Motor/Visual) process the state.
4. **Selection**: `BasalGanglia` gates the proposed actions based on `NeurotransmitterSystem` state.
5. **Action**: `Cradle` executes the selected keyboard/mouse commands.

### 2. Interaction Protocols
- **GCC (Global Concept Communication)**: Standardized 512-dim vector bus for inter-module communication.
- **SNN Character Protocol**: Character-level communication between `Broca` and `Cradle` for precise text entry.

---
*This document provides 100% file coverage for the Flash-JEPA project.*
