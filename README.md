# Neural-Symbolic Intermediate Representation (NS-IR) Compiler

**NS-IR** is a production-capable optimization system that seamlessly replaces traditional static cost models with deeply parameterized neural estimators. By integrating directly with the TIRAMISU polyhedral compilation engine, NS-IR structures program behavior as a rigorous fusion of exact algebraic polyhedra and rich semantic vector embeddings. 

This repository contains the compiler implementation and machine learning pipelines developed for the research paper: 
**"NS-IR: A Neural-Symbolic Intermediate Representation for Adaptive Compiler Optimization"**
*(Vaka Sonika Reddy, Rayana Pushkar Prabhath, Miriyala Sai Nithya)*

## 🚀 Key Results and Impact

- **Unprecedented Accuracy**: Achieves a test Mean Absolute Percentage Error (MAPE) of just **2.28%** in predicting the execution speedup of complex loop manipulations.
- **Superior Execution Speeds**: Operating as an autonomous auto-scheduler over the standard PolyBench/C suite, NS-IR yields a **1.14x geometric mean speedup** scaling directly against deeply tuned LLVM -O3 baselines.
- **Outperforming Baselines**: Demonstrates a 62.4% win rate compared to legacy static schedulers such as LLVM -O3, GCC -O3, and Intel ICC -O3. 

## 🧠 Architecture Overview

The system design fundamentally outmaneuvers deterministic heuristics in cross-platform compilation logic via the following primary modules:

1. **Polyhedral IR Parsing & Dense Embedding**: Maps both affine algebraic polyhedral definitions and data-flow variables into continuous vector embeddings while strictly preserving code transformation legality bounds. 
2. **Autonomous Feature Extraction**: Natively isolates complex features (e.g., nesting depth dependencies, stride access limitations, and theoretical cache reuse distances) without any manual feature engineering bottleneck. 
3. **Deep Learning Evaluators**: Evaluates topological code metrics by employing deep sequence Transformers scaling up to 4,096 tokens alongside advanced Graph Attention Network (GAT) convolution blocks.

## 📊 Dataset & Methodology

To guarantee accurate deep learning models natively absent of noise, we procedurally generated over 10,000 unique computational kernels. A total of millions of distinct optimization schedule iterations passed across bare-metal server infrastructure (x86-64 Xeon/EPYC platforms) isolating precise execution timing data for training ground-truth representations. 

## 🛠 Component Implementations
- Full integration routines inside the **TIRAMISU** infrastructure, effectively replacing the standalone analytical solver with explicit NS-IR intelligence metrics.
- Highly parameterized neural optimization pipeline relying upon AdamW, GELU activations, and precise Cosine Annealing.
- Modular static analysis processing capable of dynamic topological graph generations in real time.

---

For detailed insights referencing internal system architecture implementations, empirical data derivations, and network parameter definitions, please consult the full research documents located in this repository.

*Maintained under the GitHub account: [@sonikareddy24](https://github.com/sonikareddy24)*
