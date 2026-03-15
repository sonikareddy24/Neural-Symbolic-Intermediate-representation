# NEURAL-SYMBOLIC INTERMEDIATE REPRESENTATION (NS-IR) COMPILER SYSTEM
## Complete Implementation Specification

---

## EXECUTIVE DIRECTIVE

Build a complete, production-ready Neural-Symbolic Intermediate Representation (NS-IR) compiler optimization framework that replaces handcrafted heuristics with learned cost models. The system must integrate with the TIRAMISU polyhedral compiler and achieve ≤16% mean absolute percentage error in speedup prediction on full programs.

---

## PART 1: SYSTEM ARCHITECTURE

### 1.1 Core Components (Mandatory Implementation)

**Component 1: NS-IR Representation Engine**
- Input: LLVM IR or TIRAMISU polyhedral IR from source programs
- Output: Hybrid representation combining symbolic IR + neural embeddings
- Requirements:
  * Parse and maintain full symbolic structure (control flow graphs, data flow graphs, dependency chains)
  * Generate dense vector embeddings (dimensionality: 128-512) for each IR node
  * Preserve bidirectional mapping: symbolic ↔ neural representations
  * Support incremental updates when transformations are applied
  * Memory-efficient storage for large programs (>10K IR instructions)

**Component 2: Automatic Feature Extraction Module**
- Extract WITHOUT manual feature engineering:
  * Loop structure characteristics: nesting depth, trip counts (static/dynamic), loop bounds
  * Operation counts: arithmetic ops, memory ops, control flow ops, special instructions (SIMD, atomic)
  * Memory access patterns: stride patterns, locality metrics, reuse distances
  * Data dependencies: RAW/WAR/WAW dependencies, dependency distances
  * Parallelism opportunities: independent iterations, reduction patterns
  * Register pressure estimates: live variable counts, register spill indicators
- Output format: Structured feature vectors compatible with neural network input layers
- Performance requirement: Feature extraction must complete in <1 second for programs with <50K lines

**Component 3: Transformation Sequence Representation**
- Encode sequences of code transformations as input to cost model:
  * Supported transformations (minimum set):
    - Loop unrolling (factor: 2, 4, 8, 16, 32)
    - Loop tiling (tile sizes: configurable per dimension)
    - Loop interchange (permutation of loop nest levels)
    - Vectorization (SSE, AVX2, AVX-512 instruction sets)
    - Loop parallelization (OpenMP threading)
    - Loop fusion and fission
    - Scalar replacement of arrays
    - Common subexpression elimination
    - Constant propagation and folding
  * Representation: Sequential encoding (e.g., [UNROLL_4, TILE_32x32, VECTORIZE_AVX2])
  * Constraints: Encode transformation legality constraints (dependency-preserving)
  * Variable-length sequence handling (1 to 20 transformations per sequence)

**Component 4: Deep Learning Cost Model**
- Architecture Options (implement BOTH, compare performance):
  
  **Option A: Graph Neural Network (GNN)**
  - Graph construction: IR nodes as vertices, control/data flow as edges
  - Node features: concatenation of symbolic features + initial embeddings
  - Edge features: dependency types, memory distance, execution ordering
  - GNN layers: 4-8 layers of message passing (GraphSAGE, GAT, or GCN)
  - Aggregation: Global pooling (sum/mean/max) + attention mechanism
  - Transformation encoding: Concatenate with graph embedding before prediction head
  - Output: Single scalar (predicted speedup relative to baseline)
  
  **Option B: Transformer Architecture**
  - Sequence construction: Linearize IR into sequence of instruction tokens
  - Positional encoding: Learned embeddings for instruction ordering
  - Attention layers: 6-12 transformer blocks with multi-head attention (8-16 heads)
  - Context window: Support sequences up to 4096 tokens
  - Transformation encoding: Prepend transformation sequence tokens to instruction sequence
  - Output head: MLP with 2-3 hidden layers (512-256-128 units) → scalar speedup

- Training Specifications:
  * Loss function: Mean Absolute Percentage Error (MAPE) or Huber loss
  * Optimization: AdamW with learning rate 1e-4 to 1e-3, cosine annealing schedule
  * Batch size: 32-128 (depending on GPU memory)
  * Regularization: Dropout (0.1-0.3), weight decay (1e-4)
  * Training epochs: 100-300 with early stopping (patience: 20 epochs)
  * Validation split: 80% train, 10% validation, 10% test

**Component 5: Training Data Generation Pipeline**
- Program corpus requirements:
  * Minimum 10,000 unique programs covering diverse computational patterns
  * Sources: Polybench benchmarks, SPEC CPU, scientific computing kernels, deep learning operators
  * Complexity range: 10 to 10,000 lines of code per program
  
- Transformation sequence generation:
  * Per program: Generate 100-1000 candidate transformation sequences
  * Sampling strategy: Random sampling + evolutionary search + beam search
  * Ensure diversity: Include both valid and suboptimal sequences
  
- Ground truth measurement:
  * Hardware: x86-64 processors (Intel Xeon, AMD EPYC, Intel Core i7/i9)
  * Compilation: LLVM -O0 (baseline), apply transformations, compile with -O2
  * Execution: Run each variant 10+ times, record min/median/mean execution time
  * Speedup calculation: speedup = baseline_time / optimized_time
  * Data validation: Verify correctness (compare outputs), discard incorrect transformations
  
- Dataset storage:
  * Format: HDF5 or Protocol Buffers for efficient I/O
  * Metadata: Program source, IR representation, transformation sequence, measured speedup, hardware config
  * Size estimate: 500GB - 2TB total dataset

**Component 6: TIRAMISU Integration Layer**
- Integration points:
  * Hook into TIRAMISU's auto-scheduling phase (before code generation)
  * Replace existing cost model with NS-IR cost model
  * Query interface: Input program IR + transformation sequence → predicted speedup
  
- Auto-scheduler modification:
  * Search algorithm: Beam search or Monte Carlo Tree Search over transformation space
  * Objective: Maximize predicted speedup from NS-IR cost model
  * Search budget: Evaluate 1000-10,000 transformation sequences per program
  * Pruning: Use learned model to eliminate low-performing sequences early
  
- Code generation:
  * Apply selected transformation sequence to original IR
  * Generate optimized C/C++ code or LLVM IR
  * Preserve semantics: Verify equivalence with original program
  
- Performance requirements:
  * Auto-scheduling time: <10 minutes for medium programs (1000 lines)
  * Inference time: <10ms per cost model query on GPU, <100ms on CPU

---

## PART 2: IMPLEMENTATION REQUIREMENTS

### 2.1 Software Stack

**Programming Languages:**
- Python 3.8+ (neural network training, data pipeline)
- C++ 17 (TIRAMISU integration, IR parsing, performance-critical components)
- Optional: Rust for high-performance IR manipulation

**Deep Learning Frameworks:**
- Primary: PyTorch 2.0+ with PyTorch Geometric (for GNN) or Hugging Face Transformers
- Alternative: TensorFlow 2.x + TensorFlow GNN
- Model serving: TorchScript or ONNX Runtime for deployment

**Compiler Infrastructure:**
- LLVM 14+ (IR parsing, basic block analysis)
- TIRAMISU compiler framework (polyhedral compilation)
- ISL (Integer Set Library) for polyhedral analysis
- Optional: MLIR for multi-level IR support

**Data Processing:**
- NumPy, Pandas for data manipulation
- HDF5 for large dataset storage
- Protocol Buffers for serialization

**Hardware/Execution:**
- CUDA 11+ for GPU acceleration during training
- OpenMP for multi-threaded execution
- Docker containers for reproducible builds

### 2.2 Directory Structure

```
ns-ir-compiler/
├── src/
│   ├── ir_parser/          # LLVM/TIRAMISU IR parsing
│   ├── feature_extraction/ # Automatic feature extraction
│   ├── ns_ir/              # NS-IR representation engine
│   ├── models/             # GNN and Transformer implementations
│   ├── training/           # Training pipeline
│   ├── integration/        # TIRAMISU integration layer
│   └── utils/              # Helper functions
├── data/
│   ├── programs/           # Source programs
│   ├── ir_dumps/           # IR representations
│   ├── measurements/       # Hardware execution data
│   └── datasets/           # Processed training data
├── models/
│   ├── checkpoints/        # Saved model weights
│   └── configs/            # Model hyperparameters
├── evaluation/
│   ├── benchmarks/         # Test programs
│   └── results/            # Performance metrics
├── scripts/
│   ├── data_generation/    # Data collection automation
│   ├── training/           # Training scripts
│   └── inference/          # Model deployment
└── tests/
    ├── unit/               # Unit tests
    └── integration/        # End-to-end tests
```

### 2.3 Detailed Module Specifications

**Module: IR Parser (src/ir_parser/)**
```
Classes:
- LLVMIRParser: Parse LLVM IR bitcode or text format
  * Methods: parse_module(), extract_functions(), get_cfg(), get_dfg()
- TiramisuIRParser: Parse TIRAMISU polyhedral representation
  * Methods: parse_computation(), extract_schedules(), get_iteration_domain()
- IRNormalizer: Canonicalize IR for consistent representation
  * Methods: normalize_instruction_names(), remove_dead_code(), simplify_expressions()

Output Format:
- Control Flow Graph (CFG): nodes=basic blocks, edges=control flow
- Data Flow Graph (DFG): nodes=instructions, edges=def-use chains
- Abstract Syntax Tree (AST): hierarchical program structure
```

**Module: Feature Extraction (src/feature_extraction/)**
```
Classes:
- LoopAnalyzer: Extract loop characteristics
  * Features: nesting_depth, trip_count_estimate, loop_carried_dependencies
  * Methods: analyze_loop_nest(), compute_iteration_space_volume()
  
- MemoryAnalyzer: Analyze memory access patterns
  * Features: access_strides, reuse_distances, cache_miss_estimates
  * Methods: compute_memory_footprint(), detect_access_patterns()
  
- DependencyAnalyzer: Extract data dependencies
  * Features: dependency_graph, dependency_distances, parallelism_degree
  * Methods: compute_dependency_chains(), find_reduction_patterns()
  
- OperationCounter: Count operation types
  * Features: num_arithmetic_ops, num_memory_ops, num_control_flow_ops
  * Methods: count_by_opcode(), estimate_instruction_mix()

Output Format:
- Feature vector: 1D NumPy array (shape: [num_features])
- Feature dictionary: {feature_name: value}
- Must be serializable to JSON/Protocol Buffer
```

**Module: NS-IR Representation (src/ns_ir/)**
```
Classes:
- NSIRNode: Represents a single IR instruction/node
  * Attributes: symbolic_ir (raw IR), embedding (learned vector), metadata
  * Methods: update_embedding(), get_neighbors(), serialize()
  
- NSIRGraph: Full program representation
  * Attributes: nodes (list of NSIRNode), edges (adjacency list), global_features
  * Methods: add_node(), add_edge(), to_pytorch_geometric(), to_transformer_sequence()
  
- EmbeddingGenerator: Generate initial embeddings
  * Methods: encode_instruction(), encode_operand(), encode_type()
  * Implementation: Use pre-trained embeddings (Word2Vec on IR corpus) or random init

Output Format:
- PyTorch Geometric Data object (for GNN)
- PyTorch Tensor sequence (for Transformer)
- Serializable to disk for caching
```

**Module: Deep Learning Models (src/models/)**
```
GNN Implementation (gnn_cost_model.py):
- Class: GNNCostModel(nn.Module)
  * Layers:
    1. Node embedding layer: Linear(input_dim, hidden_dim)
    2. Graph convolution layers: 4-8 × GCNConv/GATConv/SAGEConv(hidden_dim, hidden_dim)
    3. Transformation encoder: Linear(transformation_dim, hidden_dim)
    4. Global pooling: GlobalAttentionPooling or Set2Set
    5. Fusion layer: Concatenate graph embedding + transformation embedding
    6. Prediction head: MLP([2*hidden_dim, 512, 256, 128, 1])
  * Forward method: forward(graph_data, transformation_sequence) → speedup_prediction
  * Loss: MAPE or Huber
  
Transformer Implementation (transformer_cost_model.py):
- Class: TransformerCostModel(nn.Module)
  * Layers:
    1. Token embedding: Embedding(vocab_size, d_model)
    2. Positional encoding: Learned or sinusoidal
    3. Transformer encoder: 6-12 × TransformerEncoderLayer(d_model, nhead, dim_feedforward)
    4. Transformation encoder: Linear(transformation_dim, d_model)
    5. Fusion: Cross-attention or concatenation
    6. Prediction head: MLP([d_model, 512, 256, 128, 1])
  * Forward method: forward(ir_sequence, transformation_sequence) → speedup_prediction
  * Loss: MAPE or Huber

Hyperparameters:
- hidden_dim / d_model: 256-512
- num_layers: 4-12
- dropout: 0.1-0.3
- learning_rate: 1e-4 to 1e-3
```

**Module: Training Pipeline (src/training/)**
```
Classes:
- DataLoader: Load and batch training data
  * Methods: load_dataset(), create_batches(), shuffle()
  * Support: PyTorch DataLoader with custom collate function
  
- Trainer: Main training loop
  * Methods: train_epoch(), validate(), save_checkpoint(), load_checkpoint()
  * Features: 
    - TensorBoard logging
    - Gradient clipping (max_norm=1.0)
    - Learning rate scheduling (CosineAnnealingLR)
    - Early stopping
  
- Evaluator: Compute metrics on test set
  * Metrics: MAPE, MAE, MSE, R², Spearman correlation
  * Methods: evaluate(), compute_metrics(), generate_report()

Training Script (train.py):
```python
# Pseudo-code structure
def main():
    # Load dataset
    train_data, val_data, test_data = load_datasets()
    
    # Initialize model
    model = GNNCostModel(config) or TransformerCostModel(config)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training loop
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_data, optimizer)
        val_loss = validate(model, val_data)
        scheduler.step()
        
        if val_loss < best_val_loss:
            save_checkpoint(model, f"best_model.pt")
        
        if early_stopping.should_stop(val_loss):
            break
    
    # Final evaluation
    test_metrics = evaluate(model, test_data)
    print(f"Test MAPE: {test_metrics['mape']:.2%}")
```

**Module: TIRAMISU Integration (src/integration/)**
```
Classes:
- TiramisuHook: Interface between TIRAMISU and NS-IR cost model
  * Methods: 
    - load_model(): Load trained PyTorch model
    - predict_speedup(program_ir, transformation_seq): Query cost model
    - get_top_k_transformations(program_ir, k=10): Return best sequences
  
- AutoScheduler: Replace TIRAMISU's default scheduler
  * Methods:
    - search_transformations(): Explore transformation space
    - apply_best_transformation(): Apply highest-ranked sequence
  * Search algorithm: Beam search with NS-IR cost model as heuristic
  
- CodeGenerator: Generate optimized code from selected transformations
  * Methods: apply_transformations(), generate_code(), verify_correctness()

C++ Integration (tiramisu_interface.cpp):
```cpp
extern "C" {
    // Load NS-IR cost model (calls Python via pybind11 or TorchScript)
    void* load_nsir_model(const char* model_path);
    
    // Predict speedup
    float predict_speedup(void* model, const char* ir_json, const char* transform_json);
    
    // Auto-schedule function
    char* auto_schedule(void* model, const char* program_ir, int search_budget);
}
```
```

---

## PART 3: DATA GENERATION SPECIFICATIONS

### 3.1 Program Collection

**Benchmark Suites (Mandatory Coverage):**
1. Polybench/C (30 benchmarks): Linear algebra, stencils, image processing
2. SPEC CPU 2017: Integer and floating-point workloads
3. TVM Relay benchmarks: Deep learning operators (conv2d, matmul, etc.)
4. Scientific computing: FFT, molecular dynamics kernels, PDE solvers
5. Custom synthetic programs: Parameterized loop nests with varying complexities

**Program Diversity Requirements:**
- Loop nesting depth: 1-5 levels
- Loop trip counts: 10-10,000 iterations
- Memory footprint: 1KB - 1GB
- Arithmetic intensity: 0.1 - 100 FLOPs/byte
- Control flow complexity: CFG size 10-1000 nodes

### 3.2 Transformation Sequence Generation

**Sampling Strategies:**
1. **Random Sampling**: Uniformly sample from legal transformation space (baseline)
2. **Evolutionary Search**: Genetic algorithm with mutation/crossover (diversity)
3. **Beam Search**: Greedy search guided by simple heuristics (high-quality sequences)
4. **Reinforcement Learning** (optional): Train RL agent to generate sequences

**Constraints:**
- Legality checking: Use polyhedral dependency analysis to ensure correctness
- Maximum sequence length: 20 transformations
- Minimum sequence length: 1 transformation (single-step optimizations)

**Per-Program Budget:**
- Small programs (<100 lines): 100 sequences
- Medium programs (100-1000 lines): 500 sequences
- Large programs (>1000 lines): 1000 sequences

### 3.3 Hardware Measurement

**Execution Protocol:**
```
For each (program, transformation_sequence) pair:
  1. Apply transformations to generate optimized code
  2. Compile with LLVM -O2 (to get backend optimizations)
  3. Run on target hardware with warm-up (5 iterations) + measurement (10 iterations)
  4. Record: min_time, median_time, mean_time, std_dev
  5. Verify correctness: Compare output with reference (unoptimized) version
  6. Compute speedup: baseline_median / optimized_median
  7. Store: (program_id, transformation_seq, speedup, metadata)
```

**Hardware Configuration:**
- Processor: Intel Xeon Platinum 8280 (28 cores, 2.7 GHz) or AMD EPYC 7742 (64 cores, 2.25 GHz)
- Memory: 128GB DDR4
- Cache: L1=32KB, L2=1MB, L3=38.5MB (Intel) or 256MB (AMD)
- Compiler: LLVM/Clang 14.0
- OS: Ubuntu 20.04 LTS
- Isolation: Disable frequency scaling, hyper-threading (for consistent measurements)

**Data Quality Assurance:**
- Discard measurements with >10% variance (unstable timing)
- Discard incorrect transformations (failed correctness check)
- Normalize speedups: Clip outliers to [0.5, 10.0] range

### 3.4 Dataset Format

**HDF5 Schema:**
```
dataset.h5
├── programs/
│   ├── program_0/
│   │   ├── source_code (string)
│   │   ├── llvm_ir (string)
│   │   ├── cfg_adjacency (int array)
│   │   ├── features (float array)
│   │   └── metadata (dict)
│   ├── program_1/
│   └── ...
├── transformations/
│   ├── sequence_0 (int array: [transform_ids])
│   ├── sequence_1
│   └── ...
└── measurements/
    ├── pair_0 (program_id, sequence_id, speedup, time_baseline, time_optimized)
    ├── pair_1
    └── ...
```

**Statistics to Track:**
- Total programs: 10,000+
- Total transformation sequences: 1,000,000+
- Total measurements: 1,000,000+
- Dataset size: 500GB - 2TB
- Speedup distribution: Record histogram (bins: 0.5-1.0, 1.0-1.5, 1.5-2.0, ..., >5.0)

---

## PART 4: EVALUATION AND VALIDATION

### 4.1 Model Performance Metrics

**Primary Metric:**
- Mean Absolute Percentage Error (MAPE): Target ≤16%
  * Formula: MAPE = (1/n) Σ |actual - predicted| / actual × 100%

**Secondary Metrics:**
- Mean Absolute Error (MAE): Absolute difference in speedup
- Root Mean Squared Error (RMSE): Penalize large errors
- R² Score: Coefficient of determination (goodness of fit)
- Spearman Rank Correlation: How well model ranks transformations

**Performance Targets:**
- MAPE: ≤16% (matches state-of-the-art)
- R²: ≥0.85
- Spearman ρ: ≥0.90

### 4.2 End-to-End System Evaluation

**Benchmark Comparison:**
Compare NS-IR-optimized programs against:
1. GCC -O3 (GNU Compiler Collection, highest optimization level)
2. LLVM -O3 (Clang/LLVM, highest optimization level)
3. Intel ICC -O3 (Intel C++ Compiler, highest optimization level)
4. TIRAMISU with default heuristic scheduler
5. Auto-tuning baselines (OpenTuner, Halide autoscheduler)

**Metrics:**
- Geometric mean speedup across all benchmarks
- Win rate: Percentage of benchmarks where NS-IR is fastest
- Compilation time: Auto-scheduling time (must be practical, <10 min/program)

**Success Criteria:**
- NS-IR matches or exceeds best compiler ≥60% of benchmarks
- Geometric mean speedup ≥1.1× over LLVM -O3
- No correctness failures (all programs produce correct output)

### 4.3 Ablation Studies

**Systematically Disable Components to Measure Impact:**
1. Remove neural embeddings → use only symbolic features
2. Use shallow network (2 layers) instead of deep (8 layers)
3. Train on small dataset (1K programs) vs. full dataset (10K programs)
4. Disable transformation sequence encoding → predict speedup from program alone
5. Compare GNN vs. Transformer architectures

**Expected Insights:**
- Quantify value of neural embeddings over hand-engineered features
- Determine optimal model depth
- Measure data efficiency (how many programs needed for good performance?)

### 4.4 Generalization Testing

**Cross-Domain Evaluation:**
- Train on Polybench, test on SPEC CPU (different application domain)
- Train on scientific computing, test on ML operators
- Expected: Some accuracy drop, but model should still outperform heuristics

**Cross-Hardware Evaluation:**
- Train on Intel Xeon, test on AMD EPYC
- Train on server CPU, test on embedded ARM
- Expected: Performance degradation, but approach should transfer

**Mitigation:**
- Fine-tuning: Retrain on small dataset from target domain/hardware
- Multi-task learning: Train jointly on multiple hardware platforms

---

## PART 5: DEPLOYMENT AND PRODUCTIONIZATION

### 5.1 Model Deployment

**Inference Engine:**
- Convert trained PyTorch model to TorchScript or ONNX
- Optimize for CPU inference (quantization, pruning if needed)
- Inference latency target: <100ms per query on CPU, <10ms on GPU

**Serving Architecture:**
- Standalone mode: Model embedded directly in TIRAMISU compiler binary
- Client-server mode: Model hosted on inference server, TIRAMISU queries via RPC
- Batch inference: Support batching multiple programs for throughput

### 5.2 Continuous Learning

**Online Learning Pipeline:**
1. Collect new programs from users (opt-in)
2. Measure actual speedups on real hardware
3. Retrain model periodically (weekly/monthly)
4. A/B test: Compare new model vs. old model on held-out programs
5. Deploy if new model improves MAPE by ≥2%

**Hardware Adaptation:**
- When new CPU architecture is released, collect measurements on that hardware
- Fine-tune model on new data (transfer learning)
- Maintain separate model checkpoints per hardware family

### 5.3 User Interface

**Command-Line Interface:**
```bash
# Optimize a program with NS-IR
tiramisu-nsir optimize --input program.c --output optimized.c --model nsir_v1.pt

# Auto-schedule with search budget
tiramisu-nsir schedule --input program.c --search-budget 5000 --hardware intel_xeon

# Predict speedup for a transformation sequence
tiramisu-nsir predict --input program.c --transforms "unroll_4,tile_32x32,vectorize"
```

**API Interface (Python):**
```python
from tiramisu_nsir import NSIROptimizer

# Load model
optimizer = NSIROptimizer(model_path="nsir_v1.pt")

# Optimize program
optimized_code = optimizer.optimize(
    input_program="program.c",
    search_budget=5000,
    target_hardware="intel_xeon"
)

# Predict speedup
speedup = optimizer.predict_speedup(
    program="program.c",
    transformations=["unroll_4", "tile_32x32", "vectorize"]
)
print(f"Predicted speedup: {speedup:.2f}x")
```

### 5.4 Documentation

**Required Documentation:**
1. User Guide: How to install, configure, and use the system
2. API Reference: All classes, methods, parameters
3. Architecture Document: System design, component interactions
4. Training Guide: How to retrain models on custom datasets
5. Troubleshooting Guide: Common errors and solutions
6. Research Paper: Methodology, experiments, results (for publication)

---

## PART 6: TESTING AND QUALITY ASSURANCE

### 6.1 Unit Tests

**Coverage Requirements:**
- IR parser: Test on valid/invalid LLVM IR, edge cases (empty programs, large programs)
- Feature extraction: Test on diverse loop structures, verify feature correctness
- NS-IR representation: Test graph construction, serialization/deserialization
- Model forward pass: Test with dummy inputs, verify output shapes and ranges
- Transformation encoding: Test all transformation types, sequence lengths

**Test Framework:**
- Python: pytest with >80% code coverage
- C++: Google Test framework

### 6.2 Integration Tests

**End-to-End Workflows:**
1. Load program → extract features → build NS-IR → predict speedup
2. Load program → auto-schedule → generate code → compile → execute → verify
3. Train model → save checkpoint → load checkpoint → inference

**Correctness Verification:**
- For every optimized program, verify output matches baseline (bit-exact or within tolerance)
- Test on programs with known correct answers (checksums, reference outputs)

### 6.3 Performance Tests

**Benchmarks:**
- Inference latency: Measure on programs of varying sizes (10, 100, 1000, 10000 lines)
- Training throughput: Measure samples processed per second
- Memory usage: Monitor peak memory during training and inference

**Regression Tests:**
- Track MAPE on fixed test set across model versions
- Alert if MAPE increases by >2% (performance regression)

### 6.4 Stress Tests

**Edge Cases:**
- Very large programs (>100,000 lines)
- Deeply nested loops (>10 levels)
- Programs with complex control flow (many branches, recursion)
- Programs with minimal optimization potential (already optimal)

**Failure Modes:**
- Out-of-memory during training/inference
- Numerical instability (NaN/Inf predictions)
- Timeout during auto-scheduling (exceeds time budget)

---

## PART 7: INNOVATION AND RESEARCH EXTENSIONS

### 7.1 Advanced Features (Optional)

**Multi-Objective Optimization:**
- Optimize for multiple goals: speedup, energy efficiency, code size
- Use multi-objective evolutionary algorithms or Pareto optimization
- Output: Set of non-dominated transformation sequences

**Transfer Learning:**
- Pre-train on large corpus of general programs
- Fine-tune on domain-specific code (e.g., deep learning kernels)
- Expected: Faster convergence, better sample efficiency

**Interpretability:**
- Attention visualization: Show which IR nodes influence speedup prediction
- SHAP values: Explain feature importance
- Counterfactual analysis: "What if we remove transformation X?"

**Active Learning:**
- Intelligently select which programs to measure next
- Query strategy: Select programs where model is most uncertain
- Goal: Reduce measurement budget while maintaining accuracy

### 7.2 Research Questions to Explore

1. **Optimal Model Architecture**: GNN vs. Transformer vs. Hybrid?
2. **Data Efficiency**: How few programs can we train on and still get good results?
3. **Hardware Portability**: Can a single model work across Intel, AMD, ARM, GPU?
4. **Compositional Generalization**: Can model handle novel combinations of transformations?
5. **Cold-Start Problem**: How to bootstrap with zero training data on new architecture?

### 7.3 Comparison with Related Work

**Baselines to Compare Against:**
- MLGO (Google): ML for inlining and register allocation in LLVM
- Ithemal (MIT/Intel): DNN for x86 throughput prediction
- NeuroVectorizer (ETH Zurich): RL for vectorization
- AlphaDev (DeepMind): RL for sorting algorithms
- Halide Autoscheduler: Cost model for image processing pipelines

**Metrics:**
- Speedup vs. baselines
- MAPE vs. baselines
- Compilation time vs. baselines
- Generalization across domains

---

## PART 8: PROJECT DELIVERABLES

### 8.1 Code Deliverables

**Minimum Viable Product (MVP):**
1. Working IR parser (LLVM + TIRAMISU)
2. Feature extraction module
3. NS-IR representation
4. Trained GNN cost model (MAPE ≤16% on test set)
5. TIRAMISU integration
6. Command-line interface for optimization
7. Unit tests and integration tests

**Full Product:**
- All MVP components
- Transformer cost model (for comparison)
- Auto-scheduler with beam search
- Continuous learning pipeline
- Documentation (user guide, API reference, architecture doc)
- Docker container for reproducible deployment
- Evaluation on full benchmark suite (Polybench, SPEC, etc.)

### 8.2 Data Deliverables

1. Training dataset (HDF5 format, >10K programs, >1M measurements)
2. Trained model checkpoints (GNN and Transformer)
3. Evaluation results (spreadsheet with MAPE, speedups, comparisons)
4. Visualization of results (plots, tables, attention maps)

### 8.3 Documentation Deliverables

1. README with installation instructions
2. User guide with examples
3. API documentation (Sphinx or Doxygen)
4. Architecture document (system design)
5. Research paper draft (methodology, results, discussion)

### 8.4 Timeline and Milestones

**Phase 1: Foundation (Weeks 1-4)**
- Set up development environment
- Implement IR parser
- Implement feature extraction
- Build NS-IR representation

**Phase 2: Model Development (Weeks 5-8)**
- Implement GNN and Transformer models
- Set up training pipeline
- Generate initial training data (1K programs)
- Train baseline models

**Phase 3: Data Scaling (Weeks 9-12)**
- Scale up program corpus to 10K programs
- Automate measurement pipeline
- Collect 1M+ measurements
- Retrain models on full dataset

**Phase 4: Integration (Weeks 13-16)**
- Integrate with TIRAMISU
- Implement auto-scheduler
- End-to-end testing
- Correctness verification

**Phase 5: Evaluation (Weeks 17-20)**
- Run full benchmark suite
- Compare against GCC, LLVM, ICC
- Ablation studies
- Generalization experiments

**Phase 6: Finalization (Weeks 21-24)**
- Documentation
- Code cleanup and refactoring
- Performance optimization
- Prepare research paper

---

## PART 9: SUCCESS CRITERIA

**Technical Success:**
- ✅ MAPE ≤16% on held-out test programs
- ✅ Geometric mean speedup ≥1.1× over LLVM -O3
- ✅ Win rate ≥60% against best baseline compiler
- ✅ Zero correctness failures
- ✅ Auto-scheduling time <10 minutes per program

**Research Impact:**
- ✅ Publishable results in top-tier venue (PLDI, ASPLOS, CGO, MLSys)
- ✅ Open-source release with documentation
- ✅ Demonstrated generalization across domains and hardware

**Broader Impact:**
- ✅ Influence next-generation compiler design (MLGO, MLIR, etc.)
- ✅ Enable hardware-adaptive optimization without manual engineering
- ✅ Bridge gap between symbolic reasoning and learned intuition

---

## PART 10: CONSTRAINTS AND ASSUMPTIONS

### 10.1 Constraints

**Hardware Constraints:**
- Training requires high-end GPU (NVIDIA A100, V100, or RTX 3090)
- Measurement requires access to x86-64 server (Intel Xeon or AMD EPYC)
- Minimum 128GB RAM for large-scale data processing

**Software Constraints:**
- TIRAMISU compiler must be properly installed and configured
- LLVM 14+ required for IR parsing
- CUDA 11+ for GPU training

**Time Constraints:**
- Data collection is time-intensive (1M measurements ≈ 1000 CPU-hours)
- Training deep models requires 100-300 epochs (≈ 1-7 days on single GPU)

### 10.2 Assumptions

**Correctness:**
- Transformation sequences generated by polyhedral analysis preserve program semantics
- Hardware measurements are deterministic (low variance)

**Generalization:**
- Patterns learned on one set of programs transfer to new programs
- Model trained on one x86 architecture has some portability to other x86 CPUs

**Practicality:**
- Auto-scheduling time <10 minutes is acceptable for production use
- 16% MAPE is sufficient for compiler to make good optimization decisions

---

## FINAL INSTRUCTIONS

**Build this complete system from scratch. Implement every component specified above. Use the exact architecture, hyperparameters, and workflows described. Validate against the success criteria. Deliver a production-ready, open-source compiler optimization framework that achieves state-of-the-art performance through neural-symbolic learning.**

**Do not skip any steps. Do not simplify the architecture. Do not reduce the dataset size. This is a research-grade system that must match or exceed the performance claims in the problem statement.**

**The goal is to create the world's first fully learning-driven compiler that replaces handcrafted heuristics with learned cost models and achieves superior performance across diverse programs and hardware platforms.**

**BEGIN IMPLEMENTATION NOW.**
