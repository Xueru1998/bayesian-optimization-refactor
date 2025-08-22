# SMAC3 RAG Optimizer - Complete User Guide

## Overview

The SMAC3 RAG Optimizer provides intelligent hyperparameter optimization for Retrieval-Augmented Generation (RAG) pipelines using **Sequential Model-based Algorithm Configuration (SMAC3)** and **Bayesian Optimization Hyperband (BOHB)**. It supports both **global optimization** (entire pipeline) and **component-wise optimization** (sequential component optimization) with advanced multi-fidelity capabilities and intelligent early stopping at both pipeline and component levels.

## Prerequisites

Before running the optimizer, ensure you have the following project structure:

```
your_project/
├── config.yaml                 # Pipeline configuration file
├── autorag_project/            # Data directory
│   ├── corpus.parquet          # Document corpus
│   └── qa_validation.parquet   # Question-answer validation set
└── smac3/                      # Optimizer code
    └── smac_runner.py          # Main entry point
```

**Required Files:**
- **`config.yaml`**: Defines your RAG pipeline components and their parameter search spaces
- **`corpus.parquet`**: Your document collection with columns: `doc_id`, `contents`, `metadata`
- **`qa_validation.parquet`**: Validation data with columns: `qid`, `query`, `retrieval_gt`, `generation_gt`

## Entry Point

The main entry point is `smac_runner.py`:

```bash
python smac3/smac_runner.py [options]
```

## Optimization Modes

### 1. Global Optimization
Optimizes the entire RAG pipeline end-to-end as a single objective function using SMAC3 or BOHB.

```bash
python smac_runner.py --optimization_mode global --n_trials 50 --optimizer smac
```

**Best for:**
- Finding optimal parameter combinations across all components simultaneously
- Components with strong interdependencies
- Multi-objective optimization (score + latency)

### 2. Component-wise Optimization 
Optimizes each RAG component sequentially, using outputs from previous components as inputs to the next.

```bash
python smac_runner.py --optimization_mode componentwise --n_trials 20 --optimizer smac
```

**Best for:**
- Large search spaces that would be computationally expensive for global optimization
- Understanding individual component contributions
- Faster optimization with intermediate results
- Early stopping at high-performing components

## Early Stopping Strategies

The optimizer provides multiple intelligent early stopping mechanisms:

### 1. Pipeline-Level Early Stopping
Stops optimization when a target score is achieved (available in both modes):

```bash
python smac_runner.py --early_stopping_threshold 0.9
```

### 2. Component-Level Early Stopping 

#### For Component-wise Mode:
Automatically skips remaining components when **performance goal is achieved** (success-based early stopping):

```bash
python smac_runner.py \
  --optimization_mode componentwise \
  --component_early_stopping \
  --early_stopping_threshold 0.85
```

**Component-wise Early Stopping Behavior:**
```
✓ query_expansion: 0.82 (optimized)
✓ passage_reranker: 0.87 → GOAL ACHIEVED! Skip remaining components 
⚡ passage_filter: skipped (goal reached)
⚡ passage_compressor: skipped (goal reached)
✓ prompt_maker_generator: 0.67 (final component, always optimized)
```

#### For Global Mode - Bad Configuration Early Stopping(built in):
Terminates individual trials early if components show **fundamentally poor performance**, indicating the configuration is not suitable for the dataset:

**Why Global Mode Needs Bad Config Early Stopping:**
- **Dataset Compatibility**: Some parameter combinations are fundamentally incompatible with your dataset
- **Cost Efficiency**: Stop bad configurations immediately rather than waste time/money on full pipeline evaluation
- **Resource Optimization**: Focus computational budget on promising parameter combinations
- **Faster Convergence**: Eliminate obviously poor configurations early to explore better parameter space

**Global Bad Config Early Stopping Behavior:**
```
Trial 15: retrieval_score=0.05 < threshold(0.1) → BAD CONFIG, STOP TRIAL 
  Reason: Retrieval fundamentally failing, no point testing reranker/generation

Trial 16: retrieval_score=0.12 ✓, reranker_score=0.08 < threshold(0.2) → BAD CONFIG, STOP TRIAL ⚡
  Reason: Reranker performance too low, indicates parameter mismatch

```

**Key Difference:**
- **Component-wise**: Early stopping when components perform **too well** (goal achieved)
- **Global**: Early stopping when components both when perform **too poorly** (bad configuration detected) and perform well.

**Important Note:** Component-wise mode does NOT have bad configuration early stopping - it only has success-based early stopping when performance goals are achieved.

### 3. Multi-Level Early Stopping
Combine both strategies for maximum efficiency in global mode(early stop bad trials is built in, no need to specify anything):

```bash
python smac_runner.py \
  --optimization_mode global \
  --early_stopping_threshold 0.9 \
```

## Optimization Algorithms

### SMAC3 (Sequential Model-based Algorithm Configuration) - Recommended
```bash
--optimizer smac
```
- **Strategy**: Gaussian Process-based Bayesian optimization with intelligent acquisition functions
- **Best for**: Mixed parameter types, conditional dependencies, efficient exploration
- **Advantages**: 
  - Excellent handling of categorical, continuous, and discrete parameters
  - Smart exploration of parameter dependencies
  - Proven performance on configuration problems
  - Built-in multi-objective support
- **Use when**: You want robust, efficient optimization for complex search spaces

### BOHB (Bayesian Optimization Hyperband)
```bash
--optimizer bohb --use_multi_fidelity
```
- **Strategy**: Combines Bayesian optimization with multi-fidelity Hyperband for efficient resource allocation
- **Best for**: Large search spaces, expensive evaluations, multi-fidelity optimization
- **Advantages**:
  - Automatically balances exploration vs exploitation
  - Efficient early stopping of poor configurations
  - Excellent for budget-constrained optimization
- **Requirements**: Automatically enables multi-fidelity optimization
- **Use when**: You have expensive evaluations and want to maximize efficiency

## Multi-Fidelity Optimization

SMAC3 supports advanced multi-fidelity optimization, allowing evaluation of configurations with different computational budgets:

### Enable Multi-Fidelity
```bash
python smac_runner.py \
  --use_multi_fidelity \
  --min_budget_percentage 0.1 \
  --max_budget_percentage 1.0 \
  --eta 3
```

**Benefits:**
- **Fast Initial Screening**: Test many configurations with small data samples
- **Progressive Refinement**: Promising configurations get more computational budget
- **Resource Efficiency**: Avoid wasting time on poor configurations
- **Better Exploration**: Evaluate more diverse configurations within time limits

**Budget Allocation:**
- `min_budget_percentage`: Start with 10% of data (fast evaluation)
- `max_budget_percentage`: Best configurations get 100% of data
- `eta`: Successive halving ratio (3 = keep top 1/3 configurations)

## Command Line Interface

### Quick Start Examples

**Basic component-wise optimization:**
```bash
python smac_runner.py
```

**High-performance global setup with component early stopping:**
```bash
python smac_runner.py \
  --optimization_mode global \
  --optimizer bohb \
  --use_multi_fidelity \
  --n_trials 50 \
  --early_stopping_threshold 0.9 \
  --global_component_early_stopping
```

**Component-wise with multi-fidelity:**
```bash
python smac_runner.py \
  --optimization_mode componentwise \
  --use_multi_fidelity \
  --min_budget_percentage 0.2 \
  --max_budget_percentage 1.0 \
  --eta 3 \
  --component_early_stopping
```

**RAGAS evaluation (Global mode only):**
```bash
python smac_runner.py \
  --optimization_mode global \
  --use_ragas \
  --ragas_llm_model gpt-4o-mini \
  --ragas_metrics context_precision context_recall answer_relevancy
```

### Essential Parameters

| Parameter | Description | Default | Config Example |
|-----------|-------------|---------|---------|
| `--optimization_mode` | Global or component-wise | `global` | `--optimization_mode componentwise` |
| `--optimizer` | SMAC3 or BOHB | `smac` | `--optimizer bohb` |
| `--n_trials` | Total trials (global) or per component | default 50 | `--n_trials 50` |
| `--use_multi_fidelity` | Enable multi-fidelity optimization | `False` | `--use_multi_fidelity` |
| `--config_path` | Configuration file | `config.yaml` | `--config_path my_config.yaml` |
| `--early_stopping_threshold` | Stop when score achieved | `0.9` | `--early_stopping_threshold 0.85` |


### Multi-Fidelity Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--min_budget_percentage` | Minimum evaluation budget | `0.1` | `0.05-0.5` |
| `--max_budget_percentage` | Maximum evaluation budget | `1.0` | `0.5-1.0` |
| `--eta` | Successive halving ratio | `3` | `2-4` |

### Advanced Features

**Email notifications:**
```bash
python smac_runner.py \
  --email_notifications \
  --email_sender your_email@gmail.com \
```

**Custom result directory:**
```bash
python smac_runner.py \
  --result_dir /path/to/custom/results \
  --study_name experiment_v2
```

**Resource management:**
```bash
python smac_runner.py \
  --cpu_per_trial 8 \
  --walltime_limit 7200 \
  --n_workers 1
```

**LLM-based compression evaluation:**
```bash
python smac_runner.py \
  --use_llm_compressor_evaluator \
  --llm_evaluator_model gpt-4o
```

## What You Get From SMAC3 Optimization

### 1. Optimized Configurations

**Global Mode Results:**
- Single best configuration optimizing the entire pipeline
- Multi-objective Pareto front (score vs latency trade-offs)
- Complete parameter settings for production deployment
- Multiple optimal solutions for different priorities
- Component-level early stopping statistics for efficiency analysis

**Component-wise Mode Results:**
- Sequential optimization results for each component
- Best configuration per component with performance scores
- Component interdependency analysis
- Early stopping insights and component bottleneck identification

### 2. Performance Analytics

**Advanced Optimization Metrics:**
- **Convergence Analysis**: SMAC3's Gaussian Process confidence intervals
- **Acquisition Function Insights**: How the optimizer explores vs exploits
- **Multi-fidelity Efficiency**: Budget allocation and early stopping statistics
- **Parameter Sensitivity**: Which parameters have the most impact
- **Early Stopping Efficiency**: Time savings from component-level early termination

### 3. Advanced Multi-Fidelity Results

**Budget Efficiency Analysis:**
- Resource allocation across different budget levels
- Early termination statistics and efficiency gains
- Configuration screening effectiveness
- Time savings compared to full-budget evaluation

### 4. Production-Ready Outputs

**Deployment Configurations:**
- Best single configuration (global mode)
- Component-wise optimized pipeline (component-wise mode)
- Multi-objective Pareto front with trade-off analysis

### 5. Component-wise Mode Advantages

**Sequential Component Analysis:**
- **Early Stopping Intelligence**: Automatically skip remaining components if performance threshold reached
- **Component Bottleneck Detection**: Identify which components limit overall performance
- **Incremental Insights**: Understand how each optimization improves the pipeline

**Smart Component Skipping:**
```
✓ query_expansion: 0.9245 (optimized)
✓ passage_reranker: 0.9012 (optimized, early stopping triggered)
⚡ passage_filter: skipped (early stopping)
⚡ passage_compressor: skipped (early stopping) 
✓ prompt_maker_generator: 0.7012 (final component, always optimized)
```

### 6. Global Mode Bad Configuration Early Stopping Benefits

**Intelligent Trial Filtering (Global Mode Only):**
- **Dataset Compatibility Detection**: Identify parameter combinations fundamentally incompatible with your dataset
- **Cost & Time Efficiency**: Stop bad configurations immediately instead of wasting resources on full pipeline evaluation  
- **Resource Optimization**: Focus computational budget on promising parameter combinations
- **Exploration Enhancement**: Test more diverse configurations within time constraints by eliminating obvious failures early

**Why Component-wise Mode Doesn't Need Bad Config Early Stopping:**
Component-wise mode optimizes one component at a time sequentially, so bad configurations are naturally filtered out during individual component optimization phases. Unlike global optimization where a bad retrieval configuration would make the entire pipeline evaluation pointless, component-wise optimization only tests parameters relevant to the current component with previous components already optimized and fixed.

**Built-in Default Thresholds:**
The optimizer comes with sensible default thresholds that work well for most RAG datasets:
- **retrieval/query_expansion**: < 0.1 (very poor retrieval performance)
- **reranker**: < 0.2 (reranking not improving results)  
- **filter**: < 0.25 (filtering removing too much useful content)
- **compressor**: < 0.3 (compression losing important information)

## Default Settings

The optimizer comes with intelligent defaults optimized for most RAG use cases:

```bash
python smac_runner.py  
```

### Key Default Behaviors

| Setting | Default Value | What It Means |
|---------|---------------|---------------|
| **Mode** | `global` | Optimizes entire pipeline with multi-objective approach |
| **Algorithm** | `smac` | Uses SMAC3 Bayesian optimization |
| **Trials** | by default 50 | Smart calculation based on search space size |
| **Weights** | `0.5/0.5` | Equal importance to retrieval and generation |
| **Multi-objective** | Global: Always ON<br/>Component-wise: Optional | Score + latency optimization |
| **Multi-fidelity** | `False` | Enable with `--use_multi_fidelity` |
| **Embeddings** | Cached | Reuses pre-computed embeddings for speed |
| **Pipeline early stopping** | `0.9` | Stops when 90% score achieved (both modes) |
| **Component early stopping** | `True` | Success-based early termination (component-wise only) |
| **Global component early stopping** | `True` | Bad config early termination (global mode only) **ENABLED BY DEFAULT** |
| **Default bad config thresholds** | Built-in | retrieval/query_expansion: 0.1, reranker: 0.2, filter: 0.25, compressor: 0.3 |
| **W&B logging** | Enabled | Automatic experiment tracking (use --no_wandb to disable wandb logging) |

### Multi-Objective Optimization

**Global Mode (Always Multi-Objective):**
```bash
python smac_runner.py --optimization_mode global --optimizer smac
```

**Component-wise Mode (Single-Objective by Default):**
```bash
python smac_runner.py --optimization_mode componentwise
```


## Constraint Handling
SMAC3 automatically handles complex parameter constraints:

```yaml
reranker_top_k: [1, 2, 3, 4, 5]  
passage_filter_method: ["pass_passage_filter"]  (When reranker top k = 1)
```
