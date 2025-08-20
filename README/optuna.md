# Optuna RAG Optimizer - Complete User Guide

## Overview

The Optuna RAG Optimizer provides intelligent hyperparameter optimization for Retrieval-Augmented Generation (RAG) pipelines using state-of-the-art Bayesian optimization and grid search techniques. It supports both **global optimization** (entire pipeline) and **component-wise optimization** (individual components).

## Prerequisites

Before running the optimizer, ensure you have the following project structure:

```
your_project/
├── config.yaml                 # Pipeline configuration file
├── autorag_project/            # Data directory
│   ├── corpus.parquet          # Document corpus
│   └── qa_validation.parquet   # Question-answer validation set
└── optuna_rag/                 # Optimizer code
    └── optuna_bo/
        └── bo_runner.py         # Main entry point
```

**Required Files:**
- **`config.yaml`**: Defines your RAG pipeline components and their parameter search spaces
- **`corpus.parquet`**: Your document collection with columns: `doc_id`, `contents`, `metadata`
- **`qa_validation.parquet`**: Validation data with columns: `qid`,`query`, `retrieval_gt`, `generation_gt`

## Entry Point

The main entry point is `bo_runner.py`:

```bash
python optuna_rag/optuna_bo/bo_runner.py [options]
```

## Optimization Modes

### 1. Global Optimization (Default)
Optimizes the entire RAG pipeline end-to-end as a single objective function.

```bash
python bo_runner.py --mode global --n_trials 50 --sampler tpe
```

**Best for:**
- Finding optimal parameter combinations across all components
- When components have strong interdependencies

**Built-in Early Stopping for Poor Configurations:**
Global optimization includes automatic component-level early stopping to skip configurations that perform poorly at intermediate stages. This feature cannot be disabled and uses these default thresholds:

- **Query Expansion**: 0.1 (stops if score < 0.1)
- **Retrieval**: 0.1 (stops if score < 0.1)  
- **Passage Reranker**: 0.2 (stops if score < 0.2)
- **Passage Filter**: 0.25 (stops if score < 0.25)
- **Passage Compressor**: 0.3 (stops if score < 0.3)

This prevents wasting computational resources on configurations that fail early in the pipeline, significantly improving optimization efficiency.

### 2. Component-wise Optimization
Optimizes each RAG component sequentially, using outputs from previous components.

```bash
python bo_runner.py --mode componentwise --n_trials_per_component 20 --sampler tpe
```

**Best for:**
- Large search spaces that would be computationally expensive for global optimization
- Understanding individual component contributions
- Faster optimization with intermediate results

## Optimization Strategies (Samplers)

### Tree-structured Parzen Estimator (TPE) - Recommended
```bash
--sampler tpe
```
- **Strategy**: Uses probabilistic models to intelligently suggest promising parameter combinations
- **Best for**: Most RAG optimization scenarios including Multi-objective optimization, especially with conditional parameters
- **Advantages**: 
  - Excellent for mixed parameter types (categorical, continuous, discrete)
  - Handles conditional dependencies well
  - Robust performance across different problem types
- **Use when**: You want the best balance of exploration and exploitation

### BoTorch (Gaussian Process Bayesian Optimization)
```bash
--sampler botorch
```
- **Strategy**: Advanced Gaussian Process models with sophisticated acquisition functions
- **Best for**: Multi-objective optimization, continuous parameter spaces
- **Advantages**:
  - State-of-the-art for continuous optimization
  - Excellent uncertainty quantification
  - Advanced acquisition functions
- **Requirements**: `pip install botorch gpytorch`
- **Use when**: You have primarily continuous parameters and want cutting-edge optimization

### Random Search
```bash
--sampler random
```
- **Strategy**: Randomly samples from the parameter space
- **Best for**: Baseline comparisons, very large search spaces
- **Advantages**: Simple, unbiased exploration
- **Use when**: You want a baseline or have unlimited computational budget

### Grid Search (Component-wise only)
```bash
--sampler grid --mode componentwise
```
- **Strategy**: Exhaustively tests all parameter combinations
- **Best for**: Small, discrete search spaces where complete exploration is feasible
- **Advantages**: Guaranteed to find local optimum in discrete space
- **Use when**: You have small search spaces and want exhaustive coverage

## Command Line Interface

### Quick Start Examples

**Basic optimization:**
```bash
python bo_runner.py
```

**High-performance setup:**
```bash
python bo_runner.py \
  --mode componentwise \
  --sampler tpe \
  --n_trials_per_component 20 \
  --early_stopping_threshold 0.9 \
```

**Grid search for exhaustive exploration:**
```bash
python bo_runner.py \
  --mode componentwise \
  --sampler grid \
```

**Multi-objective optimization:**
```bash
python bo_runner.py \
  --mode componentwise \
  --sampler botorch \
  --n_trials_per_component 20
```

### Essential Parameters

| Parameter | Description | Default | Alternative Example |
|-----------|-------------|---------|---------|
| `--mode` | Optimization mode | `global` | `--mode componentwise` |
| `--sampler` | Optimization algorithm | `tpe` | `--sampler botorch` |
| `--n_trials` | Trials for global mode | auto-calculated(by default 50) | `--n_trials 100` |
| `--n_trials_per_component` | Trials per component | `20` | `--n_trials_per_component 30` |
| `--config_path` | Configuration file | `config.yaml` | `--config_path my_config.yaml` |
| `--cpu_per_trial` | CPU cores per trial | `4` | `--cpu_per_trial 8` |

### Performance Optimization

| Parameter | Description | Impact |
|-----------|-------------|---------|
| `--use_cached_embeddings` | Use pre-computed embeddings | Significantly faster |
| `--cpu_per_trial` | Parallel processing | Faster individual trials |
| `--early_stopping_threshold` | Stop when score achieved | Reduce unnecessary trials |

### Advanced Features

**RAGAS Evaluation (Global mode only):**
```bash
python bo_runner.py \
  --mode global \
  --use_ragas \
  --ragas_llm_model gpt-4o-mini \
  --ragas_metrics context_precision context_recall answer_relevancy
```

**Resume interrupted studies:**
```bash
python bo_runner.py \
  --mode componentwise \
  --resume_study \
  --study_name my_previous_study
```

**Email notifications:**
```bash
python bo_runner.py \
  --send_email \
  --email_recipients user@example.com 
```

**Custom result directory:**
```bash
python bo_runner.py \
  --result_dir /path/to/custom/results \
  --study_name experiment_v2
```

## What You Get From Optuna Optimization

### 1. Optimized Configurations

**Best Hyperparameters:**
- Optimal settings for each RAG component
- Complete pipeline configuration ready for production
- Component-specific best parameters (in component-wise mode)
- YAML format configurations for easy deployment

**Component-wise Mode Results:**
- Sequential optimization results for each component
- Best configuration per component with performance scores
- Component interdependency analysis
- Detailed optimization path and decisions

### 2. Performance Analytics

**Early Stopping Analytics (Global Mode):**
- Component-level failure detection and skip statistics
- Time saved by avoiding poor configuration completion
- Early stopping distribution across pipeline components

**Weights & Biases Integration:**
- Real-time experiment tracking
- Performance comparison across runs
- Component-wise optimization dashboards
- Pareto frontier visualization (multi-objective)


### 3. Production-Ready Outputs

**Deployment Configurations:**
- Best single configuration (global mode)
- Component-wise optimized pipeline (component-wise mode)
- Multiple Pareto-optimal solutions (multi-objective)
- Performance vs. latency trade-off recommendations

### 4. Advanced Analytics (Component-wise Mode)

**Sequential Component Analysis:**
- How each component improvement affects downstream performance
- Component dependency mapping
- Bottleneck identification in the RAG pipeline

**Optimization Strategy Recommendations:**
- Which components benefit most from further optimization
- Suggested parameter ranges for future experiments
- Component-specific optimization stopping criteria

## Default Settings

The optimizer comes with sensible defaults that work well for most use cases:

```bash
# These are the default values (you don't need to specify them)
python bo_runner.py  # Equivalent to:
python bo_runner.py \
  --mode global \
  --sampler tpe \
  --config_path config.yaml \
  --project_dir autorag_project \
  --retrieval_weight 0.5 \
  --generation_weight 0.5 \
  --use_cached_embeddings \
  --n_trials 50 \
  --seed 42 \
  --early_stopping_threshold 0.9 \
```

### Key Default Behaviors

| Setting | Default Value | What It Means |
|---------|---------------|---------------|
| **Mode** | `global` | Optimizes entire pipeline end-to-end (always multi-objective) |
| **Sampler** | `tpe` | Uses Tree-structured Parzen Estimator |
| **Trials** | Auto-calculated | 50 (global) or 20/component |
| **Weights** | `0.5/0.5` | Equal importance to retrieval and generation |
| **Multi-objective** | Global: Always ON<br/>Component-wise: `False` | Global mode: score + latency<br/>Component-wise: configurable |
| **Embeddings** | Cached | Reuses pre-computed embeddings for speed |
| **Early stopping** | `0.9` | Stops when 90% score achieved |
| **Component early stopping** | Global: Always ON<br/>Component-wise: OFF | Built-in poor config detection (global only) |
| **W&B logging** | Enabled | Tracks experiments automatically(use no_wandb to disable wandb logging) |

### Multi-Objective Optimization

**Global Mode (Always Multi-Objective):**
```bash
# Global optimization automatically uses multi-objective (score + latency)
python bo_runner.py --mode global --sampler botorch / --sampler tpe
```

**Component-wise Mode (Configurable):**
```bash
# Single objective (default) - optimizes score only
python bo_runner.py --mode componentwise

# Multi-objective - optimizes both score AND latency
python bo_runner.py \
  --mode componentwise \
  --use_multi_objective \
  --sampler botorch
```

**Multi-objective benefits:**
- Finds trade-offs between accuracy and speed
- Provides multiple optimal solutions for different priorities
- Essential for production deployment decisions
- Creates Pareto frontier showing all non-dominated solutions

**Mode Differences:**
- **Global mode**: Always optimizes both score and latency simultaneously
- **Component-wise mode**: Can choose single-objective (score) or multi-objective (score + latency)

**Note**: Multi-objective optimization works best with the BoTorch sampler, which is specifically designed for multi-objective problems.

## Tips for Success

1. **Start with Component-wise TPE** for most use cases
2. **Use cached embeddings** to dramatically speed up optimization
3. **Set early stopping** to avoid unnecessary computation
4. **Monitor with W&B** for real-time insights
5. **Use grid search** only for small, discrete search spaces
6. **Resume studies** if interrupted - state is automatically saved
7. **Global mode automatically filters bad configs** - no configuration needed

## Troubleshooting

**Slow optimization?**
- Enable `--use_cached_embeddings`
- Reduce `--n_trials_per_component`

**Poor results?**
- Check your configuration file has sufficient parameter ranges
- Increase number of trials
- Try different samplers (TPE vs BoTorch)

**Memory issues?**
- Reduce `--cpu_per_trial`
- Use component-wise instead of global mode
- Enable embeddings caching

**Want exhaustive search?**
- Use `--sampler grid --mode componentwise`
- Note: Only feasible for small search spaces

**Many early stopped trials in global mode?**
- This is normal and beneficial - indicates the optimizer is efficiently skipping poor configurations
- Check component-specific thresholds are appropriate for your use case
- Consider if your parameter ranges might be too broad