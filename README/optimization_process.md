# Optimization Process Overview

## High-Level Architecture

The optimization framework implements Bayesian Optimization (BO) using Optuna to find optimal configurations for RAG (Retrieval-Augmented Generation) pipelines.

## Main Optimization Loop

### 1. Initialization Phase
```
BOPipelineOptimizer.__init__()
├── Load configuration (config.yaml)
├── Initialize ConfigGenerator
├── Calculate search space size
├── Initialize metrics for each component
├── Set up pipeline runner with early stopping thresholds
└── Configure optimizer (TPE/BoTorch/Random)
```

### 2. Optimization Loop Structure

```
study.optimize(objective, n_trials=N)
│
└── For each trial (1 to N):
    │
    ├── [OPTUNA] Suggest hyperparameters
    │   ├── TPE: Tree-structured Parzen Estimator
    │   ├── BoTorch: Gaussian Process-based
    │   └── Random: Random sampling
    │
    ├── [OBJECTIVE] Execute trial
    │   │
    │   ├── Generate configuration from suggestions
    │   │   └── Map Optuna suggestions to pipeline config
    │   │
    │   ├── Run RAG pipeline
    │   │   ├── Query Expansion (optional)
    │   │   ├── Retrieval
    │   │   ├── Reranking (optional)
    │   │   ├── Filtering (optional)
    │   │   ├── Compression (optional)
    │   │   ├── Prompt Making
    │   │   └── Generation
    │   │
    │   ├── Early Stopping Checks (except Random sampler)
    │   │   ├── Component-level: Stop if score < threshold
    │   │   └── Trial-level: Stop if score > 0.9
    │   │
    │   └── Evaluate performance
    │       ├── Retrieval metrics
    │       └── Generation metrics
    │
    ├── [OPTUNA] Update model
    │   └── Learn from trial results
    │
    └── [CALLBACKS]
        ├── Early stopping callback (if score > 0.9)
        └── WandB logging (optional)
```

## Component Execution Flow

### Pipeline Execution Sequence

```
RAGPipelineRunner.run_pipeline()
│
├── 1. Query Expansion (if configured)
│   ├── Expand query using LLM
│   └── Pass expanded query to retrieval
│
├── 2. Retrieval
│   ├── BM25 or VectorDB search
│   ├── Retrieve top-k documents
│   └── Early stop if score < 0.1
│
├── 3. Reranking (if configured)
│   ├── Re-score retrieved documents
│   ├── Reorder by relevance
│   └── Early stop if score < 0.2
│
├── 4. Filtering (if configured)
│   ├── Filter by threshold or percentile
│   ├── Remove low-relevance documents
│   └── Early stop if score < 0.25
│
├── 5. Compression (if configured)
│   ├── Compress document content
│   ├── Extract key information
│   └── Early stop if score < 0.3
│
├── 6. Prompt Making
│   └── Create prompts with retrieved context
│
└── 7. Generation
    └── Generate final answer using LLM
```

## Early Stopping Mechanisms

### 1. Component-Level Early Stopping
- Applied during pipeline execution
- Thresholds:
  - Retrieval: 0.1
  - Query Expansion: 0.1
  - Reranker: 0.2
  - Filter: 0.25
  - Compressor: 0.3
- **Disabled for**: Random sampler, Local optimization

### 2. Trial-Level Early Stopping
- Stops optimization if score > 0.9
- **Disabled for**: Random sampler

## Configuration Space

### Hyperparameter Types

1. **Categorical Parameters**
   - `retrieval_method`: ["bm25", "vectordb", "hybrid"]
   - `generator_model`: List of available LLMs
   - `prompt_template_idx`: Template selection

2. **Numerical Parameters**
   - `retriever_top_k`: Number of documents to retrieve
   - `generator_temperature`: LLM temperature
   - `threshold`: Filtering threshold

3. **Conditional Parameters**
   - `vectordb_name`: Only when retrieval_method="vectordb"
   - `reranker_model`: Only when reranker enabled

## Optimization Strategies

### 1. TPE (Tree-structured Parzen Estimator)
- Default optimizer
- Models P(x|y) and P(y)
- Good for mixed parameter types
- Fast convergence

### 2. BoTorch (Gaussian Process)
- Bayesian optimization with GPs
- Better for continuous parameters
- More sample efficient

### 3. Random Search
- Baseline comparison
- No early stopping
- Uniform sampling

## Local vs Global Optimization

### Global Optimization
- Optimizes entire pipeline end-to-end
- Uses combined score (retrieval + generation)
- All early stopping mechanisms active

### Local/Component-wise Optimization
- Optimizes single component
- No early stopping for bad scores
- Focused on component-specific metrics

## Score Calculation

```python
combined_score = (retrieval_weight * retrieval_score + 
                 generation_weight * generation_score)
```

Default weights:
- `retrieval_weight`: 0.5
- `generation_weight`: 0.5

## Results and Outputs

### Trial Results Structure
```
{
    "trial_number": int,
    "config": {...},
    "score": float,
    "retrieval_score": float,
    "generation_score": float,
    "last_retrieval_component": str,
    "early_stopped_at": str (if applicable),
    "execution_time": float
}
```

### Best Configuration Selection
1. Find Pareto front of all trials
2. Select trial with score > 0.8 from Pareto front
3. If none > 0.8, select maximum score

## Key Files

- `bo_optuna_integration.py`: Main optimizer class
- `objective.py`: Objective function for Optuna
- `rag_pipeline_runner.py`: Pipeline execution
- `config_generator.py`: Configuration management
- `pipeline_executor.py`: Component execution
- `pipeline_evaluator.py`: Metrics evaluation

## Usage Example

```python
from optuna_rag.optuna_bo.optuna_global_optimization import BOPipelineOptimizer

optimizer = BOPipelineOptimizer(
    config_path="config.yaml",
    qa_df=qa_dataframe,
    corpus_df=corpus_dataframe,
    project_dir="./project",
    n_trials=50,
    optimizer="tpe",  # or "botorch", "random"
    retrieval_weight=0.5,
    generation_weight=0.5,
    use_ragas=True,
    early_stopping_threshold=0.9
)

best_config = optimizer.optimize()
```

## Monitoring and Logging

- **WandB Integration**: Track experiments, hyperparameters, and metrics
- **Intermediate Results**: Saved for each component execution
- **Debug Mode**: Detailed logging of pipeline execution
- **Progress Bar**: Real-time optimization progress