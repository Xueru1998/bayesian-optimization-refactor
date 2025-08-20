# RAG Optimization Framework

A Comprehensive Framework for Retrieval-Augmented Generation (RAG) Pipeline Optimization via Bayesian Optimization with Optuna and SMAC3.

## Disclaimer and Attribution

**This research framework is built upon and extends the AutoRAG library by Marker Inc. Korea.**

- **Original Project**: [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)
- **Purpose**: This framework is developed for research purposes to explore advanced optimization techniques for RAG pipelines

### Components Using AutoRAG

The following components directly utilize or extend AutoRAG's base classes and functionality:

#### Core Pipeline Components
- **Retrieval System** (`pipeline_component/nodes/retrieval.py`)
  - Uses `autorag.nodes.retrieval.bm25.BM25`
  - Uses `autorag.nodes.retrieval.vectordb.VectorDB`
  - Extends AutoRAG's retrieval infrastructure

- **Passage Reranking** (`pipeline_component/nodes/passageReranker.py`)
  - Inherits from `autorag.nodes.passagereranker.base.BasePassageReranker`
  - Uses multiple AutoRAG reranker implementations:
    - `autorag.nodes.passagereranker.colbert.ColbertReranker`
    - `autorag.nodes.passagereranker.flag_embedding.FlagEmbeddingReranker`
    - `autorag.nodes.passagereranker.monot5.MonoT5`
    - `autorag.nodes.passagereranker.sentence_transformer.SentenceTransformerReranker`
    - And others from the AutoRAG ecosystem

- **Passage Compression** (`pipeline_component/nodes/passagecompressor.py`)
  - Extends `autorag.nodes.passagecompressor.base.BasePassageCompressor`
  - Uses `autorag.nodes.passagecompressor.base.LlamaIndexCompressor`

- **Passage Filtering** (`pipeline_component/nodes/passagefilter.py`)
  - Uses AutoRAG filter implementations:
    - `autorag.nodes.passagefilter.percentile_cutoff.PercentileCutoff`
    - `autorag.nodes.passagefilter.threshold_cutoff.ThresholdCutoff`
    - `autorag.nodes.passagefilter.similarity_threshold_cutoff.SimilarityThresholdCutoff`

- **Prompt Making** (`pipeline_component/nodes/promptmaker.py`)
  - Uses AutoRAG prompt maker classes:
    - `autorag.nodes.promptmaker.fstring.Fstring`
    - `autorag.nodes.promptmaker.long_context_reorder.LongContextReorder`
    - `autorag.nodes.promptmaker.window_replacement.WindowReplacement`

#### Evaluation Components
- **Generation Evaluation** (`pipeline_component/evaluation/generation_evaluator.py`)
  - Uses AutoRAG evaluation metrics:
    - `autorag.evaluation.metric.generation.*` (bleu, meteor, rouge, sem_score, bert_score, g_eval)
    - `autorag.schema.metricinput.MetricInput`

- **Token Evaluation** (`pipeline_component/evaluation/token_evaluation.py`)
  - Uses AutoRAG retrieval evaluation metrics:
    - `autorag.evaluation.metric.retrieval_token_f1`
    - `autorag.evaluation.metric.retrieval_token_precision`
    - `autorag.evaluation.metric.retrieval_token_recall`

#### Infrastructure Components
- **Embedding Management** (`pipeline_component/embedding/embedding_manager.py`)
  - Uses AutoRAG's embedding and vector database infrastructure:
    - `autorag.nodes.retrieval.bm25.bm25_ingest`
    - `autorag.vectordb.load_vectordb_from_yaml`
    - `autorag.nodes.retrieval.vectordb.vectordb_ingest`

- **Utility Functions**
  - Various utility functions from `autorag.utils.util`
  - Schema definitions from `autorag.schema.*`

### Novel Contributions

While building upon AutoRAG, this framework contributes:

1. **Advanced Optimization Algorithms**: Integration of Optuna and SMAC3 for hyperparameter optimization
2. **Dual Optimization Strategies**: Both global and local (component-wise) optimization approaches
3. **Enhanced Evaluation Framework**: Comprehensive evaluation strategies with multiple metrics
4. **Flexible Pipeline Management**: Modular design for easy customization and extension
5. **Research-Oriented Features**: Advanced logging, monitoring, and visualization capabilities

## Overview

This framework provides both global and local optimization strategies for RAG systems, enabling fine-tuned performance optimization across different components of your RAG pipeline. Whether you're looking to optimize embedding models, retrieval mechanisms, reranking strategies, compressor strategies or generation parameters, this toolkit offers flexible and scalable solutions.

## Key Features

- **Dual Optimization Backends**: Support for both Optuna and SMAC3 optimizers
- **Global vs Local Optimization**: Choose between optimizing the entire pipeline or individual components
- **Comprehensive RAG Components**: Built-in support for embedding, retrieval(query expansion), reranking, filtering, compression and generation
- **Advanced Evaluation**: Multiple evaluation strategies including RAGAS, LLM-based, token-based, document ID based metrics
- **Logging & Monitoring**: Integrated W&B logging and email notifications
- **Flexible Pipeline**: Modular design for easy customization and extension

## Project Structure

```
optuna_rag/
├── optuna_bo/
│   ├── optuna_global_optimization/
│   │   ├── bo_optuna_integration.py
│   │   ├── objective.py
│   │   └── plot_generator.py
│   ├── optuna_local_optimization/
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   └── base_componentwise_optimizer.py
│   │   ├── helpers/
│   │   │   ├── __init__.py
│   │   │   ├── component_grid_search_helper.py
│   │   │   ├── component_pipeline_manager.py
│   │   │   └── component_search_space_builder.py
│   │   ├── optimizers/
│   │   │   ├── componentwise_bayesian_optimization.py
│   │   │   └── componentwise_grid_search.py
│   │   ├── __init__.py
│   │   └── componentwise_optuna_optimizer.py
│   ├── __init__.py
│   └── bo_runner.py

pipeline/
├── logging/
│   ├── email/
│   │   ├── __init__.py
│   │   └── email_notifier.py
│   └── wandb/
│       ├── __init__.py
│       ├── core.py
│       ├── metrics.py
│       ├── tables.py
│       ├── utils.py
│       └── visualization.py
├── config_manager.py
├── pipeline_evaluator.py
├── pipeline_executor.py
├── rag_pipeline_runner.py
├── search_space_calculator.py
├── search_space_extractor.py
└── utils.py

pipeline_component/
├── embedding/
│   └── embedding_manager.py
├── evaluation/
│   ├── __init__.py
│   ├── generation_evaluator.py
│   ├── llm_evaluator.py
│   ├── ragas_evaluator.py
│   ├── retrieval_evaluation.py
│   └── token_evaluation.py
└── nodes/
    ├── generator.py
    ├── passagecompressor.py
    ├── passagefilter.py
    ├── passageReranker.py
    ├── promptmaker.py
    ├── query_expansion.py
    └── retrieval.py

smac3/
├── global_optimization/
│   ├── __init__.py
│   ├── config_space_builder.py
│   └── smac_rag_optimizer.py
├── local_optimization/
│   ├── __init__.py
│   ├── componentwise_config_space_builder.py
│   ├── componentwise_rag_processor.py
│   └── componentwise_smac_rag_optimizer.py
├── __init__.py
└── smac_runner.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Core Components

### Pipeline Components
- **Embedding Manager**: Handles document and query embeddings
- **Retrieval**: Implements various retrieval strategies
- **Reranking**: Cross-encoder and other reranking methods
- **Filtering & Compression**: Passage filtering and compression techniques
- **Generation**: LLM-based answer generation

### Evaluation
- **RAGAS Evaluation**: Comprehensive RAG evaluation metrics
- **LLM Evaluation**: Model-based evaluation
- **Retrieval Evaluation**: Precision, recall, and ranking metrics
- **Token Evaluation**: Token-level analysis

### Optimization Strategies
- **Global Optimization**: End-to-end pipeline optimization
- **Local Optimization**: Component-wise optimization
- **Bayesian Optimization**: Efficient hyperparameter search
- **Grid Search**: Exhaustive parameter exploration

## Logging and Monitoring

The framework includes comprehensive logging capabilities:

- **Weights & Biases Integration**: Track experiments, metrics, and visualizations
- **Email Notifications**: Get notified when optimization runs complete
- **Custom Metrics**: Define and track domain-specific evaluation metrics

## Citation

If you use this framework in your research, please cite both this work and the original AutoRAG project:

```bibtex
@software{autorag2024,
  author = {Marker Inc. Korea},
  title = {AutoRAG: Automatic Retrieval-Augmented Generation},
  url = {https://github.com/Marker-Inc-Korea/AutoRAG},
  year = {2024}
}
```

## License

This project respects and complies with the licensing terms of the original AutoRAG project. Please refer to the [AutoRAG repository](https://github.com/Marker-Inc-Korea/AutoRAG) for specific license information.