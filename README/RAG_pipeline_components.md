# RAG Pipeline Components Guide

## Overview

The RAG optimization framework provides a comprehensive set of modular components that can be configured and optimized to build high-performance retrieval-augmented generation pipelines. Each component is designed to be independently configurable with various methods and parameters.

## Prerequisites and Setup

### Vector Database Configuration

If you plan to use vector-based retrieval (`vectordb` module), you **must** provide vector database configurations at the top of your `config.yaml` file.

#### Required VectorDB Configuration
```yaml
vectordb:
  - name: bge_small
    db_type: chroma
    client_type: persistent
    embedding_model: huggingface_baai_bge_small
    collection_name: huggingface_baai_bge_small
    path: ${PROJECT_DIR}/resources/chroma
  - name: mpnet
    db_type: chroma
    client_type: persistent
    embedding_model: huggingface_all_mpnet_base_v2
    collection_name: huggingface_all_mpnet_base_v2
    path: ${PROJECT_DIR}/resources/chroma
  - name: bge_m3
    db_type: chroma
    client_type: persistent
    embedding_model: huggingface_bge_m3
    collection_name: huggingface_bge_m3
    path: ${PROJECT_DIR}/resources/chroma

node_lines:
  # Your pipeline configuration continues here...
```

#### VectorDB Configuration Parameters
- **`name`**: Unique identifier for the vector database configuration
- **`db_type`**: Database type (currently supports `chroma`)
- **`client_type`**: Client connection type (`persistent` for local storage)
- **`embedding_model`**: HuggingFace model for generating embeddings
- **`collection_name`**: Name of the collection in the vector database
- **`path`**: Storage path (use `${PROJECT_DIR}/resources/chroma` for automatic path resolution)

#### Important Notes

**Automatic Configuration Copying:**
- The vectordb configuration is automatically copied to `autorag_project/resources/vectordb.yaml`
- This enables embedding generation and vector database setup

**Configuration Changes:**
- If you modify the vectordb configuration (especially `embedding_model`), you **must** delete the existing `vectordb.yaml` file:
  ```bash
  rm autorag_project/resources/vectordb.yaml
  ```
- This prevents conflicts between old and new embedding models
- Failure to do this may result in "embedding model not found" errors or incorrect embeddings

**Embedding Generation:**
- Embeddings are automatically generated during optimization
- Generated embeddings are cached in `autorag_project/resources/chroma/`
- First run may take longer as embeddings are created

#### Available Embedding Models
| Name | Model |
|------|-------|
| `bge_small` | `huggingface_baai_bge_small` 
| `mpnet` | `huggingface_all_mpnet_base_v2` 
| `bge_m3` | `huggingface_bge_m3` | Multilingual 
| `rubert` | `huggingface_cointegrated_rubert_tiny2` 

## Pipeline Architecture

```
Input Query
     ↓
[Query Expansion] → [Retrieval] → [Passage Reranking] → [Passage Filtering] → [Passage Compression] → [Prompt Making] → [Generation]
     ↓                    ↓               ↓                    ↓                    ↓                  ↓              ↓
Output: Expanded     Retrieved      Reranked           Filtered            Compressed         Formatted      Generated
Queries             Documents      Documents          Documents           Documents          Prompts        Answer
```

## Component Configuration

Components are configured in `config.yaml` under `node_lines`. Each component can specify multiple methods and their parameters for optimization.

### Basic Structure
```yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: [component_name] (eg: retrieval)
        strategy:
          metrics: [evaluation_metrics]
        modules:
          - module_type: [method_name] (eg: bm25/vectordb)
            [method_parameters]
```

## 1. Query Expansion

Enhances the input query to improve retrieval effectiveness by generating multiple query variations or decompositions.

### Available Methods

#### Pass Query Expansion (Default)
```yaml
- module_type: pass_query_expansion
```
- **Purpose**: No expansion, uses original query
- **Use case**: Baseline or when queries are already well-formed

#### HyDE (Hypothetical Document Embeddings)
```yaml
- module_type: hyde
  generator_module_type: vllm
  model: ["Qwen/Qwen2.5-1.5B-Instruct"]
  max_token: [64, 128]
```
- **Purpose**: Generates hypothetical passages that answer the query
- **Parameters**:
  - `model`: LLM model for generation
  - `max_token`: Length of generated hypothetical documents
- **Use case**: When semantic similarity search needs better query representation

#### Query Decomposition
```yaml
- module_type: query_decompose
  generator_module_type: vllm
  model: ["Qwen/Qwen2.5-1.5B-Instruct"]
```
- **Purpose**: Breaks complex queries into simpler sub-questions
- **Use case**: Complex multi-part questions that benefit from step-by-step retrieval

#### Multi-Query Expansion
```yaml
- module_type: multi_query_expansion
  generator_module_type: vllm
  model: ["Qwen/Qwen2.5-1.5B-Instruct"]
  temperature: [0.2, 1.0]
```
- **Purpose**: Generates multiple paraphrases of the original query
- **Parameters**:
  - `model`: LLM model for expansion
  - `temperature`: Controls variation in generated queries
- **Use case**: Overcoming vocabulary mismatch between queries and documents

### Configuration Example
```yaml
- node_type: query_expansion
  strategy:
    metrics: [retrieval_f1]
    top_k: [2, 4]
    retrieval_modules:
      - module_type: bm25
        bm25_tokenizer: [porter_stemmer, space, gpt2]
      - module_type: vectordb
        vectordb: [bge_small, rubert, mpnet]
  modules:
    - module_type: pass_query_expansion
    - module_type: hyde
      generator_module_type: vllm
      model: ["Qwen/Qwen2.5-1.5B-Instruct"]
      max_token: [64, 128]
    - module_type: multi_query_expansion
      generator_module_type: vllm
      model: ["Qwen/Qwen2.5-1.5B-Instruct"]
      temperature: [0.2, 1.0]
```

## 2. Retrieval

Retrieves relevant documents from the corpus using various search methods.

### Available Methods

#### BM25 (Keyword-based)
```yaml
- module_type: bm25
  bm25_tokenizer: [porter_stemmer, space, gpt2]
```
- **Purpose**: Traditional keyword-based retrieval
- **Parameters**:
  - `bm25_tokenizer`: Text preprocessing method
    - `porter_stemmer`: Reduces words to root forms
    - `space`: Simple whitespace tokenization
    - `gpt2`: GPT-2 tokenizer
- **Use case**: When exact keyword matching is important

#### Vector Database (Semantic)
```yaml
- module_type: vectordb
  vectordb: [bge_small, rubert, mpnet, bge_m3]
  embedding_batch: 256
```
- **Purpose**: Semantic similarity-based retrieval using embeddings
- **Prerequisites**: Requires vectordb configuration at top of config.yaml (see Prerequisites section)
- **Parameters**:
  - `vectordb`: Must match a `name` from your vectordb configuration
  - `embedding_batch`: Batch size for embedding generation
- **Available Configurations** (from your vectordb section):
  - `bge_small`: Fast, good general performance
  - `mpnet`: Balanced performance and accuracy
  - `bge_m3`: Multilingual support (slower)
  - `rubert`: Russian language optimization
- **⚠️ Important**: If you change embedding models in vectordb config, delete `autorag_project/resources/vectordb.yaml`

### Configuration Example
```yaml
- node_type: retrieval
  strategy:
    metrics: [retrieval_f1]
  top_k: [2, 4]
  modules:
    - module_type: bm25
      bm25_tokenizer: [porter_stemmer, space, gpt2]
    - module_type: vectordb
      vectordb: [bge_small, rubert, mpnet, bge_m3]
      embedding_batch: 256
```

## 3. Passage Reranking

Reorders retrieved passages using more sophisticated relevance models.

### Available Methods

#### Pass Reranker (Default)
```yaml
- module_type: pass_reranker
```
- **Purpose**: No reranking, maintains original order
- **Hardware**: CPU only
- **Performance**: Instant (no processing)

#### Cross-Encoder Models
```yaml
- module_type: sentence_transformer_reranker
  model_name: [cross-encoder/ms-marco-MiniLM-L12-v2, cross-encoder/ms-marco-TinyBERT-L2-v2, cross-encoder/stsb-distilroberta-base]
```
- **Purpose**: Sophisticated relevance scoring using cross-encoders
- **Available Models**:
  - `cross-encoder/ms-marco-MiniLM-L12-v2`: Balanced accuracy/speed
  - `cross-encoder/ms-marco-TinyBERT-L2-v2`: Fastest option, CPU-friendly
  - `cross-encoder/stsb-distilroberta-base`: Good for semantic similarity
- **Hardware**: CPU-friendly (GPU optional for speed)

#### MonoT5
```yaml
- module_type: monot5
  model_name: [castorini/monot5-base-msmarco-10k, castorini/monot5-large-msmarco-10k, unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2, unicamp-dl/mt5-base-mmarco-v1]
```
- **Purpose**: T5-based reranking with strong performance
- **Models**:
  - `monot5-base`: Good balance of accuracy/speed
  - `monot5-large`: Higher accuracy, slower
  - `ptt5-base-en-pt`: Portuguese-English bilingual
  - `mt5-base-mmarco`: Multilingual support
- **Hardware**: GPU recommended (CPU possible but slow)

#### Flag Embedding Rerankers
```yaml
- module_type: flag_embedding_reranker
  model_name: [BAAI/bge-reranker-large, BAAI/bge-reranker-base]
```
- **Purpose**: BAAI's high-performance reranking models
- **Models**:
  - `bge-reranker-base`: Smaller, faster
  - `bge-reranker-large`: Higher accuracy
- **Hardware**: GPU recommended for large model, base can run on CPU

#### Flag Embedding LLM Rerankers
```yaml
- module_type: flag_embedding_llm_reranker
  model_name: [BAAI/bge-reranker-v2-gemma, BAAI/bge-reranker-v2-m3]
```
- **Purpose**: LLM-powered reranking for highest accuracy
- **Models**:
  - `bge-reranker-v2-gemma`: Gemma-based architecture
  - `bge-reranker-v2-m3`: Multilingual support
- **Hardware**: GPU strongly recommended (very slow on CPU)

#### FlashRank
```yaml
- module_type: flashrank_reranker
  model: ["ms-marco-MiniLM-L-12-v2", "ms-marco-MultiBERT-L-12", "rank-T5-flan", "ce-esci-MiniLM-L12-v2"]
```
- **Purpose**: Fast reranking with good performance
- **Models**:
  - `ms-marco-MiniLM-L-12-v2`: Fast and efficient
  - `ms-marco-MultiBERT-L-12`: Multilingual support
  - `rank-T5-flan`: T5-based ranking
  - `ce-esci-MiniLM-L12-v2`: E-commerce optimized
- **Hardware**: CPU-friendly with optimizations

#### ColBERT Reranker
```yaml
- module_type: colbert_reranker
```
- **Purpose**: Efficient late-interaction reranking using ColBERT architecture
- **Hardware**: GPU recommended for dense retrieval operations

#### UPR (Universal Passage Reranker)
```yaml
- module_type: upr
```
- **Purpose**: Universal passage reranking with learned representations
- **Hardware**: GPU recommended

## Hardware Recommendations

### CPU-Only Deployments
- pass_reranker
- sentence_transformer_reranker (TinyBERT, MiniLM)
- flashrank_reranker

### GPU Recommended
- monot5 (especially large models)
- flag_embedding_reranker
- colbert_reranker
- upr

### GPU Required
- flag_embedding_llm_reranker models
### Configuration Example
```yaml
- node_type: passage_reranker
  strategy:
    metrics: [retrieval_f1]
  top_k: [1, 3]
  modules:
    - module_type: pass_reranker
    - module_type: sentence_transformer_reranker
      model_name: [cross-encoder/ms-marco-MiniLM-L12-v2]
    - module_type: flashrank_reranker
      model: ["ms-marco-MiniLM-L-12-v2"]
```

## 4. Passage Filtering

Removes irrelevant or low-quality passages based on various criteria.

### Available Methods

#### Pass Filter (Default)
```yaml
- module_type: pass_passage_filter
```
- **Purpose**: No filtering, keeps all passages

#### Threshold-based Filtering
```yaml
- module_type: threshold_cutoff
  threshold: [0.4, 0.9]
```
- **Purpose**: Removes passages below a score threshold
- **Parameters**:
  - `threshold`: Minimum score to keep (0.0-1.0)

#### Percentile-based Filtering
```yaml
- module_type: percentile_cutoff
  percentile: [0.4, 0.9]
```
- **Purpose**: Keeps top percentage of passages
- **Parameters**:
  - `percentile`: Fraction of top passages to keep (0.0-1.0)

#### Similarity-based Filtering
```yaml
- module_type: similarity_threshold_cutoff
  threshold: [0.45, 0.95]
- module_type: similarity_percentile_cutoff
  percentile: [0.4, 0.9]
```
- **Purpose**: Filters based on semantic similarity to query
- **Use case**: When retrieval scores don't reflect query relevance

### Configuration Example
```yaml
- node_type: passage_filter
  strategy:
    metrics: [retrieval_f1]
  modules:
    - module_type: pass_passage_filter
    - module_type: percentile_cutoff
      percentile: [0.4, 0.9]
    - module_type: similarity_threshold_cutoff
      threshold: [0.45, 0.95]
```

## 5. Passage Compression

Reduces passage length while preserving important information.

### Available Methods

#### Pass Compressor (Default)
```yaml
- module_type: pass_compressor
```
- **Purpose**: No compression, uses full passages

#### LLM-based Compression
```yaml
- module_type: tree_summarize
  llm: openai
  model: gpt-3.5-turbo-16k
- module_type: refine
  llm: openai
  model: gpt-3.5-turbo-16k
```
- **Purpose**: Uses LLMs to generate summaries
- **Methods**:
  - `tree_summarize`: Hierarchical summarization
  - `refine`: Iterative refinement approach
- **Parameters**:
  - `llm`: Provider (openai)
  - `model`: Specific model name

#### LexRank Compression
```yaml
- module_type: lexrank
  compression_ratio: [0.3, 0.7]
  threshold: [0.05, 0.3]
  damping: [0.75, 0.9]
  max_iterations: [15, 40]
```
- **Purpose**: Graph-based extractive summarization
- **Parameters**:
  - `compression_ratio`: Fraction of sentences to keep
  - `threshold`: Similarity threshold for graph construction
  - `damping`: Random walk damping factor
  - `max_iterations`: Maximum LexRank iterations

#### SpaCy Compression
```yaml
- module_type: spacy
  compression_ratio: [0.3, 0.5]
  spacy_model: ["en_core_web_sm", "en_core_web_md", "en_core_web_trf",  "en_core_web_lg"]
```
- **Purpose**: NLP-based extractive summarization
- **Parameters**:
  - `compression_ratio`: Fraction of content to keep
  - `spacy_model`: SpaCy model for analysis
    - `en_core_web_sm`: Small, fast model
    - `en_core_web_md`: Medium model with vectors
    - `en_core_web_trf`: Transformer-based, most accurate

### Configuration Example
```yaml
- node_type: passage_compressor
  strategy:
    metrics: [retrieval_token_f1, retrieval_token_recall, retrieval_token_precision]
  modules:
    - module_type: pass_compressor
    - module_type: lexrank
      compression_ratio: [0.3, 0.7]
      threshold: [0.05, 0.3]
      damping: [0.75, 0.9]
      max_iterations: [15, 40]
    - module_type: spacy
      compression_ratio: [0.3, 0.5]
      spacy_model: ["en_core_web_sm", "en_core_web_md"]
```

## 6. Prompt Making

Formats the query and retrieved context into prompts for the generation model.

### Available Methods

#### F-String Formatting
```yaml
- module_type: fstring
  prompt:
    - "Answer to given questions with the following passage: {retrieved_contents} \n\n Question: {query} \n\n Answer:"
    - "Question: {query} \n\n Context: {retrieved_contents} \n\n Please provide a detailed answer:"
```
- **Purpose**: Simple template-based prompt formatting
- **Parameters**:
  - `prompt`: List of template strings with placeholders

#### Long Context Reordering
```yaml
- module_type: long_context_reorder
  prompt:
    - "Answer to given questions with the following passage: {retrieved_contents} \n\n Question: {query} \n\n Answer:"
```
- **Purpose**: Reorders passages based on relevance scores to optimize context window usage
- **Use case**: When using models with limited context windows

#### Window Replacement
```yaml
- module_type: window_replacement
  prompt:
    - "Tell me something about the question: {query} \n\n {retrieved_contents}"
```
- **Purpose**: Uses document metadata to provide additional context
- **Use case**: When documents have structured metadata

### Configuration Example
```yaml
- node_type: prompt_maker
  modules:
    - module_type: fstring
      prompt:
        - "Answer this question: {query}\n\nContext: {retrieved_contents}\n\nAnswer:"
    - module_type: long_context_reorder
      prompt:
        - "Based on the context: {retrieved_contents}\n\nQuestion: {query}\n\nProvide a comprehensive answer:"
```

## 7. Generation

Generates the final answer using language models.

### Available Models

#### OpenAI Models
```yaml
- module_type: llama_index_llm
  model: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
  temperature: [0.1, 1.0]
```

#### Open Source Models
```yaml
- module_type: vllm
  llm: [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.2-1B-Instruct", 
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  ]
  temperature: [0.1, 1.0]
  max_tokens: 512
```

### Configuration Example
```yaml
- node_type: generator
  strategy:
    metrics:
      - metric_name: bleu
      - metric_name: meteor
      - metric_name: rouge
      - metric_name: sem_score
        embedding_model: openai
  modules:
    - module_type: vllm
      llm: ["meta-llama/Llama-2-7b-chat-hf", "Qwen/Qwen2.5-1.5B-Instruct"]
      temperature: [0.1, 1.0]
      max_tokens: 512
```

## Performance Considerations

### Component Impact on Speed
1. **Query Expansion**: LLM-based methods add generation overhead
2. **Retrieval**: Vector search generally faster than BM25 for large corpora
3. **Reranking**: Cross-encoders are slower but more accurate
4. **Filtering**: Minimal impact, can improve downstream speed
5. **Compression**: LLM-based methods slow, extractive methods fast
6. **Generation**: Model size directly impacts speed

### Optimization Strategies
- **Development**: Use fast models and minimal components
- **Production**: Balance speed/accuracy based on requirements
- **Resource-Constrained**: Prioritize efficient components
- **High-Accuracy**: Use sophisticated models with comprehensive pipeline

## Best Practices

1. **Start Simple**: Begin with basic retrieval + generation
2. **Add Incrementally**: Add components based on performance analysis
3. **Match Components**: Ensure retrieval top_k > reranker top_k > filter output
4. **Resource Management**: Monitor memory usage with large models
5. **Evaluation**: Use appropriate metrics for each component
6. **Caching**: Enable embedding caching for faster optimization
7. **Model Selection**: Choose models appropriate for your language/domain