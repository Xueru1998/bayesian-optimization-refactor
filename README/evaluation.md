# RAG Pipeline Evaluation Guide

## Overview

The RAG optimization framework provides multiple evaluation strategies to measure pipeline performance across different components. You can use traditional metrics defined in your configuration file, or enable advanced LLM-based evaluation for more nuanced quality assessment.

## Evaluation Types

### 1. Traditional Metrics (Default)
Traditional metrics are fast, consistent, and configured directly in your `config.yaml` file. These metrics are averaged to produce a final score for optimization.

### 2. LLM-based Evaluation (Advanced)
LLM-based evaluation provides more sophisticated quality assessment but is slower and costs more due to API calls. When enabled, LLM scores override traditional metrics for optimization.

## Traditional Metrics Configuration

### Component-Level Metrics in config.yaml

Each component can specify metrics in its `strategy` section:

```yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_precision, retrieval_recall]
        # ... other parameters
      
      - node_type: passage_reranker
        strategy:
          metrics: [retrieval_f1]
        # ... other parameters
      
      - node_type: passage_filter
        strategy:
          metrics: [retrieval_f1, retrieval_precision]
        # ... other parameters
      
      - node_type: passage_compressor
        strategy:
          metrics: [retrieval_token_f1, retrieval_token_precision, retrieval_token_recall]
        # ... other parameters

  - node_line_name: post_retrieve_node_line
    nodes:      
      - node_type: generator
        strategy:
          metrics:
            - metric_name: bleu
            - metric_name: meteor
            - metric_name: rouge
            - metric_name: sem_score
              embedding_model: openai
        # ... other parameters
```

### Score Calculation

**Final Score = Average of all specified metrics**

For example, if you specify:
```yaml
metrics: [retrieval_f1, retrieval_precision, retrieval_recall]
```

And the results are:
- `retrieval_f1`: 0.75
- `retrieval_precision`: 0.80  
- `retrieval_recall`: 0.70

**Final Score = (0.75 + 0.80 + 0.70) / 3 = 0.75**

## Available Traditional Metrics

### Retrieval Metrics
| Metric | Description | Range | Best Value |
|--------|-------------|--------|------------|
| `retrieval_f1` | Harmonic mean of precision and recall | 0.0-1.0 | 1.0 |
| `retrieval_precision` | Fraction of retrieved docs that are relevant | 0.0-1.0 | 1.0 |
| `retrieval_recall` | Fraction of relevant docs that are retrieved | 0.0-1.0 | 1.0 |
| `retrieval_ndcg` | Normalized Discounted Cumulative Gain | 0.0-1.0 | 1.0 |
| `retrieval_map` | Mean Average Precision | 0.0-1.0 | 1.0 |
| `retrieval_mrr` | Mean Reciprocal Rank | 0.0-1.0 | 1.0 |

### Token-Level Metrics (for Compression)
| Metric | Description | Range | Best Value |
|--------|-------------|--------|------------|
| `retrieval_token_f1` | Token-level F1 between compressed and ground truth | 0.0-1.0 | 1.0 |
| `retrieval_token_precision` | Token-level precision | 0.0-1.0 | 1.0 |
| `retrieval_token_recall` | Token-level recall | 0.0-1.0 | 1.0 |

### Generation Metrics
| Metric | Description | Range | Best Value |
|--------|-------------|--------|------------|
| `bleu` | BLEU score for text similarity | 0.0-100.0* | 100.0 |
| `meteor` | METEOR score for semantic similarity | 0.0-1.0 | 1.0 |
| `rouge` | ROUGE score for content overlap | 0.0-1.0 | 1.0 |
| `sem_score` | Semantic similarity using embeddings | 0.0-1.0 | 1.0 |
| `bert_score` | BERT-based semantic similarity | 0.0-1.0 | 1.0 |
| `g_eval` | GPT-based evaluation score | 0.0-1.0 | 1.0 |

*Note: BLEU scores are automatically normalized to 0.0-1.0 range for fair averaging*

## LLM-based Evaluation

### RAGAS Evaluation

RAGAS provides comprehensive RAG-specific evaluation using LLM judges.

**Enable RAGAS:**
```bash
python bo_runner.py \
  --mode global \
  --use_ragas \
  --ragas_llm_model gpt-4o-mini \
  --ragas_embedding_model text-embedding-ada-002
```

**Available RAGAS Metrics:**
| Metric | Type | Description |
|--------|------|-------------|
| `context_precision` | Retrieval | Relevance of retrieved context to the query |
| `context_recall` | Retrieval | Coverage of ground truth information in retrieved context |
| `answer_relevancy` | Generation | How well the answer addresses the query |
| `faithfulness` | Generation | Factual consistency with retrieved context |
| `factual_correctness` | Generation | Accuracy of facts in the generated answer |
| `semantic_similarity` | Generation | Semantic alignment with ground truth |

**Specify Custom RAGAS Metrics:**
```bash
python bo_runner.py \
  --mode global \
  --use_ragas \
  --ragas_metrics context_precision context_recall answer_relevancy faithfulness
```

**When RAGAS is enabled:**
- Traditional metrics are ignored for optimization
- RAGAS provides a combined score used for Bayesian optimization
- Individual RAGAS metric scores are logged for analysis

### LLM Compressor Evaluation

For passage compression quality assessment using GPT models.

**Enable LLM Compressor Evaluation:**
```bash
python bo_runner.py \
  --mode componentwise \
  --use_llm_compressor_evaluator \
  --llm_evaluator_model gpt-4o
```

**Available Models:**
- `gpt-4o` (most accurate, recommended)
- `gpt-4o-mini` (faster, good balance)
- `gpt-3.5-turbo` (fastest, basic evaluation)

**Evaluation Criteria:**
1. **Atomic Fact Preservation (50%)**: Retention of specific facts, numbers, methods
2. **Completeness (15%)**: Ability to generate exact ground truth from compressed context
3. **Relevance & Accuracy (20%)**: Absence of irrelevant or incorrect information
4. **Efficiency & Precision (15%)**: Brevity while maintaining completeness

**When LLM Compressor Evaluation is enabled:**
- Token-level metrics (retrieval_token_*) are replaced by LLM scores
- Only applies to passage compression components
- Other components continue using traditional metrics

**LLM-based (Advanced):**
```bash
--use_llm_compressor_evaluator --llm_evaluator_model gpt-4o
```

### Text Generation
```yaml
- node_type: generator
  strategy:
    metrics:
      - metric_name: bleu
      - metric_name: meteor
      - metric_name: rouge
      - metric_name: sem_score
        embedding_model: openai
```

**Recommended combinations:**
- **Fast**: `[bleu, meteor, rouge]`
- **Semantic**: `[meteor, rouge, sem_score]`
- **Comprehensive**: `[bleu, meteor, rouge, sem_score, bert_score]`

## Evaluation Strategy Selection

### Use Traditional Metrics When:
- ✅ Fast evaluation is needed
- ✅ Consistent, reproducible results required
- ✅ Limited API budget
- ✅ Well-understood domains
- ✅ Component-wise optimization
- ⚠️ Important: For token evaluation in the compressor, in almost all cases, uncompressed text will have a higher score than compressed text. 

### Use RAGAS When:
- ✅ End-to-end quality assessment needed
- ✅ Global pipeline optimization
- ✅ Research requiring nuanced evaluation
- ✅ Complex question-answering tasks

### Use LLM Compressor Evaluation When:
- ✅ Compression quality is critical
- ✅ Token metrics don't correlate with downstream performance
- ✅ Research on compression techniques

## Performance Considerations

### Traditional Metrics
- **Speed**: Very fast (milliseconds per sample)
- **Cost**: No additional API costs
- **Consistency**: Deterministic results
- **Scalability**: Handles large datasets easily

### RAGAS Evaluation  
- **Speed**: Slow (seconds per sample)
- **Cost**: OpenAI API costs 
- **Consistency**: Some variability due to LLM responses
- **Scalability**: Limited by API rate limits

### LLM Compressor Evaluation
- **Speed**: Moderate
- **Cost**: OpenAI API costs
- **Consistency**: Stable with temperature=0
- **Scalability**: Batch processing reduces API calls

## Best Practices

1. **Start with Traditional Metrics**: Begin optimization with fast traditional metrics to explore the search space

2. **Use Multiple Metrics**: Combine complementary metrics (e.g., precision + recall + F1) for robust evaluation

3. **LLM for Final Validation**: Use RAGAS or LLM evaluation for final model validation after traditional optimization

4. **Component-Specific Choice**: 
   - Use token metrics for compression
   - Use retrieval metrics for retrieval/reranking/filtering
   - Use generation metrics for text generation

5. **Budget Considerations**: Traditional metrics for exploration, LLM metrics for exploitation

6. **Validation**: Always validate LLM-based scores with human evaluation on a sample
