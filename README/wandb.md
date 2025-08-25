# WandB Logging Overview

## Metrics Logged

### Trial Metrics
- **Score & Performance**: Trial score, execution time, latency
- **Configuration**: Config ID, trial number, status
- **Budget Information**: Budget percentage, score per budget (for multi-fidelity)

### Component Scores
- Retrieval score
- Query expansion score  
- Reranker score
- Filter score
- Compressor score
- Prompt maker score
- Generation score
- Combined/final score

### RAGAS Evaluation Metrics
- **Retrieval metrics**: Context precision, context recall
- **Generation metrics**: Answer relevancy, faithfulness, factual correctness, semantic similarity
- **Aggregated scores**: RAGAS mean score, retrieval mean, generation mean

### Summary Statistics
- Best score and latency achieved
- Total trials completed
- Optimization time (seconds/minutes/hours)
- Early stopping status
- Number of unique configurations tested
- Pareto front size (multi-objective)

## Tables

### Optimization Tables
- **Parameters Table**: All trials with configurations, scores, and execution times
- **Component Metrics Breakdown**: Detailed component-wise performance for each trial
- **Best Configs Table**: Top performing configurations (full budget evaluations only)
- **All Trials Table**: Complete trial history with config IDs and performance metrics

### RAGAS Tables
- **RAGAS Summary Table**: Comprehensive RAGAS metrics organized by trial
- **RAGAS Breakdown**: Detailed retrieval vs generation performance

### Component-wise Tables
- **Individual Component Tables**: Separate optimization results for each component
- **Component Summary Table**: Best score, trials count, and time per component
- **Final Configuration Table**: Combined best parameters from all components
- **Score Breakdown Table**: Weighted contribution of retrieval and generation to final score

## Visualizations

### Optimization Progress
- **Score History Plot**: Trial-by-trial score progression with best score line
- **Latency History Plot**: Latency trends over trials with best latency tracking
- **Dual Objective Plots**: Separate plots for score and latency in multi-objective optimization

### Component Analysis
- **Component Score Progression**: Line plot showing all component scores over trials
- **Average Component Scores**: Bar chart comparing mean performance across components
- **Component Timeline**: Horizontal bar chart showing optimization time per component

### Multi-Objective Visualization
- **Pareto Front Plot**: Scatter plot of score vs latency with Pareto-optimal solutions highlighted
- **Budget Progression**: Color-coded trials showing budget allocation (for multi-fidelity)

### RAGAS Visualizations
- **RAGAS Metrics Progression**: Multi-line plot of all RAGAS metrics over trials
- **RAGAS Category Comparison**: Bar chart comparing retrieval vs generation metrics
- **Average RAGAS Scores**: Bar chart of mean scores for each RAGAS metric

## Interactive HTML Reports
- Plotly-based interactive optimization history
- Downloadable HTML files for offline analysis
- Zoomable and hoverable data points for detailed inspection