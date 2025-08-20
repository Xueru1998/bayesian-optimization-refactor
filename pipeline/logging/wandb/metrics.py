import wandb
from typing import Dict, Any, Optional, Union
from optuna.trial import Trial
from .utils import WandBUtils


class MetricsMixin:
    @classmethod
    def log_trial_metrics(cls, trial, score, config=None, results=None, step=None):
        if wandb.run is None:
            return
        
        if step is None:
            step = cls.get_next_step()
        
        if isinstance(trial, Trial):
            trial_number = trial.number
            execution_time = trial.user_attrs.get('execution_time', 0.0)
            trial_config = trial.params
            trial_attrs = trial.user_attrs
        else:
            trial_number = trial
            trial_config = config or {}
            trial_attrs = results or {}
            execution_time = results.get('latency', results.get('execution_time', 0.0)) if results else 0.0
        
        config_id = WandBUtils.get_config_id(trial_config)
        
        metrics = {
            "trial_number": trial_number,
            "config_id": config_id,
            "score": score,
            "execution_time": execution_time,
        }
        
        if results and 'budget' in results:
            metrics["budget"] = results['budget']
            metrics["budget_percentage"] = results.get('budget_percentage', 1.0)
        
        if 'retriever_top_k' in trial_config:
            metrics["retriever_top_k"] = trial_config.get('retriever_top_k')
        elif results and 'retriever_top_k' in results:
            metrics["retriever_top_k"] = results.get('retriever_top_k')
        
        is_ragas_evaluation = results and 'ragas_mean_score' in results
        
        if is_ragas_evaluation:
            metrics["ragas_mean_score"] = results.get('ragas_mean_score', 0.0)
            
            ragas_metrics = results.get('ragas_metrics', {})
            if isinstance(ragas_metrics, dict):
                ragas_metric_names = [
                    'context_precision', 'context_recall', 
                    'answer_relevancy', 'faithfulness', 
                    'factual_correctness', 'semantic_similarity',
                    'retrieval_mean_score', 'generation_mean_score',
                    'overall_mean_score'
                ]
                
                for metric_name in ragas_metric_names:
                    if metric_name in ragas_metrics:
                        value = ragas_metrics.get(metric_name)
                        if isinstance(value, (int, float)):
                            metrics[f"ragas/{metric_name}"] = value
        else:
            
            compressor_score = (trial_attrs.get("compressor_score") or 
                            trial_attrs.get("compression_score") or 
                            (results.get("compressor_score") if results else None) or
                            (results.get("compression_score") if results else None))
            
            component_scores = {
                "retrieval_score": trial_attrs.get("retrieval_score") or (results.get("retrieval_score") if results else None),
                "reranker_score": trial_attrs.get("reranker_score") or (results.get("reranker_score") if results else None),
                "filter_score": trial_attrs.get("filter_score") or (results.get("filter_score") if results else None),
                "compressor_score": compressor_score, 
                "prompt_maker_score": trial_attrs.get("prompt_maker_score") or (results.get("prompt_maker_score") if results else None),
                "generation_score": trial_attrs.get("generation_score") or (results.get("generation_score") if results else None),
                "combined_score": trial_attrs.get("combined_score") or (results.get("combined_score") if results else None),
                "query_expansion_score": trial_attrs.get("query_expansion_score") or (results.get("query_expansion_score") if results else None),
                "last_retrieval_score": trial_attrs.get("last_retrieval_score") or (results.get("last_retrieval_score") if results else None),
            }
            
            for key, value in component_scores.items():
                if value is not None and isinstance(value, (int, float)) and value > 0:
                    metrics[key] = value
        
        if results and "multi_objective" in results:
            metrics["multi_objective/score"] = results["score"]
            metrics["multi_objective/latency"] = results["latency"]

        wandb.log(metrics, step=step)
        
        if is_ragas_evaluation:
            MetricsMixin._log_ragas_detailed_metrics(results, step=step)
        else:
            MetricsMixin._log_detailed_component_metrics(
                trial_attrs if isinstance(trial, Trial) else results, 
                step=step
            )
    
    @staticmethod
    def _log_ragas_detailed_metrics(results: Dict[str, Any], step: Optional[int] = None):
        if wandb.run is None or not results:
            return
        
        detailed_metrics = {}
        
        ragas_metrics = results.get('ragas_metrics', {})
        if isinstance(ragas_metrics, dict):
            for category in ['retrieval', 'generation']:
                category_metrics = {}
                
                if category == 'retrieval':
                    metric_names = ['context_precision', 'context_recall']
                else:
                    metric_names = ['answer_relevancy', 'faithfulness', 
                                  'factual_correctness', 'semantic_similarity']
                
                for metric_name in metric_names:
                    if metric_name in ragas_metrics:
                        value = ragas_metrics.get(metric_name)
                        if isinstance(value, (int, float)):
                            category_metrics[metric_name] = value
                
                if category_metrics:
                    for metric_name, value in category_metrics.items():
                        detailed_metrics[f"ragas_breakdown/{category}/{metric_name}"] = value
        
        if detailed_metrics:
            if step is not None:
                wandb.log(detailed_metrics, step=step)
            else:
                wandb.log(detailed_metrics)
    
    @staticmethod
    def _log_detailed_component_metrics(attrs, step=None):
        if wandb.run is None or not attrs:
            return
        
        detailed_metrics = {}
        
        if isinstance(attrs, dict) and any('_config' in key for key in attrs.keys()):
            if 'query_expansion_config' in attrs:
                detailed_metrics["query_expansion/config"] = attrs['query_expansion_config']
            if 'query_expansion_retrieval_method' in attrs:
                detailed_metrics["query_expansion/retrieval_method"] = attrs['query_expansion_retrieval_method']
                if attrs['query_expansion_retrieval_method'] == 'bm25':
                    detailed_metrics["query_expansion/bm25_tokenizer"] = attrs.get('query_expansion_bm25_tokenizer', '')
        
        components = ["retrieval", "query_expansion", "reranker", "filter", 
                    "compression", "generation", "prompt", "prompt_maker"]
        
        for component in components:
            metrics_key = f"{component}_metrics"
            if metrics_key in attrs:
                component_metrics = attrs[metrics_key]
                if isinstance(component_metrics, dict):
                    if 'metrics' in component_metrics:
                        metrics_data = component_metrics['metrics']
                    else:
                        metrics_data = component_metrics
                    
                    for metric_name, value in metrics_data.items():
                        if isinstance(value, (int, float)) and metric_name not in ["mean_accuracy", "mean_score"]:
                            detailed_metrics[f"{component}/{metric_name}"] = value
        
        if detailed_metrics:
            if step is not None:
                wandb.log(detailed_metrics, step=step)
            else:
                wandb.log(detailed_metrics)