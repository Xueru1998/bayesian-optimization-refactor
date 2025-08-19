import os
import wandb
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union
import hashlib
import json
import optuna
import optuna.visualization as vis
from optuna.trial import Trial, FrozenTrial


class WandBLogger:
    _step_counter = 0
    
    @classmethod
    def reset_step_counter(cls):
        cls._step_counter = 0
    
    @classmethod
    def get_next_step(cls):
        step = cls._step_counter
        cls._step_counter += 1
        return step
    
    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None
        elif value == '':
            return None
        elif isinstance(value, (np.integer, int)):
            return int(value)
        elif isinstance(value, (np.floating, float)):  
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, bool):
            return value
        elif isinstance(value, list):
            return str(value)
        elif isinstance(value, (str, np.str_)):
            return str(value)
        elif isinstance(value, dict):
            return str(value)
        elif hasattr(value, '__str__'):
            return str(value)
        else:
            return str(value)
    
    @staticmethod
    def _get_config_id(config: Dict[str, Any]) -> str:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        config_clean = convert_numpy(config)
        config_str = json.dumps(config_clean, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    @staticmethod
    def _format_config_for_table(config: Dict[str, Any], component: str = None) -> str:
        exclude_keys = []
        
        if component == 'query_expansion':
            exclude_keys = ['query_expansion_generator_module_type', 'query_expansion_llm']
        elif component == 'generator':
            exclude_keys = ['generator_llm', 'generator_module_type']
        
        config_items = []
        for key in sorted(config.keys()):
            if 'api_url' in key or key in exclude_keys:
                continue
            
            value = config[key]
            if isinstance(value, list):
                value_str = str(value)
            elif isinstance(value, (int, float)):
                value_str = str(value)
            else:
                value_str = str(value)
            
            config_items.append(f"{key}: {value_str}")
        
        return " | ".join(config_items)
    
    @staticmethod
    def log_trial_metrics(trial, score, config=None, results=None, step=None):
        if wandb.run is None:
            return
        
        if step is None:
            step = WandBLogger.get_next_step()
        
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
        
        config_id = WandBLogger._get_config_id(trial_config)
        
        metrics = {
            "trial_number": trial_number,
            "config_id": config_id,
            "score": score,
            "latency": execution_time,
            "execution_time": execution_time,
        }
        
        config_metrics = {}
    
        # Query Expansion configs
        if 'query_expansion_method' in trial_config:
            config_metrics['config/query_expansion_method'] = trial_config['query_expansion_method']
        if 'query_expansion_model' in trial_config:
            config_metrics['config/query_expansion_model'] = trial_config['query_expansion_model']
        if 'query_expansion_generator_module_type' in trial_config:
            config_metrics['config/query_expansion_generator_type'] = trial_config['query_expansion_generator_module_type']
        if 'query_expansion_llm' in trial_config:
            config_metrics['config/query_expansion_llm'] = trial_config['query_expansion_llm']
        if 'query_expansion_max_token' in trial_config:
            config_metrics['config/query_expansion_max_token'] = trial_config['query_expansion_max_token']
        if 'query_expansion_temperature' in trial_config:
            config_metrics['config/query_expansion_temperature'] = trial_config['query_expansion_temperature']
        
        # Retrieval configs
        if 'retrieval_method' in trial_config:
            config_metrics['config/retrieval_method'] = trial_config['retrieval_method']
        if 'retriever_top_k' in trial_config:
            config_metrics['config/retriever_top_k'] = trial_config['retriever_top_k']
        if 'bm25_tokenizer' in trial_config:
            config_metrics['config/bm25_tokenizer'] = trial_config['bm25_tokenizer']
        if 'vectordb_name' in trial_config:
            config_metrics['config/vectordb_name'] = trial_config['vectordb_name']
        
        # Reranker configs
        if 'passage_reranker_method' in trial_config:
            config_metrics['config/reranker_method'] = trial_config['passage_reranker_method']
        if 'reranker_model_name' in trial_config:
            config_metrics['config/reranker_model'] = trial_config['reranker_model_name']
        elif 'reranker_model' in trial_config:
            config_metrics['config/reranker_model'] = trial_config['reranker_model']
        if 'reranker_top_k' in trial_config:
            config_metrics['config/reranker_top_k'] = trial_config['reranker_top_k']
        
        # Filter configs
        if 'passage_filter_method' in trial_config:
            config_metrics['config/filter_method'] = trial_config['passage_filter_method']
        if 'threshold' in trial_config and 'passage_filter_method' in trial_config:
            config_metrics['config/filter_threshold'] = trial_config['threshold']
        if 'percentile' in trial_config and 'passage_filter_method' in trial_config:
            config_metrics['config/filter_percentile'] = trial_config['percentile']
        
        # Compressor configs
        if 'passage_compressor_method' in trial_config:
            method = trial_config['passage_compressor_method']
            config_metrics['config/compressor_method'] = method
            
            if method == 'lexrank':
                if 'compression_ratio' in trial_config:
                    config_metrics['config/compressor_compression_ratio'] = trial_config['compression_ratio']
                if 'threshold' in trial_config:
                    config_metrics['config/compressor_threshold'] = trial_config['threshold']
                if 'damping' in trial_config:
                    config_metrics['config/compressor_damping'] = trial_config['damping']
                if 'max_iterations' in trial_config:
                    config_metrics['config/compressor_max_iterations'] = trial_config['max_iterations']
            elif method == 'spacy':
                if 'compression_ratio' in trial_config:
                    config_metrics['config/compressor_compression_ratio'] = trial_config['compression_ratio']
                if 'spacy_model' in trial_config:
                    config_metrics['config/compressor_spacy_model'] = trial_config['spacy_model']
            elif method in ['tree_summarize', 'refine']:
                if 'compressor_model' in trial_config:
                    config_metrics['config/compressor_model'] = trial_config['compressor_model']
                if 'compressor_llm' in trial_config:
                    config_metrics['config/compressor_llm'] = trial_config['compressor_llm']
                if 'compressor_generator_module_type' in trial_config:
                    config_metrics['config/compressor_generator_type'] = trial_config['compressor_generator_module_type']
        
        # Prompt maker configs
        if 'prompt_maker_method' in trial_config:
            config_metrics['config/prompt_method'] = trial_config['prompt_maker_method']
        if 'prompt_template_idx' in trial_config:
            config_metrics['config/prompt_template_idx'] = trial_config['prompt_template_idx']
        
        # Generator configs
        if 'generator_model' in trial_config:
            config_metrics['config/generator_model'] = trial_config['generator_model']
        if 'generator_llm' in trial_config:
            config_metrics['config/generator_llm'] = trial_config['generator_llm']
        if 'generator_module_type' in trial_config:
            config_metrics['config/generator_type'] = trial_config['generator_module_type']
        if 'generator_temperature' in trial_config:
            config_metrics['config/generator_temperature'] = trial_config['generator_temperature']
        if 'generator_max_tokens' in trial_config:
            config_metrics['config/generator_max_tokens'] = trial_config['generator_max_tokens']
        
        # Add config metrics to main metrics
        metrics.update(config_metrics)
        
        if results and 'budget' in results:
            metrics["budget"] = results['budget']
            metrics["budget_percentage"] = results.get('budget_percentage', 1.0)
            metrics["score_per_budget"] = score / results.get('budget_percentage', 1.0)
        
        if 'retriever_top_k' in trial_config:
            metrics["retriever_top_k"] = trial_config.get('retriever_top_k')
        elif results and 'retriever_top_k' in results:
            metrics["retriever_top_k"] = results.get('retriever_top_k')
        
        is_ragas_evaluation = results and 'ragas_mean_score' in results
        
        if is_ragas_evaluation:
            metrics["ragas_mean_score"] = results.get('ragas_mean_score', 0.0)
            metrics["evaluation_method"] = "ragas"
            
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
            metrics["evaluation_method"] = "traditional"
            
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
            WandBLogger._log_ragas_detailed_metrics(results, step=step)
        else:
            WandBLogger._log_detailed_component_metrics(
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
    
    @staticmethod
    def create_parameters_table(trials_data: List[Dict[str, Any]], 
                            include_budget: bool = False, 
                            step: Optional[int] = None) -> wandb.Table:
        if not trials_data:
            return None

        all_columns = set()
        for row in trials_data:
            all_columns.update(row.keys())
        
        has_ragas = any('ragas_mean_score' in row for row in trials_data)
        
        if has_ragas:
            priority_columns = ["trial", "score", "ragas_mean_score", "execution_time_s", "status", "retriever_top_k"]
        else:
            priority_columns = ["trial", "score", "execution_time_s", "status", "retriever_top_k"]
            
        if include_budget:
            if has_ragas:
                priority_columns = ["trial", "config_id", "score", "ragas_mean_score", "latency", "execution_time_s", "status", "budget", "budget_percentage"]
            else:
                priority_columns = ["trial", "config_id", "score", "latency", "execution_time_s", "status", "budget", "budget_percentage"]
        
        param_columns = sorted([col for col in all_columns if col not in priority_columns])
        columns = [col for col in priority_columns if col in all_columns] + param_columns

        normalized_data = []
        for row in trials_data:
            normalized_row = []
            for col in columns:
                value = row.get(col, None)
                normalized_value = WandBLogger._normalize_value(value)
                normalized_row.append(normalized_value)
            normalized_data.append(normalized_row)
        
        return wandb.Table(columns=columns, data=normalized_data)
    
    @staticmethod
    def log_optimization_plots(study_or_facade: Any, all_trials: List[Dict[str, Any]] = None,
                            pareto_front: List[Dict[str, Any]] = None, prefix: str = "optuna"):
        if wandb.run is None:
            return
        
        is_optuna = hasattr(study_or_facade, 'trials')
        
        if is_optuna:
            study = study_or_facade
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                print("No completed trials found. Skipping plot logging.")
                return
            max_trial_number = max(t.number for t in completed_trials)
        else:
            if not all_trials:
                return
            max_trial_number = max([t.get('trial_number', 0) for t in all_trials]) if all_trials else 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plots_logged = []
            
            if is_optuna:
                WandBLogger._log_optuna_plots(study, temp_dir, prefix, plots_logged, None)
            else:
                WandBLogger._log_smac_plots(all_trials, pareto_front, temp_dir, prefix)
            
            print(f"Successfully logged optimization plots to W&B")
    
    @staticmethod
    def _log_optuna_plots(study: optuna.Study, temp_dir: str, prefix: str, 
                        plots_logged: List[str], max_trial_number: int):
        is_multi_objective = hasattr(study, 'directions') and len(study.directions) > 1
        
        try:
            if not is_multi_objective:
                trials = study.trials
                completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
                
                if completed_trials:
                    trial_numbers = [t.number for t in completed_trials]
                    scores = [t.value for t in completed_trials] 

                    best_scores = []
                    current_best = float('-inf') if study.direction == optuna.StudyDirection.MAXIMIZE else float('inf')
                    for score in scores:
                        if study.direction == optuna.StudyDirection.MAXIMIZE:
                            current_best = max(current_best, score)
                        else:
                            current_best = min(current_best, score)
                        best_scores.append(current_best)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(trial_numbers, scores, 'b-', alpha=0.7, label='Score')
                    plt.scatter(trial_numbers, scores, c='blue', s=20)
                    plt.plot(trial_numbers, best_scores, 'r--', alpha=0.7, label='Best Score')
                    
                    plt.xlabel('Trial Number')
                    plt.ylabel('Objective Value')
                    plt.title('Optimization History')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    file_path = os.path.join(temp_dir, "optimization_history.png")
                    plt.savefig(file_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    wandb.log({f"{prefix}/optimization_history": wandb.Image(file_path)})
                    plots_logged.append("optimization_history")

                fig = vis.plot_optimization_history(study)
                html_str = pio.to_html(fig, include_plotlyjs='cdn')
                html_file = os.path.join(temp_dir, "optimization_history.html")
                with open(html_file, 'w') as f:
                    f.write(html_str)
                wandb.save(html_file)
                
            else:
                for i, direction in enumerate(study.directions):
                    try:
                        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values]
                        
                        if completed_trials:
                            trial_numbers = [t.number for t in completed_trials]
                            if i == 0:  
                                scores = [-t.values[i] for t in completed_trials]
                            else:  
                                scores = [t.values[i] for t in completed_trials]
                            
                            best_scores = []
                            if i == 0:  
                                current_best = 0
                                for score in scores:
                                    current_best = max(current_best, score)
                                    best_scores.append(current_best)
                            else: 
                                current_best = float('inf')
                                for score in scores:
                                    current_best = min(current_best, score)
                                    best_scores.append(current_best)

                            plt.figure(figsize=(10, 6))
                            objective_name = "Score" if i == 0 else "Latency"
                            plt.plot(trial_numbers, scores, 'b-' if i == 0 else 'g-', alpha=0.7, label=objective_name)
                            plt.scatter(trial_numbers, scores, c='blue' if i == 0 else 'green', s=20)
                            plt.plot(trial_numbers, best_scores, 'r--', alpha=0.7, label=f'Best {objective_name}')
                            
                            plt.xlabel('Trial Number')
                            plt.ylabel(f'{objective_name} {"(Higher is better)" if i == 0 else "(Lower is better)"}')
                            plt.title(f'Optimization History - {objective_name}')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            file_path = os.path.join(temp_dir, f"optimization_history_obj{i}.png")
                            plt.savefig(file_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                            objective_name_lower = "score" if i == 0 else "latency"
                            wandb.log({f"{prefix}/optimization_history_{objective_name_lower}": wandb.Image(file_path)})
                            plots_logged.append(f"optimization_history_obj{i}")

                        fig = vis.plot_optimization_history(study, target=lambda t: t.values[i])
                        html_str = pio.to_html(fig, include_plotlyjs='cdn')
                        html_file = os.path.join(temp_dir, f"optimization_history_obj{i}.html")
                        with open(html_file, 'w') as f:
                            f.write(html_str)
                        wandb.save(html_file)
                        
                    except Exception as e:
                        print(f"Could not generate optimization history plot for objective {i}: {e}")
        except Exception as e:
            print(f"Could not generate optimization history plot: {e}")
        
        try:
            if len(study.trials) > 5:
                if not is_multi_objective:
                    print("Skipping param importance plots (requires kaleido)")
                else:
                    for i in range(len(study.directions)):
                        print(f"Skipping param importance plot for objective {i} (requires kaleido)")
        except Exception as e:
            print(f"Could not generate param importances plot: {e}")
        
        try:
            if is_multi_objective and len(study.directions) >= 2:
                print("Skipping Pareto front plot (requires kaleido)")
        except Exception as e:
            print(f"Could not generate Pareto front plot: {e}")
        
        WandBLogger.log_component_comparison_plot(study, prefix)
        WandBLogger._save_html_plots(study, plots_logged, temp_dir, prefix)
        
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if is_multi_objective and hasattr(trial, 'values') and trial.values:
                    score = -trial.values[0] if trial.values else 0.0
                    latency = trial.values[1] if len(trial.values) > 1 else 0.0
                else:
                    score = trial.value if hasattr(trial, 'value') else 0.0
                    latency = trial.user_attrs.get('execution_time', 0.0)
                
                trial_data = {
                    "trial": trial.number,
                    "score": score,
                    "execution_time_s": trial.user_attrs.get('execution_time', 0.0),
                    "status": "COMPLETE"
                }

                for param_name, param_value in trial.params.items():
                    trial_data[param_name] = param_value

                for attr_name in ['retrieval_score', 'reranker_score', 'filter_score', 
                                'compressor_score', 'compression_score', 'prompt_maker_score', 
                                'generation_score', 'generator_score', 'combined_score',
                                'query_expansion_score', 'retriever_top_k', 'ragas_mean_score']:
                    if attr_name in trial.user_attrs:
                        trial_data[attr_name] = trial.user_attrs[attr_name]

                if 'ragas_metrics' in trial.user_attrs:
                    trial_data['ragas_metrics'] = trial.user_attrs['ragas_metrics']
                    trial_data['ragas_mean_score'] = trial.user_attrs.get('ragas_mean_score', 0.0)
                
                trials_data.append(trial_data)

        if trials_data:
            WandBLogger.log_study_tables(study, trials_data, prefix=prefix)

    
    @staticmethod
    def _log_smac_plots(all_trials: List[Dict[str, Any]], pareto_front: List[Dict[str, Any]], 
                    temp_dir: str, prefix: str):
        max_trial_number = max([t.get('trial_number', 0) for t in all_trials]) if all_trials else 0

        WandBLogger._plot_optimization_history(all_trials, temp_dir, prefix)
        WandBLogger._plot_component_scores(all_trials, temp_dir, prefix)  
        WandBLogger._plot_pareto_front(all_trials, pareto_front, temp_dir, prefix)
        
        has_ragas = any('ragas_mean_score' in t for t in all_trials)
        if has_ragas:
            WandBLogger.log_ragas_comparison_plot(all_trials, prefix)
        
        if len(all_trials) > 10:
            WandBLogger._plot_parameter_importance(all_trials, temp_dir, prefix)
        
        WandBLogger._log_study_tables(all_trials, pareto_front, prefix=prefix, step=None)
    
    @staticmethod
    def _save_html_plots(study: optuna.Study, plots_logged: List[str], temp_dir: str, prefix: str):
        try:
            html_plots = []
            is_multi_objective = hasattr(study, 'directions') and len(study.directions) > 1
            
            for plot_name in plots_logged:
                try:
                    if plot_name == "optimization_history":
                        fig = vis.plot_optimization_history(study)
                    elif plot_name.startswith("optimization_history_obj"):
                        obj_idx = int(plot_name.split("obj")[1])
                        fig = vis.plot_optimization_history(study, target=lambda t: t.values[obj_idx])
                    elif plot_name == "param_importances":
                        fig = vis.plot_param_importances(study)
                    elif plot_name.startswith("param_importances_obj"):
                        obj_idx = int(plot_name.split("obj")[1])
                        fig = vis.plot_param_importances(study, target=lambda t: t.values[obj_idx])
                    elif plot_name == "pareto_front":
                        fig = vis.plot_pareto_front(study)
                    else:
                        continue
                    
                    html_str = pio.to_html(fig, include_plotlyjs='cdn')
                    html_file = os.path.join(temp_dir, f"{plot_name}.html")
                    with open(html_file, 'w') as f:
                        f.write(html_str)
                    
                    wandb.save(html_file)
                    html_plots.append(plot_name)
                    
                except Exception as e:
                    print(f"Could not save HTML for {plot_name}: {e}")
            
            if html_plots:
                print(f"Logged {len(html_plots)} interactive HTML plots to W&B Files")
        except Exception as e:
            print(f"Could not save HTML plots: {e}")
    
    @staticmethod
    def detect_active_components(study_or_trials: Union[optuna.Study, List[Dict[str, Any]]]) -> Dict[str, bool]:
        components = {
            "query_expansion": False,
            "retrieval": False,
            "reranker": False,
            "filter": False,
            "compressor": False,
            "prompt_maker": False,
            "generator": False
        }
        
        if hasattr(study_or_trials, 'trials'):
            trials = study_or_trials.trials
            for trial in trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    attrs = trial.user_attrs
                    if attrs.get("query_expansion_score") is not None and attrs.get("query_expansion_score") > 0:
                        components["query_expansion"] = True
                    if attrs.get("retrieval_score") is not None and attrs.get("retrieval_score") > 0:
                        components["retrieval"] = True
                    if attrs.get("reranker_score") is not None and attrs.get("reranker_score") > 0:
                        components["reranker"] = True
                    if attrs.get("filter_score") is not None and attrs.get("filter_score") > 0:
                        components["filter"] = True
                    if (attrs.get("compression_score") is not None and attrs.get("compression_score") > 0) or \
                    (attrs.get("compressor_score") is not None and attrs.get("compressor_score") > 0):
                        components["compressor"] = True
                    if attrs.get("prompt_maker_score") is not None and attrs.get("prompt_maker_score") > 0:
                        components["prompt_maker"] = True
                    if attrs.get("generation_score") is not None and attrs.get("generation_score") > 0:
                        components["generator"] = True
                    
                    if all(components.values()):
                        break
        else:
            for trial in study_or_trials:
                for comp in components.keys():
                    if comp == "compressor":
                        if (trial.get("compressor_score", 0) > 0 or 
                            trial.get("compression_score", 0) > 0):
                            components[comp] = True
                    elif comp == "generator":
                        if trial.get("generation_score", 0) > 0:
                            components[comp] = True
                    else:
                        score_key = f"{comp}_score"
                        if score_key in trial and trial.get(score_key, 0) > 0:
                            components[comp] = True
        
        return components
        
    @staticmethod
    def log_component_comparison_plot(study_or_trials: Union[optuna.Study, List[Dict[str, Any]]], 
                                    prefix: str = "optuna"):
        if wandb.run is None:
            return
        
        active_components = WandBLogger.detect_active_components(study_or_trials)
        
        has_query_expansion = active_components.get("query_expansion", False)
        
        if has_query_expansion:
            active_components.pop("retrieval", None)
            component_display_names = {
                "query_expansion": "Query Expansion/Retrieval",
                "reranker": "Reranker",
                "filter": "Filter", 
                "compressor": "Compressor",
                "prompt_maker": "Prompt Maker",
                "generator": "Generator"
            }
        else:
            component_display_names = {
                "retrieval": "Retrieval",
                "reranker": "Reranker",
                "filter": "Filter",
                "compressor": "Compressor", 
                "prompt_maker": "Prompt Maker",
                "generator": "Generator"
            }
        
        active_component_names = [name for name, is_active in active_components.items() if is_active]
        
        if not active_component_names:
            return
        
        component_scores = {comp: [] for comp in active_component_names}
        trial_numbers = []
        
        if hasattr(study_or_trials, 'trials'):
            for trial in study_or_trials.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    trial_numbers.append(trial.number)
                    for comp in active_component_names:
                        if comp == "query_expansion" and has_query_expansion:
                            score = trial.user_attrs.get("last_retrieval_score", 0.0)
                        elif comp == "compressor":
                            score = trial.user_attrs.get("compressor_score", 0.0) or \
                                trial.user_attrs.get("compression_score", 0.0)
                        else:
                            score_key = f"{comp}_score"
                            score = trial.user_attrs.get(score_key, 0.0)
                        component_scores[comp].append(score)
        else:
            trials = sorted(study_or_trials, key=lambda x: x.get('trial_number', 0))
            trial_numbers = [t.get('trial_number', 0) for t in trials]
            
            for trial in trials:
                for comp in active_component_names:
                    if comp == "compressor":
                        score = trial.get("compressor_score", 0.0) or trial.get("compression_score", 0.0)
                    else:
                        score_key = f"{comp}_score"
                        score = trial.get(score_key, 0.0)
                    component_scores[comp].append(score)
        
        if not trial_numbers:
            return
        
        # Use matplotlib instead of plotly to avoid kaleido
        plt.figure(figsize=(12, 6))
        
        for comp in active_component_names:
            display_name = component_display_names.get(comp, comp.replace('_', ' ').title())
            plt.plot(trial_numbers, component_scores[comp], marker='o', markersize=4, 
                    label=display_name, alpha=0.8)
        
        plt.xlabel('Trial Number')
        plt.ylabel('Score')
        plt.title('Component Score Progression')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "component_scores.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            wandb.log({f"{prefix}/component_scores": wandb.Image(file_path)})
        
        # Bar chart for average scores
        avg_scores = {comp: np.mean(component_scores[comp]) for comp in active_component_names}
        
        plt.figure(figsize=(10, 6))
        components = [component_display_names.get(k, k) for k in avg_scores.keys()]
        scores = list(avg_scores.values())
        
        bars = plt.bar(range(len(components)), scores)
        plt.xticks(range(len(components)), components, rotation=45, ha='right')
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.ylabel('Average Score')
        plt.title('Average Component Scores')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "component_avg_scores.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            wandb.log({f"{prefix}/component_avg_scores": wandb.Image(file_path)})
    
    @staticmethod
    def log_ragas_comparison_plot(all_trials: List[Dict[str, Any]], prefix: str = "ragas"):
        if wandb.run is None or not all_trials:
            return
        
        ragas_trials = [t for t in all_trials if 'ragas_mean_score' in t]
        if not ragas_trials:
            return
        
        trials = sorted(ragas_trials, key=lambda x: x.get('trial_number', 0))
        trial_numbers = [t.get('trial_number', 0) for t in trials]
        
        ragas_metrics = {
            'context_precision': [],
            'context_recall': [],
            'answer_relevancy': [],
            'faithfulness': [],
            'factual_correctness': [],
            'semantic_similarity': [],
            'ragas_mean_score': []
        }
        
        for trial in trials:
            ragas_data = trial.get('ragas_metrics', {})
            if isinstance(ragas_data, dict):
                for metric in ragas_metrics.keys():
                    if metric == 'ragas_mean_score':
                        value = trial.get('ragas_mean_score', 0.0)
                    else:
                        value = ragas_data.get(metric, 0.0)
                    ragas_metrics[metric].append(value)
        
        # Use matplotlib instead of plotly
        plt.figure(figsize=(14, 8))
        
        retrieval_metrics = ['context_precision', 'context_recall']
        generation_metrics = ['answer_relevancy', 'faithfulness', 'factual_correctness', 'semantic_similarity']
        
        colors = {
            'context_precision': 'blue',
            'context_recall': 'lightblue',
            'answer_relevancy': 'green',
            'faithfulness': 'lightgreen',
            'factual_correctness': 'orange',
            'semantic_similarity': 'red',
            'ragas_mean_score': 'purple'
        }
        
        for metric in retrieval_metrics + generation_metrics:
            if ragas_metrics[metric]:
                plt.plot(trial_numbers, ragas_metrics[metric], 
                        marker='o', markersize=4, 
                        label=metric.replace('_', ' ').title(),
                        color=colors.get(metric, 'gray'),
                        alpha=0.8)
        
        if ragas_metrics['ragas_mean_score']:
            plt.plot(trial_numbers, ragas_metrics['ragas_mean_score'], 
                    marker='s', markersize=6, linewidth=3,
                    label='RAGAS Mean Score',
                    color=colors['ragas_mean_score'])
        
        plt.xlabel('Trial Number')
        plt.ylabel('Score')
        plt.title('RAGAS Metrics Progression')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "ragas_metrics_progression.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            wandb.log({f"{prefix}/metrics_progression": wandb.Image(file_path)})
        
        # Average scores bar chart
        avg_retrieval_scores = {
            metric: np.mean(ragas_metrics[metric]) 
            for metric in retrieval_metrics if ragas_metrics[metric]
        }
        avg_generation_scores = {
            metric: np.mean(ragas_metrics[metric]) 
            for metric in generation_metrics if ragas_metrics[metric]
        }
        
        plt.figure(figsize=(12, 6))
        
        all_metrics = list(avg_retrieval_scores.keys()) + list(avg_generation_scores.keys())
        all_scores = list(avg_retrieval_scores.values()) + list(avg_generation_scores.values())
        all_colors = ['lightblue'] * len(avg_retrieval_scores) + ['lightgreen'] * len(avg_generation_scores)
        
        bars = plt.bar(range(len(all_metrics)), all_scores, color=all_colors)
        plt.xticks(range(len(all_metrics)), 
                  [m.replace('_', ' ').title() for m in all_metrics], 
                  rotation=45, ha='right')
        
        for i, (bar, score) in enumerate(zip(bars, all_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.ylabel('Average Score')
        plt.title('Average RAGAS Scores by Category')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "ragas_average_scores.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            wandb.log({f"{prefix}/average_scores": wandb.Image(file_path)})
    
    @staticmethod
    def log_ragas_summary_table(all_trials: List[Dict[str, Any]], prefix: str = "ragas"):
        if wandb.run is None or not all_trials:
            return
        
        ragas_trials = [t for t in all_trials if 'ragas_mean_score' in t]
        if not ragas_trials:
            return
        
        summary_data = []
        for trial in ragas_trials:
            trial_data = {
                'trial_number': trial.get('trial_number', 0),
                'ragas_mean_score': trial.get('ragas_mean_score', 0.0),
                'combined_score': trial.get('combined_score', 0.0),
                'latency': trial.get('latency', 0.0)
            }
            
            ragas_metrics = trial.get('ragas_metrics', {})
            if isinstance(ragas_metrics, dict):
                for metric in ['context_precision', 'context_recall', 'answer_relevancy', 
                             'faithfulness', 'factual_correctness', 'semantic_similarity']:
                    trial_data[metric] = ragas_metrics.get(metric, 0.0)
                
                trial_data['retrieval_mean'] = ragas_metrics.get('retrieval_mean_score', 0.0)
                trial_data['generation_mean'] = ragas_metrics.get('generation_mean_score', 0.0)
            
            config = trial.get('config', {})
            for key, value in config.items():
                clean_key = key.replace('/', '_').replace('\\', '_')
                trial_data[f"config_{clean_key}"] = WandBLogger._normalize_value(value)
            
            summary_data.append(trial_data)
        
        columns = ['trial_number', 'ragas_mean_score', 'retrieval_mean', 'generation_mean', 
                  'context_precision', 'context_recall', 'answer_relevancy', 
                  'faithfulness', 'factual_correctness', 'semantic_similarity',
                  'combined_score', 'latency']
        
        config_columns = sorted([col for col in set().union(*[d.keys() for d in summary_data]) 
                               if col.startswith('config_')])
        columns.extend(config_columns)
        
        table_data = []
        for row_dict in summary_data:
            row = []
            for col in columns:
                value = row_dict.get(col, None)
                row.append(WandBLogger._normalize_value(value))
            table_data.append(row)
        
        table = wandb.Table(columns=columns, data=table_data)
        wandb.log({f"{prefix}/summary_table": table})
    
    @staticmethod
    def _plot_pareto_front(all_trials: List[Dict[str, Any]], pareto_front: List[Dict[str, Any]], 
                        temp_dir: str, prefix: str):
        try:
            plt.figure(figsize=(10, 6))
            
            all_scores = [t['score'] for t in all_trials]
            all_latencies = [t['latency'] for t in all_trials]
            
            if any('budget' in t for t in all_trials):
                budgets = [t.get('budget', max(t.get('budget', 0) for t in all_trials)) for t in all_trials]
                scatter = plt.scatter(all_scores, all_latencies, c=budgets, 
                                    cmap='viridis', alpha=0.5, s=50)
                plt.colorbar(scatter, label='Budget (samples)')
            else:
                plt.scatter(all_scores, all_latencies, c='blue', alpha=0.3, s=30, label='All Trials')
            
            if pareto_front:
                pf_scores = [t['score'] for t in pareto_front]
                pf_latencies = [t['latency'] for t in pareto_front]
                plt.scatter(pf_scores, pf_latencies, c='red', marker='x', s=100, label='Pareto Front')
            
            plt.xlabel('Score')
            plt.ylabel('Latency (s)')
            plt.title('Multi-Objective Optimization: Pareto Front')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            file_path = os.path.join(temp_dir, 'pareto_front.png')
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            wandb.log({f"{prefix}/pareto_front": wandb.Image(file_path)})
        except Exception as e:
            print(f"Error plotting Pareto front: {e}")
    
    @staticmethod
    def _plot_optimization_history(all_trials: List[Dict[str, Any]], temp_dir: str, prefix: str):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            trials = sorted(all_trials, key=lambda x: x['trial_number'])
            trial_numbers = [t['trial_number'] for t in trials]
            scores = [t['score'] for t in trials]
            latencies = [t['latency'] for t in trials]
            
            ax1.plot(trial_numbers, scores, 'b-', alpha=0.7, label='Score')
            ax1.scatter(trial_numbers, scores, c='blue', s=20)
            
            best_scores = []
            current_best = 0
            for score in scores:
                current_best = max(current_best, score)
                best_scores.append(current_best)
            ax1.plot(trial_numbers, best_scores, 'r--', alpha=0.7, label='Best Score')
            
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Score')
            ax1.set_title('Score Progression')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(trial_numbers, latencies, 'g-', alpha=0.7, label='Latency')
            ax2.scatter(trial_numbers, latencies, c='green', s=20)
            
            best_latencies = []
            current_best = float('inf')
            for latency in latencies:
                current_best = min(current_best, latency)
                best_latencies.append(current_best)
            ax2.plot(trial_numbers, best_latencies, 'r--', alpha=0.7, label='Best Latency')
            
            ax2.set_xlabel('Trial Number')
            ax2.set_ylabel('Latency (s)')
            ax2.set_title('Latency Progression')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            file_path = os.path.join(temp_dir, 'optimization_history.png')
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            wandb.log({f"{prefix}/optimization_history": wandb.Image(file_path)})
        except Exception as e:
            print(f"Error plotting optimization history: {e}")

    @staticmethod
    def _plot_component_scores(all_trials: List[Dict[str, Any]], temp_dir: str, prefix: str):
        try:
            components = ["retrieval", "generation", "reranker", "filter", 
                        "compression", "prompt_maker", "query_expansion"]
            active_components = []
            
            for comp in components:
                score_key = f"{comp}_score"
                if any(score_key in trial and trial.get(score_key, 0) > 0 for trial in all_trials):
                    active_components.append(comp)
            
            if not active_components:
                return
            
            plt.figure(figsize=(12, 6))
            
            trials = sorted(all_trials, key=lambda x: x['trial_number'])
            trial_numbers = [t['trial_number'] for t in trials]
            
            for comp in active_components:
                score_key = f"{comp}_score"
                scores = [t.get(score_key, 0.0) for t in trials]
                if any(s > 0 for s in scores):
                    plt.plot(trial_numbers, scores, marker='o', markersize=4, 
                            label=comp.replace('_', ' ').title(), alpha=0.8)
            
            plt.xlabel('Trial Number')
            plt.ylabel('Score')
            plt.title('Component Score Progression')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            file_path = os.path.join(temp_dir, 'component_scores.png')
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            wandb.log({f"{prefix}/component_scores": wandb.Image(file_path)})
            
            plt.figure(figsize=(10, 6))
            
            avg_scores = {}
            for comp in active_components:
                score_key = f"{comp}_score"
                scores = [t.get(score_key, 0.0) for t in trials if t.get(score_key, 0.0) > 0]
                if scores:
                    avg_scores[comp] = np.mean(scores)
            
            if avg_scores:
                components = list(avg_scores.keys())
                scores = list(avg_scores.values())
                
                bars = plt.bar(range(len(components)), scores)
                plt.xticks(range(len(components)), [c.replace('_', ' ').title() for c in components], rotation=45)
                
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{score:.3f}', ha='center', va='bottom')
                
                plt.ylabel('Average Score')
                plt.title('Average Component Scores')
                plt.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                file_path = os.path.join(temp_dir, 'component_avg_scores.png')
                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                wandb.log({f"{prefix}/component_avg_scores": wandb.Image(file_path)})
                
        except Exception as e:
            print(f"Error plotting component scores: {e}")
    
    @staticmethod
    def _plot_parameter_importance(all_trials: List[Dict[str, Any]], temp_dir: str, prefix: str):
        try:
            configs = [t['config'] for t in all_trials]
            scores = [t['score'] for t in all_trials]
            
            all_params = set()
            for config in configs:
                all_params.update(config.keys())
            
            importances = {}
            
            for param in all_params:
                values = [config.get(param, None) for config in configs]
                
                unique_values = set(v for v in values if v is not None)
                if len(unique_values) <= 1:
                    continue
                
                value_scores = {}
                for val, score in zip(values, scores):
                    if val is not None:
                        if val not in value_scores:
                            value_scores[val] = []
                        value_scores[val].append(score)
                
                if len(value_scores) > 1:
                    mean_scores = [np.mean(scores) for scores in value_scores.values()]
                    importances[param] = np.var(mean_scores)
            
            if not importances:
                return
            
            sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            
            plt.figure(figsize=(10, 6))
            params = [p[0] for p in sorted_params]
            scores = [p[1] for p in sorted_params]
            
            bars = plt.barh(range(len(params)), scores)
            plt.yticks(range(len(params)), params)
            plt.xlabel('Importance Score (Variance)')
            plt.title('Parameter Importance (Top 10)')
            plt.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            file_path = os.path.join(temp_dir, 'parameter_importance.png')
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            wandb.log({f"{prefix}/parameter_importance": wandb.Image(file_path)})
            
        except Exception as e:
            print(f"Error plotting parameter importance: {e}")
    
    @staticmethod
    def log_summary(study_or_results: Union[optuna.Study, Dict[str, Any]], 
                additional_metrics: Optional[Dict[str, Any]] = None):
        if wandb.run is None:
            return
        
        if hasattr(study_or_results, 'trials'):
            study = study_or_results
            is_multi_objective = hasattr(study, 'directions') and len(study.directions) > 1
            
            if is_multi_objective:
                best_trials = study.best_trials if study.trials else []
                
                if best_trials:
                    wandb.run.summary["num_best_trials"] = len(best_trials)
                    
                    if best_trials:
                        best_trial = best_trials[0] 
                        wandb.run.summary["best_score"] = -best_trial.values[0] if best_trial.values else None
                        wandb.run.summary["best_latency"] = best_trial.values[1] if len(best_trial.values) > 1 else None
                        wandb.run.summary["best_trial_number"] = best_trial.number
            else:
                best_trial = study.best_trial if study.trials else None
                
                if best_trial:
                    wandb.run.summary["best_score"] = best_trial.value
                    wandb.run.summary["best_trial_number"] = best_trial.number
            
            wandb.run.summary["total_trials_completed"] = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        else:
            optimization_results = study_or_results
            wandb.run.summary["optimizer"] = optimization_results.get("optimizer", optimization_results.get("algorithm", "unknown"))
            wandb.run.summary["use_multi_fidelity"] = optimization_results.get("use_multi_fidelity", False)
            wandb.run.summary["best_score"] = optimization_results.get("best_score", 0.0)
            wandb.run.summary["best_latency"] = optimization_results.get("best_latency", float('inf'))
            wandb.run.summary["total_trials"] = optimization_results.get("total_trials", optimization_results.get("n_trials", 0))
            wandb.run.summary["optimization_time"] = optimization_results.get("optimization_time", 0.0)
            wandb.run.summary["early_stopped"] = optimization_results.get("early_stopped", False)
            
            all_trials = optimization_results.get("all_trials", [])
            ragas_trials = [t for t in all_trials if 'ragas_mean_score' in t]
            if ragas_trials:
                best_ragas_trial = max(ragas_trials, key=lambda x: x.get('ragas_mean_score', 0.0))
                wandb.run.summary["best_ragas_score"] = best_ragas_trial.get('ragas_mean_score', 0.0)
                wandb.run.summary["best_ragas_trial_number"] = best_ragas_trial.get('trial_number', 0)
                wandb.run.summary["num_ragas_trials"] = len(ragas_trials)

            if all_trials and optimization_results.get("use_multi_fidelity"):
                full_budget_count = len([t for t in all_trials if t.get('budget_percentage', 1.0) >= 0.99])
                unique_configs = len(set(WandBLogger._get_config_id(t.get('config', {})) for t in all_trials))
                wandb.run.summary["fully_evaluated_configs"] = full_budget_count
                wandb.run.summary["unique_configs_tested"] = unique_configs
                
            best_config = optimization_results.get("best_config", {})
            if isinstance(best_config, dict) and "trial_number" in best_config:
                wandb.run.summary["best_config_trial_number"] = best_config.get("trial_number")
            
            pareto_front = optimization_results.get("pareto_front", [])
            wandb.run.summary["pareto_front_size"] = len(pareto_front)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                wandb.run.summary[key] = value
    
    @staticmethod
    def log_ranked_trials_table(trials_data, table_name="ranked_trials", top_n=None, step=None):
        if wandb.run is None or not trials_data:
            return

        sorted_data = sorted(trials_data, key=lambda x: x.get('score', 0), reverse=True)

        for i, row in enumerate(sorted_data):
            row['rank'] = i + 1
        
        if top_n:
            sorted_data = sorted_data[:top_n]

        all_columns = set()
        for row in sorted_data:
            all_columns.update(row.keys())
        
        priority_columns = ["rank", "trial", "score", "execution_time_s", "query_expansion_top_k", "retriever_top_k"]
        param_columns = sorted([col for col in all_columns if col not in priority_columns + ["status"]])
        columns = [col for col in priority_columns if col in all_columns] + param_columns + ["status"]

        normalized_data = []
        for row in sorted_data:
            normalized_row = []
            for col in columns:
                value = row.get(col, None)
                normalized_value = WandBLogger._normalize_value(value)
                normalized_row.append(normalized_value)
            normalized_data.append(normalized_row)

        table = wandb.Table(columns=columns, data=normalized_data)
        
        wandb.log({table_name: table})
        
        print(f"[DEBUG] Successfully logged table {table_name}")
    
    @staticmethod
    def log_component_metrics_table(study_or_trials, table_name="component_metrics_breakdown", step=None):
        if wandb.run is None:
            return
        
        data = []
        
        if hasattr(study_or_trials, 'trials'):
            study = study_or_trials
            for trial in study.trials:
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    continue
                
                row = {
                    "trial": trial.number,
                    "combined_score": trial.value if not hasattr(study, 'directions') or len(study.directions) == 1 else -trial.values[0] if trial.values else 0.0,
                }

                if 'query_expansion_retrieval_method' in trial.params:
                    row["query_expansion_top_k"] = 10
                elif 'retriever_top_k' in trial.params:
                    row["retriever_top_k"] = trial.params.get('retriever_top_k')

                components = ["query_expansion", "retrieval", "reranker", "filter", 
                            "compression", "prompt_maker", "generation"]
                
                for comp in components:
                    score_key = f"{comp}_score"
                    if score_key in trial.user_attrs:
                        row[score_key] = trial.user_attrs.get(score_key, 0.0)

                    metrics_key = f"{comp}_metrics"
                    if metrics_key in trial.user_attrs:
                        metrics = trial.user_attrs[metrics_key]
                        if isinstance(metrics, dict):
                            if 'metrics' in metrics and isinstance(metrics['metrics'], dict):
                                for metric_name, value in metrics['metrics'].items():
                                    if isinstance(value, (int, float)):
                                        row[f"{comp}_{metric_name}"] = value
                            else:
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        row[f"{comp}_{metric_name}"] = value
                
                if 'last_retrieval_score' in trial.user_attrs:
                    row['last_retrieval_score'] = trial.user_attrs['last_retrieval_score']
                
                data.append(row)
        else:
            for trial in study_or_trials:
                row = {
                    "trial": trial.get("trial_number", 0),
                    "config_id": trial.get("config_id", WandBLogger._get_config_id(trial.get("config", {}))),
                    "combined_score": trial.get("score", 0.0)
                }
                
                if "budget" in trial:
                    row["budget"] = trial["budget"]
                    row["budget_percentage"] = trial.get("budget_percentage", 1.0)
                
                components = ["query_expansion", "retrieval", "reranker", "filter", 
                            "compression", "prompt_maker", "generation"]
                
                for comp in components:
                    score_key = f"{comp}_score"
                    if score_key in trial:
                        row[score_key] = trial[score_key]
                
                config = trial.get("config", {})
                for param_name, param_value in config.items():
                    row[param_name] = param_value
                
                data.append(row)
        
        if not data:
            return

        all_columns = set()
        for row in data:
            all_columns.update(row.keys())
        
        if hasattr(study_or_trials, 'trials'):
            columns = ["trial", "combined_score", "query_expansion_top_k", "retriever_top_k"] + \
                    sorted([col for col in all_columns if col not in ["trial", "combined_score", "query_expansion_top_k", "retriever_top_k"]])
        else:
            columns = ["trial", "config_id", "combined_score", "budget", "budget_percentage"] + \
                    sorted([col for col in all_columns if col not in ["trial", "config_id", "combined_score", "budget", "budget_percentage"]])
        
        normalized_data = []
        for row in data:
            normalized_row = []
            for col in columns:
                value = row.get(col, None)
                normalized_value = WandBLogger._normalize_value(value)
                normalized_row.append(normalized_value)
            normalized_data.append(normalized_row)
        
        table = wandb.Table(columns=columns, data=normalized_data)
        
        wandb.log({table_name: table})
    
    @staticmethod
    def log_study_tables(study_or_trials, trials_data, prefix="study", step=None):
        if wandb.run is None:
            return

        if hasattr(study_or_trials, 'trials'):
            study = study_or_trials
            for i, trial_data in enumerate(trials_data):
                if i < len(study.trials):
                    trial = study.trials[i]
                    if 'query_expansion_retrieval_method' in trial.params:
                        trial_data["query_expansion_top_k"] = 10
                    elif 'retriever_top_k' in trial.params:
                        trial_data["retriever_top_k"] = trial.params.get('retriever_top_k')

        if trials_data:
            params_table = WandBLogger.create_parameters_table(trials_data)
            if params_table:
                wandb.log({f"{prefix}/parameters_table": params_table})

        WandBLogger.log_ranked_trials_table(
            trials_data, 
            table_name=f"{prefix}/best_trials_top10",
            top_n=10,
            step=None
        )

        WandBLogger.log_component_metrics_table(
            study_or_trials,
            table_name=f"{prefix}/component_metrics",
            step=None
        )
        

    @staticmethod
    def _log_study_tables(all_trials, pareto_front, prefix="study", step=None):
        if wandb.run is None:
            return
        
        formatted_trials = []
        for trial in all_trials:
            formatted_trial = {
                'trial_number': trial.get('trial_number', 0),
                'config_id': trial.get('config_id', WandBLogger._get_config_id(trial.get('config', {}))),
                'score': trial.get('score', 0.0),
                'latency': trial.get('latency', 0.0),
                'execution_time_s': trial.get('execution_time', 0.0),
                'status': trial.get('status', 'COMPLETE')
            }
            
            if 'budget' in trial:
                formatted_trial['budget'] = trial['budget']
                formatted_trial['budget_percentage'] = trial.get('budget_percentage', 1.0)
            
            if 'ragas_mean_score' in trial:
                formatted_trial['ragas_mean_score'] = trial['ragas_mean_score']
            
            config = trial.get('config', {})
            formatted_trial['complete_config'] = WandBLogger._format_config_for_table(config)
            
            formatted_trials.append(formatted_trial)
        
        columns = ['trial_number', 'config_id', 'score', 'latency', 'execution_time_s', 'status', 'complete_config']
        if any('budget' in t for t in formatted_trials):
            columns.insert(6, 'budget')
            columns.insert(7, 'budget_percentage')
        if any('ragas_mean_score' in t for t in formatted_trials):
            columns.insert(3, 'ragas_mean_score')
        
        table_data = []
        for trial in formatted_trials:
            row = [WandBLogger._normalize_value(trial.get(col)) for col in columns]
            table_data.append(row)
        
        params_table = wandb.Table(columns=columns, data=table_data)
        wandb.log({f"{prefix}/all_trials_table": params_table})

        best_configs = WandBLogger._get_best_trials_by_config(all_trials, full_budget_only=True)
        
        best_configs_formatted = []
        for trial in best_configs:
            formatted_trial = {
                'trial_number': trial.get('trial_number', 0),
                'config_id': trial.get('config_id'),
                'score': trial.get('score', 0.0),
                'latency': trial.get('latency', 0.0),
                'num_evaluations': trial.get('num_evaluations', 1),
                'budget_progression': str(trial.get('budget_progression', [])),
                'complete_config': WandBLogger._format_config_for_table(trial.get('config', {}))
            }
            best_configs_formatted.append(formatted_trial)
        
        if best_configs_formatted:
            columns = ['config_id', 'score', 'latency', 'num_evaluations', 'budget_progression', 'complete_config']
            table_data = []
            for trial in best_configs_formatted:
                row = [WandBLogger._normalize_value(trial.get(col)) for col in columns]
                table_data.append(row)
            
            best_configs_table = wandb.Table(columns=columns, data=table_data)
            wandb.log({f"{prefix}/best_configs_table": best_configs_table})

        WandBLogger.log_ranked_trials_table(
            best_configs, 
            table_name=f"{prefix}/best_configs_ranked",
            top_n=20,
            step=None
        )
        
        WandBLogger.log_component_metrics_table(
            all_trials,
            table_name=f"{prefix}/component_metrics",
            step=None
        )
        
        has_ragas = any('ragas_mean_score' in t for t in all_trials)
        if has_ragas:
            WandBLogger.log_ragas_summary_table(all_trials, prefix=prefix)

    
    @staticmethod
    def _get_best_trials_by_config(trials_data: List[Dict[str, Any]], full_budget_only: bool = False) -> List[Dict[str, Any]]:
        config_groups = WandBLogger._group_trials_by_config(trials_data)
        
        best_trials = []
        for config_id, trials in config_groups.items():
            if full_budget_only:
                full_budget_trials = [t for t in trials if t.get('budget_percentage', 1.0) >= 0.99]
                if not full_budget_trials:
                    continue
                best_trial = max(full_budget_trials, key=lambda t: t.get('score', 0))
            else:
                best_trial = max(trials, key=lambda t: (t.get('budget', 0), t.get('score', 0)))
            
            best_trial['config_id'] = config_id
            best_trial['num_evaluations'] = len(trials)
            
            budgets = sorted([t.get('budget', 0) for t in trials])
            best_trial['budget_progression'] = budgets
            
            best_trials.append(best_trial)
        
        return best_trials
    
    @staticmethod
    def _group_trials_by_config(trials_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        config_groups = {}
        for trial in trials_data:
            config = trial.get('config', {})
            config_id = WandBLogger._get_config_id(config)
            
            if config_id not in config_groups:
                config_groups[config_id] = []
            config_groups[config_id].append(trial)
        
        return config_groups
    
    
    @staticmethod
    def log_final_tables(all_trials, pareto_front, prefix="final"):
        if wandb.run is None:
            return
        
        WandBLogger._log_study_tables(all_trials, pareto_front, prefix=prefix, step=None)
        
        if pareto_front:
            full_budget_pareto = [pf for pf in pareto_front if pf.get('budget_percentage', 1.0) >= 0.99]
            if full_budget_pareto:
                pareto_table = WandBLogger.create_parameters_table(full_budget_pareto, include_budget=True)
                if pareto_table:
                    wandb.log({f"{prefix}/pareto_front_table": pareto_table})
    
    @staticmethod
    def log_custom_metrics(metrics: Dict[str, Any]):
        if wandb.run is None:
            return
        
        wandb.log(metrics)
        
    @staticmethod
    def log_component_optimization_start(component: str, component_idx: int, 
                                    total_components: int, fixed_config: Dict[str, Any]):
        if wandb.run is None:
            return

        wandb.log({
            "current_component": component,
            "component_index": component_idx + 1,
            "total_components": total_components,
        })

    @staticmethod
    def log_component_trial(component: str, trial_number: int, config: Dict[str, Any], 
                        score: float, latency: float, step: Optional[int] = None):
        if wandb.run is None:
            return

        complete_config = WandBLogger._format_config_for_table(config, component)
        
        metrics = {
            f"{component}/trial": trial_number,
            f"{component}/score": score,
            f"{component}/latency": latency,
            f"{component}/complete_config": complete_config
        }
        
        wandb.log(metrics)

    @staticmethod
    def log_component_summary(component: str, best_config: Dict[str, Any], 
                            best_score: float, n_trials: int, 
                            optimization_time: float):
        if wandb.run is None:
            return

        wandb.run.summary[f"{component}_best_score"] = best_score
        wandb.run.summary[f"{component}_n_trials"] = n_trials
        wandb.run.summary[f"{component}_time_seconds"] = optimization_time

    @staticmethod
    def log_component_optimization_table(component: str, trials_data: List[Dict[str, Any]], 
                                    detailed_metrics: Optional[Union[Dict[int, Dict], List[Dict]]] = None):
        if wandb.run is None:
            print(f"[DEBUG] wandb.run is None for component {component}")
            return
            
        if not trials_data:
            print(f"[DEBUG] No trials data for component {component}")
            return
        
        print(f"[DEBUG] Starting table creation for {component} with {len(trials_data)} trials")
        
        table_data = []
        
        for i, trial in enumerate(trials_data):
            trial_number = trial.get('trial_number', trial.get('trial', i))
            
            row = {
                'trial': int(trial_number),
                'score': float(trial.get('score', 0.0)),
                'latency': float(trial.get('latency', 0.0))
            }

            config = trial.get('full_config', trial.get('config', {}))
            row['complete_config'] = WandBLogger._format_config_for_table(config, component)

            if detailed_metrics:
                trial_metrics = None
                if isinstance(detailed_metrics, dict):
                    trial_metrics = detailed_metrics.get(i)
                elif isinstance(detailed_metrics, list) and i < len(detailed_metrics):
                    trial_metrics = detailed_metrics[i]
                
                if trial_metrics and isinstance(trial_metrics, dict):
                    for metric_name, metric_value in trial_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            row[f"metric_{metric_name}"] = metric_value
            
            table_data.append(row)
        
        if not table_data:
            print(f"[DEBUG] No table data generated for {component}")
            return

        columns = ['trial', 'score', 'latency', 'complete_config']

        all_columns = set()
        for row in table_data:
            all_columns.update(row.keys())
        
        metric_columns = sorted([col for col in all_columns if col.startswith('metric_')])
        columns.extend(metric_columns)

        normalized_data = []
        for row in table_data:
            normalized_row = []
            for col in columns:
                value = row.get(col, None)
                if value == '':
                    value = None
                normalized_row.append(WandBLogger._normalize_value(value))
            normalized_data.append(normalized_row)

        print(f"[DEBUG] Creating wandb.Table with {len(columns)} columns and {len(normalized_data)} rows")
        table = wandb.Table(columns=columns, data=normalized_data)
        
        table_key = f"{component}/optimization_trials"
        
        try:
            wandb.log({table_key: table})
            print(f"[DEBUG] Successfully logged {table_key} table")
        except Exception as e:
            print(f"[ERROR] Failed to log table {table_key}: {e}")
            import traceback
            traceback.print_exc()
            
    @staticmethod
    def log_dynamic_component_table(component: str, trials_so_far: List[Dict[str, Any]], wandb_enabled: bool = True):
        if not wandb_enabled or wandb.run is None:
            return
        
        table_data = []
        for idx, trial in enumerate(trials_so_far):
            row = {
                'trial': idx + 1,
                'score': trial.get('score', 0.0),
                'latency': trial.get('latency', 0.0),
                'status': trial.get('status', 'COMPLETE')
            }

            config = trial.get('full_config', trial.get('config', {}))
            row['complete_config'] = WandBLogger._format_config_for_table(config, component)
            
            table_data.append(row)

        table_data.sort(key=lambda x: x['trial'])

        columns = ['trial', 'score', 'latency', 'status', 'complete_config']

        normalized_data = []
        for row in table_data:
            normalized_row = [WandBLogger._normalize_value(row.get(col)) for col in columns]
            normalized_data.append(normalized_row)
        
        table = wandb.Table(columns=columns, data=normalized_data)
        wandb.log({f"{component}/trials_progress": table})

    
    @staticmethod
    def log_final_componentwise_summary(all_results: Dict[str, Any]):
        if wandb.run is None:
            return

        all_trials_data = []
        component_metrics = {}
        
        for comp in all_results.get('component_order', []):
            comp_result = all_results.get('component_results', {}).get(comp, {})
            comp_trials = comp_result.get('all_trials', [])
            
            if comp not in component_metrics:
                component_metrics[comp] = []
            
            for trial in comp_trials:
                trial_row = {
                    'component': comp,
                    'trial_number': trial.get('trial_number', 0),
                    'score': trial.get('score', 0.0),
                    'latency': trial.get('latency', 0.0),
                }
                
                if comp in ['retrieval', 'query_expansion']:
                    trial_row['retrieval_score'] = trial.get('score', 0.0)
                elif comp == 'generator':
                    trial_row['generation_score'] = trial.get('score', 0.0)
                
                config = trial.get('config', {})
                for param, value in config.items():
                    clean_param = param.replace('/', '_').replace('\\', '_')
                    trial_row[clean_param] = str(value) if isinstance(value, (list, dict)) else value
                
                all_trials_data.append(trial_row)
                component_metrics[comp].append(trial)

            WandBLogger.log_component_optimization_table(
                comp, 
                comp_trials,
                comp_result.get('detailed_metrics')
            )

        if all_trials_data:
            columns = sorted(set().union(*[d.keys() for d in all_trials_data]))
            priority_cols = ['component', 'trial_number', 'score', 'latency', 'retrieval_score', 'generation_score']
            other_cols = [c for c in columns if c not in priority_cols]
            columns = [c for c in priority_cols if c in set().union(*[d.keys() for d in all_trials_data])] + sorted(other_cols)
            
            table_data = []
            for trial in all_trials_data:
                row = []
                for col in columns:
                    value = trial.get(col, None)
                    if value == '':
                        value = None
                    elif isinstance(value, (np.integer, np.int64, np.int32)):
                        value = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        value = float(value)
                    elif hasattr(value, 'item'):  
                        value = value.item()
                    elif value is not None and not isinstance(value, (str, int, float, bool)):
                        value = str(value)
                    row.append(value)
                table_data.append(row)
            
            all_trials_table = wandb.Table(columns=columns, data=table_data)
            wandb.log({"componentwise/all_trials": all_trials_table})
        
        summary_data = []
        total_time = 0
        total_trials = 0
        
        best_retrieval_score = 0.0
        best_generation_score = 0.0
        
        for comp in all_results.get('component_order', []):
            comp_result = all_results.get('component_results', {}).get(comp, {})
            if comp_result:
                comp_time = comp_result.get('optimization_time', 0.0)
                comp_trials = comp_result.get('n_trials', 0)
                best_score = comp_result.get('best_score', 0.0)
                
                hours = int(comp_time // 3600)
                minutes = int((comp_time % 3600) // 60)
                seconds = int(comp_time % 60)
                
                if hours > 0:
                    time_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    time_str = f"{minutes}m {seconds}s"
                else:
                    time_str = f"{seconds}s"
                
                summary_data.append([
                    comp.replace('_', ' ').title(),
                    f"{best_score:.4f}",
                    comp_trials,
                    time_str,
                    f"{comp_time:.1f}" 
                ])
                
                total_time += comp_time
                total_trials += comp_trials

                if comp in ['retrieval', 'query_expansion']:
                    best_retrieval_score = max(best_retrieval_score, best_score)
                elif comp == 'prompt_maker_generator' or comp == 'generator':
                    best_generation_score = max(best_generation_score, best_score)

        total_hours = int(total_time // 3600)
        total_minutes = int((total_time % 3600) // 60)
        total_seconds = int(total_time % 60)
        
        if total_hours > 0:
            total_time_str = f"{total_hours}h {total_minutes}m {total_seconds}s"
        elif total_minutes > 0:
            total_time_str = f"{total_minutes}m {total_seconds}s"
        else:
            total_time_str = f"{total_seconds}s"
        
        summary_data.append([
            "**TOTAL**",
            "-",
            total_trials,
            total_time_str,
            f"{total_time:.1f}"
        ])
        
        summary_table = wandb.Table(
            columns=["Component", "Best Score", "Trials", "Time", "Time (seconds)"],
            data=summary_data
        )
        wandb.log({"componentwise/optimization_summary": summary_table})
        
        # Get the last retrieval-related component's score and generation score
        last_retrieval_score = 0.0
        last_retrieval_component = None
        
        # Retrieval-related components in order of precedence (reverse order of pipeline)
        retrieval_components = ['passage_compressor', 'passage_filter', 'passage_reranker', 'retrieval', 'query_expansion']

        for comp in retrieval_components:
            if comp in all_results.get('component_results', {}):
                comp_result = all_results.get('component_results', {}).get(comp, {})
                if comp_result and not comp_result.get('skipped', False) and comp_result.get('n_trials', 0) > 0:
                    last_retrieval_score = comp_result.get('best_score', 0.0)
                    last_retrieval_component = comp
                    break

        if last_retrieval_score > 0 or best_generation_score > 0:
            retrieval_weight = all_results.get('retrieval_weight', 0.5)
            generation_weight = all_results.get('generation_weight', 0.5)
            
            # If no generation component, use full weight for retrieval
            if best_generation_score == 0:
                combined_score = last_retrieval_score
            else:
                combined_score = (retrieval_weight * last_retrieval_score + 
                                generation_weight * best_generation_score)

            score_breakdown_data = []
            
            if last_retrieval_component:
                retrieval_display_name = f"Retrieval ({last_retrieval_component.replace('_', ' ').title()})"
                score_breakdown_data.append([
                    retrieval_display_name, 
                    f"{last_retrieval_score:.4f}", 
                    f"{retrieval_weight*100:.0f}%", 
                    f"{last_retrieval_score * retrieval_weight:.4f}"
                ])
            
            if best_generation_score > 0:
                score_breakdown_data.append([
                    "Generation", 
                    f"{best_generation_score:.4f}", 
                    f"{generation_weight*100:.0f}%", 
                    f"{best_generation_score * generation_weight:.4f}"
                ])
            
            score_breakdown_data.append([
                "**Combined Score**", 
                "-", 
                "-", 
                f"**{combined_score:.4f}**"
            ])
            
            score_breakdown_table = wandb.Table(
                columns=["Component", "Best Score", "Weight", "Weighted Score"],
                data=score_breakdown_data
            )
            wandb.log({"componentwise/final_score_breakdown": score_breakdown_table})

            wandb.run.summary["last_retrieval_component"] = last_retrieval_component
            wandb.run.summary["last_retrieval_score"] = last_retrieval_score
            wandb.run.summary["best_generation_score"] = best_generation_score
            wandb.run.summary["final_combined_score"] = combined_score

        best_configs_data = []
        for comp in all_results.get('component_order', []):
            comp_result = all_results.get('component_results', {}).get(comp, {})
            best_config = all_results.get('best_configs', {}).get(comp, {})
            
            if comp_result and best_config:
                config_str = ", ".join([f"{k}={v}" for k, v in sorted(best_config.items())[:3]])
                if len(best_config) > 3:
                    config_str += f" ... (+{len(best_config)-3} more)"
                
                best_configs_data.append([
                    comp.replace('_', ' ').title(),
                    f"{comp_result.get('best_score', 0.0):.4f}",
                    config_str
                ])
        
        if best_configs_data:
            best_configs_table = wandb.Table(
                columns=["Component", "Best Score", "Best Configuration (Summary)"],
                data=best_configs_data
            )
            wandb.log({"componentwise/best_configurations_summary": best_configs_table})

        final_config = {}
        for component in all_results.get('component_order', []):
            comp_config = all_results.get('best_configs', {}).get(component, {})
            final_config.update(comp_config)
        
        if final_config:
            config_data = [[k, str(v)] for k, v in sorted(final_config.items())]
            final_config_table = wandb.Table(columns=["Parameter", "Value"], data=config_data)
            wandb.log({"componentwise/final_best_configuration": final_config_table})
        
        wandb.run.summary["total_optimization_time_seconds"] = all_results.get('optimization_time', 0.0)
        wandb.run.summary["total_optimization_time_minutes"] = all_results.get('optimization_time', 0.0) / 60
        wandb.run.summary["total_optimization_time_hours"] = all_results.get('optimization_time', 0.0) / 3600
        wandb.run.summary["total_components"] = len(all_results.get('component_order', []))
        wandb.run.summary["total_trials"] = total_trials
        wandb.run.summary["early_stopped"] = all_results.get('early_stopped', False)