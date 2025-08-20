import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import wandb
from typing import Dict, Any, List, Optional, Union
import optuna
import optuna.visualization as vis
from .utils import WandBUtils
from .tables import TableMixin


class VisualizationMixin:
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
                VisualizationMixin._log_optuna_plots(study, temp_dir, prefix, plots_logged, None)
            else:
                VisualizationMixin._log_smac_plots(all_trials, pareto_front, temp_dir, prefix)
            
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
        
        VisualizationMixin.log_component_comparison_plot(study, prefix)
        VisualizationMixin._save_html_plots(study, plots_logged, temp_dir, prefix)
        
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
            TableMixin.log_study_tables(study, trials_data, prefix=prefix)
    
    @staticmethod
    def _log_smac_plots(all_trials: List[Dict[str, Any]], pareto_front: List[Dict[str, Any]], 
                    temp_dir: str, prefix: str):
        from .tables import TableMixin
        
        max_trial_number = max([t.get('trial_number', 0) for t in all_trials]) if all_trials else 0

        VisualizationMixin._plot_optimization_history(all_trials, temp_dir, prefix)
        VisualizationMixin._plot_component_scores(all_trials, temp_dir, prefix)  
        VisualizationMixin._plot_pareto_front(all_trials, pareto_front, temp_dir, prefix)
        
        has_ragas = any('ragas_mean_score' in t for t in all_trials)
        if has_ragas:
            VisualizationMixin.log_ragas_comparison_plot(all_trials, prefix)
        
        TableMixin._log_study_tables(all_trials, pareto_front, prefix=prefix, step=None)
    
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
    def log_component_comparison_plot(study_or_trials: Union[optuna.Study, List[Dict[str, Any]]], 
                                    prefix: str = "optuna"):
        if wandb.run is None:
            return
        
        active_components = WandBUtils.detect_active_components(study_or_trials)
        
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