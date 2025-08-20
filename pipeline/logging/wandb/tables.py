import numpy as np
import wandb
from typing import Dict, Any, List, Optional, Union
import optuna
from .utils import WandBUtils


class TableMixin:
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
                normalized_value = WandBUtils.normalize_value(value)
                normalized_row.append(normalized_value)
            normalized_data.append(normalized_row)
        
        return wandb.Table(columns=columns, data=normalized_data)
    
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
            params_table = TableMixin.create_parameters_table(trials_data)
            if params_table:
                wandb.log({f"{prefix}/parameters_table": params_table})

        TableMixin.log_component_metrics_table(
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
                'config_id': trial.get('config_id', WandBUtils.get_config_id(trial.get('config', {}))),
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
            formatted_trial['complete_config'] = WandBUtils.format_config_for_table(config)
            
            formatted_trials.append(formatted_trial)
        
        columns = ['trial_number', 'config_id', 'score', 'latency', 'execution_time_s', 'status', 'complete_config']
        if any('budget' in t for t in formatted_trials):
            columns.insert(6, 'budget')
            columns.insert(7, 'budget_percentage')
        if any('ragas_mean_score' in t for t in formatted_trials):
            columns.insert(3, 'ragas_mean_score')
        
        table_data = []
        for trial in formatted_trials:
            row = [WandBUtils.normalize_value(trial.get(col)) for col in columns]
            table_data.append(row)
        
        params_table = wandb.Table(columns=columns, data=table_data)
        wandb.log({f"{prefix}/all_trials_table": params_table})

        best_configs = WandBUtils.get_best_trials_by_config(all_trials, full_budget_only=True)
        
        best_configs_formatted = []
        for trial in best_configs:
            formatted_trial = {
                'trial_number': trial.get('trial_number', 0),
                'config_id': trial.get('config_id'),
                'score': trial.get('score', 0.0),
                'latency': trial.get('latency', 0.0),
                'num_evaluations': trial.get('num_evaluations', 1),
                'budget_progression': str(trial.get('budget_progression', [])),
                'complete_config': WandBUtils.format_config_for_table(trial.get('config', {}))
            }
            best_configs_formatted.append(formatted_trial)
        
        if best_configs_formatted:
            columns = ['config_id', 'score', 'latency', 'num_evaluations', 'budget_progression', 'complete_config']
            table_data = []
            for trial in best_configs_formatted:
                row = [WandBUtils.normalize_value(trial.get(col)) for col in columns]
                table_data.append(row)
            
            best_configs_table = wandb.Table(columns=columns, data=table_data)
            wandb.log({f"{prefix}/best_configs_table": best_configs_table})

        TableMixin.log_component_metrics_table(
            all_trials,
            table_name=f"{prefix}/component_metrics",
            step=None
        )
        
        has_ragas = any('ragas_mean_score' in t for t in all_trials)
        if has_ragas:
            TableMixin.log_ragas_summary_table(all_trials, prefix=prefix)
    
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

                components = ["query_expansion", "retrieval", "reranker", "filter", 
                            "compression", "prompt_maker", "generation"]
                
                for comp in components:
                    score_key = f"{comp}_score"
                    if score_key in trial.user_attrs:
                        score_value = trial.user_attrs.get(score_key, 0.0)
                        if isinstance(score_value, str):
                            try:
                                row[score_key] = float(score_value)
                            except:
                                row[score_key] = 0.0
                        else:
                            row[score_key] = float(score_value) if score_value is not None else 0.0

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
                    score_value = trial.user_attrs['last_retrieval_score']
                    if isinstance(score_value, str):
                        try:
                            row['last_retrieval_score'] = float(score_value)
                        except:
                            row['last_retrieval_score'] = 0.0
                    else:
                        row['last_retrieval_score'] = float(score_value) if score_value is not None else 0.0
                
                data.append(row)
        else:
            for trial in study_or_trials:
                row = {
                    "trial": trial.get("trial_number", 0),
                    "config_id": trial.get("config_id", WandBUtils.get_config_id(trial.get("config", {}))),
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
                        score_value = trial[score_key]
                        if isinstance(score_value, str):
                            try:
                                row[score_key] = float(score_value)
                            except:
                                row[score_key] = 0.0
                        else:
                            row[score_key] = float(score_value) if score_value is not None else 0.0
                
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
                if '_score' in col and value is not None:
                    if isinstance(value, str):
                        try:
                            normalized_value = float(value)
                        except:
                            normalized_value = 0.0
                    else:
                        normalized_value = float(value) if value is not None else 0.0
                else:
                    normalized_value = WandBUtils.normalize_value(value)
                normalized_row.append(normalized_value)
            normalized_data.append(normalized_row)
        
        table = wandb.Table(columns=columns, data=normalized_data)
        
        wandb.log({table_name: table})
    
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
                trial_data[f"config_{clean_key}"] = WandBUtils.normalize_value(value)
            
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
                row.append(WandBUtils.normalize_value(value))
            table_data.append(row)
        
        table = wandb.Table(columns=columns, data=table_data)
        wandb.log({f"{prefix}/summary_table": table})
    
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
            row['complete_config'] = WandBUtils.format_config_for_table(config, component)

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
                normalized_row.append(WandBUtils.normalize_value(value))
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
            row['complete_config'] = WandBUtils.format_config_for_table(config, component)
            
            table_data.append(row)

        table_data.sort(key=lambda x: x['trial'])

        columns = ['trial', 'score', 'latency', 'status', 'complete_config']

        normalized_data = []
        for row in table_data:
            normalized_row = [WandBUtils.normalize_value(row.get(col)) for col in columns]
            normalized_data.append(normalized_row)
        
        table = wandb.Table(columns=columns, data=normalized_data)
        wandb.log({f"{component}/trials_progress": table})
    
    @staticmethod
    def log_final_tables(all_trials, pareto_front, prefix="final"):
        if wandb.run is None:
            return
        
        TableMixin._log_study_tables(all_trials, pareto_front, prefix=prefix, step=None)
    
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

            TableMixin.log_component_optimization_table(
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
        
        last_retrieval_score = 0.0
        last_retrieval_component = None
        
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