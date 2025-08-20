import wandb
import optuna
from typing import Dict, Any, List, Optional, Union
from .utils import WandBUtils
from .visualization import VisualizationMixin
from .tables import TableMixin
from .metrics import MetricsMixin


class WandBLogger(VisualizationMixin, TableMixin, MetricsMixin):
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
        return WandBUtils.normalize_value(value)
    
    @staticmethod
    def _get_config_id(config: Dict[str, Any]) -> str:
        return WandBUtils.get_config_id(config)
    
    @staticmethod
    def _format_config_for_table(config: Dict[str, Any], component: str = None) -> str:
        return WandBUtils.format_config_for_table(config, component)
    
    @staticmethod
    def _group_trials_by_config(trials_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        return WandBUtils.group_trials_by_config(trials_data)
    
    @staticmethod
    def _get_best_trials_by_config(trials_data: List[Dict[str, Any]], full_budget_only: bool = False) -> List[Dict[str, Any]]:
        return WandBUtils.get_best_trials_by_config(trials_data, full_budget_only)
    
    @staticmethod
    def detect_active_components(study_or_trials: Union[Any, List[Dict[str, Any]]]) -> Dict[str, bool]:
        return WandBUtils.detect_active_components(study_or_trials)
    
    @staticmethod
    def log_summary(study_or_results: Union[Any, Dict[str, Any]], 
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
                unique_configs = len(set(WandBUtils.get_config_id(t.get('config', {})) for t in all_trials))
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
    def log_custom_metrics(metrics: Dict[str, Any]):
        if wandb.run is None:
            return
        
        wandb.log(metrics)

    @staticmethod
    def log_component_trial(component: str, trial_number: int, config: Dict[str, Any], 
                        score: float, latency: float, step: Optional[int] = None):
        if wandb.run is None:
            return

        complete_config = WandBUtils.format_config_for_table(config, component)
        
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