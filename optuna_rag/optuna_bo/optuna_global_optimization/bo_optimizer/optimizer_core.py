import time
import optuna
import wandb
import logging
from typing import Dict, Any, List

from pipeline.utils import Utils
from pipeline.logging.wandb import WandBLogger

logger = logging.getLogger(__name__)


class OptimizerCore:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def optimize(self) -> Dict[str, Any]:
        if self.optimizer.optimizer == "grid":
            return self._run_grid_search()
        else:
            return self._run_bayesian_optimization()

    def _run_grid_search(self) -> Dict[str, Any]:
        from global_grid_search import GlobalGridSearchOptimizer
        
        grid_optimizer = GlobalGridSearchOptimizer(
            config_path=self.optimizer.config_path,
            qa_df=self.optimizer.qa_df,
            corpus_df=self.optimizer.corpus_df,
            project_dir=self.optimizer.project_dir,
            sample_percentage=self.optimizer.sample_percentage,
            cpu_per_trial=self.optimizer.cpu_per_trial,
            retrieval_weight=self.optimizer.retrieval_weight,
            generation_weight=self.optimizer.generation_weight,
            use_cached_embeddings=self.optimizer.use_cached_embeddings,
            result_dir=self.optimizer.result_dir,
            study_name=self.optimizer.study_name,
            use_wandb=self.optimizer.use_wandb,
            wandb_project=self.optimizer.wandb_project,
            wandb_entity=self.optimizer.wandb_entity,
            wandb_run_name=self.optimizer.wandb_run_name,
            use_ragas=self.optimizer.use_ragas,
            ragas_llm_model=self.optimizer.ragas_llm_model,
            ragas_embedding_model=self.optimizer.ragas_embedding_model,
            ragas_metrics=self.optimizer.ragas_metrics,
            use_llm_compressor_evaluator=self.optimizer.use_llm_compressor_evaluator,
            llm_evaluator_model=self.optimizer.llm_evaluator_model,
            max_trials=self.optimizer.n_trials
        )
        
        return grid_optimizer.optimize()

    def _run_bayesian_optimization(self) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.optimizer.use_wandb:
            self._initialize_wandb()
        
        early_stopping = self._create_early_stopping_callback()
        
        try:
            sampler = self._create_sampler()
            
            study = optuna.create_study(
                directions=["minimize", "minimize"],
                sampler=sampler,
                study_name=self.optimizer.study_name
            )
            
            callbacks = [early_stopping] if self.optimizer.optimizer != "random" else []
            
            try:
                study.optimize(
                    self.optimizer.objective,
                    n_trials=self.optimizer.n_trials,
                    callbacks=callbacks,
                    show_progress_bar=True
                )
            except:
                if self.optimizer.optimizer != "random" and early_stopping.should_stop:
                    print("Optimization stopped early due to achieving target score.")
                else:
                    raise
            
            end_time = time.time()
            total_time = end_time - start_time
            time_str = Utils.format_time_duration(total_time)
            
            self.optimizer.save_study_results(study)
            self.optimizer.results_manager.generate_plots(study)
            
            if self.optimizer.use_wandb and wandb.run is not None:
                self._log_wandb_results(study)
            
            best_config = self.optimizer.results_manager.find_best_config(study)
            
            results = self._prepare_results(best_config, total_time, early_stopping)
            
            self._print_results_summary(time_str, results)
            
            Utils.save_results_to_json(self.optimizer.result_dir, "optimization_summary.json", results)
            
            if self.optimizer.use_wandb:
                self._update_wandb_summary(results, best_config)
                wandb.finish()
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            if self.optimizer.use_wandb:
                wandb.finish(exit_code=1)
            raise
    
    def _initialize_wandb(self):
        search_space_filtered = {}
        for param_name, param_value in self.optimizer.search_space.items():
            if isinstance(param_value, list):
                search_space_filtered[param_name] = param_value
            elif isinstance(param_value, tuple) and len(param_value) == 2:
                search_space_filtered[param_name] = f"[{param_value[0]}, {param_value[1]}]"
            else:
                search_space_filtered[param_name] = str(param_value)
        
        optimizer_type = "GRID" if self.optimizer.optimizer == "grid" else f"BO-{self.optimizer.optimizer.upper()}"
        
        wandb_config = {
            "search_type": "grid_search" if self.optimizer.optimizer == "grid" else f"bayesian_optimization_{self.optimizer.optimizer}",
            "optimizer": optimizer_type,
            "n_trials": self.optimizer.n_trials,
            "retrieval_weight": self.optimizer.retrieval_weight,
            "generation_weight": self.optimizer.generation_weight,
            "search_space": search_space_filtered,
            "search_space_size": self.optimizer.search_space_calculator.calculate_total_combinations(),
            "study_name": self.optimizer.study_name,
            "evaluation_method": "RAGAS" if self.optimizer.use_ragas else "Traditional",
            "ragas_enabled": self.optimizer.use_ragas,
            "ragas_llm_model": self.optimizer.ragas_llm_model if self.optimizer.use_ragas else None,
            "ragas_embedding_model": self.optimizer.ragas_embedding_model if self.optimizer.use_ragas else None,
            "ragas_metrics": self.optimizer.ragas_metrics if self.optimizer.use_ragas else None,
            "component_early_stopping_enabled": self.optimizer.component_early_stopping_enabled if self.optimizer.optimizer != "grid" else False,
            "component_early_stopping_thresholds": self.optimizer.component_early_stopping_thresholds if self.optimizer.component_early_stopping_enabled and self.optimizer.optimizer != "grid" else None
        }
        
        wandb.init(
            project=self.optimizer.wandb_project,
            entity=self.optimizer.wandb_entity,
            name=self.optimizer.wandb_run_name,
            config=wandb_config,
            reinit=True
        )
    
    def _create_early_stopping_callback(self):
        class EarlyStoppingCallback:
            def __init__(self, score_threshold=0.9):
                self.score_threshold = score_threshold
                self.should_stop = False
            
            def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    score = -trial.values[0] if trial.values else 0.0
                    
                    if score > self.score_threshold:
                        print(f"\n*** Early stopping triggered! ***")
                        print(f"Trial {trial.number} achieved score {score:.4f} > {self.score_threshold}")
                        self.should_stop = True
                        study.stop()

        threshold = getattr(self.optimizer, 'early_stopping_threshold', getattr(self.optimizer, '_temp_early_stopping', 0.9))
        return EarlyStoppingCallback(score_threshold=threshold)
    
    def _create_sampler(self):
        if self.optimizer.optimizer == "tpe":
            from optuna.samplers import TPESampler
            sampler = TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42,
                multivariate=True,
                constant_liar=True,
                warn_independent_sampling=False
            )
            print("Using TPE (Tree-structured Parzen Estimator) sampler")
            
        elif self.optimizer.optimizer == "botorch":
            from optuna.integration import BoTorchSampler
            sampler = BoTorchSampler(
                n_startup_trials=10,
                seed=42
            )
            print("Using BoTorch (Gaussian Process-based) sampler")
                            
        elif self.optimizer.optimizer == "random":
            from optuna.samplers import RandomSampler
            sampler = RandomSampler(seed=42)
            print("Using Random sampler")
            
        elif self.optimizer.optimizer == "grid":
            from optuna.samplers import GridSampler
            grid_values = self._get_grid_values_from_search_space()
            sampler = GridSampler(search_space=grid_values)
            print("Using Grid sampler")
            
        else:
            from optuna.samplers import TPESampler
            sampler = TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42,
                multivariate=True,
                constant_liar=True,
                warn_independent_sampling=False
            )
            print(f"Unknown sampler type '{self.optimizer.optimizer}', using TPE as default")
        
        return sampler
    
    def _get_grid_values_from_search_space(self) -> Dict[str, List[Any]]:
        grid_values = {}
        
        for param_name, param_spec in self.optimizer.search_space.items():
            if isinstance(param_spec, list):
                grid_values[param_name] = param_spec
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                min_val, max_val = param_spec
                if isinstance(min_val, int):
                    step = max(1, (max_val - min_val) // 5)
                    grid_values[param_name] = list(range(min_val, max_val + 1, step))
                else:
                    grid_values[param_name] = [min_val, (min_val + max_val) / 2, max_val]
            else:
                grid_values[param_name] = [param_spec]
        
        return grid_values
    
    def _log_wandb_results(self, study):
        print("\nGenerating Optuna visualization plots for W&B...")
        
        if self.optimizer.use_ragas:
            WandBLogger.log_ragas_comparison_plot(self.optimizer.all_trials, prefix="bo")
            WandBLogger.log_ragas_summary_table(self.optimizer.all_trials, prefix="bo")

        WandBLogger.log_optimization_plots(study, self.optimizer.all_trials, None, prefix="bo_plots")
        WandBLogger.log_final_tables(self.optimizer.all_trials, None, prefix="final")
        WandBLogger.log_summary(study)
    
    def _prepare_results(self, best_config, total_time, early_stopping):
        early_stopped_count = sum(1 for t in self.optimizer.all_trials if t.get('early_stopped', False))
        valid_trials = [t for t in self.optimizer.all_trials if not t.get('early_stopped', False)]
        pareto_front = Utils.find_pareto_front(valid_trials)
        
        results = {
            "best_config": best_config,
            "best_score_config": self.optimizer.best_score["config"],
            "best_score": self.optimizer.best_score["value"],
            "best_score_latency": self.optimizer.best_score["latency"],
            "best_latency_config": self.optimizer.best_latency["config"],
            "best_latency": self.optimizer.best_latency["value"],
            "best_latency_score": self.optimizer.best_latency["score"],
            "pareto_front": pareto_front,
            "optimization_time": total_time,
            "n_trials": len(self.optimizer.all_trials),
            "early_stopped_trials": early_stopped_count,
            "completed_trials": len(self.optimizer.all_trials) - early_stopped_count,
            "early_stopped": self.optimizer.optimizer not in ["random", "grid"] and early_stopping.should_stop,
            "optimizer": self.optimizer.optimizer,
            "total_trials": len(self.optimizer.all_trials),
            "all_trials": self.optimizer.all_trials,
            "component_early_stopping_enabled": self.optimizer.component_early_stopping_enabled if self.optimizer.optimizer != "grid" else False,
            "component_early_stopping_thresholds": self.optimizer.component_early_stopping_thresholds if self.optimizer.component_early_stopping_enabled and self.optimizer.optimizer != "grid" else None
        }

        if best_config:
            results["best_config"] = {
                "config": best_config.get("config", {}),
                "score": best_config.get("score", 0.0),
                "latency": best_config.get("latency", float('inf')),
                "trial_number": best_config.get("trial_number", -1)
            }
        
        return results
    
    def _print_results_summary(self, time_str, results):
        early_stopped_count = results['early_stopped_trials']
        
        print("\n===== Optimization Results =====")
        print(f"Total optimization time: {time_str}")
        print(f"Total trials: {len(self.optimizer.all_trials)}")
        print(f"Early stopped trials: {early_stopped_count}")
        print(f"Completed trials: {len(self.optimizer.all_trials) - early_stopped_count}")
        print(f"Sampler used: {self.optimizer.optimizer.upper()}")
        
        if results.get('best_config'):
            best_config = results['best_config']
            print("\nBest configuration (considering score > 0.9 with minimum latency):")
            print(f"  Trial: {best_config.get('trial_number', 'N/A')}")
            print(f"  Score: {best_config.get('score', 0.0):.4f}")
            print(f"  Latency: {best_config.get('latency', float('inf')):.2f}s")
            print(f"  Config: {best_config.get('config', {})}")
        
        print("\nBest trial by score only:")
        print(f"  Score: {self.optimizer.best_score['value']:.4f}")
        print(f"  Latency: {self.optimizer.best_score['latency']:.2f}s")
        print(f"  Config: {self.optimizer.best_score['config']}")
        
        print("\nBest trial by latency only:")
        print(f"  Score: {self.optimizer.best_latency['score']:.4f}")
        print(f"  Latency: {self.optimizer.best_latency['value']:.2f}s")
        print(f"  Config: {self.optimizer.best_latency['config']}")
        
        pareto_front = results['pareto_front']
        print(f"\nPareto optimal solutions: {len(pareto_front)}")
        for i, trial in enumerate(pareto_front[:5]):
            print(f"  Solution {i+1}: Score={trial['score']:.4f}, Latency={trial['latency']:.2f}s (Trial #{trial['trial_number']})")
            print(f"    Config: {trial['config']}")
    
    def _update_wandb_summary(self, results, best_config):
        wandb.summary["best_score"] = self.optimizer.best_score["value"]
        wandb.summary["best_latency"] = self.optimizer.best_latency["value"]
        wandb.summary["total_trials"] = len(self.optimizer.all_trials)
        wandb.summary["early_stopped_trials"] = results['early_stopped_trials']
        wandb.summary["completed_trials"] = results['completed_trials']
        wandb.summary["optimization_time"] = results['optimization_time']
        wandb.summary["early_stopped"] = results['early_stopped']
        wandb.summary["optimizer"] = self.optimizer.optimizer
        wandb.summary["evaluation_method"] = "RAGAS" if self.optimizer.use_ragas else "Traditional"
        
        if best_config and isinstance(best_config, dict) and 'config' in best_config:
            for key, value in best_config['config'].items():
                wandb.summary[f"best_config_{key}"] = value