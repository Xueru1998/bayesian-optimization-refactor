import time
import optuna
import wandb
import logging
from typing import Dict, Any, Tuple

from pipeline.utils import Utils
from pipeline.logging.wandb import WandBLogger

logger = logging.getLogger(__name__)


class OptimizerCore:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def optimize(self) -> Dict[str, Any]:
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
                if early_stopping.should_stop:
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
    
    def _calculate_total_search_space(self) -> Tuple[int, str]:
        total_combinations = 1
        combination_note = ""
        
        components = [
            'query_expansion', 'retrieval', 'passage_filter', 
            'passage_reranker', 'passage_compressor', 'prompt_maker_generator'
        ]
        
        has_active_qe = False
        if self.optimizer.config_generator.node_exists("query_expansion"):
            qe_config = self.optimizer.config_generator.extract_node_config("query_expansion")
            if qe_config and qe_config.get("modules", []):
                for module in qe_config.get("modules", []):
                    if module.get("module_type") != "pass_query_expansion":
                        has_active_qe = True
                        break
        
        for component in components:
            if component == 'retrieval' and has_active_qe:
                continue
            
            combos, note = self.optimizer.combination_calculator.calculate_component_combinations(
                component, 
                search_space=self.optimizer.search_space
            )
            
            if combos > 0:
                total_combinations *= combos
                combination_note = note
        
        return total_combinations, combination_note
    
    def _initialize_wandb(self):
        search_space_size, combination_note = self._calculate_total_search_space()
        
        search_space_filtered = {}
        for param_name, param_value in self.optimizer.search_space.items():
            if isinstance(param_value, list):
                search_space_filtered[param_name] = param_value
            elif isinstance(param_value, tuple) and len(param_value) == 2:
                search_space_filtered[param_name] = f"[{param_value[0]}, {param_value[1]}]"
            else:
                search_space_filtered[param_name] = str(param_value)
        
        wandb_config = {
            "search_type": f"bayesian_optimization_{self.optimizer.optimizer}",
            "optimizer": f"BO-{self.optimizer.optimizer.upper()}",
            "n_trials": self.optimizer.n_trials,
            "retrieval_weight": self.optimizer.retrieval_weight,
            "generation_weight": self.optimizer.generation_weight,
            "search_space": search_space_filtered,
            "search_space_size": search_space_size,
            "search_space_note": combination_note,
            "study_name": self.optimizer.study_name,
            "evaluation_method": "RAGAS" if self.optimizer.use_ragas else "Traditional",
            "ragas_enabled": self.optimizer.use_ragas,
            "ragas_llm_model": self.optimizer.ragas_llm_model if self.optimizer.use_ragas else None,
            "ragas_embedding_model": self.optimizer.ragas_embedding_model if self.optimizer.use_ragas else None,
            "ragas_metrics": self.optimizer.ragas_metrics if self.optimizer.use_ragas else None
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
                constant_liar=True
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
            
        else:
            from optuna.samplers import TPESampler
            sampler = TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42,
                multivariate=True,
                constant_liar=True
            )
            print(f"Unknown sampler type '{self.optimizer.optimizer}', using TPE as default")
        
        return sampler
    
    def _log_wandb_results(self, study):
        print("\nGenerating Optuna visualization plots for W&B...")
        
        if self.optimizer.use_ragas:
            WandBLogger.log_ragas_comparison_plot(self.optimizer.all_trials, prefix="bo")
            WandBLogger.log_ragas_summary_table(self.optimizer.all_trials, prefix="bo")

        WandBLogger.log_optimization_plots(study, self.optimizer.all_trials, None, prefix="bo_plots")
        WandBLogger.log_final_tables(self.optimizer.all_trials, None, prefix="final")
        WandBLogger.log_summary(study)
    
    def _prepare_results(self, best_config, total_time, early_stopping):
        pareto_front = Utils.find_pareto_front(self.optimizer.all_trials)
        
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
            "early_stopped": early_stopping.should_stop,
            "optimizer": self.optimizer.optimizer,
            "total_trials": len(self.optimizer.all_trials),
            "all_trials": self.optimizer.all_trials
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
        print("\n===== Optimization Results =====")
        print(f"Total optimization time: {time_str}")
        print(f"Total trials: {len(self.optimizer.all_trials)}")
        print(f"Sampler used: {self.optimizer.optimizer.upper()}")
        
        print(f"Early stopped trials: {len(self.optimizer.early_stopped_trials)}")

        if self.optimizer.early_stopped_trials:
            print("\nEarly stopping summary:")
            component_counts = {}
            for trial in self.optimizer.early_stopped_trials:
                component = trial.get('stopped_at', 'unknown')
                component_counts[component] = component_counts.get(component, 0) + 1
            
            for component, count in component_counts.items():
                print(f"  {component}: {count} trials stopped")
        
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
        wandb.summary["optimization_time"] = results['optimization_time']
        wandb.summary["early_stopped"] = results['early_stopped']
        wandb.summary["optimizer"] = self.optimizer.optimizer
        wandb.summary["evaluation_method"] = "RAGAS" if self.optimizer.use_ragas else "Traditional"
        
        if best_config and isinstance(best_config, dict) and 'config' in best_config:
            for key, value in best_config['config'].items():
                wandb.summary[f"best_config_{key}"] = value