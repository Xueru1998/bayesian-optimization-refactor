import time
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import wandb
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband, SuccessiveHalving
from smac.initial_design import SobolInitialDesign
from smac.callback import Callback
from pipeline.logging.wandb import WandBLogger
from pipeline.utils import Utils


class OptimizationCore:
    
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.use_wandb:
            self._initialize_wandb()
        
        cs = self.config_space_builder.build_configuration_space()
        early_stopping_callback = self._create_early_stopping_callback()
        
        scenario = self._create_scenario(cs)
        initial_design = self._create_initial_design(scenario)
        
        target_function = (self.target_function_multifidelity if self.use_multi_fidelity 
                          else self.target_function_standard)
        
        smac = self._create_optimizer(scenario, target_function, initial_design, [early_stopping_callback])
        
        self._print_optimization_start_info()
        
        incumbents = self._run_optimization(smac, early_stopping_callback)
        pareto_front = self._extract_pareto_front(smac, incumbents)
        
        optimization_results = self._create_optimization_results(
            pareto_front, early_stopping_callback.should_stop, incumbents, time.time() - start_time
        )
        
        self._save_and_log_results(optimization_results, pareto_front)
        self._print_optimization_summary(optimization_results, pareto_front)
        
        return optimization_results
    
    def _initialize_wandb(self):
        WandBLogger.reset_step_counter()
        
        cs = self.config_space_builder.build_configuration_space()
        search_space_info = self.config_space_builder.get_search_space_info()
        
        wandb_config = {
            "optimizer": f"{self.optimizer.upper()}{' Multi-Fidelity' if self.use_multi_fidelity else ''}",
            "n_trials": self.n_trials,
            "retrieval_weight": self.retrieval_weight,
            "generation_weight": self.generation_weight,
            "search_space_size": search_space_info['n_hyperparameters'],
            "study_name": self.study_name,
            "early_stopping_threshold": self.early_stopping_threshold,
            "component_early_stopping_enabled": self.component_early_stopping_enabled,
            "component_early_stopping_thresholds": self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None,
            "n_workers": self.n_workers,
            "walltime_limit": self.walltime_limit if self.walltime_limit is not None else "No limit",
            "use_multi_fidelity": self.use_multi_fidelity,
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
            "eta": self.eta if self.use_multi_fidelity else None
        }
        
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.wandb_run_name or self.study_name,
            config=wandb_config,
            reinit=True
        )
    
    def _create_early_stopping_callback(self):
        class EarlyStoppingCallback(Callback):
            def __init__(self, threshold: float):
                super().__init__()
                self.threshold = threshold
                self.should_stop = False
                
            def on_tell(self, smbo, info, value):
                if info and value:
                    if hasattr(value, 'cost') and isinstance(value.cost, list):
                        score = -value.cost[0]
                    else:
                        score = 0
                    
                    if score >= self.threshold:
                        print(f"\n*** Early stopping triggered! Score {score:.4f} >= {self.threshold} ***")
                        self.should_stop = True
                        smbo._stop = True
        
        return EarlyStoppingCallback(self.early_stopping_threshold)
    
    def _create_scenario(self, cs) -> Scenario:
        base_params = {
            'configspace': cs,
            'deterministic': True,
            'n_trials': self.n_trials,
            'n_workers': self.n_workers,
            'seed': self.seed,
            'objectives': ["score", "latency"],
            'output_directory': self.result_dir,
            'name': self.study_name
        }
        
        if self.walltime_limit is not None:
            base_params['walltime_limit'] = self.walltime_limit
        
        if self.use_multi_fidelity:
            base_params['min_budget'] = self.min_budget
            base_params['max_budget'] = self.max_budget
        
        return Scenario(**base_params)
    
    def _create_initial_design(self, scenario: Scenario) -> SobolInitialDesign:
        n_init = min(self.n_trials // 4, 10)
        n_init = max(n_init, 2)
        
        print(f"Using {n_init} initial random configurations")
        
        return SobolInitialDesign(
            scenario=scenario,
            n_configs=n_init,
            max_ratio=1.0,
            additional_configs=[]
        )
    
    def _create_optimizer(self, scenario, target_function, initial_design, callbacks):
        if self.use_multi_fidelity:
            return self._create_multi_fidelity_optimizer(scenario, target_function, initial_design, callbacks)
        else:
            return self._create_standard_optimizer(scenario, target_function, initial_design, callbacks)
    
    def _create_multi_fidelity_optimizer(self, scenario, target_function, initial_design, callbacks):
        if self.optimizer == "bohb":
            intensifier = Hyperband(
                scenario=scenario,
                incumbent_selection="highest_budget",
                eta=self.eta
            )
            print(f"Using BOHB (Bayesian Optimization Hyperband) with eta={self.eta}")
        else:
            intensifier = SuccessiveHalving(
                scenario=scenario,
                incumbent_selection="highest_budget",
                eta=self.eta
            )
            print(f"Using SMAC3 with Successive Halving, eta={self.eta}")
        
        return MultiFidelityFacade(
            scenario=scenario,
            target_function=target_function,
            intensifier=intensifier,
            callbacks=callbacks,
            initial_design=initial_design,
            overwrite=True
        )
    
    def _create_standard_optimizer(self, scenario, target_function, initial_design, callbacks):
        return HPOFacade(
            scenario=scenario,
            target_function=target_function,
            multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
                scenario,
                objective_weights=[self.generation_weight, self.retrieval_weight]
            ),
            callbacks=callbacks,
            initial_design=initial_design,
            overwrite=True
        )
    
    def _print_optimization_start_info(self):
        if self.use_multi_fidelity:
            print(f"\nStarting {self.optimizer.upper()} multi-fidelity optimization")
            print(f"Budget range: {self.min_budget} to {self.max_budget} samples")
            print(f"Budget percentage: {self.min_budget_percentage:.1%} to {self.max_budget_percentage:.1%}")
        else:
            print(f"\nStarting standard SMAC3 optimization (no multi-fidelity)")
        
        print(f"Total trials: {self.n_trials}")
        print(f"Objectives: score (weight={self.generation_weight}), latency (weight={self.retrieval_weight})")
        print(f"Early stopping threshold: {self.early_stopping_threshold}")
    
    def _run_optimization(self, smac, early_stopping_callback):
        try:
            incumbents = smac.optimize()
        except Exception as e:
            print(f"Optimization stopped: {e}")
            incumbents = self._extract_incumbents_from_smac(smac)
        
        return incumbents if isinstance(incumbents, list) else ([incumbents] if incumbents else [])
    
    def _extract_incumbents_from_smac(self, smac):
        incumbents = []
        try:
            if hasattr(smac, 'intensifier') and hasattr(smac.intensifier, 'get_incumbents'):
                incumbents = smac.intensifier.get_incumbents()
            elif hasattr(smac, 'get_incumbents'):
                incumbents = smac.get_incumbents()
            elif hasattr(smac, 'runhistory'):
                incumbents = self._extract_from_runhistory(smac)
        except Exception as e:
            print(f"Could not retrieve incumbents: {e}")
        return incumbents
    
    def _extract_from_runhistory(self, smac):
        incumbents = []
        if hasattr(smac.runhistory, 'get_incumbents'):
            incumbents = smac.runhistory.get_incumbents()
        else:
            configs = smac.runhistory.get_configs()
            if configs:
                for config in configs[:20]:
                    try:
                        cost = smac.runhistory.get_cost(config)
                        if cost is not None:
                            incumbents.append(config)
                    except:
                        continue
        return incumbents
    
    def _extract_pareto_front(self, smac, incumbents):
        pareto_front = []
        for incumbent in incumbents:
            try:
                cost = self._get_incumbent_cost(smac, incumbent)
                if cost is None:
                    continue
                
                score_val, latency_val = self._extract_cost_values(cost)
                pareto_solution = self._create_pareto_solution(incumbent, score_val, latency_val)
                pareto_front.append(pareto_solution)
                
            except Exception as e:
                print(f"Error processing incumbent: {e}")
                continue
        
        self._update_pareto_front_trial_numbers(pareto_front)
        return pareto_front
    
    def _get_incumbent_cost(self, smac, incumbent):
        if hasattr(smac, 'validate'):
            return smac.validate(incumbent)
        elif hasattr(smac, 'runhistory'):
            try:
                incumbent_cost = smac.runhistory.get_cost(incumbent)
                if isinstance(incumbent_cost, list):
                    return {"score": incumbent_cost[0], "latency": incumbent_cost[1]}
                else:
                    return {"score": incumbent_cost, "latency": 0.0}
            except:
                return None
        return None
    
    def _extract_cost_values(self, cost):
        if isinstance(cost, dict):
            score_val = cost.get("score", 0.0)
            latency_val = cost.get("latency", 0.0)
        elif isinstance(cost, (list, np.ndarray)):
            if len(cost) >= 2:
                score_val = float(cost[0])
                latency_val = float(cost[1])
            else:
                score_val = float(cost[0]) if len(cost) > 0 else 0.0
                latency_val = 0.0
        else:
            score_val = float(cost)
            latency_val = 0.0
        
        if hasattr(score_val, '__len__') and not isinstance(score_val, str):
            score_val = float(score_val[0] if len(score_val) > 0 else 0.0)
        if hasattr(latency_val, '__len__') and not isinstance(latency_val, str):
            latency_val = float(latency_val[0] if len(latency_val) > 0 else 0.0)
        
        return score_val, latency_val
    
    def _create_pareto_solution(self, incumbent, score_val, latency_val):
        config_dict = dict(incumbent)
        pareto_solution = {
            'config': config_dict,
            'score': -float(score_val) if score_val else 0.0,
            'latency': float(latency_val) if latency_val else 0.0,
            'trial_number': None
        }
        
        for trial in self.all_trials:
            if trial['config'] == config_dict:
                pareto_solution.update({
                    'score': trial['score'],
                    'latency': trial['latency'],
                    'trial_number': trial['trial_number'],
                    'budget': trial.get('budget', self.max_budget),
                    'budget_percentage': trial.get('budget_percentage', 1.0)
                })
                break
        
        return pareto_solution
    
    def _update_pareto_front_trial_numbers(self, pareto_front):
        for pf_solution in pareto_front:
            for trial in self.all_trials:
                if trial['config'] == pf_solution['config']:
                    pf_solution['trial_number'] = trial['trial_number']
                    break
    
    def _create_optimization_results(self, pareto_front, early_stopped, incumbents, total_time):
        best_configs = self._find_best_configurations(pareto_front)
        
        valid_trials = [t for t in self.all_trials if not t.get('early_stopped', False)]
        
        results = {
            'optimizer': self.optimizer,
            'use_multi_fidelity': bool(self.use_multi_fidelity),
            'min_budget': int(self.min_budget),
            'max_budget': int(self.max_budget),
            'best_config': self._convert_numpy_types(best_configs['best_balanced']),
            'best_score_config': self._convert_numpy_types(best_configs['best_score']['config']),
            'best_score': float(best_configs['best_score']['score']),
            'best_score_latency': float(best_configs['best_score']['latency']),
            'best_latency_config': self._convert_numpy_types(best_configs['best_latency']['config']),
            'best_latency': float(best_configs['best_latency']['latency']),
            'best_latency_score': float(best_configs['best_latency']['score']),
            'pareto_front': [self._convert_numpy_types(pf) for pf in pareto_front],
            'optimization_time': float(total_time),
            'n_trials': int(self.trial_counter),
            'total_trials': int(self.trial_counter),
            'early_stopped_trials': int(self.early_stopped_trials_count),
            'completed_trials': int(self.trial_counter - self.early_stopped_trials_count),
            'early_stopped': bool(early_stopped),
            'incumbents': [self._convert_numpy_types(dict(inc)) for inc in incumbents],
            'all_trials': [self._convert_numpy_types(trial) for trial in self.all_trials],
            'component_early_stopping_enabled': self.component_early_stopping_enabled,
            'component_early_stopping_thresholds': self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None
        }
        
        return results
    
    def _find_best_configurations(self, pareto_front):
        valid_trials = [t for t in self.all_trials if not t.get('early_stopped', False)]
        
        if not valid_trials:
            default_trial = {'config': {}, 'score': 0.0, 'latency': float('inf')}
            return {
                'best_score': default_trial,
                'best_latency': default_trial,
                'best_balanced': default_trial
            }
        
        if self.use_multi_fidelity:
            return self._find_best_multifidelity_configs_with_early_stopping(pareto_front, valid_trials)
        else:
            return self._find_best_standard_configs_with_early_stopping(pareto_front, valid_trials)
    
    def _find_best_multifidelity_configs_with_early_stopping(self, pareto_front, valid_trials):
        full_budget_trials = [
            t for t in valid_trials 
            if t.get('budget_percentage', 1.0) >= 0.99
        ]
        
        if full_budget_trials:
            best_score_trial = max(full_budget_trials, key=lambda x: x['score'])
            best_latency_trial = min(full_budget_trials, key=lambda x: x['latency'])
            
            high_score_trials = [t for t in full_budget_trials if t['score'] > 0.9]
            if high_score_trials:
                best_balanced = min(high_score_trials, key=lambda x: x['latency'])
            else:
                best_balanced = max(full_budget_trials, key=lambda x: x['score'])
        else:
            best_score_trial = max(valid_trials, key=lambda x: x['score'])
            best_latency_trial = min(valid_trials, key=lambda x: x['latency'])
            best_balanced = best_score_trial
        
        return {
            'best_score': best_score_trial,
            'best_latency': best_latency_trial,
            'best_balanced': best_balanced
        }

    def _find_best_standard_configs_with_early_stopping(self, pareto_front, valid_trials):
        best_score_trial = max(valid_trials, key=lambda x: x['score'])
        best_latency_trial = min(valid_trials, key=lambda x: x['latency'])
        
        high_score_trials = [t for t in valid_trials if t['score'] > 0.9]
        if high_score_trials:
            best_balanced = min(high_score_trials, key=lambda x: x['latency'])
        else:
            valid_pareto = [p for p in pareto_front if not any(t.get('early_stopped', False) for t in self.all_trials if t['config'] == p['config'])]
            best_balanced = max(valid_pareto, key=lambda x: x['score']) if valid_pareto else best_score_trial
        
        return {
            'best_score': best_score_trial,
            'best_latency': best_latency_trial,
            'best_balanced': best_balanced
        }
    
    def _save_and_log_results(self, optimization_results, pareto_front):
        Utils.save_results_to_json(self.result_dir, "optimization_summary.json", optimization_results)
        
        if self.all_trials:
            converted_trials = self._convert_trials_for_csv(self.all_trials)
            Utils.save_results_to_csv(self.result_dir, "all_trials.csv", converted_trials)
        
        if self.use_wandb:
            WandBLogger.log_optimization_plots(None, self.all_trials, pareto_front, prefix=self.optimizer)
            WandBLogger.log_final_tables(self.all_trials, pareto_front, prefix="final")
            WandBLogger.log_summary(optimization_results)
            
            wandb.run.summary["tables_logged"] = True
            
            wandb.finish()
    
    def _convert_trials_for_csv(self, trials):
        converted_trials = []
        for trial in trials:
            converted_trial = {}
            for k, v in trial.items():
                if isinstance(v, np.integer):
                    converted_trial[k] = int(v)
                elif isinstance(v, np.floating):
                    converted_trial[k] = float(v)
                elif isinstance(v, np.ndarray):
                    converted_trial[k] = v.tolist()
                elif isinstance(v, dict):
                    converted_trial[k] = str(v)
                else:
                    converted_trial[k] = v
            converted_trials.append(converted_trial)
        return converted_trials
    
    def _print_optimization_summary(self, optimization_results, pareto_front):
        time_str = Utils.format_time_duration(optimization_results['optimization_time'])
        
        print(f"\n{'='*60}")
        print(f"{self.optimizer.upper()} Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total optimization time: {time_str}")
        print(f"Total trials: {self.trial_counter}")
        print(f"Early stopped trials: {self.early_stopped_trials_count}")
        print(f"Completed trials: {self.trial_counter - self.early_stopped_trials_count}")
        
        if self.use_multi_fidelity and self.all_trials:
            self._print_multifidelity_summary()
        
        if optimization_results['early_stopped']:
            print("⚡ Optimization stopped early due to achieving target score!")
        
        self._print_best_config_summary(optimization_results['best_config'])
        self._print_pareto_front_summary(pareto_front)
        
        print(f"\nResults saved to: {self.result_dir}")
    
    def _print_multifidelity_summary(self):
        full_budget_count = len([
            t for t in self.all_trials 
            if t.get('budget_percentage', 1.0) >= 0.99
        ])
        unique_configs = len(set(
            self._get_config_hash(t.get('config', {})) 
            for t in self.all_trials
        ))
        print(f"Unique configurations tested: {unique_configs}")
        print(f"Fully evaluated configurations: {full_budget_count}")
    
    def _print_best_config_summary(self, best_config):
        if best_config and isinstance(best_config, dict):
            print("\nBest balanced configuration (high score with low latency):")
            print(f"  Score: {best_config.get('score', 'N/A')}")
            print(f"  Latency: {best_config.get('latency', 'N/A')}")
            if 'budget' in best_config:
                print(f"  Budget: {best_config['budget']} samples")
            if 'config' in best_config:
                print(f"  Config: {best_config['config']}")
    
    def _print_pareto_front_summary(self, pareto_front):
        print(f"\nPareto front contains {len(pareto_front)} solutions")
        if pareto_front:
            print("Top 5 Pareto optimal solutions:")
            for i, solution in enumerate(sorted(pareto_front, key=lambda x: -x['score'])[:5]):
                budget_str = f", Budget: {solution.get('budget', 'N/A')}" if 'budget' in solution else ""
                print(f"  {i+1}. Score: {solution['score']:.4f}, Latency: {solution['latency']:.2f}s{budget_str}")
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        config_str = json.dumps(dict(sorted(config.items())), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]