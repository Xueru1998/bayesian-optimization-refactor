import os
import itertools
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from ConfigSpace import ConfigurationSpace, Configuration

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband, SuccessiveHalving
from smac.initial_design import SobolInitialDesign
from smac.callback import Callback


class SMACRunner:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def ensure_smac_directories(self, component: str):
        component_dir = os.path.join(self.optimizer.result_dir, component)
        os.makedirs(component_dir, exist_ok=True)
        
        smac_run_dir = os.path.join(component_dir, f"{self.optimizer.study_name}_{component}")
        os.makedirs(smac_run_dir, exist_ok=True)
        
        seed_dir = os.path.join(smac_run_dir, str(self.optimizer.seed))
        os.makedirs(seed_dir, exist_ok=True)
        
        for subdir in ['incumbent', 'logs', 'runhistory', 'stats', 'trajectories']:
            sub_path = os.path.join(seed_dir, subdir)
            os.makedirs(sub_path, exist_ok=True)
        
        print(f"[{component}] Created SMAC directories at: {seed_dir}")
        return component_dir, smac_run_dir, seed_dir
    
    def run_smac_optimization(self, component: str, cs: ConfigurationSpace, 
                             fixed_config: Dict[str, Any], n_trials: int) -> Configuration:
        print(f"\n[{component}] Using SMAC Bayesian optimization with seed {self.optimizer.seed}")
        
        scenario = self.create_scenario(cs, component, n_trials)
        
        initial_design = SobolInitialDesign(
            scenario=scenario,
            n_configs=min(n_trials // 4, 5),
            max_ratio=1.0,
            seed=self.optimizer.seed
        )
        
        if self.optimizer.use_multi_fidelity:
            def target_function(config: Configuration, seed: int = 0, budget: float = None) -> float:
                return self.optimizer.trial_manager.component_target_function(config, seed, component, fixed_config, budget)
        else:
            def target_function(config: Configuration, seed: int = 0) -> float:
                return self.optimizer.trial_manager.component_target_function(config, seed, component, fixed_config, None)
        
        smac = self.create_optimizer(scenario, target_function, initial_design, component)
        
        try:
            incumbent = smac.optimize()
        except Exception as e:
            print(f"[ERROR] SMAC optimization failed for {component}: {str(e)}")
            
            component_dir, smac_run_dir, seed_dir = self.ensure_smac_directories(component)            
            raise
        
        return incumbent
    
    def create_scenario(self, cs: ConfigurationSpace, component: str, n_trials: int) -> Scenario:
        component_dir, smac_run_dir, seed_dir = self.ensure_smac_directories(component)
        
        base_params = {
            'configspace': cs,
            'deterministic': True,
            'n_trials': n_trials,
            'n_workers': self.optimizer.n_workers,
            'seed': self.optimizer.seed,
            'output_directory': Path(component_dir),
            'name': f"{self.optimizer.study_name}_{component}"
        }
        
        if self.optimizer.use_multi_objective:
            base_params['objectives'] = ["score", "latency"]
        
        if self.optimizer.walltime_limit_per_component is not None:
            base_params['walltime_limit'] = self.optimizer.walltime_limit_per_component
        
        if self.optimizer.use_multi_fidelity:
            base_params['min_budget'] = self.optimizer.min_budget
            base_params['max_budget'] = self.optimizer.max_budget
        
        return Scenario(**base_params)
    
    def create_optimizer(self, scenario, target_function, initial_design, component):
        callbacks = []
        
        if self.optimizer.early_stopping_threshold < 1.0:
            early_stopping_callback = self.create_early_stopping_callback()
            callbacks.append(early_stopping_callback)
        
        if self.optimizer.use_multi_fidelity:
            if self.optimizer.optimizer == "bohb":
                intensifier = Hyperband(
                    scenario=scenario,
                    incumbent_selection="highest_budget",
                    eta=self.optimizer.eta
                )
                print(f"[{component}] Using BOHB (Hyperband) with eta={self.optimizer.eta}")
            else:
                intensifier = SuccessiveHalving(
                    scenario=scenario,
                    incumbent_selection="highest_budget",
                    eta=self.optimizer.eta
                )
                print(f"[{component}] Using Successive Halving with eta={self.optimizer.eta}")
            
            return MultiFidelityFacade(
                scenario=scenario,
                target_function=target_function,
                intensifier=intensifier,
                callbacks=callbacks,
                initial_design=initial_design,
                overwrite=True
            )
        else:
            if self.optimizer.use_multi_objective:
                return HPOFacade(
                    scenario=scenario,
                    target_function=target_function,
                    multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
                        scenario,
                        objective_weights=[self.optimizer.generation_weight, self.optimizer.retrieval_weight]
                    ),
                    callbacks=callbacks,
                    initial_design=initial_design,
                    overwrite=True
                )
            else:
                return HPOFacade(
                    scenario=scenario,
                    target_function=target_function,
                    callbacks=callbacks,
                    initial_design=initial_design,
                    overwrite=True
                )
    
    def create_early_stopping_callback(self):
        class EarlyStoppingCallback(Callback):
            def __init__(self, threshold: float):
                super().__init__()
                self.threshold = threshold
                self.should_stop = False
                
            def on_tell(self, smbo, info, value):
                if info and value:
                    if hasattr(value, 'cost'):
                        if isinstance(value.cost, list):
                            score = -value.cost[0]
                        else:
                            score = -value.cost
                    else:
                        score = 0
                    
                    if score >= self.threshold:
                        self.should_stop = True
                        smbo._stop = True
        
        return EarlyStoppingCallback(self.optimizer.early_stopping_threshold)
    
    def run_grid_search(self, component: str, cs: ConfigurationSpace, 
                        fixed_config: Dict[str, Any]) -> Configuration:
        print(f"\n[{component}] Using GRID SEARCH optimization")
        print(f"[{component}] Generating all possible configurations...")

        all_configs = self.generate_grid_search_configs(cs)
        
        if not all_configs:
            print(f"[WARNING] No valid configurations generated for {component}")
            return None
        
        print(f"[{component}] Evaluating {len(all_configs)} configurations...")

        best_score = -float('inf')
        best_config = None
        
        for i, config in enumerate(all_configs):
            print(f"\n[{component}] Grid search {i+1}/{len(all_configs)}")

            score = self.optimizer.trial_manager.component_target_function(config, seed=42, component=component, 
                                                fixed_components=fixed_config, budget=None)

            actual_score = -score
            
            if actual_score > best_score:
                best_score = actual_score
                best_config = config
                print(f"[{component}] New best score: {best_score:.4f}")

        return best_config
    
    def generate_grid_search_configs(self, cs: ConfigurationSpace) -> List[Configuration]:
        configs = []

        hyperparameters = cs.get_hyperparameters()
        
        if not hyperparameters:
            return []

        unconditional_params = {}
        conditional_params = {}
        
        for hp in hyperparameters:
            if cs.get_parents_of(hp):
                conditional_params[hp.name] = hp
            else:
                if hasattr(hp, 'choices'): 
                    unconditional_params[hp.name] = hp.choices
                elif hasattr(hp, 'lower') and hasattr(hp, 'upper'): 
                    if isinstance(hp.lower, int) and isinstance(hp.upper, int):
                        if hp.upper - hp.lower <= 20:
                            unconditional_params[hp.name] = list(range(hp.lower, hp.upper + 1))
                        else:
                            unconditional_params[hp.name] = np.linspace(hp.lower, hp.upper, 10, dtype=int).tolist()
                    else:
                        unconditional_params[hp.name] = np.linspace(hp.lower, hp.upper, 10).tolist()
                else:
                    unconditional_params[hp.name] = [hp.default_value]

        if unconditional_params:
            keys = list(unconditional_params.keys())
            values = list(unconditional_params.values())
            
            for combination in itertools.product(*values):
                base_config = dict(zip(keys, combination))

                try:
                    partial_config = Configuration(cs, values=base_config, allow_inactive=True)

                    configs_with_conditionals = [base_config.copy()]
                    
                    for cond_name, cond_hp in conditional_params.items():
                        parents = cs.get_parents_of(cond_hp)
                        is_active = True
                        
                        for parent in parents:
                            parent_value = base_config.get(parent.name)
                            conditions = cs.get_children_of(parent)

                            for child, condition in conditions:
                                if child.name == cond_name:
                                    if hasattr(condition, 'value') and parent_value != condition.value:
                                        is_active = False
                                        break
                                    elif hasattr(condition, 'values') and parent_value not in condition.values:
                                        is_active = False
                                        break
                        
                        if is_active:
                            new_configs = []
                            cond_values = []
                            
                            if hasattr(cond_hp, 'choices'):
                                cond_values = cond_hp.choices
                            elif hasattr(cond_hp, 'lower') and hasattr(cond_hp, 'upper'):
                                if isinstance(cond_hp.lower, int) and isinstance(cond_hp.upper, int):
                                    if cond_hp.upper - cond_hp.lower <= 20:
                                        cond_values = list(range(cond_hp.lower, cond_hp.upper + 1))
                                    else:
                                        cond_values = np.linspace(cond_hp.lower, cond_hp.upper, 10, dtype=int).tolist()
                                else:
                                    cond_values = np.linspace(cond_hp.lower, cond_hp.upper, 10).tolist()
                            else:
                                cond_values = [cond_hp.default_value]
                            
                            for existing_config in configs_with_conditionals:
                                for cond_value in cond_values:
                                    new_config = existing_config.copy()
                                    new_config[cond_name] = cond_value
                                    new_configs.append(new_config)
                            
                            if new_configs:
                                configs_with_conditionals = new_configs

                    for final_config in configs_with_conditionals:
                        try:
                            config = Configuration(cs, values=final_config)
                            configs.append(config)
                        except:
                            continue
                            
                except:
                    continue
        
        print(f"Generated {len(configs)} configurations for grid search")
        return configs