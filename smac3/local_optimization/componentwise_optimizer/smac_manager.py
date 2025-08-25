import os
from pathlib import Path
from typing import Dict, Any, Tuple
from ConfigSpace import ConfigurationSpace, Configuration
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband, SuccessiveHalving
from smac.initial_design import SobolInitialDesign
from smac.callback import Callback


class SMACManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def ensure_directories(self, component: str) -> Tuple[str, str, str]:
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
    
    def run_optimization(self, component: str, cs: ConfigurationSpace, 
                        fixed_config: Dict[str, Any], n_trials: int) -> Configuration:
        print(f"\n[{component}] Using SMAC Bayesian optimization with seed {self.optimizer.seed}")
        
        scenario = self._create_scenario(cs, component, n_trials)
        
        initial_design = SobolInitialDesign(
            scenario=scenario,
            n_configs=min(n_trials // 4, 5),
            max_ratio=1.0,
            seed=self.optimizer.seed
        )
        
        if self.optimizer.use_multi_fidelity:
            def target_function(config: Configuration, seed: int = 0, budget: float = None) -> float:
                return self.optimizer.trial_runner.component_target_function(config, seed, component, fixed_config, budget)
        else:
            def target_function(config: Configuration, seed: int = 0) -> float:
                return self.optimizer.trial_runner.component_target_function(config, seed, component, fixed_config, None)
        
        smac = self._create_optimizer(scenario, target_function, initial_design, component)
        
        try:
            incumbent = smac.optimize()
        except Exception as e:
            print(f"[ERROR] SMAC optimization failed for {component}: {str(e)}")            
            raise
        
        return incumbent
    
    def _create_scenario(self, cs: ConfigurationSpace, component: str, n_trials: int) -> Scenario:
        component_dir, smac_run_dir, seed_dir = self.ensure_directories(component)
        
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
    
    def _create_optimizer(self, scenario, target_function, initial_design, component):
        callbacks = []
        
        if self.optimizer.early_stopping_threshold < 1.0:
            early_stopping_callback = self._create_early_stopping_callback()
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
    
    def _create_early_stopping_callback(self):
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