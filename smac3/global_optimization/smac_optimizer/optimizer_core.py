import time
from typing import Dict, Any
import wandb
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband, SuccessiveHalving
from smac.initial_design import SobolInitialDesign
from smac.callback import Callback
from pipeline.logging.wandb import WandBLogger


class OptimizerCore:
    
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
        
        search_space_size, combination_note = self._calculate_total_search_space()
        cs = self.config_space_builder.build_configuration_space()
        search_space_info = self.config_space_builder.get_search_space_info()
        
        wandb_config = {
            "optimizer": f"{self.optimizer.upper()}{' Multi-Fidelity' if self.use_multi_fidelity else ''}",
            "n_trials": self.n_trials,
            "retrieval_weight": self.retrieval_weight,
            "generation_weight": self.generation_weight,
            "search_space_size": search_space_size,
            "search_space_note": combination_note,
            "n_hyperparameters": search_space_info['n_hyperparameters'],
            "study_name": self.study_name,
            "early_stopping_threshold": self.early_stopping_threshold,
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