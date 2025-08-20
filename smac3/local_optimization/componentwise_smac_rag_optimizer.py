import os
import json
import time
from pathlib import Path
import yaml
import shutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import wandb
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband, SuccessiveHalving
from smac.initial_design import SobolInitialDesign
from smac.callback import Callback
import itertools
from ConfigSpace import ConfigurationSpace, Configuration

from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_calculator import CombinationCalculator
from pipeline.utils import Utils
from smac3.local_optimization.componentwise_config_space_builder import ComponentwiseSMACConfigSpaceBuilder
from pipeline.logging.wandb import WandBLogger
from smac3.local_optimization.componentwise_rag_processor import ComponentwiseRAGProcessor


class ComponentwiseSMACOptimizer:
    
    COMPONENT_ORDER = [
        'query_expansion',
        'retrieval', 
        'passage_reranker',
        'passage_filter',
        'passage_compressor',
        'prompt_maker_generator' 
    ]
    
    def __init__(
        self,
        config_template: Dict[str, Any],
        qa_data: pd.DataFrame,
        corpus_data: pd.DataFrame,
        project_dir: str,
        n_trials_per_component: Optional[int] = None,
        sample_percentage: float = 0.1,
        cpu_per_trial: int = 4,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        walltime_limit_per_component: int = 1800,
        n_workers: int = 1,
        seed: int = 42,
        early_stopping_threshold: float = 0.9,
        use_wandb: bool = True,
        wandb_project: str = "Componentwise Optimization",
        wandb_entity: Optional[str] = None,
        optimizer: str = "smac", 
        use_multi_fidelity: bool = False,
        min_budget_percentage: float = 0.1,
        max_budget_percentage: float = 1.0,
        eta: int = 3,
        use_multi_objective: bool = False,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_config: Optional[Dict[str, Any]] = None,
        search_type: str = "bo",
    ):
        print(f"[ComponentwiseSMACOptimizer] Initializing with:")
        print(f"  use_llm_compressor_evaluator: {use_llm_compressor_evaluator}")
        print(f"  llm_evaluator_config: {llm_evaluator_config}")
        print(f"  seed: {seed}")
            
        self.config_template = config_template
        self.qa_data = qa_data
        self.corpus_data = corpus_data
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        self.total_samples = len(qa_data)
        
        self.n_trials_per_component = n_trials_per_component
        self.sample_percentage = sample_percentage
        self.cpu_per_trial = cpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.walltime_limit_per_component = walltime_limit_per_component
        self.n_workers = n_workers
        self.seed = seed
        self.early_stopping_threshold = early_stopping_threshold
        
        self.use_wandb = use_wandb
        self.wandb_enabled = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.optimizer = optimizer.lower()
        
        self.use_multi_fidelity = use_multi_fidelity
        self.min_budget_percentage = min_budget_percentage
        self.max_budget_percentage = max_budget_percentage
        self.eta = eta
        self.use_multi_objective = use_multi_objective
        
        self.min_budget = max(1, int(self.total_samples * self.min_budget_percentage))
        self.max_budget = max(self.min_budget, int(self.total_samples * self.max_budget_percentage))
        
        self.component_detailed_metrics = {} 
        
        self.study_name = study_name if study_name else f"componentwise_smac_sap_api_{int(time.time())}"
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                f"componentwise_optimization_results"
            )
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.search_type = search_type
        self.config_generator = ConfigGenerator(self.config_template)
        self.combination_calculator = CombinationCalculator(
            config_generator=self.config_generator,
            search_type=self.search_type
        )
        self.config_space_builder = ComponentwiseSMACConfigSpaceBuilder(self.config_generator, seed=self.seed)
        
        self.rag_processor = ComponentwiseRAGProcessor(
            config_generator=self.config_generator,
            retrieval_weight=self.retrieval_weight,
            generation_weight=self.generation_weight,
            project_dir=self.project_dir,
            corpus_data=self.corpus_data,
            qa_data=self.qa_data,
            use_llm_evaluator=use_llm_compressor_evaluator,
            llm_evaluator_config=llm_evaluator_config,
            search_type=self.search_type,
        )
        
        self.component_results = {}
        self.best_configs = {}
        self.component_dataframes = {}
        self.current_trial = 0
        self.trial_results = []
        
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_config = llm_evaluator_config or {}
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
        
    def _ensure_smac_directories(self, component: str):
        component_dir = os.path.join(self.result_dir, component)
        os.makedirs(component_dir, exist_ok=True)
        
        smac_run_dir = os.path.join(component_dir, f"{self.study_name}_{component}")
        os.makedirs(smac_run_dir, exist_ok=True)
        
        seed_dir = os.path.join(smac_run_dir, str(self.seed))
        os.makedirs(seed_dir, exist_ok=True)
        
        for subdir in ['incumbent', 'logs', 'runhistory', 'stats', 'trajectories']:
            sub_path = os.path.join(seed_dir, subdir)
            os.makedirs(sub_path, exist_ok=True)
        
        print(f"[{component}] Created SMAC directories at: {seed_dir}")
        return component_dir, smac_run_dir, seed_dir
    
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        components_with_small_space, component_combinations, component_notes = self._validate_all_components()
        
        if components_with_small_space:
            n_trials = self.n_trials_per_component if self.n_trials_per_component else 20
            print(f"\n{'='*70}")
            print(f"WARNING: Small search spaces detected!")
            print(f"The following components have search spaces smaller than n_trials ({n_trials}):")
            for comp in components_with_small_space:
                print(f"  - {comp}: only {component_combinations[comp]} combinations")
                print(f"    Note: {component_notes[comp]}")
            print(f"\nOptimization will continue, but may sample repeated configurations.")
            print(f"Consider:")
            print(f"  1. Adding more hyperparameters or values to these components")
            print(f"  2. Reducing n_trials_per_component to {min(component_combinations[comp] for comp in components_with_small_space)}")
            print(f"{'='*70}\n")
        
        all_results = {
            'study_name': self.study_name,
            'component_results': {},
            'best_configs': {},
            'optimization_time': 0,
            'component_order': [],
            'early_stopped': False,
            'retrieval_weight': self.retrieval_weight,
            'generation_weight': self.generation_weight,
            'components_with_small_space': components_with_small_space,
            'component_combinations': component_combinations,
            'component_notes': component_notes
        }
        
        active_components = self._get_active_components()
        early_stopped = False
        stopped_at_component = None
        
        for component_idx, component in enumerate(active_components):
            if early_stopped and component != 'prompt_maker_generator':
                print(f"\n[Component-wise] Skipping {component} optimization due to early stopping at {stopped_at_component}")
                if component == 'passage_filter':
                    self.best_configs[component] = {'passage_filter_method': 'pass_passage_filter'}
                elif component == 'passage_compressor':
                    self.best_configs[component] = {'passage_compressor_method': 'pass_compressor'}
                
                all_results['component_results'][component] = {
                    'component': component,
                    'best_config': self.best_configs[component],
                    'best_score': all_results['component_results'][stopped_at_component]['best_score'],
                    'n_trials': 0,
                    'optimization_time': 0.0,
                    'skipped': True,
                    'skip_reason': f'Early stopped at {stopped_at_component}'
                }
                
                all_results['best_configs'][component] = self.best_configs[component]
                all_results['component_order'].append(component)
                continue

            if self._should_skip_component(component):
                continue
            
            if self.use_wandb:
                self._init_wandb_for_component(component, component_idx, len(active_components))
            
            component_result = self._optimize_component(
                component, 
                component_idx,
                active_components
            )
            
            all_results['component_results'][component] = component_result
            all_results['best_configs'][component] = component_result['best_config']
            all_results['component_order'].append(component)
            
            self.component_results[component] = component_result
            self.best_configs[component] = component_result['best_config']

            if component_result.get('best_score', 0) >= self.early_stopping_threshold and component != 'prompt_maker_generator':
                early_stopped = True
                stopped_at_component = component
                component_result['early_stopped'] = True
                print(f"\n[Component-wise] Early stopping triggered at {component} with score {component_result['best_score']:.4f}")
            
            if self.use_wandb and wandb.run is not None:
                self._log_wandb_component_summary(component, component_result)
        
        all_results['optimization_time'] = time.time() - start_time
        all_results['early_stopped'] = early_stopped
        all_results['stopped_at_component'] = stopped_at_component
        all_results['total_trials'] = sum(
            comp.get('n_trials', 0) for comp in all_results['component_results'].values()
        )
        
        all_results['study_name'] = self.study_name
        
        if self.use_wandb:
            self._log_wandb_final_summary(all_results)
        
        self._save_final_results(all_results)
        self._print_final_summary(all_results)
        
        return all_results
    
    def _optimize_component(self, component: str, component_idx: int, 
                  active_components: List[str]) -> Dict[str, Any]:
        component_start_time = time.time()

        component_dir, smac_run_dir, seed_dir = self._ensure_smac_directories(component)
        
        trial_component_dir = os.path.join(self.result_dir, f"{component_idx}_{component}")
        os.makedirs(trial_component_dir, exist_ok=True)
        
        fixed_config = self.rag_processor.get_fixed_config(component, self.best_configs, self.COMPONENT_ORDER)
        
        cs = self.config_space_builder.build_component_space(component, fixed_config)
        
        if cs is None or len(cs.get_hyperparameters()) == 0:
            return {
                'component': component,
                'best_config': {},
                'best_score': 0.0,
                'n_trials': 0,
                'optimization_time': 0.0
            }

        n_trials = self._calculate_component_trials(component, cs)

        self.current_component = component
        self.current_fixed_config = fixed_config
        self.component_trial_counter = 0
        self.component_trials = []
        self.component_detailed_metrics[component] = []
        self.component_best_scores = {}
        self.component_best_outputs = {}
        
        incumbent = self._run_smac_optimization(component, cs, fixed_config, n_trials)

        best_config = dict(incumbent) if incumbent else {}
        
        best_config = self.rag_processor.parse_component_config(component, best_config)

        best_trial = self._find_best_trial(component)
        
        if best_trial and best_trial.get('output_parquet') and os.path.exists(best_trial['output_parquet']):
            self.component_dataframes[component] = best_trial['output_parquet']
            print(f"[{component}] Best configuration found with score {best_trial['score']:.4f} and latency {best_trial.get('latency', 0.0):.2f}s")
            print(f"[{component}] Best output saved at: {best_trial['output_parquet']}")
            print(f"[{component}] Best config: {best_trial['config']}")
        
        if self.use_wandb:
            self._log_wandb_component_table(component)
        
        result = self._create_component_result(
            component, best_trial, best_config, fixed_config, 
            cs, component_start_time
        )

        self._save_component_result(trial_component_dir, result)
        
        return result

    def _component_target_function(self, config: Configuration, seed: int, component: str, 
                             fixed_components: Dict[str, Any], budget: float = None) -> float:
        trial_config = dict(config)
        
        trial_config, is_pass_component = self.rag_processor.parse_trial_config(component, trial_config)
        
        full_config = {**fixed_components, **trial_config}
        
        print(f"\n[DEBUG] Trial {self.current_trial + 1} config from SMAC: {dict(config)}")
        print(f"[DEBUG] Is pass component: {is_pass_component}")
        
        cleaned_config = self.config_space_builder.clean_trial_config(full_config)
        
        print(f"[DEBUG] Cleaned config: {cleaned_config}")
        
        self.current_trial += 1
        self.component_trial_counter += 1
        
        trial_id = f"trial_{self.current_trial:04d}"
        trial_dir = os.path.join(self.result_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        
        start_time = time.time()
        
        try:
            if budget:
                qa_subset = self._sample_data(int(budget), seed)
            else:
                qa_subset = self.qa_data.copy()
            
            qa_subset = self.rag_processor.load_previous_outputs(
                component, qa_subset, self.component_dataframes, trial_dir
            )
                    
            self._copy_corpus_data(trial_dir)
            
            trial_config_yaml = self.config_generator.generate_trial_config(cleaned_config)
            
            with open(os.path.join(trial_dir, "config.yaml"), 'w') as f:
                yaml.dump(trial_config_yaml, f)

            results = self.rag_processor.run_pipeline(
                cleaned_config,
                trial_dir,
                qa_subset,
                component,
                self.component_results
            )

            
            working_df = results.pop('working_df', qa_subset)

            detailed_metrics = self.rag_processor.extract_detailed_metrics(component, results)

            if component not in self.component_detailed_metrics:
                self.component_detailed_metrics[component] = []
            
            self.component_detailed_metrics[component].append(detailed_metrics)

            score = self.rag_processor.calculate_component_score(
                component, results, is_pass_component, self.component_results
            )
            
            end_time = time.time()
            latency = end_time - start_time

            output_parquet_path = self.rag_processor.save_component_output(
                component, trial_dir, results, working_df
            )
            
            current_best_score = self.component_results.get(component, {}).get('best_score', -float('inf'))

            if len(self.component_trials) == 0:
                current_best_score = -float('inf')

            print(f"[{component}] Trial score: {score:.4f}, Current best: {current_best_score:.4f}")

            if score > current_best_score:
                self.component_dataframes[component] = output_parquet_path
                print(f"[{component}] New best score: {score:.4f}, saving output to: {output_parquet_path}")

                if component not in self.component_results:
                    self.component_results[component] = {}
                self.component_results[component]['best_score'] = score
                self.component_results[component]['best_output_path'] = output_parquet_path
                self.component_results[component]['best_config'] = cleaned_config.copy()

            trial_result = self._create_trial_result(
                cleaned_config, score, latency, 
                int(budget) if budget else len(qa_subset),
                budget / self.total_samples if budget else 1.0,
                results, component, output_parquet_path
            )

            for metric_key in ['retrieval_score', 'query_expansion_score', 'reranker_score', 
                            'filter_score', 'compressor_score', 'compression_score',
                            'prompt_maker_score', 'generation_score', 'last_retrieval_score']:
                if metric_key in results:
                    trial_result[metric_key] = results[metric_key]
            
            self.component_trials.append(trial_result)
            
            if self.wandb_enabled:
                WandBLogger.log_component_trial(component, self.component_trial_counter, 
                                            cleaned_config, score, latency)
            
            self._save_trial_results(trial_dir, results, cleaned_config, component, latency)
            
            print(f"\n[Trial {self.current_trial}] Score: {score:.4f} | Time: {latency:.2f}s")
            
            return -score
            
        except Exception as e:
            print(f"\n[ERROR] Trial {self.current_trial} failed:")
            print(f"  Component: {component}")
            print(f"  Config: {trial_config}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self._save_error_results(trial_dir, cleaned_config, component, e)

            empty_output_df = qa_subset.copy() if 'qa_subset' in locals() else self.qa_data.head(0).copy()
            output_path = os.path.join(trial_dir, f"{component}_output.parquet")
            empty_output_df.to_parquet(output_path)
            
            return 0.0
    
    def _create_scenario(self, cs: ConfigurationSpace, component: str, n_trials: int) -> Scenario:
        component_dir, smac_run_dir, seed_dir = self._ensure_smac_directories(component)
        
        base_params = {
            'configspace': cs,
            'deterministic': True,
            'n_trials': n_trials,
            'n_workers': self.n_workers,
            'seed': self.seed,
            'output_directory': Path(component_dir),
            'name': f"{self.study_name}_{component}"
        }
        
        if self.use_multi_objective:
            base_params['objectives'] = ["score", "latency"]
        
        if self.walltime_limit_per_component is not None:
            base_params['walltime_limit'] = self.walltime_limit_per_component
        
        if self.use_multi_fidelity:
            base_params['min_budget'] = self.min_budget
            base_params['max_budget'] = self.max_budget
        
        return Scenario(**base_params)
    
    def _create_optimizer(self, scenario, target_function, initial_design, component):
        callbacks = []
        
        if self.early_stopping_threshold < 1.0:
            early_stopping_callback = self._create_early_stopping_callback()
            callbacks.append(early_stopping_callback)
        
        if self.use_multi_fidelity:
            if self.optimizer == "bohb":
                intensifier = Hyperband(
                    scenario=scenario,
                    incumbent_selection="highest_budget",
                    eta=self.eta
                )
                print(f"[{component}] Using BOHB (Hyperband) with eta={self.eta}")
            else:
                intensifier = SuccessiveHalving(
                    scenario=scenario,
                    incumbent_selection="highest_budget",
                    eta=self.eta
                )
                print(f"[{component}] Using Successive Halving with eta={self.eta}")
            
            return MultiFidelityFacade(
                scenario=scenario,
                target_function=target_function,
                intensifier=intensifier,
                callbacks=callbacks,
                initial_design=initial_design,
                overwrite=True
            )
        else:
            if self.use_multi_objective:
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
            else:
                return HPOFacade(
                    scenario=scenario,
                    target_function=target_function,
                    callbacks=callbacks,
                    initial_design=initial_design,
                    overwrite=True
                )

    def _get_component_combinations(self, component: str) -> Tuple[int, str]:
        combinations, note = self.rag_processor.get_component_combinations(
            component, 
            self.config_space_builder,
            self.best_configs
        )
        return combinations, note
    
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
        
        return EarlyStoppingCallback(self.early_stopping_threshold)
    
    def _validate_all_components(self) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
        active_components = []
        has_active_query_expansion = False
        
        for comp in self.COMPONENT_ORDER:
            if comp == 'query_expansion':
                if not self.config_generator.node_exists(comp):
                    continue 
                    
                qe_config = self.config_generator.extract_node_config("query_expansion")
                qe_methods = []
                for module in qe_config.get("modules", []):
                    method = module.get("module_type")
                    if method and method != "pass_query_expansion":
                        qe_methods.append(method)
                if qe_methods:
                    has_active_query_expansion = True
                    active_components.append(comp)
            elif comp == 'retrieval':
                if has_active_query_expansion:
                    continue
                else:
                    if self.config_generator.node_exists(comp):
                        active_components.append(comp)
            elif comp == 'prompt_maker_generator':
                if self.config_generator.node_exists('prompt_maker') or self.config_generator.node_exists('generator'):
                    active_components.append(comp)
            else:
                if self.config_generator.node_exists(comp):
                    active_components.append(comp)
        
        component_combinations = {}
        component_notes = {}
        components_with_small_space = []
        
        n_trials = self.n_trials_per_component if self.n_trials_per_component else 20
        
        for component in active_components:
            combinations, note = self._get_component_combinations(component)
            component_combinations[component] = combinations
            component_notes[component] = note
            
            if combinations < n_trials and combinations > 0:
                components_with_small_space.append(component)
        
        return components_with_small_space, component_combinations, component_notes
    
    def _get_active_components(self) -> List[str]:
        active_components = []
        has_active_query_expansion = False
        
        for comp in self.COMPONENT_ORDER:
            if comp == 'query_expansion':
                if not self.config_generator.node_exists(comp):
                    continue  
                    
                qe_config = self.config_generator.extract_node_config("query_expansion")
                qe_methods = []
                for module in qe_config.get("modules", []):
                    method = module.get("module_type")
                    if method and method != "pass_query_expansion":
                        qe_methods.append(method)
                if qe_methods:
                    has_active_query_expansion = True
                    active_components.append(comp)
            elif comp == 'retrieval':
                if not has_active_query_expansion:
                    if self.config_generator.node_exists(comp):
                        active_components.append(comp)
                else:
                    print("[Component-wise] Skipping retrieval component since query expansion includes retrieval")
            elif comp == 'prompt_maker_generator':
                if self.config_generator.node_exists('prompt_maker') or self.config_generator.node_exists('generator'):
                    active_components.append(comp)
            else:
                if self.config_generator.node_exists(comp):
                    active_components.append(comp)
        
        return active_components
    
    def _should_skip_component(self, component: str) -> bool:
        if component == 'passage_filter' and 'passage_reranker' in self.best_configs:
            reranker_config = self.best_configs['passage_reranker']
            if reranker_config.get('reranker_top_k') == 1:
                print(f"\n[Component-wise] Skipping filter optimization because reranker_top_k=1")
                self.best_configs[component] = {'passage_filter_method': 'pass_passage_filter'}
                return True
        return False
    
    def _calculate_component_trials(self, component: str, cs: ConfigurationSpace) -> int:
        total_combinations, note = self._get_component_combinations(component)
        
        print(f"[{component}] {note}")
        print(f"[{component}] Total combinations: {total_combinations}")
        
        if self.n_trials_per_component:
            if total_combinations < self.n_trials_per_component:
                print(f"[{component}] WARNING: Only {total_combinations} unique combinations available, but {self.n_trials_per_component} trials requested.")
                print(f"[{component}] Some configurations may be sampled multiple times.")
            return self.n_trials_per_component
        
        return min(20, total_combinations)
    
    def _run_smac_optimization(self, component: str, cs: ConfigurationSpace, 
                             fixed_config: Dict[str, Any], n_trials: int) -> Configuration:
        print(f"\n[{component}] Using SMAC Bayesian optimization with seed {self.seed}")
        
        scenario = self._create_scenario(cs, component, n_trials)
        
        initial_design = SobolInitialDesign(
            scenario=scenario,
            n_configs=min(n_trials // 4, 5),
            max_ratio=1.0,
            seed=self.seed
        )
        
        if self.use_multi_fidelity:
            def target_function(config: Configuration, seed: int = 0, budget: float = None) -> float:
                return self._component_target_function(config, seed, component, fixed_config, budget)
        else:
            def target_function(config: Configuration, seed: int = 0) -> float:
                return self._component_target_function(config, seed, component, fixed_config, None)
        
        smac = self._create_optimizer(scenario, target_function, initial_design, component)
        
        try:
            incumbent = smac.optimize()
        except Exception as e:
            print(f"[ERROR] SMAC optimization failed for {component}: {str(e)}")            
            raise
        
        return incumbent
    
    def _find_best_trial(self, component: str) -> Optional[Dict]:
        return Utils.find_best_trial_from_component(self.component_trials, component)

    def _create_component_result(self, component: str, best_trial: Optional[Dict[str, Any]], 
                               best_config: Dict[str, Any], fixed_config: Dict[str, Any],
                               cs: ConfigurationSpace, start_time: float) -> Dict[str, Any]:
        if not best_trial and not self.component_trials:
            print(f"[WARNING] No successful trials for component {component}")
            return {
                'component': component,
                'best_config': {},
                'best_score': 0.0,
                'best_trial': None,
                'all_trials': [],
                'n_trials': 0,
                'fixed_config': fixed_config,
                'search_space_size': len(cs.get_hyperparameters()),
                'optimization_time': time.time() - start_time,
                'detailed_metrics': [],
                'optimization_method': 'smac'
            }

        return {
            'component': component,
            'best_config': best_trial['config'] if best_trial else best_config,
            'best_score': best_trial['score'] if best_trial else 0.0,
            'best_latency': best_trial.get('latency', 0.0) if best_trial else 0.0,
            'best_trial': best_trial,
            'all_trials': self.component_trials,
            'n_trials': len(self.component_trials),
            'fixed_config': fixed_config,
            'search_space_size': len(cs.get_hyperparameters()),
            'optimization_time': time.time() - start_time,
            'detailed_metrics': self.component_detailed_metrics.get(component, []),
            'best_output_path': best_trial.get('output_parquet') if best_trial else None,
            'optimization_method': 'smac'
        }
    
    def _create_trial_result(self, config_dict, score, latency, budget, budget_percentage, 
                           results, component, output_parquet_path):
        trial_result = {
            "trial_number": int(self.component_trial_counter),
            "component": component,
            "config": self._convert_numpy_types(config_dict),
            "full_config": self._convert_numpy_types({**self.current_fixed_config, **config_dict}),
            "score": float(score),
            "latency": float(latency),
            "budget": int(budget),
            "budget_percentage": float(budget_percentage),
            "results": results,
            "output_parquet": output_parquet_path,
            "timestamp": float(time.time())
        }
        
        for k, v in results.items():
            if k.endswith('_score') or k.endswith('_metrics'):
                trial_result[k] = self._convert_numpy_types(v)
        
        return trial_result
    
    def _sample_data(self, budget: int, seed: int) -> pd.DataFrame:
        actual_samples = min(budget, self.total_samples)
        if actual_samples < self.total_samples:
            return self.qa_data.sample(n=actual_samples, random_state=seed)
        return self.qa_data
    
    def _copy_corpus_data(self, trial_dir: str):
        centralized_corpus_path = os.path.join(self.project_dir, "data", "corpus.parquet")
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
    
    def _convert_numpy_types(self, obj):
        return Utils.convert_numpy_types(obj)
    
    def _save_component_result(self, component_dir: str, result: Dict[str, Any]):
        result_for_json = result.copy()

        if 'best_trial' in result_for_json and result_for_json['best_trial']:
            if 'results' in result_for_json['best_trial']:
                trial_results = result_for_json['best_trial']['results'].copy()
                keys_to_remove = [k for k, v in trial_results.items() if isinstance(v, pd.DataFrame)]
                for key in keys_to_remove:
                    trial_results.pop(key, None)
                result_for_json['best_trial']['results'] = trial_results
                
        if 'all_trials' in result_for_json:
            cleaned_trials = []
            for trial in result_for_json['all_trials']:
                trial_copy = trial.copy()
                if 'results' in trial_copy:
                    trial_results = trial_copy['results'].copy()
                    keys_to_remove = [k for k, v in trial_results.items() if isinstance(v, pd.DataFrame)]
                    for key in keys_to_remove:
                        trial_results.pop(key, None)
                    trial_copy['results'] = trial_results
                cleaned_trials.append(trial_copy)
            result_for_json['all_trials'] = cleaned_trials

        result_serializable = self._convert_numpy_types(result_for_json)

        with open(os.path.join(component_dir, "optimization_result.json"), 'w') as f:
            json.dump(result_serializable, f, indent=2)
    
    def _save_trial_results(self, trial_dir: str, results: Dict[str, Any], 
                          cleaned_config: Dict[str, Any], component: str, latency: float):
        results['trial_number'] = self.current_trial
        results['time_taken'] = latency
        results['config'] = cleaned_config
        results['component'] = component
        
        results_for_json = results.copy()

        keys_to_remove = []
        for key, value in results_for_json.items():
            if isinstance(value, pd.DataFrame):
                keys_to_remove.append(key)
            elif isinstance(value, dict):
                nested_keys_to_remove = []
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, pd.DataFrame):
                        nested_keys_to_remove.append(nested_key)
                for nested_key in nested_keys_to_remove:
                    value.pop(nested_key, None)

        for key in keys_to_remove:
            results_for_json.pop(key, None)
        
        results_serializable = self._convert_numpy_types(results_for_json)
        
        with open(os.path.join(trial_dir, "results.json"), 'w') as f:
            json.dump(results_serializable, f, indent=2)
    
    def _save_error_results(self, trial_dir: str, cleaned_config: Dict[str, Any], 
                          component: str, error: Exception):
        import traceback
        
        error_results = {
            'trial_number': self.current_trial,
            'config': cleaned_config,
            'component': component,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        
        error_results_serializable = self._convert_numpy_types(error_results)
        
        with open(os.path.join(trial_dir, "error.json"), 'w') as f:
            json.dump(error_results_serializable, f, indent=2)
    
    def _save_final_results(self, results: Dict[str, Any]):
        Utils.save_component_optimization_results(
            self.result_dir, results, self.config_generator
        )
    
    def _print_final_summary(self, results: Dict[str, Any]):
        print(f"\nTotal optimization time: {Utils.format_time_duration(results['optimization_time'])}")
        
        if results.get('validation_failed', False):
            print("\nOptimization failed due to insufficient search space.")
        else:
            for component in results['component_order']:
                comp_result = results['component_results'][component]
                print(f"\n{component.upper()}:")
                print(f"  Best score: {comp_result['best_score']:.4f}")
                print(f"  Best config: {comp_result['best_config']}")
                print(f"  Trials run: {comp_result['n_trials']}")
    
    def _init_wandb_for_component(self, component: str, component_idx: int, total_components: int):
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARNING] Error finishing previous W&B run: {e}")

            time.sleep(2)
        
        wandb_run_name = f"{self.study_name}_{component}"
        try:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=wandb_run_name,
                config={
                    "component": component,
                    "stage": f"{component_idx + 1}/{total_components}",
                    "n_trials": self.n_trials_per_component,
                    "use_multi_fidelity": self.use_multi_fidelity,
                    "use_multi_objective": self.use_multi_objective
                },
                reinit=True,
                force=True
            )
            WandBLogger.reset_step_counter()
        except Exception as e:
            print(f"[WARNING] Failed to initialize W&B for {component}: {e}")
            print(f"[WARNING] Continuing without W&B logging for {component}")
            self.wandb_enabled = False
    
    def _log_wandb_component_summary(self, component: str, component_result: Dict[str, Any]):
        try:
            WandBLogger.log_component_summary(
                component, 
                component_result['best_config'],
                component_result['best_score'],
                component_result['n_trials'],
                component_result.get('optimization_time', 0.0)
            )
            wandb.finish()
        except Exception as e:
            print(f"[WARNING] Error logging W&B summary for {component}: {e}")

        if not self.wandb_enabled:
            self.wandb_enabled = True
    
    def _log_wandb_component_table(self, component: str):
        detailed_metrics_dict = {}
        if component in self.component_detailed_metrics:
            for i, metrics in enumerate(self.component_detailed_metrics[component]):
                detailed_metrics_dict[i] = metrics
        
        WandBLogger.log_component_optimization_table(
            component, 
            self.component_trials,
            detailed_metrics_dict
        )
    
    def _log_wandb_final_summary(self, all_results: Dict[str, Any]):
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARNING] Error finishing component W&B run: {e}")
            
            time.sleep(2)
        
        try:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=f"{self.study_name}_summary",
                config={
                    "optimization_mode": "componentwise",
                    "total_components": len(all_results['component_order']),
                    "total_time": all_results['optimization_time'],
                    "early_stopped": all_results.get('early_stopped', False),
                    "stopped_at_component": all_results.get('stopped_at_component'),
                    "use_multi_fidelity": self.use_multi_fidelity,
                    "use_multi_objective": self.use_multi_objective
                },
                reinit=True,
                force=True
            )

            WandBLogger.log_final_componentwise_summary(all_results)
            wandb.finish()
        except Exception as e:
            print(f"[WARNING] Failed to log final W&B summary: {e}")