import os
import time
from pathlib import Path
import yaml
import shutil
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_calculator import CombinationCalculator
from pipeline.utils import Utils
from ..config_space_builder import ComponentwiseSMACConfigSpaceBuilder
from smac3.local_optimization.componentwise_rag_processor import ComponentwiseRAGProcessor

from .trial_runner import TrialRunner
from .results_manager import ResultsManager
from .wandb_manager import WandBManager
from .smac_manager import SMACManager


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
        
        self.trial_runner = TrialRunner(self)
        self.results_manager = ResultsManager(self)
        self.wandb_manager = WandBManager(self)
        self.smac_manager = SMACManager(self)
    
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
                self.wandb_manager.init_for_component(component, component_idx, len(active_components))
            
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
            
            if self.use_wandb:
                self.wandb_manager.log_component_summary(component, component_result)
        
        all_results['optimization_time'] = time.time() - start_time
        all_results['early_stopped'] = early_stopped
        all_results['stopped_at_component'] = stopped_at_component
        all_results['total_trials'] = sum(
            comp.get('n_trials', 0) for comp in all_results['component_results'].values()
        )
        
        all_results['study_name'] = self.study_name
        
        if self.use_wandb:
            self.wandb_manager.log_final_summary(all_results)
        
        self.results_manager.save_final_results(all_results)
        self.results_manager.print_final_summary(all_results)
        
        return all_results
    
    def _optimize_component(self, component: str, component_idx: int, 
                  active_components: List[str]) -> Dict[str, Any]:
        component_start_time = time.time()

        component_dir, smac_run_dir, seed_dir = self.smac_manager.ensure_directories(component)
        
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
        
        incumbent = self.smac_manager.run_optimization(component, cs, fixed_config, n_trials)

        best_config = dict(incumbent) if incumbent else {}
        
        best_config = self.rag_processor.parse_component_config(component, best_config)

        best_trial = self.trial_runner.find_best_trial(component)
        
        if best_trial and best_trial.get('output_parquet') and os.path.exists(best_trial['output_parquet']):
            self.component_dataframes[component] = best_trial['output_parquet']
            print(f"[{component}] Best configuration found with score {best_trial['score']:.4f} and latency {best_trial.get('latency', 0.0):.2f}s")
            print(f"[{component}] Best output saved at: {best_trial['output_parquet']}")
            print(f"[{component}] Best config: {best_trial['config']}")
        
        if self.use_wandb:
            self.wandb_manager.log_component_table(component)
        
        result = self.results_manager.create_component_result(
            component, best_trial, best_config, fixed_config, 
            cs, component_start_time
        )

        self.results_manager.save_component_result(trial_component_dir, result)
        
        return result
    
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
    
    def _calculate_component_trials(self, component: str, cs) -> int:
        total_combinations, note = self._get_component_combinations(component)
        
        print(f"[{component}] {note}")
        print(f"[{component}] Total combinations: {total_combinations}")
        
        if self.n_trials_per_component:
            if total_combinations < self.n_trials_per_component:
                print(f"[{component}] WARNING: Only {total_combinations} unique combinations available, but {self.n_trials_per_component} trials requested.")
                print(f"[{component}] Some configurations may be sampled multiple times.")
            return self.n_trials_per_component
        
        return min(20, total_combinations)
    
    def _get_component_combinations(self, component: str) -> Tuple[int, str]:
        combinations, note = self.rag_processor.get_component_combinations(
            component, 
            self.config_space_builder,
            self.best_configs
        )
        return combinations, note