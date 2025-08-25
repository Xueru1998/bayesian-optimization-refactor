import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import wandb

from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_calculator import SearchSpaceCalculator
from pipeline.utils import Utils
from ..config_space_builder import ComponentwiseSMACConfigSpaceBuilder
from pipeline.logging.wandb import WandBLogger
from smac3.local_optimization.componentwise_rag_processor import ComponentwiseRAGProcessor

from .trial_manager import TrialManager
from .smac_runner import SMACRunner
from .component_manager import ComponentManager
from .results_manager import ResultsManager
from .wandb_manager import WandBManager


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
        wandb_project: str = "Component-wise SMAC Optimization",
        wandb_entity: Optional[str] = None,
        optimizer: str = "smac", 
        use_multi_fidelity: bool = False,
        min_budget_percentage: float = 0.1,
        max_budget_percentage: float = 1.0,
        eta: int = 3,
        use_multi_objective: bool = False,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o" 
    ):
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
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
        self.component_detailed_metrics = {} 
        
        self.study_name = study_name if study_name else f"componentwise_smac_local_model_{int(time.time())}"
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                f"componentwise_optimization_results"
            )
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.config_generator = ConfigGenerator(self.config_template)
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        self.config_space_builder = ComponentwiseSMACConfigSpaceBuilder(self.config_generator, seed=self.seed)
        
        self.rag_processor = ComponentwiseRAGProcessor(
            config_generator=self.config_generator,
            retrieval_weight=self.retrieval_weight,
            generation_weight=self.generation_weight,
            project_dir=self.project_dir,
            corpus_data=self.corpus_data,
            qa_data=self.qa_data,
            use_llm_compressor_evaluator=use_llm_compressor_evaluator,
            llm_evaluator_model=self.llm_evaluator_model,
        )
        
        self.component_results = {}
        self.best_configs = {}
        self.component_dataframes = {}
        self.current_trial = 0
        self.trial_results = []
        
        self.trial_manager = TrialManager(self)
        self.smac_runner = SMACRunner(self)
        self.component_manager = ComponentManager(self)
        self.results_manager = ResultsManager(self)
        self.wandb_manager = WandBManager(self)
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
        
        self.current_component = None
        self.current_fixed_config = None
        self.component_trial_counter = 0
        self.component_trials = []
        self.component_best_scores = {}
        self.component_best_outputs = {}
    
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        self.component_manager.validate_all_components()
        
        all_results = {
            'study_name': self.study_name,
            'component_results': {},
            'best_configs': {},
            'optimization_time': 0,
            'component_order': [],
            'early_stopped': False,
            'retrieval_weight': self.retrieval_weight,
            'generation_weight': self.generation_weight
        }
        
        active_components = self.component_manager.get_active_components()
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

            if self.component_manager.should_skip_component(component):
                continue
            
            if self.use_wandb:
                self.wandb_manager.init_wandb_for_component(component, component_idx, len(active_components))
            
            component_result = self.optimize_component(
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
                self.wandb_manager.log_wandb_component_summary(component, component_result)
        
        all_results['optimization_time'] = time.time() - start_time
        all_results['early_stopped'] = early_stopped
        all_results['stopped_at_component'] = stopped_at_component
        all_results['total_trials'] = sum(
            comp.get('n_trials', 0) for comp in all_results['component_results'].values()
        )
        
        all_results['study_name'] = self.study_name
        
        if self.use_wandb:
            self.wandb_manager.log_wandb_final_summary(all_results)
        
        self.results_manager.save_final_results(all_results)
        self.results_manager.print_final_summary(all_results)
        
        return all_results
    
    def optimize_component(self, component: str, component_idx: int, 
                  active_components: List[str]) -> Dict[str, Any]:
        component_start_time = time.time()

        component_dir, smac_run_dir, seed_dir = self.smac_runner.ensure_smac_directories(component)
        
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

        n_trials = self.component_manager.calculate_component_trials(component, cs)

        self.current_component = component
        self.current_fixed_config = fixed_config
        self.component_trial_counter = 0
        self.component_trials = []
        self.component_detailed_metrics[component] = []
        self.component_best_scores = {}
        self.component_best_outputs = {}
        
        incumbent = self.smac_runner.run_smac_optimization(component, cs, fixed_config, n_trials)

        best_config = dict(incumbent) if incumbent else {}
        
        best_config = self.rag_processor.parse_component_config(component, best_config)

        best_trial = self.trial_manager.find_best_trial(component)
        
        if best_trial and best_trial.get('output_parquet') and os.path.exists(best_trial['output_parquet']):
            self.component_dataframes[component] = best_trial['output_parquet']
            print(f"[{component}] Best configuration found with score {best_trial['score']:.4f} and latency {best_trial.get('latency', 0.0):.2f}s")
            print(f"[{component}] Best output saved at: {best_trial['output_parquet']}")
            print(f"[{component}] Best config: {best_trial['config']}")
        
        if self.use_wandb:
            self.wandb_manager.log_wandb_component_table(component)
        
        result = self.results_manager.create_component_result(
            component, best_trial, best_config, fixed_config, 
            cs, component_start_time
        )

        self.results_manager.save_component_result(trial_component_dir, result)
        
        return result