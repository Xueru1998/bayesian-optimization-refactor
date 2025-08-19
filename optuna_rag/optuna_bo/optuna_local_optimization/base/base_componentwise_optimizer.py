import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import wandb

from pipeline.config_manager import ConfigGenerator
from pipeline.rag_pipeline_runner import RAGPipelineRunner
from pipeline.search_space_calculator import CombinationCalculator
from pipeline.utils import Utils
from pipeline.wandb_logger import WandBLogger
from optuna_rag.config_extractor import OptunaConfigExtractor
from ..helpers.component_pipeline_manager import ComponentPipelineManager
from ..helpers.component_search_space_builder import ComponentSearchSpaceBuilder


class BaseComponentwiseOptimizer:
    """Base class containing shared logic for both Grid Search and Bayesian Optimization"""
    
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
        walltime_limit_per_component: Optional[int] = None,
        n_workers: int = 1,
        seed: int = 42,
        early_stopping_threshold: float = 0.9,
        use_wandb: bool = True,
        wandb_project: str = "Component-wise Optuna Optimization",
        wandb_entity: Optional[str] = None,
        optimizer: str = "tpe",
        use_multi_objective: bool = False,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_config: Optional[Dict[str, Any]] = None,
        resume_study: bool = False,
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
        self.use_multi_objective = use_multi_objective

        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_config = llm_evaluator_config or {}
        
        self.component_detailed_metrics = {}
        
        self.study_name = study_name if study_name else f"componentwise_optuna_{int(time.time())}"
        self.resume_study = resume_study
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                f"componentwise_optuna_results_{self.study_name}"
            )
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.global_trial_counter = 0
        self.global_trial_state_file = os.path.join(self.result_dir, "global_trial_state.json")
        
        if self.resume_study:
            self._load_previous_state()
            self._load_global_trial_state()
        
        self.config_generator = ConfigGenerator(self.config_template)
        search_type = 'grid' if self.optimizer.lower() in ['grid', 'grid_search'] else 'bo'
        self.combination_calculator = CombinationCalculator(
            self.config_generator, 
            search_type=search_type
        )
        self.config_extractor = OptunaConfigExtractor(self.config_generator, search_type=search_type)
        self.search_space_builder = ComponentSearchSpaceBuilder(self.config_generator, self.config_extractor)
        self.pipeline_manager = ComponentPipelineManager(self.config_generator, self.project_dir, self.corpus_data, self.qa_data)

        self._setup_runner()
        
        self.component_results = {}
        self.best_configs = {}
        self.component_dataframes = {}
        self.current_trial = self.global_trial_counter
        self.trial_results = []
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
    
    def _load_global_trial_state(self):
        if os.path.exists(self.global_trial_state_file):
            with open(self.global_trial_state_file, 'r') as f:
                state = json.load(f)
                self.global_trial_counter = state.get('global_trial_counter', 0)
                self.current_trial = self.global_trial_counter
                print(f"[RESUME] Loaded global trial counter: {self.global_trial_counter}")
        else:
            self.global_trial_counter = 0
            self.current_trial = 0
    
    def _save_global_trial_state(self):
        state = {
            'global_trial_counter': self.global_trial_counter,
            'timestamp': time.time()
        }
        with open(self.global_trial_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _setup_runner(self):
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config('prompt_maker')
        self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()
        
        early_stopping_thresholds = {
            'retrieval': -1.0,
            'query_expansion': -1.0,
            'reranker': -1.0,
            'filter': -1.0,
            'compressor': -1.0
        }
        
        self.pipeline_runner = RAGPipelineRunner(
            config_generator=self.config_generator,
            retrieval_metrics=self.retrieval_metrics,
            filter_metrics=self.filter_metrics,
            compressor_metrics=self.compressor_metrics,
            generation_metrics=self.generation_metrics,
            prompt_maker_metrics=self.prompt_maker_metrics,
            query_expansion_metrics=self.query_expansion_metrics,
            reranker_metrics=self.reranker_metrics,
            retrieval_weight=self.retrieval_weight,
            generation_weight=self.generation_weight,
            use_ragas=False,
            ragas_config=None,
            use_llm_evaluator=self.use_llm_compressor_evaluator,
            llm_evaluator_config=self.llm_evaluator_config,
            early_stopping_thresholds=early_stopping_thresholds
        )
    
    def _validate_all_components(self) -> Tuple[bool, List[str], Dict[str, int], str]:
        active_components = []
        has_active_query_expansion = False
        
        for comp in self.COMPONENT_ORDER:
            if comp == 'query_expansion' and self.config_generator.node_exists(comp):
                qe_config = self.config_generator.extract_node_config("query_expansion")
                qe_methods = []
                for module in qe_config.get("modules", []):
                    method = module.get("module_type")
                    if method and method != "pass_query_expansion":
                        qe_methods.append(method)
                if qe_methods:
                    has_active_query_expansion = True
                    active_components.append(comp)
            elif comp == 'retrieval' and has_active_query_expansion:
                continue
            elif comp == 'prompt_maker_generator':
                if self.config_generator.node_exists('prompt_maker') or self.config_generator.node_exists('generator'):
                    active_components.append(comp)
            elif self.config_generator.node_exists(comp):
                active_components.append(comp)
        
        component_combinations = {}
        combination_note = ""
        
        for component in active_components:
            fixed_config = self._get_fixed_config(component, active_components)
            search_space = self.search_space_builder.build_component_search_space(component, fixed_config)
            
            if search_space:
                combinations, note = self.combination_calculator.calculate_component_combinations(
                    component, 
                    search_space, 
                    fixed_config, 
                    self.best_configs
                )
                combination_note = note
            else:
                combinations = 0
            
            component_combinations[component] = combinations
        
        return True, [], component_combinations, combination_note
    
    def _load_previous_state(self):
        summary_file = os.path.join(self.result_dir, "component_optimization_summary.json")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                previous_results = json.load(f)

            self.best_configs = previous_results.get('best_configs', {})
            self.component_results = previous_results.get('component_results', {})

            for component, result in previous_results.get('component_results', {}).items():
                if 'best_output_path' in result and os.path.exists(result['best_output_path']):
                    self.component_dataframes[component] = result['best_output_path']
            
            print(f"\n[RESUME] Loaded previous state from {summary_file}")
            print(f"[RESUME] Previously completed components: {list(self.best_configs.keys())}")
        else:
            print(f"\n[RESUME] No previous state found. Starting fresh optimization.")
            self.resume_study = False
    
    def _get_fixed_config(self, component: str, active_components: List[str]) -> Dict[str, Any]:
        fixed_config = {}
        
        if component not in ['query_expansion', 'retrieval']:
            original_retriever_top_k = 10
            
            if 'query_expansion' in self.best_configs:
                qe_config = self.best_configs['query_expansion']
                if 'retriever_top_k' in qe_config:
                    original_retriever_top_k = qe_config['retriever_top_k']
            
            if 'retrieval' in self.best_configs:
                retrieval_config = self.best_configs['retrieval']
                if 'retriever_top_k' in retrieval_config:
                    original_retriever_top_k = retrieval_config['retriever_top_k']

                if 'retrieval_method' in retrieval_config:
                    fixed_config['retrieval_method'] = retrieval_config['retrieval_method']
                if retrieval_config.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in retrieval_config:
                    fixed_config['bm25_tokenizer'] = retrieval_config['bm25_tokenizer']
                elif retrieval_config.get('retrieval_method') == 'vectordb' and 'vectordb_name' in retrieval_config:
                    fixed_config['vectordb_name'] = retrieval_config['vectordb_name']

            if component in ['passage_compressor', 'prompt_maker_generator']:
                if 'passage_filter' in self.best_configs:
                    filter_config = self.best_configs['passage_filter']
                    if filter_config.get('passage_filter_method') != 'pass_passage_filter':
                        fixed_config['retriever_top_k'] = original_retriever_top_k
                    else:
                        if 'passage_reranker' in self.best_configs:
                            reranker_config = self.best_configs['passage_reranker']
                            if reranker_config.get('passage_reranker_method') != 'pass_reranker' and 'reranker_top_k' in reranker_config:
                                fixed_config['retriever_top_k'] = original_retriever_top_k
                                fixed_config['effective_top_k'] = reranker_config['reranker_top_k']
                            else:
                                fixed_config['retriever_top_k'] = original_retriever_top_k
                        else:
                            fixed_config['retriever_top_k'] = original_retriever_top_k
                else:
                    if 'passage_reranker' in self.best_configs:
                        reranker_config = self.best_configs['passage_reranker']
                        if reranker_config.get('passage_reranker_method') != 'pass_reranker' and 'reranker_top_k' in reranker_config:
                            fixed_config['retriever_top_k'] = original_retriever_top_k
                            fixed_config['effective_top_k'] = reranker_config['reranker_top_k']
                        else:
                            fixed_config['retriever_top_k'] = original_retriever_top_k
                    else:
                        fixed_config['retriever_top_k'] = original_retriever_top_k
            else:
                fixed_config['retriever_top_k'] = original_retriever_top_k
        
        current_component_position = self.COMPONENT_ORDER.index(component) if component in self.COMPONENT_ORDER else len(self.COMPONENT_ORDER)
        
        for prev_comp in self.COMPONENT_ORDER[:current_component_position]:
            if prev_comp in self.best_configs:
                best_config = self.best_configs[prev_comp]
                
                if prev_comp == 'query_expansion':
                    if 'query_expansion_method' in best_config:
                        fixed_config['query_expansion_method'] = best_config['query_expansion_method']
                    
                    if 'retrieval_method' in best_config:
                        fixed_config['retrieval_method'] = best_config['retrieval_method']
                    if 'bm25_tokenizer' in best_config:
                        fixed_config['bm25_tokenizer'] = best_config['bm25_tokenizer']
                    if 'vectordb_name' in best_config:
                        fixed_config['vectordb_name'] = best_config['vectordb_name']
                    
                    if best_config.get('query_expansion_method') != 'pass_query_expansion':
                        for key in ['query_expansion_model', 'query_expansion_temperature', 'query_expansion_max_token',
                                    'query_expansion_api_url', 'query_expansion_llm']:
                            if key in best_config:
                                fixed_config[key] = best_config[key]
                
                elif prev_comp == 'retrieval':
                    if 'retrieval_method' in best_config:
                        fixed_config['retrieval_method'] = best_config['retrieval_method']
                    if best_config.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in best_config:
                        fixed_config['bm25_tokenizer'] = best_config['bm25_tokenizer']
                    elif best_config.get('retrieval_method') == 'vectordb' and 'vectordb_name' in best_config:
                        fixed_config['vectordb_name'] = best_config['vectordb_name']
                
                elif prev_comp == 'passage_reranker':
                    if 'passage_reranker_method' in best_config:
                        fixed_config['passage_reranker_method'] = best_config['passage_reranker_method']
                    if 'reranker_top_k' in best_config:
                        fixed_config['reranker_top_k'] = best_config['reranker_top_k']
                    if 'reranker_model' in best_config:
                        fixed_config['reranker_model'] = best_config['reranker_model']
                    if 'reranker_api_url' in best_config:
                        fixed_config['reranker_api_url'] = best_config['reranker_api_url']
                    
                    if best_config.get('passage_reranker_method') == 'sap_api':
                        reranker_config = self.config_generator.extract_node_config("passage_reranker")
                        for module in reranker_config.get("modules", []):
                            if module.get("module_type") == "sap_api":
                                fixed_config['reranker_api_url'] = module.get('api_url')
                                break
                
                elif prev_comp == 'passage_filter':
                    if 'passage_filter_method' in best_config:
                        fixed_config['passage_filter_method'] = best_config['passage_filter_method']
                    for key, value in best_config.items():
                        if key in ['threshold', 'percentile'] or key.startswith('threshold') or key.startswith('percentile'):
                            fixed_config[key] = value
                
                elif prev_comp == 'passage_compressor':
                    if 'passage_compressor_method' in best_config:
                        fixed_config['passage_compressor_method'] = best_config['passage_compressor_method']
                    if 'compressor_generator_module_type' in best_config:
                        fixed_config['compressor_generator_module_type'] = best_config['compressor_generator_module_type']
                    if 'compressor_llm' in best_config:
                        fixed_config['compressor_llm'] = best_config['compressor_llm']
                    if 'compressor_model' in best_config:
                        fixed_config['compressor_model'] = best_config['compressor_model']
                    if 'compressor_api_url' in best_config:
                        fixed_config['compressor_api_url'] = best_config['compressor_api_url']
                    if 'compressor_batch' in best_config:
                        fixed_config['compressor_batch'] = best_config['compressor_batch']
                    if 'compressor_temperature' in best_config:
                        fixed_config['compressor_temperature'] = best_config['compressor_temperature']
                    if 'compressor_max_tokens' in best_config:
                        fixed_config['compressor_max_tokens'] = best_config['compressor_max_tokens']
                    
                    for key in ['lexrank_compression_ratio', 'lexrank_threshold', 'lexrank_damping', 'lexrank_max_iterations',
                            'spacy_compression_ratio', 'spacy_model']:
                        if key in best_config:
                            fixed_config[key] = best_config[key]
        
        return fixed_config
    
    def _calculate_component_trials(self, component: str, search_space: Dict[str, Any]) -> int:
        if self.n_trials_per_component:
            return self.n_trials_per_component
        
        return min(20, max(10, len(search_space) * 3))
    
    def _create_trial_result(self, config_dict, score, latency, budget, budget_percentage, 
            results, component, output_parquet_path):
        trial_number_in_component = int(self.component_trial_counter)
        
        component_config = {}
        
        if component == 'passage_compressor':
            compressor_params = [
                'passage_compressor_config', 'passage_compressor_method',
                'compressor_generator_module_type', 'compressor_model', 
                'compressor_llm', 'compressor_api_url', 'compressor_batch',
                'compressor_temperature', 'compressor_max_tokens',
                'lexrank_compression_ratio', 'lexrank_threshold', 'lexrank_damping', 'lexrank_max_iterations',
                'spacy_compression_ratio', 'spacy_model'
            ]
            for param in compressor_params:
                if param in config_dict and param not in self.current_fixed_config:
                    component_config[param] = config_dict[param]
            
            if not component_config and 'passage_compressor_method' in config_dict:
                component_config['passage_compressor_method'] = config_dict['passage_compressor_method']
        
        elif component == 'query_expansion':
            qe_params = [
                'query_expansion_config', 'query_expansion_method',
                'query_expansion_model', 'query_expansion_temperature',
                'query_expansion_max_token', 'retriever_top_k',
                'retrieval_method', 'bm25_tokenizer', 'vectordb_name',
                'query_expansion_api_url', 'query_expansion_llm'
            ]
            for param in qe_params:
                if param in config_dict and param not in self.current_fixed_config:
                    component_config[param] = config_dict[param]
                    
        elif component == 'passage_reranker':
            reranker_params = [
                'passage_reranker_method', 'reranker_top_k',
                'reranker_model', 'reranker_model_name', 'reranker_api_url'
            ]
            for param in reranker_params:
                if param in config_dict and param not in self.current_fixed_config:
                    component_config[param] = config_dict[param]
                    
        elif component == 'passage_filter':
            filter_params = [
                'passage_filter_method', 'threshold', 'percentile'
            ]
            for param in filter_params:
                if param in config_dict and param not in self.current_fixed_config:
                    component_config[param] = config_dict[param]
                    
        elif component == 'prompt_maker_generator':
            generator_params = [
                'prompt_maker_method', 'prompt_template_idx',
                'generator_config', 'generator_model', 'generator_temperature',
                'generator_module_type', 'generator_llm', 'generator_api_url'
            ]
            for param in generator_params:
                if param in config_dict and param not in self.current_fixed_config:
                    component_config[param] = config_dict[param]
        else:
            for key, value in config_dict.items():
                if key not in self.current_fixed_config:
                    component_config[key] = value
        
        trial_result = {
            "trial": trial_number_in_component, 
            "trial_number": trial_number_in_component,  
            "global_trial_number": int(self.current_trial),  
            "component": component,
            "config": self._convert_numpy_types(component_config),
            "full_config": self._convert_numpy_types(config_dict),
            "score": float(score),
            "latency": float(latency),
            "execution_time_s": float(latency),
            "budget": int(budget),
            "budget_percentage": float(budget_percentage),
            "status": "COMPLETE",
            "results": results,
            "output_parquet": output_parquet_path,
            "timestamp": float(time.time())
        }

        for k, v in results.items():
            if k.endswith('_score') or k.endswith('_metrics'):
                trial_result[k] = self._convert_numpy_types(v)

        if 'retriever_top_k' in config_dict:
            trial_result['retriever_top_k'] = config_dict['retriever_top_k']
        elif 'retriever_top_k' in self.current_fixed_config:
            trial_result['retriever_top_k'] = self.current_fixed_config['retriever_top_k']

        if 'effective_top_k' in self.current_fixed_config:
            trial_result['effective_top_k'] = self.current_fixed_config['effective_top_k']
        
        return trial_result
    
    def _convert_numpy_types(self, obj):
        return Utils.convert_numpy_types(obj)
    
    def _find_best_trial(self, trials) -> Optional[Dict]:
        return Utils.find_best_trial_from_component(trials, self.current_component)
    
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
                if component in results['component_results']:
                    comp_result = results['component_results'][component]
                    print(f"\n{component.upper()}:")
                    print(f"  Best score: {comp_result['best_score']:.4f}")
                    print(f"  Best config: {comp_result['best_config']}")
                    print(f"  Trials run: {comp_result['n_trials']}")
                else:
                    print(f"\n{component.upper()}:")
                    print(f"  Skipped (no optimization needed)")
                    if component in results.get('best_configs', {}):
                        print(f"  Config: {results['best_configs'][component]}")
    
    def _clean_config(self, config: Dict[str, Any]):
        composite_params = [
            'query_expansion_config',
            'passage_compressor_config',
            'generator_config'
        ]
        
        for param in composite_params:
            config.pop(param, None)
        
        config.pop('compressor_bearer_token', None)
        config.pop('generator_bearer_token', None)
        config.pop('query_expansion_bearer_token', None)
        
        if config.get('passage_compressor_method') in ['refine', 'tree_summarize']:
            config.pop('compressor_temperature', None)
            config.pop('compressor_max_tokens', None)

        for key in sorted(config.keys()):
            if 'api_url' in key:
                print(f"  {key}: {config[key]}")
    
    def _parse_composite_configs(self, component: str, trial_config: Dict[str, Any], 
                            full_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        trial_config = trial_config.copy()
        full_config = full_config.copy()
        
        if component == 'query_expansion' and 'query_expansion_config' in trial_config:
            qe_config_str = trial_config['query_expansion_config']
            parts = qe_config_str.split('::', 2)
            
            if len(parts) >= 3:
                method, gen_type, model = parts
                trial_config['query_expansion_method'] = method
                trial_config['query_expansion_generator_module_type'] = gen_type
                trial_config['query_expansion_model'] = model
                full_config['query_expansion_method'] = method
                full_config['query_expansion_generator_module_type'] = gen_type
                full_config['query_expansion_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                for gen_config in unified_params.get('generator_configs', []):
                    if (gen_config['method'] == method and 
                        gen_config['generator_module_type'] == gen_type and 
                        model in gen_config['models']):
                        if gen_type == 'sap_api':
                            trial_config['query_expansion_api_url'] = gen_config.get('api_url')
                            trial_config['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                            full_config['query_expansion_api_url'] = gen_config.get('api_url')
                            full_config['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                        else:
                            trial_config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                            full_config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                        break
            elif qe_config_str == 'pass_query_expansion':
                trial_config['query_expansion_method'] = 'pass_query_expansion'
                full_config['query_expansion_method'] = 'pass_query_expansion'
            
            trial_config.pop('query_expansion_config', None)
            full_config.pop('query_expansion_config', None)
        
        if component == 'retrieval' and 'retrieval_config' in trial_config:
            parsed_config = self.pipeline_runner._parse_retrieval_config(trial_config['retrieval_config'])
            trial_config.update(parsed_config)
            full_config.update(parsed_config)
            trial_config.pop('retrieval_config', None)
            full_config.pop('retrieval_config', None)
        
        if component == 'passage_reranker':
            if 'sap_api_models' in trial_config:
                trial_config['reranker_model'] = trial_config['sap_api_models']
                full_config['reranker_model'] = trial_config['sap_api_models']
                trial_config.pop('sap_api_models', None)
                full_config.pop('sap_api_models', None)
            
            if trial_config.get('passage_reranker_method') == 'sap_api' or full_config.get('passage_reranker_method') == 'sap_api':
                reranker_config = self.config_generator.extract_node_config("passage_reranker")
                api_url_found = False
                
                for module in reranker_config.get("modules", []):
                    if module.get("module_type") == "sap_api":
                        api_url = module.get('api-url') or module.get('api_url')
                        if api_url:
                            trial_config['reranker_api_url'] = api_url
                            full_config['reranker_api_url'] = api_url
                            api_url_found = True
                            print(f"[DEBUG] Set reranker_api_url from config: {api_url}")
                        break
                
                if not api_url_found:
                    unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
                    api_endpoints = unified_params.get('api_endpoints', {})
                    if 'sap_api' in api_endpoints:
                        api_url = api_endpoints['sap_api']
                        trial_config['reranker_api_url'] = api_url
                        full_config['reranker_api_url'] = api_url
                        print(f"[DEBUG] Set reranker_api_url from unified params: {api_url}")
        
        if component == 'passage_compressor':
            if 'lexrank_compression_ratio' in trial_config or 'lexrank_threshold' in trial_config:
                trial_config['passage_compressor_method'] = 'lexrank'
                full_config['passage_compressor_method'] = 'lexrank'
                print("[DEBUG] Set passage_compressor_method to 'lexrank' based on lexrank parameters")
            
            elif 'spacy_compression_ratio' in trial_config or 'spacy_model' in trial_config:
                trial_config['passage_compressor_method'] = 'spacy'
                full_config['passage_compressor_method'] = 'spacy'
                print("[DEBUG] Set passage_compressor_method to 'spacy' based on spacy parameters")
            
            elif 'passage_compressor_config' in trial_config:
                pc_config_str = trial_config['passage_compressor_config']
                
                if pc_config_str == 'pass_compressor':
                    trial_config['passage_compressor_method'] = 'pass_compressor'
                    full_config['passage_compressor_method'] = 'pass_compressor'
                
                elif pc_config_str == 'lexrank':
                    trial_config['passage_compressor_method'] = 'lexrank'
                    full_config['passage_compressor_method'] = 'lexrank'
                
                elif pc_config_str.startswith('spacy::'):
                    parts = pc_config_str.split('::', 1)
                    trial_config['passage_compressor_method'] = 'spacy'
                    full_config['passage_compressor_method'] = 'spacy'
                    if len(parts) > 1:
                        trial_config['spacy_model'] = parts[1]
                        full_config['spacy_model'] = parts[1]
                
                elif pc_config_str in ['sentence_rank', 'keyword_extraction', 'query_focused']:
                    trial_config['passage_compressor_method'] = pc_config_str
                    full_config['passage_compressor_method'] = pc_config_str
                
                else:
                    parts = pc_config_str.split('::', 3)
                    if len(parts) >= 2:
                        method = parts[0]
                        trial_config['passage_compressor_method'] = method
                        full_config['passage_compressor_method'] = method
                        
                        if method in ['tree_summarize', 'refine'] and len(parts) >= 3:
                            gen_type = parts[1]
                            model = parts[2]
                            trial_config['compressor_generator_module_type'] = gen_type
                            trial_config['compressor_model'] = model
                            full_config['compressor_generator_module_type'] = gen_type
                            full_config['compressor_model'] = model
                            
                            compressor_config = self.config_generator.extract_node_config("passage_compressor")
                            for module in compressor_config.get("modules", []):
                                if module.get("module_type") == method:
                                    if 'generator' in module:
                                        for gen_module in module.get("generator", {}).get("modules", []):
                                            if gen_module.get("module_type") == gen_type:
                                                models = gen_module.get('model', gen_module.get('llm', []))
                                                if not isinstance(models, list):
                                                    models = [models]
                                                if model in models:
                                                    if gen_type == 'sap_api':
                                                        api_url = gen_module.get('api_url') or gen_module.get('api-url')
                                                        if api_url:
                                                            trial_config['compressor_api_url'] = api_url
                                                            full_config['compressor_api_url'] = api_url
                                                            print(f"[DEBUG] Set compressor_api_url from config: {api_url}")
                                                        trial_config['compressor_llm'] = gen_module.get('llm', 'mistralai')
                                                        full_config['compressor_llm'] = gen_module.get('llm', 'mistralai')
                                                    else:
                                                        trial_config['compressor_llm'] = gen_module.get('llm', 'openai')
                                                        full_config['compressor_llm'] = gen_module.get('llm', 'openai')
                                                    break
                                    else:
                                        if gen_type == 'sap_api':
                                            api_url = module.get('api_url') or module.get('api-url')
                                            if api_url:
                                                trial_config['compressor_api_url'] = api_url
                                                full_config['compressor_api_url'] = api_url
                                            trial_config['compressor_llm'] = module.get('llm', 'mistralai')
                                            full_config['compressor_llm'] = module.get('llm', 'mistralai')
                                        else:
                                            trial_config['compressor_llm'] = module.get('llm', 'openai')
                                            full_config['compressor_llm'] = module.get('llm', 'openai')
                                        
                                        if 'model' in module:
                                            trial_config['compressor_model'] = module['model']
                                            full_config['compressor_model'] = module['model']
                                    break
                
                trial_config.pop('passage_compressor_config', None)
                full_config.pop('passage_compressor_config', None)
        
        if component == 'prompt_maker_generator' and 'generator_config' in trial_config:
            gen_config_str = trial_config['generator_config']
            parts = gen_config_str.split('::', 2)
            if len(parts) >= 2:
                gen_type = parts[0]
                model = parts[1]
                trial_config['generator_module_type'] = gen_type
                trial_config['generator_model'] = model
                full_config['generator_module_type'] = gen_type
                full_config['generator_model'] = model
                
                gen_node_config = self.config_generator.extract_node_config("generator")
                for module in gen_node_config.get("modules", []):
                    if module.get("module_type") == gen_type:
                        models = module.get('model', module.get('llm', []))
                        if not isinstance(models, list):
                            models = [models]
                        if model in models:
                            if gen_type == 'sap_api':
                                api_url = module.get('api_url') or module.get('api-url')
                                if api_url:
                                    trial_config['generator_api_url'] = api_url
                                    full_config['generator_api_url'] = api_url
                                    print(f"[DEBUG] Set generator_api_url from config: {api_url}")
                                trial_config['generator_llm'] = module.get('llm', 'mistralai')
                                full_config['generator_llm'] = module.get('llm', 'mistralai')
                            else:
                                trial_config['generator_llm'] = module.get('llm', 'openai')
                                full_config['generator_llm'] = module.get('llm', 'openai')
                            break
            
            trial_config.pop('generator_config', None)
            full_config.pop('generator_config', None)
        
        return trial_config, full_config

    def _parse_best_config_composite(self, component: str, best_config: Dict[str, Any]) -> Dict[str, Any]:
        final_config = best_config.copy()
        
        if component == 'query_expansion' and 'query_expansion_config' in final_config:
            qe_config_str = final_config['query_expansion_config']
            if qe_config_str != 'pass_query_expansion':
                parts = qe_config_str.split('::', 2)
                if len(parts) >= 3:
                    method, gen_type, model = parts
                    final_config['query_expansion_method'] = method
                    final_config['query_expansion_generator_module_type'] = gen_type
                    final_config['query_expansion_model'] = model
                    
                    unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                    for gen_config in unified_params.get('generator_configs', []):
                        if (gen_config['method'] == method and 
                            gen_config['generator_module_type'] == gen_type and 
                            model in gen_config['models']):
                            if gen_type == 'sap_api':
                                final_config['query_expansion_api_url'] = gen_config.get('api_url')
                                final_config['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                            else:
                                final_config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                            break
                    
                    final_config.pop('query_expansion_config', None)
            else:
                final_config['query_expansion_method'] = 'pass_query_expansion'
                final_config.pop('query_expansion_config', None)
        
        if component == 'retrieval' and 'retrieval_config' in final_config:
            parsed_config = self.pipeline_runner._parse_retrieval_config(final_config['retrieval_config'])
            final_config.update(parsed_config)
            final_config.pop('retrieval_config', None)
        
        if component == 'passage_compressor' and 'passage_compressor_config' in final_config:
            pc_config_str = final_config['passage_compressor_config']
            if pc_config_str != 'pass_compressor':
                parts = pc_config_str.split('::', 3)
                if len(parts) >= 2:
                    method = parts[0]
                    final_config['passage_compressor_method'] = method
                    
                    if method in ['tree_summarize', 'refine'] and len(parts) >= 3:
                        gen_type = parts[1]
                        model = parts[2]
                        final_config['compressor_generator_module_type'] = gen_type
                        final_config['compressor_model'] = model
                        
                        unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                        for gen_config in unified_params.get('generator_configs', []):
                            if (gen_config['method'] == method and 
                                gen_config['generator_module_type'] == gen_type and 
                                model in gen_config['models']):
                                if gen_type == 'sap_api':
                                    final_config['compressor_api_url'] = gen_config.get('api_url')
                                    final_config['compressor_llm'] = gen_config.get('llm', 'mistralai')
                                else:
                                    final_config['compressor_llm'] = gen_config.get('llm', 'openai')
                                break
                    
                    final_config.pop('passage_compressor_config', None)
            else:
                final_config['passage_compressor_method'] = 'pass_compressor'
                final_config.pop('passage_compressor_config', None)
        
        if component == 'prompt_maker_generator' and 'generator_config' in final_config:
            gen_config_str = final_config['generator_config']
            parts = gen_config_str.split('::', 2)
            if len(parts) >= 2:
                gen_type = parts[0]
                model = parts[1]
                final_config['generator_module_type'] = gen_type
                final_config['generator_model'] = model
                
                gen_node_config = self.config_generator.extract_node_config("generator")
                for module in gen_node_config.get("modules", []):
                    if module.get("module_type") == gen_type:
                        models = module.get('model', module.get('llm', []))
                        if not isinstance(models, list):
                            models = [models]
                        if model in models:
                            if gen_type == 'sap_api':
                                final_config['generator_api_url'] = module.get('api_url')
                                final_config['generator_llm'] = module.get('llm', 'mistralai')
                            else:
                                final_config['generator_llm'] = module.get('llm', 'openai')
                            break
            
            final_config.pop('generator_config', None)
        
        return final_config