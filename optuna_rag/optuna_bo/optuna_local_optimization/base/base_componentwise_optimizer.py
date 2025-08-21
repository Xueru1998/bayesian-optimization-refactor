import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union


from pipeline.config_manager import ConfigGenerator
from pipeline.rag_pipeline_runner import RAGPipelineRunner
from pipeline.search_space_calculator import SearchSpaceCalculator
from pipeline.utils import Utils
from optuna_rag.config_extractor import OptunaConfigExtractor
from ..helpers.component_pipeline_manager import ComponentPipelineManager
from ..helpers.component_search_space_builder import ComponentSearchSpaceBuilder
from ..helpers.component_grid_search_helper import ComponentGridSearchHelper


class BaseComponentwiseOptimizer:
    
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
        llm_evaluator_model: str = "gpt-4o",
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
        self.llm_evaluator_model = llm_evaluator_model
        
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
        
        self.component_results = {}
        self.best_configs = {}
        self.component_dataframes = {}
        self.current_trial = 0
        self.trial_results = []
        
        if self.resume_study:
            self._load_previous_state()
            self._load_global_trial_state()
        else:
            self.current_trial = self.global_trial_counter
        
        self.config_generator = ConfigGenerator(self.config_template)
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        
        search_type = 'grid' if self.optimizer == 'grid' else 'bo'
        self.config_extractor = OptunaConfigExtractor(self.config_generator, search_type=search_type)
        
        self.search_space_builder = ComponentSearchSpaceBuilder(self.config_generator, self.config_extractor)
        self.pipeline_manager = ComponentPipelineManager(self.config_generator, self.project_dir, self.corpus_data, self.qa_data)
        
        self.grid_search_helper = ComponentGridSearchHelper()
        
        self._setup_runner()
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
        
        self.current_component = None
        self.current_fixed_config = {}
        self.component_trial_counter = 0
        self.component_trials = []
    
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
    
    def _save_intermediate_state(self):
        summary_file = os.path.join(self.result_dir, "component_optimization_summary.json")
        
        state = {
            'study_name': self.study_name,
            'component_results': self.component_results,
            'best_configs': self.best_configs,
            'component_order': list(self.component_results.keys()),
            'retrieval_weight': self.retrieval_weight,
            'generation_weight': self.generation_weight
        }
        
        state_serializable = self._convert_numpy_types(state)
        
        with open(summary_file, 'w') as f:
            json.dump(state_serializable, f, indent=2)
        
        print(f"[CHECKPOINT] Saved intermediate state to {summary_file}")
    
    def _setup_runner(self):
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config('prompt_maker')
        self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()
        
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
            use_llm_compressor_evaluator=self.use_llm_compressor_evaluator,
            llm_evaluator_model=self.llm_evaluator_model
        )
    
    def _validate_all_components(self) -> Tuple[bool, List[str], Dict[str, int]]:
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
        
        for component in active_components:
            fixed_config = self._get_fixed_config(component, active_components)
            search_space = self.search_space_builder.build_component_search_space(component, fixed_config)
            
            if component == 'prompt_maker_generator':
                print(f"\n[DEBUG] {component} search space for {self.optimizer}:")
                for key, value in search_space.items():
                    print(f"  {key}: {value} (type: {type(value)})")
            
            if search_space:
                combinations = self.grid_search_helper.calculate_grid_search_combinations(
                    component, search_space, fixed_config
                )
            else:
                combinations = 0
            
            component_combinations[component] = combinations
        
        n_trials = self.n_trials_per_component if self.n_trials_per_component else 10
        
        if self.optimizer == 'grid':
            print(f"\n{'='*70}")
            for comp, combos in component_combinations.items():
                print(f"  - {comp}: {combos} combinations")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"COMPONENT VALIDATION PASSED")
            print(f"All components have sufficient search space for optimization:")
            
            for component in active_components:
                combos = component_combinations[component]
                
                if combos == 0:
                    print(f"  - {component}: No search space (will be skipped)")
                    continue
                    
                print(f"  - {component}: {combos} combinations")
                
                if combos < n_trials and combos > 0:
                    if component in ['query_expansion', 'retrieval', 'passage_reranker']:
                        print(f"    ⚠️  WARNING: Search space ({combos}) < n_trials ({n_trials})")
                        print(f"       May encounter duplicate sampling during optimization")
                    elif component == 'passage_filter':
                        print(f"    ℹ️  INFO: For filters, the actual search space is continuous")
                        print(f"       between threshold/percentile boundaries")
                    elif component == 'prompt_maker_generator':
                        print(f"    ℹ️  INFO: For generator, the actual search space is continuous")
                        print(f"       selecting from temperature and max token ranges")
                    else:
                        print(f"    ℹ️  INFO: Search space ({combos}) < n_trials ({n_trials})")
                        print(f"       Consider adding more parameter values or reducing n_trials")
            
            print(f"{'='*70}\n")
        
        return True, [], component_combinations
    
    def _get_fixed_config(self, component: str, active_components: List[str]) -> Dict[str, Any]:
        fixed_config = {}
                
        original_retriever_top_k = 10
        
        if 'query_expansion' in self.best_configs:
            qe_config = self.best_configs['query_expansion']
            print(f"[DEBUG] Query expansion best config: {qe_config}")
            
            if 'retriever_top_k' in qe_config:
                original_retriever_top_k = qe_config['retriever_top_k']
                
            if 'retrieval_method' in qe_config:
                fixed_config['retrieval_method'] = qe_config['retrieval_method']
            
            if qe_config.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in qe_config:
                fixed_config['bm25_tokenizer'] = qe_config['bm25_tokenizer']
            elif qe_config.get('retrieval_method') == 'vectordb' and 'vectordb_name' in qe_config:
                fixed_config['vectordb_name'] = qe_config['vectordb_name']
        
        if 'retrieval' in self.best_configs:
            retrieval_config = self.best_configs['retrieval']
            print(f"[DEBUG] Retrieval best config: {retrieval_config}")
            
            if 'retriever_top_k' in retrieval_config:
                original_retriever_top_k = retrieval_config['retriever_top_k']
                
            if 'retrieval_method' in retrieval_config:
                fixed_config['retrieval_method'] = retrieval_config['retrieval_method']
            
            if retrieval_config.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in retrieval_config:
                fixed_config['bm25_tokenizer'] = retrieval_config['bm25_tokenizer']
            elif retrieval_config.get('retrieval_method') == 'vectordb' and 'vectordb_name' in retrieval_config:
                fixed_config['vectordb_name'] = retrieval_config['vectordb_name']
        
        if 'retrieval_method' not in fixed_config and component not in ['query_expansion', 'retrieval']:
            retrieval_node = self.config_generator.extract_node_config("retrieval")
            if retrieval_node:
                for module in retrieval_node.get("modules", []):
                    method = module.get("module_type")
                    if method:
                        fixed_config['retrieval_method'] = method                        
                        if method == 'bm25':
                            tokenizer = module.get('tokenizer', module.get('bm25_tokenizer'))
                            if tokenizer:
                                if isinstance(tokenizer, list):
                                    fixed_config['bm25_tokenizer'] = tokenizer[0] if tokenizer else None
                                else:
                                    fixed_config['bm25_tokenizer'] = tokenizer
                        elif method == 'vectordb':
                            vectordb = module.get('vectordb_name', module.get('name'))
                            if vectordb:
                                if isinstance(vectordb, list):
                                    fixed_config['vectordb_name'] = vectordb[0] if vectordb else None
                                else:
                                    fixed_config['vectordb_name'] = vectordb
                        break
        
        if component not in ['query_expansion', 'retrieval']:
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
                    if 'retriever_top_k' in best_config:
                        fixed_config['retriever_top_k'] = best_config['retriever_top_k']
                    
                    if best_config.get('query_expansion_method') != 'pass_query_expansion':
                        for key in ['query_expansion_model', 'query_expansion_temperature', 'query_expansion_max_token']:
                            if key in best_config:
                                fixed_config[key] = best_config[key]
                
                elif prev_comp == 'retrieval':
                    if 'retrieval_method' in best_config:
                        fixed_config['retrieval_method'] = best_config['retrieval_method']
                    if 'retriever_top_k' in best_config:
                        fixed_config['retriever_top_k'] = best_config['retriever_top_k']
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
                    if 'reranker_model_name' in best_config:
                        fixed_config['reranker_model_name'] = best_config['reranker_model_name']
                
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
        
        return fixed_config
    
    def _calculate_component_trials(self, component: str, search_space: Dict[str, Any]) -> int:
        if self.n_trials_per_component:
            return self.n_trials_per_component
        
        return min(20, max(10, len(search_space) * 3))
    
    def _calculate_total_combinations(self, component: str, search_space: Dict[str, Any]) -> int:
        if self.optimizer == 'grid':
            return self.grid_search_helper.calculate_grid_search_combinations(
                component, search_space, self.current_fixed_config
            )
        else:
            fixed_config = self._get_fixed_config(component, self.COMPONENT_ORDER)
            combinations, _ = self.search_space_calculator.calculate_component_combinations(
                component, 
                search_space, 
                fixed_config, 
                self.best_configs
            )
            return combinations
    
    def _find_best_trial(self, trials: List[Dict]) -> Optional[Dict]:
        if not trials:
            return None

        score_groups = {}
        for trial in trials:
            if trial.get('status') != 'FAILED':
                score = trial['score']
                if score not in score_groups:
                    score_groups[score] = []
                score_groups[score].append(trial)
        
        if not score_groups:
            return None

        max_score = max(score_groups.keys())
        trials_with_max_score = score_groups[max_score]

        if len(trials_with_max_score) > 1:
            print(f"Found {len(trials_with_max_score)} trials with score {max_score:.4f}, selecting by latency")
            best_trial = min(trials_with_max_score, key=lambda t: t.get('latency', float('inf')))
            print(f"Selected trial {best_trial['trial_number']} with latency {best_trial['latency']:.2f}s")
            return best_trial
        else:
            return trials_with_max_score[0]
    
    def _create_trial_result(self, config_dict, score, latency, budget, budget_percentage, 
                        results, component, output_parquet_path):
        trial_number_in_component = int(self.component_trial_counter)
        
        component_config = {}
        
        if component == 'passage_compressor':
            compressor_params = [
                'passage_compressor_config', 'passage_compressor_method',
                'compressor_generator_module_type', 'compressor_model', 
                'compressor_llm', 'compressor_api_url', 'compressor_batch',
                'compressor_temperature', 'compressor_max_tokens'
            ]
            for param in compressor_params:
                if param in config_dict and param not in self.current_fixed_config:
                    component_config[param] = config_dict[param]
            
            if not component_config:
                if config_dict.get('passage_compressor_method') == 'pass_compressor':
                    component_config['passage_compressor_method'] = 'pass_compressor'
                elif config_dict.get('passage_compressor_config') == 'pass_compressor':
                    component_config['passage_compressor_method'] = 'pass_compressor'
        
        elif component == 'query_expansion':
            qe_params = [
                'query_expansion_config', 'query_expansion_method',
                'query_expansion_model', 'query_expansion_temperature',
                'query_expansion_max_token', 'retriever_top_k',
                'retrieval_method', 'bm25_tokenizer', 'vectordb_name'
            ]
            for param in qe_params:
                if param in config_dict and param not in self.current_fixed_config:
                    component_config[param] = config_dict[param]
        elif component == 'passage_reranker':
            reranker_params = [
                'passage_reranker_method', 'reranker_top_k',
                'reranker_model', 'reranker_model_name'
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
        if isinstance(obj, pd.DataFrame):
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                converted = self._convert_numpy_types(v)
                if converted is not None: 
                    result[k] = converted
            return result
        elif isinstance(obj, list):
            result = []
            for item in obj:
                converted = self._convert_numpy_types(item)
                if converted is not None:
                    result.append(converted)
            return result
        else:
            return obj
    
    def _clean_config(self, config: Dict[str, Any]):
        composite_params = [
            'query_expansion_config',
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
    
    def _save_final_results(self, results: Dict[str, Any]):
        summary_file = os.path.join(self.result_dir, "component_optimization_summary.json")
        
        results_serializable = self._convert_numpy_types(results)
        
        with open(summary_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        if not results.get('validation_failed', False):
            final_config = {}
            for component in results['component_order']:
                if component in results['best_configs'] and results['best_configs'][component]:
                    final_config.update(results['best_configs'][component])
            
            if final_config:
                final_config_file = os.path.join(self.result_dir, "final_best_config.yaml")
                final_config_serializable = self._convert_numpy_types(final_config)
                with open(final_config_file, 'w') as f:
                    yaml.dump(self.config_generator.generate_trial_config(final_config_serializable), f)
    
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
    
    def _get_component_combinations(self, component: str) -> int:
        fixed_config = self._get_fixed_config(component, self.COMPONENT_ORDER)
        search_space = self.search_space_builder.build_component_search_space(component, fixed_config)
        
        if not search_space:
            return 0
        
        combinations, _ = self.search_space_calculator.calculate_component_combinations(
            component, 
            search_space, 
            fixed_config, 
            self.best_configs
        )
        
        return combinations