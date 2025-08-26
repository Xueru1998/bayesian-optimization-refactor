import itertools
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import yaml
import tempfile
import shutil
import json
from pipeline.config_manager import ConfigGenerator
from pipeline.utils import Utils
from pipeline.search_space_calculator import SearchSpaceCalculator
from optuna_rag.config_extractor import OptunaConfigExtractor




class GlobalGridSearchOptimizer:
    def __init__(
        self,
        config_path: str,
        qa_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        project_dir: str,
        sample_percentage: float = 0.1,
        cpu_per_trial: int = 4,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        use_wandb: bool = True,
        wandb_project: str = "Grid Search & AutoRAG",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        use_ragas: bool = False,
        ragas_llm_model: str = "gpt-4o-mini",
        ragas_embedding_model: str = "text-embedding-ada-002",
        ragas_metrics: Optional[Dict[str, List[str]]] = None,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o",
        max_trials: Optional[int] = None,
        grid_reduction_factor: float = 0.1
    ):
        self.start_time = time.time()
        
        self.project_root = Utils.find_project_root()
        self.config_path = Utils.get_centralized_config_path(config_path)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        print(f"Grid Search using config file: {self.config_path}")
        
        self.qa_df = qa_df
        self.corpus_df = corpus_df
        
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        print(f"Grid Search using project directory: {self.project_dir}")
        
        with open(self.config_path, 'r') as f:
            self.config_template = yaml.safe_load(f)
        
        self.config_generator = ConfigGenerator(self.config_template)
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        
        self.sample_percentage = sample_percentage
        self.cpu_per_trial = cpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.study_name = study_name if study_name else f"Grid_search_rag_opt_{int(time.time())}"
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name or self.study_name
        self.max_trials = max_trials
        self.grid_reduction_factor = grid_reduction_factor
        
        self.use_ragas = use_ragas
        self.ragas_llm_model = ragas_llm_model
        self.ragas_embedding_model = ragas_embedding_model
        
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
        if ragas_metrics is None and use_ragas:
            self.ragas_metrics = {
                'retrieval': ['context_precision', 'context_recall'],
                'generation': ['answer_relevancy', 'faithfulness', 'factual_correctness', 'semantic_similarity']
            }
        else:
            self.ragas_metrics = ragas_metrics or {}
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
        
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.best_score = {"value": 0.0, "config": None, "latency": float('inf')}
        self.best_latency = {"value": float('inf'), "config": None, "score": 0.0}
        self.all_trials = []
        
        self.config_extractor = OptunaConfigExtractor(self.config_generator, search_type='grid')
        self.search_space = self.config_extractor.extract_search_space()
        
        self.grid_combinations = self._generate_grid_combinations()
        self.current_trial_index = 0
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        from pipeline.pipeline_runner.rag_pipeline_runner import RAGPipelineRunner
        
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.query_expansion_metrics = []
        if self.config_generator.node_exists("query_expansion"):
            self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()
        
        self.filter_metrics = []
        if self.config_generator.node_exists("passage_filter"):
            self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        
        self.compressor_metrics = []
        if self.config_generator.node_exists("passage_compressor"):
            self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        
        self.reranker_metrics = []
        if self.config_generator.node_exists("passage_reranker"):
            self.reranker_metrics = self.config_generator.extract_metrics_from_config(node_type='passage_reranker')
        
        self.generation_metrics = []
        if self.config_generator.node_exists("generator"):
            self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        
        self.prompt_maker_metrics = []
        if self.config_generator.node_exists("prompt_maker"):
            self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config(node_type='prompt_maker')
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_df, self.qa_df)
        
        if self.use_ragas:
            ragas_config = {
                'llm_model': self.ragas_llm_model,
                'embedding_model': self.ragas_embedding_model,
                'retrieval_metrics': self.ragas_metrics.get('retrieval', []),
                'generation_metrics': self.ragas_metrics.get('generation', [])
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
                use_ragas=True,
                ragas_config=ragas_config,
                use_llm_compressor_evaluator=self.use_llm_compressor_evaluator,
                llm_evaluator_model=self.llm_evaluator_model,
                early_stopping_thresholds=None
            )
        else:
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
                use_llm_compressor_evaluator=self.use_llm_compressor_evaluator,
                llm_evaluator_model=self.llm_evaluator_model,
                early_stopping_thresholds=None
            )
        
        self._print_initialization_summary()
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        from optuna_rag.optuna_bo.optuna_local_optimization.helpers.component_grid_search_helper import ComponentGridSearchHelper
        
        grid_helper = ComponentGridSearchHelper()
        
        grid_space = self._convert_search_space_to_grid()
        
        all_combinations = self._generate_valid_combinations(grid_space)
        
        if self.max_trials and len(all_combinations) > self.max_trials:
            all_combinations = self._reduce_grid_combinations(all_combinations, self.max_trials)
        
        return all_combinations
    
    def _convert_search_space_to_grid(self) -> Dict[str, List[Any]]:
        grid_space = {}
        
        for param_name, param_spec in self.search_space.items():
            if isinstance(param_spec, list):
                grid_space[param_name] = param_spec
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                if 'top_k' in param_name or 'batch' in param_name or 'iterations' in param_name:
                    min_val, max_val = param_spec
                    if isinstance(min_val, int):
                        step = max(1, (max_val - min_val) // 3)
                        grid_space[param_name] = list(range(min_val, max_val + 1, step))
                    else:
                        grid_space[param_name] = [min_val, (min_val + max_val) / 2, max_val]
                elif 'temperature' in param_name or 'threshold' in param_name or 'ratio' in param_name:
                    min_val, max_val = param_spec
                    grid_space[param_name] = [min_val, (min_val + max_val) / 2, max_val]
                else:
                    grid_space[param_name] = [param_spec[0], param_spec[1]]
            else:
                grid_space[param_name] = [param_spec]
        
        return grid_space
    
    def _generate_valid_combinations(self, grid_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        valid_combinations = []
        
        qe_methods = grid_space.get('query_expansion_method', ['pass_query_expansion'])
        retrieval_methods = grid_space.get('retrieval_method', [])
        bm25_tokenizers = grid_space.get('bm25_tokenizer', [])
        vectordb_names = grid_space.get('vectordb_name', [])
        retriever_top_k_values = grid_space.get('retriever_top_k', [10])
        
        for qe_method in qe_methods:
            if qe_method == 'pass_query_expansion':
                for retrieval_method in retrieval_methods:
                    for top_k in retriever_top_k_values:
                        base_retrieval = {
                            'query_expansion_method': qe_method,
                            'retrieval_method': retrieval_method,
                            'retriever_top_k': top_k
                        }
                        
                        if retrieval_method == 'bm25':
                            for tokenizer in bm25_tokenizers:
                                config = base_retrieval.copy()
                                config['bm25_tokenizer'] = tokenizer
                                self._add_downstream_components(config, grid_space, valid_combinations, top_k)
                        elif retrieval_method == 'vectordb':
                            for vdb_name in vectordb_names:
                                config = base_retrieval.copy()
                                config['vectordb_name'] = vdb_name
                                self._add_downstream_components(config, grid_space, valid_combinations, top_k)
                        else:
                            self._add_downstream_components(base_retrieval, grid_space, valid_combinations, top_k)
            else:
                for top_k in retriever_top_k_values:
                    base_qe = {
                        'query_expansion_method': qe_method,
                        'retriever_top_k': top_k
                    }
                    
                    if 'query_expansion_model' in grid_space:
                        for model in grid_space['query_expansion_model']:
                            config = base_qe.copy()
                            config['query_expansion_model'] = model
                            
                            if qe_method == 'hyde' and 'query_expansion_max_token' in grid_space:
                                for max_token in grid_space['query_expansion_max_token']:
                                    hyde_config = config.copy()
                                    hyde_config['query_expansion_max_token'] = max_token
                                    self._add_downstream_components(hyde_config, grid_space, valid_combinations, top_k)
                            elif qe_method == 'multi_query_expansion' and 'query_expansion_temperature' in grid_space:
                                for temp in grid_space['query_expansion_temperature']:
                                    mq_config = config.copy()
                                    mq_config['query_expansion_temperature'] = temp
                                    self._add_downstream_components(mq_config, grid_space, valid_combinations, top_k)
                            else:
                                self._add_downstream_components(config, grid_space, valid_combinations, top_k)
                    else:
                        self._add_downstream_components(base_qe, grid_space, valid_combinations, top_k)
        
        return valid_combinations
    
    def _add_downstream_components(self, base_config: Dict[str, Any], grid_space: Dict[str, List[Any]], 
                                   valid_combinations: List[Dict[str, Any]], retriever_top_k: int):
        reranker_methods = grid_space.get('passage_reranker_method', ['pass_reranker'])
        filter_methods = grid_space.get('passage_filter_method', ['pass_passage_filter'])
        compressor_methods = grid_space.get('passage_compressor_method', ['pass_compressor'])
        generator_models = grid_space.get('generator_model', [])
        generator_temps = grid_space.get('generator_temperature', [0.7])
        
        for reranker_method in reranker_methods:
            reranker_configs = []
            
            if reranker_method == 'pass_reranker':
                reranker_configs.append({'passage_reranker_method': reranker_method})
            else:
                reranker_top_k_values = grid_space.get('reranker_top_k', [])
                valid_reranker_top_k = [k for k in reranker_top_k_values if k <= retriever_top_k]
                
                if not valid_reranker_top_k:
                    valid_reranker_top_k = [min(5, retriever_top_k)]
                
                for reranker_top_k in valid_reranker_top_k:
                    reranker_config = {
                        'passage_reranker_method': reranker_method,
                        'reranker_top_k': reranker_top_k
                    }
                    
                    model_key = f"{reranker_method}_model"
                    if model_key in grid_space:
                        for model in grid_space[model_key]:
                            model_config = reranker_config.copy()
                            model_config[model_key] = model
                            reranker_configs.append(model_config)
                    else:
                        reranker_configs.append(reranker_config)
            
            for reranker_config in reranker_configs:
                for filter_method in filter_methods:
                    filter_configs = []
                    
                    if filter_method == 'pass_passage_filter':
                        filter_configs.append({'passage_filter_method': filter_method})
                    else:
                        if filter_method == 'threshold_cutoff' and 'threshold_cutoff_threshold' in grid_space:
                            for threshold in grid_space['threshold_cutoff_threshold']:
                                filter_configs.append({
                                    'passage_filter_method': filter_method,
                                    'threshold_cutoff_threshold': threshold
                                })
                        elif filter_method == 'percentile_cutoff' and 'percentile_cutoff_percentile' in grid_space:
                            for percentile in grid_space['percentile_cutoff_percentile']:
                                filter_configs.append({
                                    'passage_filter_method': filter_method,
                                    'percentile_cutoff_percentile': percentile
                                })
                        else:
                            filter_configs.append({'passage_filter_method': filter_method})
                    
                    for filter_config in filter_configs:
                        for compressor_method in compressor_methods:
                            compressor_configs = []
                            
                            if compressor_method == 'pass_compressor':
                                compressor_configs.append({'passage_compressor_method': compressor_method})
                            else:
                                compressor_configs.append({'passage_compressor_method': compressor_method})
                            
                            for compressor_config in compressor_configs:
                                for generator_model in generator_models:
                                    for generator_temp in generator_temps:
                                        final_config = base_config.copy()
                                        final_config.update(reranker_config)
                                        final_config.update(filter_config)
                                        final_config.update(compressor_config)
                                        final_config['generator_model'] = generator_model
                                        final_config['generator_temperature'] = generator_temp
                                        
                                        if 'prompt_maker_method' in grid_space:
                                            for prompt_method in grid_space['prompt_maker_method']:
                                                prompt_config = final_config.copy()
                                                prompt_config['prompt_maker_method'] = prompt_method
                                                valid_combinations.append(prompt_config)
                                        else:
                                            valid_combinations.append(final_config)
    
    def _reduce_grid_combinations(self, combinations: List[Dict[str, Any]], max_trials: int) -> List[Dict[str, Any]]:
        import random
        random.seed(42)
        
        if len(combinations) <= max_trials:
            return combinations
        
        step = len(combinations) // max_trials
        reduced_combinations = combinations[::step][:max_trials]
        
        print(f"Reduced grid from {len(combinations)} to {len(reduced_combinations)} combinations")
        
        return reduced_combinations
    
    def _print_initialization_summary(self):
        total_combinations = len(self.grid_combinations)
        
        print("\n===== Global Grid Search Optimizer Initialized =====")
        print(f"Total grid combinations: {total_combinations}")
        
        if self.max_trials and total_combinations > self.max_trials:
            print(f"Limited to {self.max_trials} trials (sampling every {total_combinations // self.max_trials} combinations)")
        
        if self.use_ragas:
            print(f"\nEvaluation Method: RAGAS")
            print(f"  LLM Model: {self.ragas_llm_model}")
            print(f"  Embedding Model: {self.ragas_embedding_model}")
        else:
            print(f"\nEvaluation Method: Traditional (component-wise)")
        
        print(f"\nScore weights - Retrieval: {self.retrieval_weight}, Generation: {self.generation_weight}")
        
        print("\nSearch space summary (grid values):")
        grid_space = self._convert_search_space_to_grid()
        for param, values in grid_space.items():
            if len(values) <= 5:
                print(f"  {param}: {values}")
            else:
                print(f"  {param}: {values[:3]} ... ({len(values)} values)")
    
    def run_trial(self, config: Dict[str, Any], trial_number: int) -> Dict[str, Any]:
        from pipeline.pipeline_runner.pipeline_utils import EarlyStoppingException
        
        trial_dir = tempfile.mkdtemp(prefix=f"grid_trial_{trial_number}_")
        
        try:
            start_time = time.time()
            
            complete_config = config.copy()
            
            complete_config['save_intermediate_results'] = True
            
            if 'query_expansion_method' in config and config['query_expansion_method'] != 'pass_query_expansion':
                if 'retrieval_method' in complete_config and 'query_expansion_retrieval_method' not in complete_config:
                    complete_config['query_expansion_retrieval_method'] = complete_config.pop('retrieval_method')
                elif 'query_expansion_retrieval_method' not in complete_config and 'query_expansion_retrieval_method' in self.search_space:
                    complete_config['query_expansion_retrieval_method'] = self.search_space['query_expansion_retrieval_method'][0]
                
                if complete_config.get('query_expansion_retrieval_method') == 'bm25':
                    if 'bm25_tokenizer' in complete_config and 'query_expansion_bm25_tokenizer' not in complete_config:
                        complete_config['query_expansion_bm25_tokenizer'] = complete_config.pop('bm25_tokenizer')
                    elif 'query_expansion_bm25_tokenizer' not in complete_config and 'query_expansion_bm25_tokenizer' in self.search_space:
                        complete_config['query_expansion_bm25_tokenizer'] = self.search_space['query_expansion_bm25_tokenizer'][0]
                elif complete_config.get('query_expansion_retrieval_method') == 'vectordb':
                    if 'vectordb_name' in complete_config and 'query_expansion_vectordb_name' not in complete_config:
                        complete_config['query_expansion_vectordb_name'] = complete_config.pop('vectordb_name')
                    elif 'query_expansion_vectordb_name' not in complete_config and 'query_expansion_vectordb_name' in self.search_space:
                        complete_config['query_expansion_vectordb_name'] = self.search_space['query_expansion_vectordb_name'][0]
                
                if 'query_expansion_model' not in complete_config and 'query_expansion_model' in self.search_space:
                    complete_config['query_expansion_model'] = self.search_space['query_expansion_model'][0]
                if 'query_expansion_llm' not in complete_config and 'query_expansion_llm' in self.search_space:
                    complete_config['query_expansion_llm'] = self.search_space['query_expansion_llm'][0]
                if 'query_expansion_temperature' not in complete_config and 'query_expansion_temperature' in self.search_space:
                    complete_config['query_expansion_temperature'] = self.search_space['query_expansion_temperature'][0]
                
                if 'retrieval_method' in complete_config:
                    del complete_config['retrieval_method']
                if 'bm25_tokenizer' in complete_config:
                    del complete_config['bm25_tokenizer']
                if 'vectordb_name' in complete_config:
                    del complete_config['vectordb_name']
            else:
                if 'retrieval_method' not in complete_config and 'retrieval_method' in self.search_space:
                    complete_config['retrieval_method'] = self.search_space['retrieval_method'][0]
                
                if complete_config.get('retrieval_method') == 'bm25':
                    if 'bm25_tokenizer' not in complete_config and 'bm25_tokenizer' in self.search_space:
                        complete_config['bm25_tokenizer'] = self.search_space['bm25_tokenizer'][0]
                elif complete_config.get('retrieval_method') == 'vectordb':
                    if 'vectordb_name' not in complete_config and 'vectordb_name' in self.search_space:
                        complete_config['vectordb_name'] = self.search_space['vectordb_name'][0]
            
            if 'passage_reranker_method' in complete_config and complete_config['passage_reranker_method'] != 'pass_reranker':
                reranker_method = complete_config['passage_reranker_method']
                model_keys = [f'{reranker_method}_model_name', f'{reranker_method}_model', 'reranker_model_name', 'reranker_model']
                for key in model_keys:
                    if key not in complete_config and key in self.search_space:
                        complete_config['reranker_model_name'] = self.search_space[key][0]
                        break
            
            if 'passage_filter_method' in complete_config:
                filter_method = complete_config['passage_filter_method']
                if filter_method == 'threshold_cutoff' and 'threshold_cutoff_threshold' not in complete_config:
                    if 'threshold_cutoff_threshold' in self.search_space:
                        complete_config['threshold_cutoff_threshold'] = self.search_space['threshold_cutoff_threshold'][0]
                elif filter_method == 'percentile_cutoff' and 'percentile_cutoff_percentile' not in complete_config:
                    if 'percentile_cutoff_percentile' in self.search_space:
                        complete_config['percentile_cutoff_percentile'] = self.search_space['percentile_cutoff_percentile'][0]
                elif filter_method == 'similarity_threshold_cutoff' and 'similarity_threshold_cutoff_threshold' not in complete_config:
                    if 'similarity_threshold_cutoff_threshold' in self.search_space:
                        complete_config['similarity_threshold_cutoff_threshold'] = self.search_space['similarity_threshold_cutoff_threshold'][0]
                elif filter_method == 'similarity_percentile_cutoff' and 'similarity_percentile_cutoff_percentile' not in complete_config:
                    if 'similarity_percentile_cutoff_percentile' in self.search_space:
                        complete_config['similarity_percentile_cutoff_percentile'] = self.search_space['similarity_percentile_cutoff_percentile'][0]
            
            if 'passage_compressor_method' in complete_config and complete_config['passage_compressor_method'] != 'pass_compressor':
                if 'compressor_llm' not in complete_config and 'compressor_llm' in self.search_space:
                    complete_config['compressor_llm'] = self.search_space['compressor_llm'][0]
                if 'compressor_model' not in complete_config and 'compressor_model' in self.search_space:
                    complete_config['compressor_model'] = self.search_space['compressor_model'][0]
            
            if 'prompt_template_idx' not in complete_config and 'prompt_template_idx' in self.search_space:
                complete_config['prompt_template_idx'] = self.search_space['prompt_template_idx'][0]
            
            if 'generator_module_type' not in complete_config and 'generator_module_type' in self.search_space:
                complete_config['generator_module_type'] = self.search_space['generator_module_type'][0]
            
            print(f"\nRunning trial {trial_number} with complete config:")
            for key, value in sorted(complete_config.items()):
                print(f"  {key}: {value}")
            
            from pipeline_component.nodes.retrieval import RetrievalModule
            retrieval_module = RetrievalModule(
                base_project_dir=trial_dir,
                use_pregenerated_embeddings=True,
                centralized_project_dir=self.pipeline_runner._get_centralized_project_dir()
            )
            retrieval_module.prepare_project_dir(trial_dir, self.corpus_df, self.qa_df)
            
            original_thresholds = self.pipeline_runner.early_stopping_handler.early_stopping_thresholds
            self.pipeline_runner.early_stopping_handler.early_stopping_thresholds = None
            
            try:
                results = self.pipeline_runner.run_pipeline(complete_config, trial_dir, self.qa_df)
            except EarlyStoppingException as e:
                results = {
                    'combined_score': e.score,
                    'score': e.score,
                    'early_stopped': True,
                    'early_stopped_at': e.component,
                    'error': str(e)
                }
            finally:
                self.pipeline_runner.early_stopping_handler.early_stopping_thresholds = original_thresholds
            
            execution_time = time.time() - start_time
            
            score = results.get("combined_score", results.get("score", 0.0))
            
            intermediate_results_dir = os.path.join(trial_dir, "debug_intermediate_results")
            if os.path.exists(intermediate_results_dir):
                save_dir = os.path.join(self.result_dir, "grid_intermediate_results", f"trial_{trial_number}")
                os.makedirs(save_dir, exist_ok=True)
                shutil.copytree(intermediate_results_dir, save_dir, dirs_exist_ok=True)
                print(f"[Grid Search] Intermediate results saved to {save_dir}")
            
            trial_result = {
                "trial_number": trial_number,
                "config": complete_config,
                "score": score,
                "latency": execution_time,
                "retrieval_score": results.get("retrieval_score", 0.0),
                "generation_score": results.get("generation_score", 0.0),
                "combined_score": score,
                "timestamp": time.time(),
                "intermediate_results_path": os.path.join(self.result_dir, "grid_intermediate_results", f"trial_{trial_number}")
            }
            
            if self.use_ragas and 'ragas_mean_score' in results:
                trial_result['ragas_mean_score'] = results['ragas_mean_score']
                trial_result['ragas_metrics'] = results.get('ragas_metrics', {})
            
            for key, value in results.items():
                if key not in trial_result and not key.startswith('working'):
                    trial_result[key] = value
            
            return trial_result
            
        except Exception as e:
            print(f"Error in trial {trial_number}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "trial_number": trial_number,
                "config": config,
                "score": 0.0,
                "latency": float('inf'),
                "error": str(e)
            }
        finally:
            if os.path.exists(trial_dir):
                try:
                    if not complete_config.get('save_intermediate_results', True):
                        shutil.rmtree(trial_dir)
                except:
                    pass


    def save_consolidated_intermediate_results(self):        
        intermediate_dir = os.path.join(self.result_dir, "grid_intermediate_results")
        if not os.path.exists(intermediate_dir):
            print("No intermediate results found to consolidate")
            return
        
        consolidated_data = []
        
        for trial_dir in sorted(os.listdir(intermediate_dir)):
            if trial_dir.startswith("trial_"):
                trial_number = int(trial_dir.split("_")[1])
                trial_path = os.path.join(intermediate_dir, trial_dir)
                
                trial_summary = {
                    "trial_number": trial_number,
                    "components": {}
                }
                
                for file in os.listdir(trial_path):
                    if file.endswith("_summary.json"):
                        component_name = file.replace("_summary.json", "")
                        with open(os.path.join(trial_path, file), 'r') as f:
                            summary_data = json.load(f)
                            trial_summary["components"][component_name] = {
                                "score": summary_data.get("score", 0.0),
                                "execution_time": summary_data.get("execution_time", 0.0)
                            }
                
                if os.path.exists(os.path.join(trial_path, "pipeline_summary.json")):
                    with open(os.path.join(trial_path, "pipeline_summary.json"), 'r') as f:
                        pipeline_summary = json.load(f)
                        trial_summary["final_score"] = pipeline_summary.get("combined_score", 0.0)
                        trial_summary["last_retrieval_component"] = pipeline_summary.get("last_retrieval_component")
                        trial_summary["generation_score"] = pipeline_summary.get("generation_score", 0.0)
                
                consolidated_data.append(trial_summary)
        
        consolidated_df = pd.DataFrame(consolidated_data)
        consolidated_df.to_csv(os.path.join(self.result_dir, "grid_intermediate_summary.csv"), index=False)
        
        with open(os.path.join(self.result_dir, "grid_intermediate_summary.json"), 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        
        print(f"Consolidated intermediate results saved to {self.result_dir}")
        return consolidated_data
    
    def optimize(self) -> Dict[str, Any]:
        import os
        import yaml
        
        start_time = time.time()
        
        if self.use_wandb:
            import wandb
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.wandb_run_name,
                config={
                    "search_type": "grid_search",
                    "optimizer": "GRID",
                    "total_combinations": len(self.grid_combinations),
                    "retrieval_weight": self.retrieval_weight,
                    "generation_weight": self.generation_weight,
                    "study_name": self.study_name,
                    "evaluation_method": "RAGAS" if self.use_ragas else "Traditional"
                },
                reinit=True
            )
        
        print(f"\nStarting Grid Search with {len(self.grid_combinations)} combinations...")
        
        for i, config in enumerate(self.grid_combinations):
            print(f"\n{'='*80}")
            print(f"Trial {i+1}/{len(self.grid_combinations)}")
            print(f"{'='*80}")
            
            trial_result = self.run_trial(config, i)
            
            self.all_trials.append(trial_result)
            
            if trial_result['score'] > self.best_score['value']:
                self.best_score = {
                    'value': trial_result['score'],
                    'config': trial_result['config'],
                    'latency': trial_result['latency']
                }
            
            if trial_result['latency'] < self.best_latency['value']:
                self.best_latency = {
                    'value': trial_result['latency'],
                    'config': trial_result['config'],
                    'score': trial_result['score']
                }
            
            print(f"\nTrial {i+1} completed:")
            print(f"  Score: {trial_result['score']:.4f}")
            print(f"  Latency: {trial_result['latency']:.2f}s")
            print(f"  Best score so far: {self.best_score['value']:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "trial": i,
                    "score": trial_result['score'],
                    "latency": trial_result['latency'],
                    "best_score": self.best_score['value']
                })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        pareto_front = []
        if self.all_trials:
            valid_trials = [t for t in self.all_trials if t['score'] > 0 and t['latency'] < float('inf')]
            if valid_trials:
                for trial in valid_trials:
                    is_dominated = False
                    for other in valid_trials:
                        if other != trial:
                            if other['score'] >= trial['score'] and other['latency'] <= trial['latency']:
                                if other['score'] > trial['score'] or other['latency'] < trial['latency']:
                                    is_dominated = True
                                    break
                    if not is_dominated:
                        pareto_front.append(trial)
        
        best_config = None
        if self.best_score['value'] > 0:
            best_config = {
                "config": self.best_score['config'],
                "score": self.best_score['value'],
                "latency": self.best_score['latency'],
                "trial_number": next((t['trial_number'] for t in self.all_trials 
                                    if t['score'] == self.best_score['value']), 0)
            }
        
        results = {
            "best_config": best_config,
            "best_score_config": self.best_score['config'],
            "best_score": self.best_score['value'],
            "best_score_latency": self.best_score['latency'],
            "best_latency_config": self.best_latency['config'],
            "best_latency": self.best_latency['value'],
            "best_latency_score": self.best_latency['score'],
            "pareto_front": pareto_front,
            "optimization_time": total_time,
            "n_trials": len(self.all_trials),
            "early_stopped_trials": 0,
            "completed_trials": len(self.all_trials),
            "early_stopped": False,
            "optimizer": "grid",
            "total_trials": len(self.all_trials),
            "all_trials": self.all_trials,
            "component_early_stopping_enabled": False,
            "component_early_stopping_thresholds": None
        }
        
        results_file = os.path.join(self.result_dir, "grid_search_results.yaml")
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        
        print(f"\n{'='*80}")
        print("Grid Search Completed!")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best score: {self.best_score['value']:.4f}")
        print(f"Best config: {self.best_score['config']}")
        
        if self.use_wandb:
            wandb.summary["best_score"] = self.best_score['value']
            wandb.summary["best_latency"] = self.best_latency['value']
            wandb.summary["total_time"] = total_time
            wandb.finish()
        
        return results