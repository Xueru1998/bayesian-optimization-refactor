import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from pipeline.rag_pipeline_runner import RAGPipelineRunner
from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_calculator import CombinationCalculator
from pipeline.pipeline_executor import RAGPipelineExecutor


class ComponentwiseRAGProcessor:
    
    def __init__(
        self,
        config_generator: ConfigGenerator,
        retrieval_weight: float,
        generation_weight: float,
        project_dir: str,
        corpus_data: pd.DataFrame,
        qa_data: pd.DataFrame,
        use_llm_evaluator: bool = False,
        llm_evaluator_config: Optional[Dict[str, Any]] = None,
        search_type: str = 'bo',
    ):
        self.config_generator = config_generator
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.project_dir = project_dir
        self.corpus_data = corpus_data
        self.qa_data = qa_data
        
        self.use_llm_evaluator = use_llm_evaluator
        self.llm_evaluator_config = llm_evaluator_config or {}
        self.search_type = search_type
        
        self.combination_calculator = CombinationCalculator(
            config_generator=config_generator,
            search_type=search_type
        )
        
        self._setup_metrics()
        self._setup_pipeline_runner()
        
    
    def _setup_metrics(self):
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config('prompt_maker')
        self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()
    
    def _setup_pipeline_runner(self):
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
            use_llm_evaluator=self.use_llm_evaluator,
            llm_evaluator_config=self.llm_evaluator_config,
        )

    def parse_trial_config(self, component: str, trial_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        is_pass_component = False
        
        if component == 'retrieval' and 'retrieval_config' in trial_config:
            parsed = RAGPipelineExecutor.parse_retrieval_config(trial_config['retrieval_config'])
            trial_config.update(parsed)
            trial_config.pop('retrieval_config', None)
        
        elif component == 'query_expansion' and 'query_expansion_config' in trial_config:
            qe_config_str = trial_config['query_expansion_config']
            
            if qe_config_str == 'pass_query_expansion':
                is_pass_component = True
                trial_config['query_expansion_method'] = 'pass_query_expansion'
                trial_config.pop('query_expansion_config', None)
            else:
                parts = qe_config_str.split('::', 2)
                if len(parts) == 3:
                    method, gen_type, model = parts
                    trial_config['query_expansion_method'] = method
                    trial_config['query_expansion_generator_module_type'] = gen_type
                    trial_config['query_expansion_model'] = model
                    
                    unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                    for gen_config in unified_params.get('generator_configs', []):
                        if (gen_config['method'] == method and 
                            gen_config['generator_module_type'] == gen_type and 
                            model in gen_config['models']):
                            trial_config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                            if gen_type == 'sap_api':
                                trial_config['query_expansion_api_url'] = gen_config.get('api_url')
                            break
                else:
                    parsed = RAGPipelineExecutor.parse_query_expansion_config(qe_config_str)
                    trial_config.update(parsed)
            
            trial_config.pop('query_expansion_config', None)
        
        elif component == 'passage_reranker' and 'reranker_config' in trial_config:
            reranker_config_str = trial_config['reranker_config']
            
            if reranker_config_str == 'pass_reranker':
                is_pass_component = True
                trial_config['passage_reranker_method'] = 'pass_reranker'
            elif reranker_config_str == 'sap_api':
                trial_config['passage_reranker_method'] = 'sap_api'
                unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
                api_endpoints = unified_params.get('api_endpoints', {})
                models = unified_params.get('models', {})
                
                if 'sap_api' in models and models['sap_api']:
                    trial_config['reranker_model_name'] = models['sap_api'][0]
                else:
                    trial_config['reranker_model_name'] = 'cohere-rerank-v3.5'
                
                if 'sap_api' in api_endpoints:
                    trial_config['reranker_api_url'] = api_endpoints['sap_api']
            elif '::' in reranker_config_str:
                parts = reranker_config_str.split('::', 1)
                trial_config['passage_reranker_method'] = parts[0]
                trial_config['reranker_model_name'] = parts[1]
            else:
                parsed = RAGPipelineExecutor.parse_reranker_config(reranker_config_str)
                trial_config.update(parsed)
            
            trial_config.pop('reranker_config', None)
        
        elif component == 'passage_compressor' and 'passage_compressor_config' in trial_config:
            comp_config_str = trial_config['passage_compressor_config']
            if comp_config_str == 'pass_compressor':
                is_pass_component = True
                trial_config['passage_compressor_method'] = 'pass_compressor'
            else:
                parts = comp_config_str.split('::', 3)
                
                if parts[0] in ['lexrank', 'spacy']:
                    method = parts[0]
                    trial_config['passage_compressor_method'] = method

                    if 'compression_ratio' in trial_config:
                        trial_config['compression_ratio'] = trial_config['compression_ratio']
                    
                    if method == 'lexrank':
                        if 'lexrank_threshold' in trial_config:
                            trial_config['threshold'] = trial_config.pop('lexrank_threshold')
                        if 'lexrank_damping' in trial_config:
                            trial_config['damping'] = trial_config.pop('lexrank_damping')
                        if 'lexrank_max_iterations' in trial_config:
                            trial_config['max_iterations'] = trial_config.pop('lexrank_max_iterations')
                            
                    elif method == 'spacy':
                        if len(parts) > 1:
                            trial_config['spacy_model'] = parts[1]
                        elif 'spacy_model' in trial_config:
                            pass
                            
                elif len(parts) == 3:
                    method, gen_type, model = parts
                    
                    trial_config['passage_compressor_method'] = method
                    trial_config['compressor_generator_module_type'] = gen_type
                    trial_config['compressor_model'] = model
                    
                    unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                    for comp_config in unified_params.get('compressor_configs', []):
                        if (comp_config['method'] == method and 
                            comp_config['generator_module_type'] == gen_type and 
                            model in comp_config['models']):
                            
                            trial_config['compressor_llm'] = comp_config.get('llm', 'openai')
                            
                            if gen_type == 'sap_api':
                                trial_config['compressor_api_url'] = comp_config.get('api_url')
                            elif gen_type == 'vllm':
                                trial_config['compressor_llm'] = model
                                if 'tensor_parallel_size' in comp_config:
                                    trial_config['compressor_tensor_parallel_size'] = comp_config['tensor_parallel_size']
                                if 'gpu_memory_utilization' in comp_config:
                                    trial_config['compressor_gpu_memory_utilization'] = comp_config['gpu_memory_utilization']
                            
                            if 'batch' in comp_config:
                                trial_config['compressor_batch'] = comp_config['batch']
                            break
                
                elif len(parts) == 2:
                    method = parts[0]
                    
                    if method in ['lexrank', 'spacy']:
                        trial_config['passage_compressor_method'] = method
                        if method == 'spacy':
                            trial_config['spacy_model'] = parts[1]
                    else:
                        llm = parts[1]
                        trial_config['passage_compressor_method'] = method
                        trial_config['compressor_llm'] = llm
                        
                        unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                        for comp_config in unified_params.get('compressor_configs', []):
                            if comp_config['method'] == method and comp_config['llm'] == llm:
                                if comp_config['models']:
                                    trial_config['compressor_model'] = comp_config['models'][0]
                                if comp_config.get('generator_module_type') == 'sap_api':
                                    trial_config['compressor_api_url'] = comp_config.get('api_url')
                                break
                else:
                    parsed = RAGPipelineExecutor.parse_compressor_config(comp_config_str)
                    trial_config.update(parsed)
            
            trial_config.pop('passage_compressor_config', None)
            
        elif component == 'generator' and 'generator_config' in trial_config:
            gen_config_str = trial_config['generator_config']
            parts = gen_config_str.split('::', 1)
            if len(parts) == 2:
                module_type, model = parts
                trial_config['generator_module_type'] = module_type
                trial_config['generator_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('generator')
                for module_config in unified_params.get('module_configs', []):
                    if module_config['module_type'] == module_type and model in module_config['models']:
                        if module_type == 'sap_api':
                            trial_config['generator_api_url'] = module_config.get('api_url')
                            trial_config['generator_llm'] = module_config.get('llm', 'mistralai')
                        elif module_type == 'vllm':
                            trial_config['generator_llm'] = model
                        else:
                            trial_config['generator_llm'] = module_config.get('llm', 'openai')
                        break
            
            trial_config.pop('generator_config', None)
        
        elif component == 'prompt_maker_generator' and 'generator_config' in trial_config:
            gen_config_str = trial_config['generator_config']
            parts = gen_config_str.split('::', 1)
            if len(parts) == 2:
                module_type, model = parts
                trial_config['generator_module_type'] = module_type
                trial_config['generator_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('generator')
                for module_config in unified_params.get('module_configs', []):
                    if module_config['module_type'] == module_type and model in module_config['models']:
                        if module_type == 'sap_api':
                            trial_config['generator_api_url'] = module_config.get('api_url')
                            trial_config['generator_llm'] = module_config.get('llm', 'mistralai')
                        elif module_type == 'vllm':
                            trial_config['generator_llm'] = model
                        else:
                            trial_config['generator_llm'] = module_config.get('llm', 'openai')
                        break
            
            trial_config.pop('generator_config', None)
        
        elif component == 'passage_filter' and 'passage_filter_config' in trial_config:
            filter_config_str = trial_config['passage_filter_config']
            if filter_config_str == 'pass_passage_filter':
                is_pass_component = True
            parsed_config = RAGPipelineExecutor.parse_filter_config(filter_config_str)
            trial_config.update(parsed_config)
            trial_config.pop('passage_filter_config', None)
        
        elif component == 'prompt_maker_generator' and 'prompt_config' in trial_config:
            prompt_config_str = trial_config['prompt_config']
            if prompt_config_str == 'pass_prompt_maker':
                is_pass_component = True
            parsed_config = RAGPipelineExecutor.parse_prompt_config(prompt_config_str)
            trial_config.update(parsed_config)
            trial_config.pop('prompt_config', None)

        if not is_pass_component:
            if component == 'passage_filter' and trial_config.get('passage_filter_method') == 'pass_passage_filter':
                is_pass_component = True
            elif component == 'passage_reranker' and trial_config.get('passage_reranker_method') == 'pass_reranker':
                is_pass_component = True
            elif component == 'passage_compressor' and trial_config.get('passage_compressor_method') == 'pass_compressor':
                is_pass_component = True
            elif component == 'query_expansion' and trial_config.get('query_expansion_method') == 'pass_query_expansion':
                is_pass_component = True
        
        return trial_config, is_pass_component
    
    def parse_component_config(self, component: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if component == 'retrieval' and 'retrieval_config' in config:
            parsed_config = RAGPipelineExecutor.parse_retrieval_config(config['retrieval_config'])
            config.update(parsed_config)
            config.pop('retrieval_config', None)
        return config
    
    def get_fixed_config(self, component: str, best_configs: Dict[str, Any], 
                        component_order: List[str]) -> Dict[str, Any]:
        fixed_config = {}
        
        for prev_component in component_order[:component_order.index(component)]:
            if prev_component in best_configs:
                fixed_config.update(best_configs[prev_component])
        
        if component == 'retrieval' and 'query_expansion_method' in fixed_config:
            if fixed_config['query_expansion_method'] != 'pass_query_expansion':
                fixed_config['retrieval_method'] = 'pass_retrieval'
        
        if 'query_expansion' in best_configs and best_configs['query_expansion']:
            qe_best = best_configs['query_expansion']
            if qe_best.get('query_expansion_method') != 'pass_query_expansion':
                if 'retrieval_config' in qe_best:
                    parsed = RAGPipelineExecutor.parse_retrieval_config(qe_best['retrieval_config'])
                    fixed_config.update(parsed)
                if 'retriever_top_k' in qe_best:
                    fixed_config['retriever_top_k'] = qe_best['retriever_top_k']
        
        return fixed_config
    
    def load_previous_outputs(self, component: str, qa_subset: pd.DataFrame, 
                            component_dataframes: Dict[str, str], 
                            trial_dir: str) -> pd.DataFrame:
        prev_components = ['query_expansion', 'retrieval', 'passage_reranker', 
                          'passage_filter', 'passage_compressor', 'prompt_maker_generator']
        
        best_output_loaded = False
        
        if component not in ['query_expansion', 'retrieval']:
            if 'query_expansion' in component_dataframes:
                print(f"[{component}] Using best query_expansion output (which includes retrieval)")
                best_parquet_path = component_dataframes['query_expansion']
                if os.path.exists(best_parquet_path):
                    best_prev_df = pd.read_parquet(best_parquet_path)
                    
                    if len(best_prev_df) == len(qa_subset):
                        for col in best_prev_df.columns:
                            if col not in ['query', 'retrieval_gt', 'generation_gt', 'qid']:
                                qa_subset[col] = best_prev_df[col]
                        best_output_loaded = True

                    trial_prev_path = os.path.join(trial_dir, "query_expansion_output.parquet")
                    best_prev_df.to_parquet(trial_prev_path)
            
            if not best_output_loaded:
                for prev_comp in reversed(prev_components[:prev_components.index(component)]):
                    if prev_comp in component_dataframes:
                        best_parquet_path = component_dataframes[prev_comp]
                        if os.path.exists(best_parquet_path):
                            print(f"[{component}] Using best {prev_comp} output from: {best_parquet_path}")
                            best_prev_df = pd.read_parquet(best_parquet_path)
                            print(f"[{component}] Loaded {prev_comp} output with columns: {list(best_prev_df.columns)}")
                            
                            if len(best_prev_df) == len(qa_subset):
                                for col in best_prev_df.columns:
                                    if col not in ['query', 'retrieval_gt', 'generation_gt', 'qid']:
                                        qa_subset[col] = best_prev_df[col]
                                print(f"[{component}] Merged columns from {prev_comp}: {[col for col in best_prev_df.columns if col not in ['query', 'retrieval_gt', 'generation_gt', 'qid']]}")
                            
                            trial_prev_path = os.path.join(trial_dir, f"{prev_comp}_output.parquet")
                            best_prev_df.to_parquet(trial_prev_path)
                            break
        
        print(f"[{component}] Final qa_subset columns before pipeline run: {list(qa_subset.columns)}")
        return qa_subset
    
    def run_pipeline(self, config: Dict[str, Any], trial_dir: str, 
                    qa_data: pd.DataFrame, component: str,
                    component_results: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(self.pipeline_runner, 'component_results'):
            self.pipeline_runner.component_results = component_results
        else:
            setattr(self.pipeline_runner, 'component_results', component_results)

        return self.pipeline_runner.run_pipeline(
            config, 
            trial_dir, 
            qa_data,
            is_local_optimization=True,
            current_component=component
        )
    
    def extract_detailed_metrics(self, component: str, results: Dict[str, Any]) -> Dict[str, Any]:
        detailed_metrics = {}
        
        if component == 'retrieval' or component == 'query_expansion':
            if 'retrieval_metrics' in results:
                detailed_metrics.update(results['retrieval_metrics'])
            if 'query_expansion_metrics' in results:
                detailed_metrics.update(results['query_expansion_metrics'])
        elif component == 'passage_reranker':
            if 'reranker_metrics' in results:
                detailed_metrics.update(results['reranker_metrics'])
        elif component == 'passage_filter':
            if 'filter_metrics' in results:
                detailed_metrics.update(results['filter_metrics'])
        elif component == 'passage_compressor':
            if 'compression_metrics' in results:
                detailed_metrics.update(results['compression_metrics'])
            if 'compressor_metrics' in results:
                detailed_metrics.update(results['compressor_metrics'])
        elif component == 'prompt_maker_generator':
            if 'prompt_maker_metrics' in results:
                detailed_metrics.update(results['prompt_maker_metrics'])
            if 'generation_metrics' in results:
                detailed_metrics.update(results['generation_metrics'])
        
        return detailed_metrics
    
    def calculate_component_score(self, component: str, results: Dict[str, Any], 
                                is_pass_component: bool, 
                                component_results: Dict[str, Any]) -> float:
        if is_pass_component:
            if component == 'query_expansion':
                print(f"[{component}] Pass query expansion detected, will run retrieval only")
                score = results.get('last_retrieval_score', 0.0)
            elif component == 'retrieval':
                print(f"[{component}] Retrieval component - no inheritance needed")
                score = results.get('last_retrieval_score', 0.0)
            else:
                print(f"[{component}] Pass component detected, using pipeline results")
                score = results.get('last_retrieval_score', 0.0)

                if score == 0.0:
                    prev_best_scores = []
                    for prev_comp in reversed(list(component_results.keys())):
                        if prev_comp in component_results and component_results[prev_comp].get('best_score', 0) > 0:
                            prev_best_scores.append(component_results[prev_comp]['best_score'])
                            print(f"[{component}] Using score from {prev_comp}: {component_results[prev_comp]['best_score']}")
                            break
                    
                    if prev_best_scores:
                        score = prev_best_scores[0]
                    else:
                        score = results.get('score', 0.0)
                    
                print(f"[{component}] Pass component final score: {score}")
        else:
            if component == 'retrieval':
                score = results.get('last_retrieval_score', 0.0)
            elif component == 'query_expansion':
                if results.get('query_expansion_score', 0.0) > 0:
                    score = results.get('query_expansion_score', 0.0)
                else:
                    score = results.get('retrieval_score', results.get('last_retrieval_score', 0.0))
            elif component == 'passage_reranker':
                score = results.get('reranker_score', results.get('last_retrieval_score', 0.0))
            elif component == 'passage_filter':
                score = results.get('filter_score', results.get('last_retrieval_score', 0.0))
            elif component == 'passage_compressor':
                score = results.get('compression_score', results.get('last_retrieval_score', 0.0))
            elif component == 'prompt_maker_generator':
                if 'generation_score' in results and results['generation_score'] > 0:
                    score = results['combined_score']
                else:
                    score = results.get('prompt_maker_score', results.get('last_retrieval_score', 0.0))
            else:
                score = results.get('combined_score', 0.0)
        
        return score
    
    def save_component_output(self, component: str, trial_dir: str, 
                        results: Dict[str, Any], qa_data: pd.DataFrame) -> str:
    
        output_df = qa_data.copy()
        
        print(f"[{component}] Initial columns in qa_data: {list(qa_data.columns)}")

        if component == 'query_expansion':
            if 'retrieval_df' in results:
                retrieval_df = results['retrieval_df']
                if isinstance(retrieval_df, pd.DataFrame):
                    for col in ['retrieved_ids', 'retrieved_contents', 'retrieve_scores']:
                        if col in retrieval_df.columns:
                            output_df[col] = retrieval_df[col]
                    if 'queries' in retrieval_df.columns:
                        output_df['queries'] = retrieval_df['queries']

            for col in ['retrieved_ids', 'retrieved_contents', 'retrieve_scores', 'queries']:
                if col in qa_data.columns and col not in output_df.columns:
                    output_df[col] = qa_data[col]

        elif component == 'retrieval':
            retrieval_columns = ['retrieved_ids', 'retrieved_contents', 'retrieve_scores']
            for col in retrieval_columns:
                if col in qa_data.columns:
                    output_df[col] = qa_data[col]
                else:
                    if col in results:
                        output_df[col] = results[col]
                    elif f'{col[:-1]}_list' in results:
                        output_df[col] = results[f'{col[:-1]}_list']

            if 'queries' in qa_data.columns:
                output_df['queries'] = qa_data['queries']

        elif component in ['passage_reranker', 'passage_filter', 'passage_compressor']:
            base_columns = ['query', 'retrieval_gt', 'generation_gt', 'qid']
            for col in qa_data.columns:
                if col not in base_columns and col not in output_df.columns:
                    output_df[col] = qa_data[col]

            if component == 'passage_compressor' and 'compressor_score' in qa_data.columns:
                output_df['compressor_score'] = qa_data['compressor_score']

        elif component == 'prompt_maker_generator':
            for col in qa_data.columns:
                if col not in output_df.columns:
                    output_df[col] = qa_data[col]

            if 'eval_df' in results and isinstance(results['eval_df'], pd.DataFrame):
                eval_df = results['eval_df']
                for col in ['generated_texts', 'prompts']:
                    if col in eval_df.columns:
                        output_df[col] = eval_df[col]
        
        output_path = os.path.join(trial_dir, f"{component}_output.parquet")
        output_df.to_parquet(output_path)

        print(f"[{component}] Saved output with columns: {list(output_df.columns)}")
        
        return output_path
    
    def get_component_combinations(self, component: str, config_space_builder, 
                                 best_configs: Dict[str, Any]) -> Tuple[int, str]:
        search_space = None
        fixed_config = None
        
        if hasattr(config_space_builder, 'get_unified_space'):
            unified_space = config_space_builder.get_unified_space()
            if unified_space:
                search_space = self._extract_component_search_space(component, unified_space)
        
        fixed_config = self.get_fixed_config(component, best_configs, 
                                           ['query_expansion', 'retrieval', 'passage_reranker', 
                                            'passage_filter', 'passage_compressor', 'prompt_maker_generator'])
        
        combinations, note = self.combination_calculator.calculate_component_combinations(
            component=component,
            search_space=search_space,
            fixed_config=fixed_config,
            best_configs=best_configs
        )
        
        return combinations, note
    
    def _extract_component_search_space(self, component: str, unified_space: Dict[str, Any]) -> Dict[str, Any]:
        search_space = {}

        is_componentwise = any(key.endswith('_config') for key in unified_space.keys())
        
        if is_componentwise and component == 'passage_reranker':
            if 'reranker_config' in unified_space and 'values' in unified_space['reranker_config']:
                configs = unified_space['reranker_config']['values']
                methods = []
                models_by_method = {}
                
                for config in configs:
                    if config == 'pass_reranker':
                        methods.append('pass_reranker')
                    elif config == 'sap_api':
                        methods.append('sap_api')
                    elif '::' in config:
                        method, model = config.split('::', 1)
                        if method not in methods:
                            methods.append(method)
                        if method not in models_by_method:
                            models_by_method[method] = []
                        models_by_method[method].append(model)
                    else:
                        if config not in methods:
                            methods.append(config)
                
                search_space['passage_reranker_method'] = methods
                for method, models in models_by_method.items():
                    search_space[f'{method}_models'] = models

                if 'reranker_top_k' in unified_space and 'values' in unified_space['reranker_top_k']:
                    search_space['reranker_top_k'] = unified_space['reranker_top_k']['values']
                    
        elif is_componentwise and component == 'passage_filter':
            if 'passage_filter_config' in unified_space and 'values' in unified_space['passage_filter_config']:
                configs = unified_space['passage_filter_config']['values']
                methods = []
                
                for config in configs:
                    if '::' in config:
                        method = config.split('::')[0]
                    else:
                        method = config
                    
                    if method not in methods:
                        methods.append(method)
                
                search_space['passage_filter_method'] = methods

                for param_name, param_info in unified_space.items():
                    if ('threshold' in param_name or 'percentile' in param_name) and 'values' in param_info:
                        search_space[param_name] = param_info['values']
                        
        elif is_componentwise and component == 'passage_compressor':
            if 'passage_compressor_config' in unified_space and 'values' in unified_space['passage_compressor_config']:
                configs = unified_space['passage_compressor_config']['values']
                search_space['passage_compressor_config'] = configs

                for param_name, param_info in unified_space.items():
                    if (param_name.startswith('lexrank') or param_name.startswith('spacy') or 
                        param_name.startswith('compression')) and 'values' in param_info:
                        search_space[param_name] = param_info['values']
        else:
            for param_name, param_info in unified_space.items():
                add_param = False
                
                if component == 'query_expansion':
                    if param_name.startswith('query_expansion') or param_name == 'retriever_top_k':
                        add_param = True
                        
                elif component == 'retrieval':
                    if (param_name.startswith('retrieval') or 
                        param_name.startswith('bm25') or 
                        param_name.startswith('vectordb') or
                        param_name == 'retriever_top_k'):
                        add_param = True
                        
                elif component == 'passage_reranker':
                    if (param_name.startswith('passage_reranker') or 
                        param_name.startswith('reranker') or
                        param_name == 'reranker_config'):
                        add_param = True
                        
                elif component == 'passage_filter':
                    if (param_name.startswith('passage_filter') or 
                        param_name.startswith('filter') or
                        'threshold' in param_name or
                        'percentile' in param_name):
                        add_param = True
                        
                elif component == 'passage_compressor':
                    if (param_name.startswith('passage_compressor') or 
                        param_name.startswith('compressor') or
                        param_name.startswith('compression') or
                        param_name.startswith('lexrank') or
                        param_name.startswith('spacy')):
                        add_param = True
                        
                elif component == 'prompt_maker_generator':
                    if (param_name.startswith('prompt_maker') or 
                        param_name.startswith('prompt') or
                        param_name.startswith('generator')):
                        add_param = True
                
                if add_param and 'values' in param_info:
                    search_space[param_name] = param_info['values']
        
        return search_space