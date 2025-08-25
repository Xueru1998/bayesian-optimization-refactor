import optuna
from typing import Dict, Any, Union, Tuple


class ParameterSuggester:
    def __init__(self, search_space, config_generator, has_query_expansion, has_retrieval):
        self.search_space = search_space
        self.config_generator = config_generator
        self.has_query_expansion = has_query_expansion
        self.has_retrieval = has_retrieval
    
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        
        self._suggest_query_expansion_params(trial, params)
        self._suggest_retrieval_params(trial, params)
        self._suggest_reranker_params(trial, params)
        self._suggest_filter_params(trial, params)
        self._suggest_compressor_params(trial, params)
        self._suggest_prompt_maker_params(trial, params)
        self._suggest_generator_params(trial, params)
        
        return params
    
    def _suggest_value(self, trial: optuna.Trial, param_name: str, 
                  param_spec: Union[list, Tuple[float, float]], 
                  param_type: str = 'categorical') -> Any:
        if isinstance(param_spec, list):
            return trial.suggest_categorical(param_name, param_spec)
        elif isinstance(param_spec, tuple) and len(param_spec) == 2:
            if param_type == 'int':
                return trial.suggest_int(param_name, param_spec[0], param_spec[1])
            else:
                return trial.suggest_float(param_name, param_spec[0], param_spec[1])
        else:
            raise ValueError(f"Invalid parameter specification for {param_name}: {param_spec}")
    
    def _suggest_query_expansion_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'query_expansion_config' in self.search_space:
            qe_config_str = trial.suggest_categorical('query_expansion_config',
                                                    self.search_space['query_expansion_config'])
            
            if qe_config_str == 'pass_query_expansion':
                params['query_expansion_method'] = 'pass_query_expansion'
                return

            parts = qe_config_str.split('::')
            if len(parts) >= 3:
                method, gen_type, model = parts[0], parts[1], parts[2]
                params['query_expansion_method'] = method
                params['query_expansion_generator_module_type'] = gen_type
                params['query_expansion_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                for gen_config in unified_params.get('generator_configs', []):
                    if (gen_config['method'] == method and 
                        gen_config['generator_module_type'] == gen_type and 
                        model in gen_config['models']):
                        if gen_type == 'sap_api':
                            params['query_expansion_api_url'] = gen_config.get('api_url')
                            params['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                        else:
                            params['query_expansion_llm'] = gen_config.get('llm', 'openai')
                        break
                
                if 'query_expansion_temperature' in self.search_space:
                    params['query_expansion_temperature'] = self._suggest_value(
                        trial, 'query_expansion_temperature', 
                        self.search_space['query_expansion_temperature'], 'float'
                    )
                
                if 'query_expansion_max_token' in self.search_space:
                    params['query_expansion_max_token'] = self._suggest_value(
                        trial, 'query_expansion_max_token', 
                        self.search_space['query_expansion_max_token'], 'int'
                    )
                
                self._add_query_expansion_retrieval_params(trial, params)
            
            return

        if 'query_expansion_method' not in self.search_space:
            return
            
        params['query_expansion_method'] = trial.suggest_categorical('query_expansion_method', 
                                                                self.search_space['query_expansion_method'])
        
        if params['query_expansion_method'] == 'pass_query_expansion':
            return

        if 'query_expansion_generator_module_type' in self.search_space:
            params['query_expansion_generator_module_type'] = trial.suggest_categorical(
                'query_expansion_generator_module_type', 
                self.search_space['query_expansion_generator_module_type']
            )
        
        if 'query_expansion_llm' in self.search_space:
            params['query_expansion_llm'] = trial.suggest_categorical(
                'query_expansion_llm', 
                self.search_space['query_expansion_llm']
            )
        
        if 'query_expansion_model' in self.search_space:
            params['query_expansion_model'] = trial.suggest_categorical(
                'query_expansion_model', 
                self.search_space['query_expansion_model']
            )

        if 'query_expansion_temperature' in self.search_space:
            params['query_expansion_temperature'] = self._suggest_value(
                trial, 'query_expansion_temperature', 
                self.search_space['query_expansion_temperature'], 'float'
            )
        
        if 'query_expansion_max_token' in self.search_space:
            params['query_expansion_max_token'] = self._suggest_value(
                trial, 'query_expansion_max_token', 
                self.search_space['query_expansion_max_token'], 'int'
            )
        
        self._add_query_expansion_retrieval_params(trial, params)

    def _add_query_expansion_retrieval_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'query_expansion_retrieval_method' in self.search_space:
            params['query_expansion_retrieval_method'] = trial.suggest_categorical(
                'query_expansion_retrieval_method', 
                self.search_space['query_expansion_retrieval_method']
            )
            
            if params['query_expansion_retrieval_method'] == 'bm25' and 'query_expansion_bm25_tokenizer' in self.search_space:
                params['query_expansion_bm25_tokenizer'] = trial.suggest_categorical(
                    'query_expansion_bm25_tokenizer', 
                    self.search_space['query_expansion_bm25_tokenizer']
                )
            elif params['query_expansion_retrieval_method'] == 'vectordb' and 'query_expansion_vectordb_name' in self.search_space:
                params['query_expansion_vectordb_name'] = trial.suggest_categorical(
                    'query_expansion_vectordb_name', 
                    self.search_space['query_expansion_vectordb_name']
                )
                    
    def _suggest_retrieval_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'retriever_top_k' in self.search_space:
            params['retriever_top_k'] = self._suggest_value(
                trial, 'retriever_top_k', 
                self.search_space['retriever_top_k'], 'int'
            )
        
        is_qe_active = False
        if 'query_expansion_config' in params:
            is_qe_active = params['query_expansion_config'] != 'pass_query_expansion'
        elif 'query_expansion_method' in params:
            is_qe_active = params['query_expansion_method'] != 'pass_query_expansion'
        
        if is_qe_active:
            print(f"[DEBUG] Active query expansion detected, skipping retrieval method params")
            return
        
        if 'retrieval_config' in self.search_space:
            params['retrieval_config'] = trial.suggest_categorical('retrieval_config', 
                                                                self.search_space['retrieval_config'])
            return
        
        if 'retrieval_method' in self.search_space:
            params['retrieval_method'] = trial.suggest_categorical('retrieval_method', 
                                                                self.search_space['retrieval_method'])
            
            if params.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in self.search_space:
                params['bm25_tokenizer'] = trial.suggest_categorical('bm25_tokenizer', 
                                                                self.search_space['bm25_tokenizer'])
            elif params.get('retrieval_method') == 'vectordb' and 'vectordb_name' in self.search_space:
                params['vectordb_name'] = trial.suggest_categorical('vectordb_name', 
                                                                self.search_space['vectordb_name'])
    
    def _suggest_filter_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        reranker_top_k = params.get('reranker_top_k', None)
        if reranker_top_k == 1:
            params['passage_filter_config'] = 'pass_passage_filter'
            params['passage_filter_method'] = 'pass_passage_filter'
            print(f"  Auto-setting filter to 'pass' because reranker_top_k=1")
            return
        
        if 'passage_filter_method' in self.search_space:
            params['passage_filter_method'] = trial.suggest_categorical(
                'passage_filter_method', 
                self.search_space['passage_filter_method']
            )
            
            if params['passage_filter_method'] == 'pass_passage_filter':
                return
            
            filter_method = params['passage_filter_method']

            if filter_method == 'threshold_cutoff' and 'threshold_cutoff_threshold' in self.search_space:
                threshold_range = self.search_space['threshold_cutoff_threshold']
                params['threshold'] = trial.suggest_float(
                    'threshold_cutoff_threshold',
                    threshold_range[0], threshold_range[1]
                )
            elif filter_method == 'percentile_cutoff' and 'percentile_cutoff_percentile' in self.search_space:
                percentile_range = self.search_space['percentile_cutoff_percentile']
                params['percentile'] = trial.suggest_float(
                    'percentile_cutoff_percentile',
                    percentile_range[0], percentile_range[1]
                )
            elif filter_method == 'similarity_threshold_cutoff' and 'similarity_threshold_cutoff_threshold' in self.search_space:
                threshold_range = self.search_space['similarity_threshold_cutoff_threshold']
                params['threshold'] = trial.suggest_float(
                    'similarity_threshold_cutoff_threshold',
                    threshold_range[0], threshold_range[1]
                )
            elif filter_method == 'similarity_percentile_cutoff' and 'similarity_percentile_cutoff_percentile' in self.search_space:
                percentile_range = self.search_space['similarity_percentile_cutoff_percentile']
                params['percentile'] = trial.suggest_float(
                    'similarity_percentile_cutoff_percentile',
                    percentile_range[0], percentile_range[1]
                )
            
            return
        
        if 'passage_filter_config' in self.search_space:
            filter_config = trial.suggest_categorical('passage_filter_config', 
                                                    self.search_space['passage_filter_config'])
            params['passage_filter_config'] = filter_config
            return
    
    def _suggest_reranker_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'reranker_config' in self.search_space:
            params['reranker_config'] = trial.suggest_categorical('reranker_config',
                                                                self.search_space['reranker_config'])
            
            if 'reranker_top_k' in self.search_space:
                if 'retriever_top_k' in params:
                    reranker_range = self.search_space['reranker_top_k']
                    if isinstance(reranker_range, tuple):
                        max_reranker_k = min(reranker_range[1], params['retriever_top_k'])
                        params['reranker_top_k'] = trial.suggest_int('reranker_top_k', 
                                                                    reranker_range[0], 
                                                                    max_reranker_k)
                    else:
                        params['reranker_top_k'] = self._suggest_value(
                            trial, 'reranker_top_k', 
                            self.search_space['reranker_top_k'], 'int'
                        )
                else:
                    params['reranker_top_k'] = self._suggest_value(
                        trial, 'reranker_top_k', 
                        self.search_space['reranker_top_k'], 'int'
                    )
            return

        if 'passage_reranker_method' not in self.search_space:
            return
            
        params['passage_reranker_method'] = trial.suggest_categorical('passage_reranker_method', 
                                                                    self.search_space['passage_reranker_method'])
        
        if params['passage_reranker_method'] == 'pass_reranker':
            return
        
        reranker_method = params['passage_reranker_method']

        if reranker_method == 'sap_api':
            unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
            api_endpoints = unified_params.get('api_endpoints', {})
            
            if 'sap_api' in api_endpoints:
                params['reranker_api_url'] = api_endpoints['sap_api']

            params['reranker_model_name'] = 'cohere-rerank-v3.5'
        else:
            model_key = f"{reranker_method}_model_name"
            if model_key in self.search_space:
                params['reranker_model_name'] = trial.suggest_categorical(model_key,
                                                                        self.search_space[model_key])
            else:
                model_key = f"{reranker_method}_model"
                if model_key in self.search_space:
                    params['reranker_model'] = trial.suggest_categorical(model_key,
                                                                    self.search_space[model_key])
            
        if 'reranker_top_k' in self.search_space:
            if 'retriever_top_k' in params:
                reranker_range = self.search_space['reranker_top_k']
                if isinstance(reranker_range, tuple):
                    max_reranker_k = min(reranker_range[1], params['retriever_top_k'])
                    params['reranker_top_k'] = trial.suggest_int('reranker_top_k', 
                                                                reranker_range[0], 
                                                                max_reranker_k)
                else:
                    params['reranker_top_k'] = self._suggest_value(
                        trial, 'reranker_top_k', 
                        self.search_space['reranker_top_k'], 'int'
                    )
            else:
                params['reranker_top_k'] = self._suggest_value(
                    trial, 'reranker_top_k', 
                    self.search_space['reranker_top_k'], 'int'
                )
    
    def _suggest_compressor_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'passage_compressor_config' in self.search_space:
            comp_config_str = trial.suggest_categorical('passage_compressor_config',
                                                    self.search_space['passage_compressor_config'])
            
            if comp_config_str != 'pass_compressor':
                if comp_config_str == 'lexrank':
                    params['passage_compressor_method'] = 'lexrank'

                    if 'lexrank_compression_ratio' in self.search_space:
                        params['compression_ratio'] = self._suggest_value(
                            trial, 'lexrank_compression_ratio',
                            self.search_space['lexrank_compression_ratio'], 'float'
                        )
                    elif 'compression_ratio' in self.search_space:
                        params['compression_ratio'] = self._suggest_value(
                            trial, 'compression_ratio',
                            self.search_space['compression_ratio'], 'float'
                        )
                    
                    if 'lexrank_threshold' in self.search_space:
                        params['threshold'] = self._suggest_value(
                            trial, 'lexrank_threshold',
                            self.search_space['lexrank_threshold'], 'float'
                        )
                    
                    if 'lexrank_damping' in self.search_space:
                        params['damping'] = self._suggest_value(
                            trial, 'lexrank_damping',
                            self.search_space['lexrank_damping'], 'float'
                        )
                    
                    if 'lexrank_max_iterations' in self.search_space:
                        params['max_iterations'] = self._suggest_value(
                            trial, 'lexrank_max_iterations',
                            self.search_space['lexrank_max_iterations'], 'int'
                        )
                
                elif comp_config_str.startswith('spacy'):
                    if '::' in comp_config_str:
                        parts = comp_config_str.split('::', 1)
                        params['passage_compressor_method'] = 'spacy'
                        if len(parts) > 1:
                            params['spacy_model'] = parts[1]
                    else:
                        params['passage_compressor_method'] = 'spacy'

                    if 'spacy_compression_ratio' in self.search_space:
                        params['compression_ratio'] = self._suggest_value(
                            trial, 'spacy_compression_ratio',
                            self.search_space['spacy_compression_ratio'], 'float'
                        )
                    elif 'compression_ratio' in self.search_space:
                        params['compression_ratio'] = self._suggest_value(
                            trial, 'compression_ratio',
                            self.search_space['compression_ratio'], 'float'
                        )
                
                elif '::' in comp_config_str:
                    parts = comp_config_str.split('::', 3)
                    if len(parts) >= 3:
                        method, gen_type, model = parts[0], parts[1], parts[2]
                        params['passage_compressor_method'] = method
                        params['compressor_generator_module_type'] = gen_type
                        params['compressor_model'] = model
                        
                        unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                        for comp_config in unified_params.get('compressor_configs', []):
                            if (comp_config['method'] == method and 
                                comp_config['generator_module_type'] == gen_type and 
                                model in comp_config['models']):
                                params['compressor_llm'] = comp_config.get('llm', 'openai')
                                if gen_type == 'sap_api':
                                    params['compressor_api_url'] = comp_config.get('api_url')
                                elif gen_type == 'vllm':
                                    params['compressor_llm'] = model
                                break
                else:
                    params['passage_compressor_method'] = comp_config_str
            else:
                params['passage_compressor_method'] = comp_config_str
            
            if 'compressor_batch' in self.search_space:
                params['compressor_batch'] = trial.suggest_categorical('compressor_batch',
                                                                    self.search_space['compressor_batch'])
            return

        if 'passage_compressor_method' not in self.search_space:
            return
            
        params['passage_compressor_method'] = trial.suggest_categorical('passage_compressor_method', 
                                                                    self.search_space['passage_compressor_method'])
        
        if params['passage_compressor_method'] == 'pass_compressor':
            return

        method = params['passage_compressor_method']
        
        if method == 'lexrank':
            if 'compression_ratio' in self.search_space:
                params['compression_ratio'] = self._suggest_value(
                    trial, 'compression_ratio',
                    self.search_space['compression_ratio'], 'float'
                )
        
        elif method == 'spacy':
            if 'compression_ratio' in self.search_space:
                params['compression_ratio'] = self._suggest_value(
                    trial, 'compression_ratio',
                    self.search_space['compression_ratio'], 'float'
                )
            if 'spacy_model' in self.search_space:
                params['spacy_model'] = trial.suggest_categorical(
                    'spacy_model',
                    self.search_space['spacy_model']
                )
        
        elif method in ['tree_summarize', 'refine']:
            if 'compressor_generator_module_type' in self.search_space:
                params['compressor_generator_module_type'] = trial.suggest_categorical(
                    'compressor_generator_module_type', 
                    self.search_space['compressor_generator_module_type']
                )
            
            if 'compressor_llm' in self.search_space:
                params['compressor_llm'] = trial.suggest_categorical('compressor_llm', 
                                                                self.search_space['compressor_llm'])
            
            if 'compressor_model' in self.search_space:
                params['compressor_model'] = trial.suggest_categorical('compressor_model', 
                                                                    self.search_space['compressor_model'])
            
            if 'compressor_batch' in self.search_space:
                params['compressor_batch'] = self._suggest_value(
                    trial, 'compressor_batch',
                    self.search_space['compressor_batch'], 'int'
                )
    
    def _suggest_prompt_maker_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'prompt_config' in self.search_space:
            prompt_configs = self.search_space['prompt_config']
            if prompt_configs:
                params['prompt_config'] = trial.suggest_categorical('prompt_config', prompt_configs)
            return
        
        if 'prompt_maker_method' not in self.search_space:
            return
        
        prompt_methods = self.search_space.get('prompt_maker_method', [])
        if not prompt_methods:
            return
            
        params['prompt_maker_method'] = trial.suggest_categorical('prompt_maker_method', prompt_methods)
        
        if params['prompt_maker_method'] == 'pass_prompt_maker':
            return
            
        if 'prompt_template_idx' in self.search_space:
            template_indices = self.search_space.get('prompt_template_idx', [])
            if template_indices:
                params['prompt_template_idx'] = self._suggest_value(
                    trial, 'prompt_template_idx',
                    template_indices, 'int'
                )
    
    def _suggest_generator_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'generator_config' in self.search_space:
            generator_configs = self.search_space.get('generator_config', {})
            
            if isinstance(generator_configs, dict) and 'values' in generator_configs:
                gen_config_str = trial.suggest_categorical('generator_config', 
                                                        generator_configs['values'])
                
                metadata = generator_configs.get('metadata', {})
                if metadata and gen_config_str in metadata:
                    config_metadata = metadata[gen_config_str]
                    
                    module_type, model = gen_config_str.split('::', 1)
                    params['generator_module_type'] = module_type
                    params['generator_model'] = model
                    
                    if 'api_url' in config_metadata:
                        params['generator_api_url'] = config_metadata['api_url']
                    if 'llm' in config_metadata:
                        params['generator_llm'] = config_metadata['llm']
            else:
                gen_config_str = trial.suggest_categorical('generator_config', generator_configs)
                module_type, model = gen_config_str.split('::', 1)
                params['generator_module_type'] = module_type
                params['generator_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('generator')
                for module_config in unified_params.get('module_configs', []):
                    if module_config['module_type'] == module_type and model in module_config['models']:
                        if module_type == 'sap_api':
                            params['generator_api_url'] = module_config.get('api_url')
                            params['generator_llm'] = module_config.get('llm', 'mistralai')
                        elif module_type == 'vllm':
                            params['generator_llm'] = model
                        else:
                            params['generator_llm'] = module_config.get('llm', 'openai')
                        break
            
            if 'generator_temperature' in self.search_space:
                temp_spec = self.search_space.get('generator_temperature')
                if temp_spec:
                    temp_value = self._suggest_value(
                        trial, 'generator_temperature',
                        temp_spec, 'float'
                    )
                    params['generator_temperature'] = round(float(temp_value), 4)
            
            if 'generator_max_tokens' in self.search_space:
                if params.get('generator_module_type') == 'sap_api':
                    params['generator_max_tokens'] = trial.suggest_int(
                        'generator_max_tokens',
                        self.search_space['generator_max_tokens'][0],
                        self.search_space['generator_max_tokens'][1]
                    )
            
            return
        
        if 'generator_module_type' in self.search_space:
            module_types = self.search_space.get('generator_module_type', [])
            if module_types:
                params['generator_module_type'] = trial.suggest_categorical('generator_module_type', module_types)

        if 'generator_model' in self.search_space:
            models = self.search_space.get('generator_model', [])
            if not models:
                raise ValueError("No generator models available in search space")
            params['generator_model'] = trial.suggest_categorical('generator_model', models)
            
        if 'generator_temperature' in self.search_space:
            temp_spec = self.search_space.get('generator_temperature')
            if temp_spec:
                temp_value = self._suggest_value(
                    trial, 'generator_temperature',
                    temp_spec, 'float'
                )
                params['generator_temperature'] = round(float(temp_value), 4)