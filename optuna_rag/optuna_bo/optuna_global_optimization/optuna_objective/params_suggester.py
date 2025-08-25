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

        if not self.has_query_expansion or params.get('query_expansion_method') == 'pass_query_expansion':
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
            params['query_expansion_config'] = trial.suggest_categorical('query_expansion_config',
                                                                    self.search_space['query_expansion_config'])
            return
        
        if 'query_expansion_method' not in self.search_space:
            return
            
        params['query_expansion_method'] = trial.suggest_categorical('query_expansion_method', 
                                                                self.search_space['query_expansion_method'])
        
        if params['query_expansion_method'] == 'pass_query_expansion':
            return

        if 'retriever_top_k' in self.search_space:
            params['retriever_top_k'] = self._suggest_value(
                trial, 'retriever_top_k', 
                self.search_space['retriever_top_k'], 'int'
            )

        if 'retrieval_method' in params:
            del params['retrieval_method']
        if 'bm25_tokenizer' in params:
            del params['bm25_tokenizer']
        if 'vectordb_name' in params:
            del params['vectordb_name']
        
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
        
        if params['query_expansion_method'] == 'hyde' and 'query_expansion_max_token' in self.search_space:
            params['query_expansion_max_token'] = self._suggest_value(
                trial, 'query_expansion_max_token', 
                self.search_space['query_expansion_max_token'], 'int'
            )
        elif params['query_expansion_method'] == 'multi_query_expansion' and 'query_expansion_temperature' in self.search_space:
            params['query_expansion_temperature'] = self._suggest_value(
                trial, 'query_expansion_temperature', 
                self.search_space['query_expansion_temperature'], 'float'
            )
        
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
        if self.has_query_expansion and params.get('query_expansion_method') and params.get('query_expansion_method') != 'pass_query_expansion':
            print(f"[DEBUG] Skipping ALL retrieval params - query expansion is active: {params.get('query_expansion_method')}")
            return

        if 'retriever_top_k' in self.search_space:
            params['retriever_top_k'] = self._suggest_value(
                trial, 'retriever_top_k', 
                self.search_space['retriever_top_k'], 'int'
            )

        if 'retrieval_config' in self.search_space:
            params['retrieval_config'] = trial.suggest_categorical('retrieval_config', 
                                                                self.search_space['retrieval_config'])
            return

        if 'retrieval_method' not in self.search_space:
            if self.has_retrieval and 'retrieval_method' not in params:
                retrieval_params = self.config_generator.extract_unified_parameters('retrieval')
                if retrieval_params.get('methods'):
                    params['retrieval_method'] = retrieval_params['methods'][0]
                    
                    if params['retrieval_method'] == 'bm25' and retrieval_params.get('bm25_tokenizers'):
                        params['bm25_tokenizer'] = retrieval_params['bm25_tokenizers'][0]
                    elif params['retrieval_method'] == 'vectordb' and retrieval_params.get('vectordb_names'):
                        params['vectordb_name'] = retrieval_params['vectordb_names'][0]
            return
        
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
        
        if 'passage_filter_method' not in self.search_space:
            return
            
        params['passage_filter_method'] = trial.suggest_categorical('passage_filter_method', 
                                                                self.search_space['passage_filter_method'])
        
        if params['passage_filter_method'] == 'pass_passage_filter':
            return
            
        filter_method = params['passage_filter_method']

        if filter_method == 'threshold_cutoff' and 'threshold_cutoff_threshold' in self.search_space:
            params['threshold'] = self._suggest_value(
                trial, 'threshold_cutoff_threshold',
                self.search_space['threshold_cutoff_threshold'], 'float'
            )
        elif filter_method == 'similarity_threshold_cutoff' and 'similarity_threshold_cutoff_threshold' in self.search_space:
            params['threshold'] = self._suggest_value(
                trial, 'similarity_threshold_cutoff_threshold',
                self.search_space['similarity_threshold_cutoff_threshold'], 'float'
            )
        elif filter_method == 'percentile_cutoff' and 'percentile_cutoff_percentile' in self.search_space:
            params['percentile'] = self._suggest_value(
                trial, 'percentile_cutoff_percentile',
                self.search_space['percentile_cutoff_percentile'], 'float'
            )
        elif filter_method == 'similarity_percentile_cutoff' and 'similarity_percentile_cutoff_percentile' in self.search_space:
            params['percentile'] = self._suggest_value(
                trial, 'similarity_percentile_cutoff_percentile',
                self.search_space['similarity_percentile_cutoff_percentile'], 'float'
            )
    
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
            params['passage_compressor_config'] = trial.suggest_categorical('passage_compressor_config',
                                                                self.search_space['passage_compressor_config'])
            return
        
        if 'compressor_config' in self.search_space:
            params['compressor_config'] = trial.suggest_categorical('compressor_config',
                                                                self.search_space['compressor_config'])
            
            if 'compressor_batch' in self.search_space:
                params['compressor_batch'] = self._suggest_value(
                    trial, 'compressor_batch',
                    self.search_space['compressor_batch'], 'int'
                )
            return
        
        if 'passage_compressor_method' not in self.search_space:
            return
            
        params['passage_compressor_method'] = trial.suggest_categorical('passage_compressor_method', 
                                                                    self.search_space['passage_compressor_method'])
        
        if params['passage_compressor_method'] == 'pass_compressor':
            return
        
        if params['passage_compressor_method'] in ['tree_summarize', 'refine']:
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
        
        elif params['passage_compressor_method'] == 'lexrank':
            if 'compressor_compression_ratio' in self.search_space:
                params['compressor_compression_ratio'] = self._suggest_value(
                    trial, 'compressor_compression_ratio',
                    self.search_space['compressor_compression_ratio'], 'float'
                )
            if 'compressor_threshold' in self.search_space:
                params['compressor_threshold'] = self._suggest_value(
                    trial, 'compressor_threshold', 
                    self.search_space['compressor_threshold'], 'float'
                )
            if 'compressor_damping' in self.search_space:
                params['compressor_damping'] = self._suggest_value(
                    trial, 'compressor_damping',
                    self.search_space['compressor_damping'], 'float'
                )
            if 'compressor_max_iterations' in self.search_space:
                params['compressor_max_iterations'] = self._suggest_value(
                    trial, 'compressor_max_iterations',
                    self.search_space['compressor_max_iterations'], 'int'
                )
        
        elif params['passage_compressor_method'] == 'spacy':
            if 'compressor_compression_ratio' in self.search_space:
                params['compressor_compression_ratio'] = self._suggest_value(
                    trial, 'compressor_compression_ratio',
                    self.search_space['compressor_compression_ratio'], 'float'
                )
            if 'compressor_spacy_model' in self.search_space:
                params['compressor_spacy_model'] = trial.suggest_categorical('compressor_spacy_model',
                                                                        self.search_space['compressor_spacy_model'])
    
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