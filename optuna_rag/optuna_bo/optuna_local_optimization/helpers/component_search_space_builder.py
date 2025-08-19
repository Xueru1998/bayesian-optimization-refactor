import optuna
from typing import Dict, Any, List, Union, Tuple


class ComponentSearchSpaceBuilder:
    
    def __init__(self, config_generator, config_extractor):
        self.config_generator = config_generator
        self.config_extractor = config_extractor
        self.current_component = None 
    
    def build_component_search_space(self, component: str, fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        self.current_component = component  

        if hasattr(self.config_extractor, 'set_optimizing_component'):
            self.config_extractor.set_optimizing_component(component)
        
        full_search_space = self.config_extractor.extract_search_space()
        component_search_space = {}
        
        if component == 'query_expansion':
            component_search_space = self._extract_query_expansion_search_space(full_search_space, fixed_config)
        elif component == 'retrieval':
            component_search_space = self._extract_retrieval_search_space(full_search_space, fixed_config)
        elif component == 'passage_reranker':
            component_search_space = self._extract_reranker_search_space(full_search_space, fixed_config)
        elif component == 'passage_filter':
            component_search_space = self._extract_filter_search_space(full_search_space, fixed_config)
        elif component == 'passage_compressor':
            component_search_space = self._extract_compressor_search_space(full_search_space, fixed_config)
        elif component == 'prompt_maker_generator':
            component_search_space = self._extract_prompt_generator_search_space(full_search_space, fixed_config)

        if hasattr(self.config_extractor, 'set_optimizing_component'):
            self.config_extractor.set_optimizing_component(None)
        
        return component_search_space
    
    def suggest_component_params(self, trial: optuna.Trial, component: str, search_space: Dict[str, Any], fixed_config: Dict[str, Any] = None) -> Dict[str, Any]:
        if fixed_config is None:
            fixed_config = {}
        trial_config = {}
        
        if component == 'query_expansion':
            if 'query_expansion_config' in search_space:
                qe_config = trial.suggest_categorical('query_expansion_config',
                                                    search_space['query_expansion_config'])
                trial_config['query_expansion_config'] = qe_config
                
                if qe_config == 'pass_query_expansion':
                    if 'retriever_top_k' in search_space:
                        if isinstance(search_space['retriever_top_k'], list):
                            trial_config['retriever_top_k'] = trial.suggest_categorical('retriever_top_k', search_space['retriever_top_k'])
                        else:
                            trial_config['retriever_top_k'] = trial.suggest_int(
                                'retriever_top_k',
                                search_space['retriever_top_k'][0],
                                search_space['retriever_top_k'][1]
                            )
                    
                    if 'retrieval_method' in search_space:
                        retrieval_method = trial.suggest_categorical('retrieval_method', search_space['retrieval_method'])
                        trial_config['retrieval_method'] = retrieval_method
                        
                        if retrieval_method == 'bm25' and 'bm25_tokenizer' in search_space:
                            trial_config['bm25_tokenizer'] = trial.suggest_categorical('bm25_tokenizer', search_space['bm25_tokenizer'])
                        elif retrieval_method == 'vectordb' and 'vectordb_name' in search_space:
                            trial_config['vectordb_name'] = trial.suggest_categorical('vectordb_name', search_space['vectordb_name'])
                    
                    return trial_config
                
                parts = qe_config.split('::')
                if len(parts) >= 3:
                    method = parts[0]
                    gen_type = parts[1]
                    model = parts[2]
                    
                    trial_config['query_expansion_method'] = method
                    trial_config['query_expansion_generator_module_type'] = gen_type
                    trial_config['query_expansion_model'] = model
                    
                    unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                    for gen_config in unified_params.get('generator_configs', []):
                        if (gen_config['method'] == method and 
                            gen_config['generator_module_type'] == gen_type and 
                            model in gen_config['models']):
                            if gen_type == 'sap_api':
                                trial_config['query_expansion_api_url'] = gen_config.get('api_url')
                                trial_config['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                            else:
                                trial_config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                            break
                    
                    if method == 'hyde' and 'query_expansion_max_token' in search_space:
                        if isinstance(search_space['query_expansion_max_token'], list):
                            trial_config['query_expansion_max_token'] = trial.suggest_categorical(
                                'query_expansion_max_token', search_space['query_expansion_max_token']
                            )
                        else:
                            trial_config['query_expansion_max_token'] = trial.suggest_int(
                                'query_expansion_max_token', 
                                search_space['query_expansion_max_token'][0],
                                search_space['query_expansion_max_token'][1]
                            )
                    elif method == 'multi_query_expansion' and 'query_expansion_temperature' in search_space:
                        if isinstance(search_space['query_expansion_temperature'], list):
                            trial_config['query_expansion_temperature'] = trial.suggest_categorical(
                                'query_expansion_temperature', search_space['query_expansion_temperature']
                            )
                        else:
                            trial_config['query_expansion_temperature'] = trial.suggest_float(
                                'query_expansion_temperature',
                                search_space['query_expansion_temperature'][0],
                                search_space['query_expansion_temperature'][1]
                            )
                
                if 'query_expansion_retrieval_method' in search_space:
                    trial_config['query_expansion_retrieval_method'] = trial.suggest_categorical(
                        'query_expansion_retrieval_method', 
                        search_space['query_expansion_retrieval_method']
                    )
                    
                    if trial_config['query_expansion_retrieval_method'] == 'bm25' and 'query_expansion_bm25_tokenizer' in search_space:
                        trial_config['query_expansion_bm25_tokenizer'] = trial.suggest_categorical(
                            'query_expansion_bm25_tokenizer', 
                            search_space['query_expansion_bm25_tokenizer']
                        )
                    elif trial_config['query_expansion_retrieval_method'] == 'vectordb' and 'query_expansion_vectordb_name' in search_space:
                        trial_config['query_expansion_vectordb_name'] = trial.suggest_categorical(
                            'query_expansion_vectordb_name', 
                            search_space['query_expansion_vectordb_name']
                        )
                
                if 'retriever_top_k' in search_space:
                    if isinstance(search_space['retriever_top_k'], list):
                        trial_config['retriever_top_k'] = trial.suggest_categorical('retriever_top_k', search_space['retriever_top_k'])
                    else:
                        trial_config['retriever_top_k'] = trial.suggest_int(
                            'retriever_top_k',
                            search_space['retriever_top_k'][0],
                            search_space['retriever_top_k'][1]
                        )
                
                return trial_config
            
            elif 'query_expansion_method' in search_space:
                qe_method = trial.suggest_categorical('query_expansion_method', search_space['query_expansion_method'])
                trial_config['query_expansion_method'] = qe_method
                
                if qe_method != 'pass_query_expansion':
                    for key, value in search_space.items():
                        if key.startswith('query_expansion_') and key not in ['query_expansion_method'] and key not in trial_config:
                            if isinstance(value, list):
                                trial_config[key] = trial.suggest_categorical(key, value)
                            elif isinstance(value, tuple) and len(value) == 2:
                                if isinstance(value[0], float):
                                    trial_config[key] = trial.suggest_float(key, value[0], value[1])
                                else:
                                    trial_config[key] = trial.suggest_int(key, value[0], value[1])
            
            if 'retriever_top_k' in search_space and 'retriever_top_k' not in trial_config:
                if isinstance(search_space['retriever_top_k'], list):
                    trial_config['retriever_top_k'] = trial.suggest_categorical('retriever_top_k', search_space['retriever_top_k'])
                else:
                    trial_config['retriever_top_k'] = trial.suggest_int(
                        'retriever_top_k',
                        search_space['retriever_top_k'][0],
                        search_space['retriever_top_k'][1]
                    )
            
            if 'retrieval_method' in search_space:
                retrieval_method = trial.suggest_categorical('retrieval_method', search_space['retrieval_method'])
                trial_config['retrieval_method'] = retrieval_method
                
                if retrieval_method == 'bm25' and 'bm25_tokenizer' in search_space:
                    trial_config['bm25_tokenizer'] = trial.suggest_categorical('bm25_tokenizer', search_space['bm25_tokenizer'])
                elif retrieval_method == 'vectordb' and 'vectordb_name' in search_space:
                    trial_config['vectordb_name'] = trial.suggest_categorical('vectordb_name', search_space['vectordb_name'])
            else:
                fixed_retrieval_method = fixed_config.get('retrieval_method')
                if fixed_retrieval_method:
                    if fixed_retrieval_method == 'bm25' and 'bm25_tokenizer' in search_space:
                        trial_config['bm25_tokenizer'] = trial.suggest_categorical('bm25_tokenizer', search_space['bm25_tokenizer'])
                    elif fixed_retrieval_method == 'vectordb' and 'vectordb_name' in search_space:
                        trial_config['vectordb_name'] = trial.suggest_categorical('vectordb_name', search_space['vectordb_name'])
        
        elif component == 'passage_reranker':
            if 'passage_reranker_method' in search_space:
                reranker_method = trial.suggest_categorical('passage_reranker_method', search_space['passage_reranker_method'])
                trial_config['passage_reranker_method'] = reranker_method
                
                if reranker_method != 'pass_reranker':
                    if reranker_method == 'sap_api':
                        unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
                        api_endpoints = unified_params.get('api_endpoints', {})
                        
                        if 'sap_api' in api_endpoints:
                            trial_config['reranker_api_url'] = api_endpoints['sap_api']

                        trial_config['reranker_model_name'] = 'cohere-rerank-v3.5'
                    else:
                        method_models_key = f"{reranker_method}_models"
                        if method_models_key in search_space:
                            model = trial.suggest_categorical(method_models_key, search_space[method_models_key])
                            if reranker_method == 'flashrank_reranker':
                                trial_config['reranker_model'] = model
                            else:
                                trial_config['reranker_model_name'] = model
                    
                    if 'reranker_top_k' in search_space:
                        if isinstance(search_space['reranker_top_k'], list):
                            trial_config['reranker_top_k'] = trial.suggest_categorical('reranker_top_k', search_space['reranker_top_k'])
                        else:
                            trial_config['reranker_top_k'] = trial.suggest_int(
                                'reranker_top_k',
                                search_space['reranker_top_k'][0],
                                search_space['reranker_top_k'][1]
                            )
        
        elif component == 'passage_filter' and 'passage_filter_method' in search_space:
            filter_method = trial.suggest_categorical('passage_filter_method', search_space['passage_filter_method'])
            trial_config['passage_filter_method'] = filter_method
            
            if filter_method != 'pass_passage_filter':
                if filter_method in ['threshold_cutoff', 'similarity_threshold_cutoff']:
                    param_name = f"{filter_method}_threshold"
                    if param_name in search_space:
                        if isinstance(search_space[param_name], list):
                            threshold = trial.suggest_categorical(param_name, search_space[param_name])
                        else:
                            threshold = trial.suggest_float(param_name, 
                                                        search_space[param_name][0], 
                                                        search_space[param_name][1])
                        trial_config['threshold'] = threshold
                
                elif filter_method in ['percentile_cutoff', 'similarity_percentile_cutoff']:
                    param_name = f"{filter_method}_percentile"
                    if param_name in search_space:
                        if isinstance(search_space[param_name], list):
                            percentile = trial.suggest_categorical(param_name, search_space[param_name])
                        else:
                            percentile = trial.suggest_float(param_name, 
                                                        search_space[param_name][0], 
                                                        search_space[param_name][1])
                        trial_config['percentile'] = percentile
        
        elif component == 'retrieval':
            if 'retrieval_method' in search_space:
                retrieval_method = trial.suggest_categorical('retrieval_method', search_space['retrieval_method'])
                trial_config['retrieval_method'] = retrieval_method
                
                if retrieval_method == 'bm25' and 'bm25_tokenizer' in search_space:
                    trial_config['bm25_tokenizer'] = trial.suggest_categorical('bm25_tokenizer', search_space['bm25_tokenizer'])
                elif retrieval_method == 'vectordb' and 'vectordb_name' in search_space:
                    trial_config['vectordb_name'] = trial.suggest_categorical('vectordb_name', search_space['vectordb_name'])
            else:
                fixed_retrieval_method = fixed_config.get('retrieval_method')
                if fixed_retrieval_method:
                    if fixed_retrieval_method == 'bm25' and 'bm25_tokenizer' in search_space:
                        trial_config['bm25_tokenizer'] = trial.suggest_categorical('bm25_tokenizer', search_space['bm25_tokenizer'])
                    elif fixed_retrieval_method == 'vectordb' and 'vectordb_name' in search_space:
                        trial_config['vectordb_name'] = trial.suggest_categorical('vectordb_name', search_space['vectordb_name'])
            
            if 'retriever_top_k' in search_space:
                if isinstance(search_space['retriever_top_k'], list):
                    trial_config['retriever_top_k'] = trial.suggest_categorical('retriever_top_k', search_space['retriever_top_k'])
                else:
                    trial_config['retriever_top_k'] = trial.suggest_int(
                        'retriever_top_k',
                        search_space['retriever_top_k'][0],
                        search_space['retriever_top_k'][1]
                    )
        
        elif component == 'passage_compressor':
            if 'passage_compressor_config' in search_space:
                comp_config = trial.suggest_categorical('passage_compressor_config',
                                                    search_space['passage_compressor_config'])
                trial_config['passage_compressor_config'] = comp_config

                if comp_config == 'pass_compressor':
                    trial_config['passage_compressor_method'] = 'pass_compressor'
                    print(f"[DEBUG] Setting passage_compressor_method to 'pass_compressor'")
                elif comp_config == 'lexrank':
                    trial_config['passage_compressor_method'] = 'lexrank'
                    print(f"[DEBUG] Setting passage_compressor_method to 'lexrank'")

                    if 'lexrank_threshold' in search_space:
                        if isinstance(search_space['lexrank_threshold'], list):
                            trial_config['threshold'] = trial.suggest_categorical('lexrank_threshold', search_space['lexrank_threshold'])
                        else:
                            trial_config['threshold'] = trial.suggest_float('lexrank_threshold', 
                                                        search_space['lexrank_threshold'][0], 
                                                        search_space['lexrank_threshold'][1])
                    
                    if 'lexrank_damping' in search_space:
                        if isinstance(search_space['lexrank_damping'], list):
                            trial_config['damping'] = trial.suggest_categorical('lexrank_damping', search_space['lexrank_damping'])
                        else:
                            trial_config['damping'] = trial.suggest_float('lexrank_damping',
                                                        search_space['lexrank_damping'][0],
                                                        search_space['lexrank_damping'][1])
                    
                    if 'lexrank_max_iterations' in search_space:
                        if isinstance(search_space['lexrank_max_iterations'], list):
                            trial_config['max_iterations'] = trial.suggest_categorical('lexrank_max_iterations', search_space['lexrank_max_iterations'])
                        else:
                            trial_config['max_iterations'] = trial.suggest_int('lexrank_max_iterations',
                                                            search_space['lexrank_max_iterations'][0],
                                                            search_space['lexrank_max_iterations'][1])
                    
                    if 'lexrank_compression_ratio' in search_space:
                        if isinstance(search_space['lexrank_compression_ratio'], list):
                            trial_config['compression_ratio'] = trial.suggest_categorical('lexrank_compression_ratio', search_space['lexrank_compression_ratio'])
                        else:
                            trial_config['compression_ratio'] = trial.suggest_float('lexrank_compression_ratio',
                                                                search_space['lexrank_compression_ratio'][0],
                                                                search_space['lexrank_compression_ratio'][1])
                    elif 'compression_ratio' in search_space:
                        if isinstance(search_space['compression_ratio'], list):
                            trial_config['compression_ratio'] = trial.suggest_categorical('compression_ratio', search_space['compression_ratio'])
                        else:
                            trial_config['compression_ratio'] = trial.suggest_float('compression_ratio',
                                                                search_space['compression_ratio'][0],
                                                                search_space['compression_ratio'][1])
                                                                
                elif comp_config.startswith('spacy::'):
                    parts = comp_config.split('::', 1)
                    trial_config['passage_compressor_method'] = 'spacy'
                    trial_config['spacy_model'] = parts[1] if len(parts) > 1 else 'en_core_web_sm'
                    print(f"[DEBUG] Setting passage_compressor_method to 'spacy' with model '{trial_config['spacy_model']}'")

                    if 'spacy_compression_ratio' in search_space:
                        if isinstance(search_space['spacy_compression_ratio'], list):
                            trial_config['compression_ratio'] = trial.suggest_categorical('spacy_compression_ratio', search_space['spacy_compression_ratio'])
                        else:
                            trial_config['compression_ratio'] = trial.suggest_float('spacy_compression_ratio',
                                                                search_space['spacy_compression_ratio'][0],
                                                                search_space['spacy_compression_ratio'][1])
                    elif 'compression_ratio' in search_space:
                        if isinstance(search_space['compression_ratio'], list):
                            trial_config['compression_ratio'] = trial.suggest_categorical('compression_ratio', search_space['compression_ratio'])
                        else:
                            trial_config['compression_ratio'] = trial.suggest_float('compression_ratio',
                                                                search_space['compression_ratio'][0],
                                                                search_space['compression_ratio'][1])
                                                                
                elif '::' in comp_config:
                    parts = comp_config.split('::', 3)
                    
                    if len(parts) >= 3:
                        method, gen_type, model = parts[0], parts[1], parts[2]
                        trial_config['passage_compressor_method'] = method
                        trial_config['compressor_generator_module_type'] = gen_type
                        trial_config['compressor_model'] = model
                        print(f"[DEBUG] Setting passage_compressor_method to '{method}' with gen_type '{gen_type}' and model '{model}'")

                        if hasattr(self, 'search_space_metadata') and 'passage_compressor_config_metadata' in self.search_space_metadata:
                            metadata = self.search_space_metadata['passage_compressor_config_metadata']
                            if metadata and comp_config in metadata:
                                config_metadata = metadata[comp_config]
                                trial_config['compressor_llm'] = config_metadata.get('llm', 'openai')
                                if gen_type == 'sap_api' and 'api_url' in config_metadata:
                                    trial_config['compressor_api_url'] = config_metadata['api_url']
                                elif gen_type == 'vllm':
                                    trial_config['compressor_llm'] = model
                                if 'batch' in config_metadata:
                                    trial_config['compressor_batch'] = config_metadata['batch']
                        else:
                            unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                            for comp_config_item in unified_params.get('compressor_configs', []):
                                if (comp_config_item['method'] == method and 
                                    comp_config_item['generator_module_type'] == gen_type and 
                                    model in comp_config_item['models']):

                                    trial_config['compressor_llm'] = comp_config_item.get('llm', 'openai')

                                    if gen_type == 'sap_api':
                                        trial_config['compressor_api_url'] = comp_config_item.get('api_url')
                                    elif gen_type == 'vllm':
                                        trial_config['compressor_llm'] = model

                                    if 'batch' in comp_config_item:
                                        trial_config['compressor_batch'] = comp_config_item['batch']
                                    break
                else:
                    trial_config['passage_compressor_method'] = comp_config
                    print(f"[DEBUG] Setting passage_compressor_method to '{comp_config}'")

            elif 'passage_compressor_method' in search_space:
                method = trial.suggest_categorical('passage_compressor_method', 
                                                search_space['passage_compressor_method'])
                trial_config['passage_compressor_method'] = method
                
                if method != 'pass_compressor':
                    if 'compressor_generator_module_type' in search_space:
                        trial_config['compressor_generator_module_type'] = trial.suggest_categorical(
                            'compressor_generator_module_type', 
                            search_space['compressor_generator_module_type']
                        )
                    
                    if 'compressor_llm' in search_space:
                        trial_config['compressor_llm'] = trial.suggest_categorical(
                            'compressor_llm', search_space['compressor_llm']
                        )
                    
                    if 'compressor_model' in search_space:
                        trial_config['compressor_model'] = trial.suggest_categorical(
                            'compressor_model', search_space['compressor_model']
                        )
                
        elif component == 'prompt_maker_generator':
            if 'prompt_maker_method' in search_space:
                trial_config['prompt_maker_method'] = trial.suggest_categorical(
                    'prompt_maker_method', search_space['prompt_maker_method']
                )
            
            if 'prompt_template_idx' in search_space:
                if isinstance(search_space['prompt_template_idx'], list):
                    trial_config['prompt_template_idx'] = trial.suggest_categorical(
                        'prompt_template_idx', search_space['prompt_template_idx']
                    )
                else:
                    trial_config['prompt_template_idx'] = trial.suggest_int(
                        'prompt_template_idx',
                        search_space['prompt_template_idx'][0],
                        search_space['prompt_template_idx'][1]
                    )
            
            if 'generator_config' in search_space:
                gen_config = trial.suggest_categorical('generator_config',
                                                    search_space['generator_config'])
                trial_config['generator_config'] = gen_config
                
                parts = gen_config.split('::', 1)
                if len(parts) == 2:
                    module_type, model = parts
                    trial_config['generator_module_type'] = module_type
                    trial_config['generator_model'] = model

                    gen_node = self.config_generator.extract_node_config('generator')
                    if gen_node and 'modules' in gen_node:
                        for module in gen_node['modules']:
                            if (module.get('module_type') == module_type and 
                                model in self.config_generator._ensure_list(module.get('model', []))):
                                
                                if module_type == 'sap_api':
                                    trial_config['generator_api_url'] = module.get('api_url')
                                    trial_config['generator_llm'] = module.get('llm', 'mistralai')
                                elif module_type == 'vllm':
                                    trial_config['generator_llm'] = model
                                else:
                                    trial_config['generator_llm'] = module.get('llm', 'openai')
                                break
            
            elif 'generator_model' in search_space:
                trial_config['generator_model'] = trial.suggest_categorical(
                    'generator_model', search_space['generator_model']
                )
            
            if 'generator_temperature' in search_space:
                if isinstance(search_space['generator_temperature'], list):
                    trial_config['generator_temperature'] = trial.suggest_categorical(
                        'generator_temperature', search_space['generator_temperature']
                    )
                else:
                    trial_config['generator_temperature'] = trial.suggest_float(
                        'generator_temperature',
                        search_space['generator_temperature'][0],
                        search_space['generator_temperature'][1]
                    )
        
        else:
            for param_name, param_values in search_space.items():
                if isinstance(param_values, list):
                    trial_config[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    if isinstance(param_values[0], float):
                        trial_config[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
                    else:
                        trial_config[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
        
        self._apply_logical_defaults(trial_config, component)
        
        return trial_config
            
    def _apply_logical_defaults(self, config: Dict[str, Any], component: str):
        if component == 'query_expansion':
            if config.get('query_expansion_method') == 'pass_query_expansion':
                pass
            
            if 'retrieval_method' in config:
                if config['retrieval_method'] == 'bm25':
                    if 'vectordb_name' in config:
                        del config['vectordb_name']
                    if 'bm25_tokenizer' not in config:
                        config['bm25_tokenizer'] = 'porter_stemmer'
                elif config['retrieval_method'] == 'vectordb':
                    if 'bm25_tokenizer' in config:
                        del config['bm25_tokenizer']
                    if 'vectordb_name' not in config:
                        config['vectordb_name'] = 'default'
        
        elif component == 'retrieval':
            if 'retrieval_method' in config:
                if config['retrieval_method'] == 'bm25':
                    if 'vectordb_name' in config:
                        del config['vectordb_name']
                    if 'bm25_tokenizer' not in config:
                        config['bm25_tokenizer'] = 'porter_stemmer'
                elif config['retrieval_method'] == 'vectordb':
                    if 'bm25_tokenizer' in config:
                        del config['bm25_tokenizer']
                    if 'vectordb_name' not in config:
                        config['vectordb_name'] = 'default'
        

    
    def _apply_component_constraints(self, trial: optuna.Trial, config: Dict[str, Any], 
                                    component: str, search_space: Dict[str, Any]):
        if component == 'query_expansion':
            if 'query_expansion_method' in config and config['query_expansion_method'] == 'pass_query_expansion':
                if 'retrieval_method' in config:
                    pass
                if 'bm25_tokenizer' in config and 'retrieval_method' in config:
                    if config['retrieval_method'] != 'bm25':
                        config.pop('bm25_tokenizer', None)  
                if 'vectordb_name' in config and 'retrieval_method' in config:
                    if config['retrieval_method'] != 'vectordb':
                        config.pop('vectordb_name', None)  
            
            if 'query_expansion_retrieval_method' in config:
                qe_method = config.get('query_expansion_method')
                if qe_method == 'pass_query_expansion':
                    trial.suggest_categorical('query_expansion_retrieval_method', ['bm25'])
                
                if 'query_expansion_vectordb_name' in config:
                    if config['query_expansion_retrieval_method'] != 'vectordb':
                        config.pop('query_expansion_vectordb_name', None)  
                
                if 'query_expansion_bm25_tokenizer' in config:
                    if config['query_expansion_retrieval_method'] != 'bm25':
                        config.pop('query_expansion_bm25_tokenizer', None) 
        
        elif component == 'retrieval':
            if 'bm25_tokenizer' in config and 'retrieval_method' in config:
                if config['retrieval_method'] != 'bm25':
                    config.pop('bm25_tokenizer', None) 
            if 'vectordb_name' in config and 'retrieval_method' in config:
                if config['retrieval_method'] != 'vectordb':
                    config.pop('vectordb_name', None)  
        
        elif component == 'passage_compressor':
            if 'passage_compressor_method' in config:
                method = config['passage_compressor_method']
                if method == 'pass_compressor':
                    if 'compressor_llm' in config:
                        config.pop('compressor_llm', None)  
                    if 'compressor_model' in config:
                        config.pop('compressor_model', None)  
    
    def _extract_query_expansion_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'query_expansion_config' in full_space and 'query_expansion_config' not in fixed_config:
            space['query_expansion_config'] = full_space['query_expansion_config']
        
        if 'query_expansion_config' not in space and 'query_expansion_method' in full_space and 'query_expansion_method' not in fixed_config:
            space['query_expansion_method'] = full_space['query_expansion_method']

        if 'retriever_top_k' in full_space and 'retriever_top_k' not in fixed_config:
            top_k_values = full_space['retriever_top_k']

            if isinstance(top_k_values, list):
                if len(top_k_values) == 2 and all(isinstance(x, int) for x in top_k_values):
                    if top_k_values[1] > top_k_values[0]:
                        space['retriever_top_k'] = list(range(top_k_values[0], top_k_values[1] + 1))
                    else:
                        space['retriever_top_k'] = top_k_values
                else:
                    space['retriever_top_k'] = top_k_values
            elif isinstance(top_k_values, tuple) and len(top_k_values) == 2:
                space['retriever_top_k'] = list(range(top_k_values[0], top_k_values[1] + 1))
            else:
                space['retriever_top_k'] = top_k_values
        
        if 'query_expansion_temperature' in full_space and 'query_expansion_temperature' not in fixed_config:
            space['query_expansion_temperature'] = full_space['query_expansion_temperature']
        if 'query_expansion_max_token' in full_space and 'query_expansion_max_token' not in fixed_config:
            space['query_expansion_max_token'] = full_space['query_expansion_max_token']
        
        if 'query_expansion_retrieval_method' in full_space and 'query_expansion_retrieval_method' not in fixed_config:
            space['query_expansion_retrieval_method'] = full_space['query_expansion_retrieval_method']
        if 'query_expansion_bm25_tokenizer' in full_space and 'query_expansion_bm25_tokenizer' not in fixed_config:
            space['query_expansion_bm25_tokenizer'] = full_space['query_expansion_bm25_tokenizer']
        if 'query_expansion_vectordb_name' in full_space and 'query_expansion_vectordb_name' not in fixed_config:
            space['query_expansion_vectordb_name'] = full_space['query_expansion_vectordb_name']
        
        if 'retrieval_method' in full_space and 'retrieval_method' not in fixed_config:
            space['retrieval_method'] = full_space['retrieval_method']
        if 'bm25_tokenizer' in full_space and 'bm25_tokenizer' not in fixed_config:
            space['bm25_tokenizer'] = full_space['bm25_tokenizer']
        if 'vectordb_name' in full_space and 'vectordb_name' not in fixed_config:
            space['vectordb_name'] = full_space['vectordb_name']
        
        return space
    
    def _extract_retrieval_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'retrieval_method' in full_space and 'retrieval_method' not in fixed_config:
            space['retrieval_method'] = full_space['retrieval_method']

        if 'retriever_top_k' in full_space and 'retriever_top_k' not in fixed_config:
            top_k_values = full_space['retriever_top_k']

            if isinstance(top_k_values, list):
                if len(top_k_values) == 2 and all(isinstance(x, int) for x in top_k_values):
                    if top_k_values[1] > top_k_values[0]:
                        space['retriever_top_k'] = list(range(top_k_values[0], top_k_values[1] + 1))
                    else:
                        space['retriever_top_k'] = top_k_values
                else:
                    space['retriever_top_k'] = top_k_values
            elif isinstance(top_k_values, tuple) and len(top_k_values) == 2:
                space['retriever_top_k'] = list(range(top_k_values[0], top_k_values[1] + 1))
            else:
                space['retriever_top_k'] = top_k_values
        
        if 'bm25_tokenizer' in full_space and 'bm25_tokenizer' not in fixed_config:
            space['bm25_tokenizer'] = full_space['bm25_tokenizer']
        
        if 'vectordb_name' in full_space and 'vectordb_name' not in fixed_config:
            space['vectordb_name'] = full_space['vectordb_name']
        
        return space
    
    def _extract_reranker_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'passage_reranker_method' in full_space and 'passage_reranker_method' not in fixed_config:
            space['passage_reranker_method'] = full_space['passage_reranker_method']
        
        if 'reranker_top_k' in full_space and 'reranker_top_k' not in fixed_config:
            prev_top_k = fixed_config.get('retriever_top_k', 10)
            reranker_top_k_values = full_space['reranker_top_k']

            if isinstance(reranker_top_k_values, list):
                if len(reranker_top_k_values) == 2 and all(isinstance(x, int) for x in reranker_top_k_values):
                    if reranker_top_k_values[1] > reranker_top_k_values[0]:
                        expanded_range = list(range(reranker_top_k_values[0], reranker_top_k_values[1] + 1))
                        valid_values = [k for k in expanded_range if k <= prev_top_k]
                    else:
                        valid_values = [k for k in reranker_top_k_values if k <= prev_top_k]
                else:
                    valid_values = [k for k in reranker_top_k_values if k <= prev_top_k]
                
                if valid_values:
                    space['reranker_top_k'] = valid_values
                    
            elif isinstance(reranker_top_k_values, tuple):
                min_k, max_k = reranker_top_k_values
                max_k = min(max_k, prev_top_k)
                if min_k <= max_k:
                    space['reranker_top_k'] = list(range(min_k, max_k + 1))

        reranker_config = self.config_generator.extract_node_config("passage_reranker")
        if reranker_config and reranker_config.get("modules"):
            for module in reranker_config["modules"]:
                method = module.get("module_type")
                if not method or method == "pass_reranker":
                    continue
                
                model_field = 'model' if method == 'flashrank_reranker' else 'model_name'
                if model_field in module:
                    method_models_key = f"{method}_models"
                    if method_models_key not in fixed_config:
                        models = module[model_field]
                        if isinstance(models, list):
                            space[method_models_key] = models
                        else:
                            space[method_models_key] = [models]
        
        return space
    
    def _extract_filter_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'passage_filter_method' in full_space and 'passage_filter_method' not in fixed_config:
            space['passage_filter_method'] = full_space['passage_filter_method']
        
        filter_config = self.config_generator.extract_node_config("passage_filter")
        if filter_config and filter_config.get("modules"):
            for module in filter_config["modules"]:
                method = module.get("module_type")
                if not method or method == "pass_passage_filter":
                    continue
                
                if method in ["threshold_cutoff", "similarity_threshold_cutoff"]:
                    param_name = f"{method}_threshold"
                    if param_name in full_space and param_name not in fixed_config:
                        space[param_name] = full_space[param_name]
                    elif "threshold" in module and param_name not in fixed_config:
                        threshold_vals = module["threshold"]
                        if isinstance(threshold_vals, list) and len(threshold_vals) == 2:
                            space[param_name] = (threshold_vals[0], threshold_vals[1])
                        elif isinstance(threshold_vals, list):
                            space[param_name] = threshold_vals
                        else:
                            space[param_name] = [threshold_vals]
                
                elif method in ["percentile_cutoff", "similarity_percentile_cutoff"]:
                    param_name = f"{method}_percentile"
                    if param_name in full_space and param_name not in fixed_config:
                        space[param_name] = full_space[param_name]
                    elif "percentile" in module and param_name not in fixed_config:
                        percentile_vals = module["percentile"]
                        if isinstance(percentile_vals, list) and len(percentile_vals) == 2:
                            space[param_name] = (percentile_vals[0], percentile_vals[1])
                        elif isinstance(percentile_vals, list):
                            space[param_name] = percentile_vals
                        else:
                            space[param_name] = [percentile_vals]
        
        return space
    
    def _extract_compressor_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'passage_compressor_config' in full_space and 'passage_compressor_config' not in fixed_config:
            space['passage_compressor_config'] = full_space['passage_compressor_config']
        
        if 'passage_compressor_config' not in space and 'passage_compressor_method' in full_space and 'passage_compressor_method' not in fixed_config:
            space['passage_compressor_method'] = full_space['passage_compressor_method']
        
        if 'compressor_generator_module_type' in full_space and 'compressor_generator_module_type' not in fixed_config:
            space['compressor_generator_module_type'] = full_space['compressor_generator_module_type']
        
        if 'compressor_llm' in full_space and 'compressor_llm' not in fixed_config:
            space['compressor_llm'] = full_space['compressor_llm']
        
        if 'compressor_model' in full_space and 'compressor_model' not in fixed_config:
            space['compressor_model'] = full_space['compressor_model']
        
        if 'compression_ratio' in full_space and 'compression_ratio' not in fixed_config:
            space['compression_ratio'] = full_space['compression_ratio']
        
        compressor_config = self.config_generator.extract_node_config("passage_compressor")
        if compressor_config and compressor_config.get("modules"):
            for module in compressor_config["modules"]:
                method = module.get("module_type")
                if not method or method == "pass_compressor":
                    continue
                
                if method == 'lexrank':
                    if 'threshold' in module and f'{method}_threshold' not in fixed_config:
                        threshold_vals = module['threshold']
                        if isinstance(threshold_vals, list) and len(threshold_vals) == 2:
                            space['lexrank_threshold'] = (threshold_vals[0], threshold_vals[1])
                        elif isinstance(threshold_vals, list):
                            space['lexrank_threshold'] = threshold_vals
                        else:
                            space['lexrank_threshold'] = [threshold_vals]
                    
                    if 'damping' in module and f'{method}_damping' not in fixed_config:
                        damping_vals = module['damping']
                        if isinstance(damping_vals, list) and len(damping_vals) == 2:
                            space['lexrank_damping'] = (damping_vals[0], damping_vals[1])
                        elif isinstance(damping_vals, list):
                            space['lexrank_damping'] = damping_vals
                        else:
                            space['lexrank_damping'] = [damping_vals]
                    
                    if 'max_iterations' in module and f'{method}_max_iterations' not in fixed_config:
                        max_iter_vals = module['max_iterations']
                        if isinstance(max_iter_vals, list) and len(max_iter_vals) == 2:
                            space['lexrank_max_iterations'] = (max_iter_vals[0], max_iter_vals[1])
                        elif isinstance(max_iter_vals, list):
                            space['lexrank_max_iterations'] = max_iter_vals
                        else:
                            space['lexrank_max_iterations'] = [max_iter_vals]
                
                if method in ['lexrank', 'spacy', 'sentence_rank', 'keyword_extraction', 'query_focused']:
                    param_name = f'{method}_compression_ratio'
                    if 'compression_ratio' in module and param_name not in fixed_config:
                        ratio_vals = module['compression_ratio']
                        if isinstance(ratio_vals, list) and len(ratio_vals) == 2:
                            space[param_name] = (ratio_vals[0], ratio_vals[1])
                        elif isinstance(ratio_vals, list):
                            space[param_name] = ratio_vals
                        else:
                            space[param_name] = [ratio_vals]
        
        return space
        
    def _extract_compressor_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('passage_compressor')
        
        if not params.get('methods'):
            return

        comp_options = []
        option_metadata = {}
        seen_configs = set()
        
        if 'pass_compressor' in params['methods']:
            comp_options.append('pass_compressor')
            option_metadata['pass_compressor'] = {'method': 'pass_compressor'}
            seen_configs.add('pass_compressor')
        
        for comp_config in params.get('compressor_configs', []):
            method = comp_config['method']

            if method in ['tree_summarize', 'refine']:
                gen_type = comp_config['generator_module_type']
                for model in comp_config['models']:
                    option_key = f"{method}::{gen_type}::{model}"
                    if option_key not in seen_configs:
                        seen_configs.add(option_key)
                        comp_options.append(option_key)
                        
                        option_metadata[option_key] = {
                            'method': method,
                            'generator_module_type': gen_type,
                            'llm': comp_config.get('llm'),
                            'model': model,
                            'api_url': comp_config.get('api_url')
                        }

            elif method in ['lexrank', 'spacy', 'sentence_rank', 'keyword_extraction', 'query_focused']:
                if method == 'spacy':
                    spacy_models = comp_config.get('spacy_model', ['en_core_web_sm'])
                    if isinstance(spacy_models, str):
                        spacy_models = [spacy_models]
                    
                    for model in spacy_models:
                        option_key = f"{method}::{model}"
                        if option_key not in seen_configs:
                            seen_configs.add(option_key)
                            comp_options.append(option_key)
                            option_metadata[option_key] = {
                                'method': method,
                                'spacy_model': model
                            }
                else:
                    option_key = method
                    if option_key not in seen_configs:
                        seen_configs.add(option_key)
                        comp_options.append(option_key)
                        option_metadata[option_key] = {'method': method}
        
        if comp_options:
            space['passage_compressor_config'] = {
                'type': 'categorical',
                'values': comp_options,
                'metadata': option_metadata
            }

        for comp_config in params.get('compressor_configs', []):
            method = comp_config['method']

            if method in ['lexrank', 'spacy', 'sentence_rank', 'keyword_extraction', 'query_focused']:
                if 'compression_ratio' in comp_config:
                    comp_ratios = comp_config['compression_ratio']
                    if isinstance(comp_ratios, list) and len(comp_ratios) >= 2:
                        space[f'{method}_compression_ratio'] = {
                            'type': 'float',
                            'values': [min(comp_ratios), max(comp_ratios)],
                            'condition': ('passage_compressor_config', 'contains', method)
                        }

            if method == 'lexrank':
                if 'threshold' in comp_config:
                    thresholds = comp_config['threshold']
                    if isinstance(thresholds, list) and len(thresholds) >= 2:
                        space['lexrank_threshold'] = {
                            'type': 'float',
                            'values': [min(thresholds), max(thresholds)],
                            'condition': ('passage_compressor_config', 'equals', 'lexrank')
                        }
                
                if 'damping' in comp_config:
                    dampings = comp_config['damping']
                    if isinstance(dampings, list) and len(dampings) >= 2:
                        space['lexrank_damping'] = {
                            'type': 'float',
                            'values': [min(dampings), max(dampings)],
                            'condition': ('passage_compressor_config', 'equals', 'lexrank')
                        }
                
                if 'max_iterations' in comp_config:
                    max_iters = comp_config['max_iterations']
                    if isinstance(max_iters, list) and len(max_iters) >= 2:
                        space['lexrank_max_iterations'] = {
                            'type': 'int',
                            'values': [min(max_iters), max(max_iters)],
                            'condition': ('passage_compressor_config', 'equals', 'lexrank')
                        }

            elif method in ['tree_summarize', 'refine']:
                pass
    
    def _extract_prompt_generator_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'prompt_maker_method' in full_space and 'prompt_maker_method' not in fixed_config:
            space['prompt_maker_method'] = full_space['prompt_maker_method']
        
        if 'prompt_template_idx' in full_space and 'prompt_template_idx' not in fixed_config:
            space['prompt_template_idx'] = full_space['prompt_template_idx']
        
        if 'generator_config' in full_space and 'generator_config' not in fixed_config:
            space['generator_config'] = full_space['generator_config']
            
            if hasattr(self.config_extractor, 'search_space_metadata'):
                metadata = self.config_extractor.search_space_metadata.get('generator_config_metadata', {})
                self.search_space_metadata = {'generator_config_metadata': metadata}
        
        if 'generator_temperature' in full_space and 'generator_temperature' not in fixed_config:
            space['generator_temperature'] = full_space['generator_temperature']
        
        return space