import optuna
from typing import Dict, Any, List, Union, Tuple


class ComponentSearchSpaceBuilder:
    
    def __init__(self, config_generator, config_extractor):
        self.config_generator = config_generator
        self.config_extractor = config_extractor
        self.search_type = config_extractor.search_type 
    
    def build_component_search_space(self, component: str, fixed_config: Dict[str, Any]) -> Dict[str, Any]:
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
        
        return component_search_space
    
    def suggest_component_params(self, trial: optuna.Trial, component: str, search_space: Dict[str, Any], fixed_config: Dict[str, Any] = None) -> Dict[str, Any]:
        if fixed_config is None:
            fixed_config = {}
        trial_config = {}
        
        if component == 'passage_compressor':
            if 'passage_compressor_method' in search_space:
                method = trial.suggest_categorical('passage_compressor_method', search_space['passage_compressor_method'])
                trial_config['passage_compressor_method'] = method
                
                if method in ['tree_summarize', 'refine']:
                    if 'compressor_llm' in search_space:
                        trial_config['compressor_llm'] = trial.suggest_categorical('compressor_llm', search_space['compressor_llm'])
                    if 'compressor_model' in search_space:
                        trial_config['compressor_model'] = trial.suggest_categorical('compressor_model', search_space['compressor_model'])
                    if 'compressor_batch' in search_space:
                        if isinstance(search_space['compressor_batch'], list):
                            trial_config['compressor_batch'] = trial.suggest_categorical('compressor_batch', search_space['compressor_batch'])
                        else:
                            trial_config['compressor_batch'] = trial.suggest_int('compressor_batch', 
                                                                                search_space['compressor_batch'][0],
                                                                                search_space['compressor_batch'][1])
                
                elif method == 'lexrank':
                    if 'compressor_compression_ratio' in search_space:
                        if isinstance(search_space['compressor_compression_ratio'], list):
                            trial_config['compressor_compression_ratio'] = trial.suggest_categorical('compressor_compression_ratio', 
                                                                                                    search_space['compressor_compression_ratio'])
                        else:
                            trial_config['compressor_compression_ratio'] = trial.suggest_float('compressor_compression_ratio',
                                                                                            search_space['compressor_compression_ratio'][0],
                                                                                            search_space['compressor_compression_ratio'][1])
                    
                    if 'compressor_threshold' in search_space:
                        if isinstance(search_space['compressor_threshold'], list):
                            trial_config['compressor_threshold'] = trial.suggest_categorical('compressor_threshold', 
                                                                                            search_space['compressor_threshold'])
                        else:
                            trial_config['compressor_threshold'] = trial.suggest_float('compressor_threshold',
                                                                                    search_space['compressor_threshold'][0],
                                                                                    search_space['compressor_threshold'][1])
                    
                    if 'compressor_damping' in search_space:
                        if isinstance(search_space['compressor_damping'], list):
                            trial_config['compressor_damping'] = trial.suggest_categorical('compressor_damping', 
                                                                                        search_space['compressor_damping'])
                        else:
                            trial_config['compressor_damping'] = trial.suggest_float('compressor_damping',
                                                                                    search_space['compressor_damping'][0],
                                                                                    search_space['compressor_damping'][1])
                    
                    if 'compressor_max_iterations' in search_space:
                        if isinstance(search_space['compressor_max_iterations'], list):
                            trial_config['compressor_max_iterations'] = trial.suggest_categorical('compressor_max_iterations', 
                                                                                                search_space['compressor_max_iterations'])
                        else:
                            trial_config['compressor_max_iterations'] = trial.suggest_int('compressor_max_iterations',
                                                                                        search_space['compressor_max_iterations'][0],
                                                                                        search_space['compressor_max_iterations'][1])
                
                elif method == 'spacy':
                    if 'compressor_compression_ratio' in search_space:
                        if isinstance(search_space['compressor_compression_ratio'], list):
                            trial_config['compressor_compression_ratio'] = trial.suggest_categorical('compressor_compression_ratio', 
                                                                                                    search_space['compressor_compression_ratio'])
                        else:
                            trial_config['compressor_compression_ratio'] = trial.suggest_float('compressor_compression_ratio',
                                                                                            search_space['compressor_compression_ratio'][0],
                                                                                            search_space['compressor_compression_ratio'][1])
                    
                    if 'compressor_spacy_model' in search_space:
                        trial_config['compressor_spacy_model'] = trial.suggest_categorical('compressor_spacy_model', 
                                                                                        search_space['compressor_spacy_model'])
        
        elif component == 'query_expansion':
            if 'query_expansion_method' in search_space:
                qe_method = trial.suggest_categorical('query_expansion_method', search_space['query_expansion_method'])
                trial_config['query_expansion_method'] = qe_method
                
                if qe_method != 'pass_query_expansion':
                    if 'query_expansion_model' in search_space:
                        trial_config['query_expansion_model'] = trial.suggest_categorical('query_expansion_model', search_space['query_expansion_model'])

                    if qe_method == 'hyde' and 'query_expansion_max_token' in search_space:
                        if isinstance(search_space['query_expansion_max_token'], list):
                            trial_config['query_expansion_max_token'] = trial.suggest_categorical('query_expansion_max_token', search_space['query_expansion_max_token'])
                        elif isinstance(search_space['query_expansion_max_token'], tuple):
                            trial_config['query_expansion_max_token'] = trial.suggest_int('query_expansion_max_token', 
                                                                                        search_space['query_expansion_max_token'][0], 
                                                                                        search_space['query_expansion_max_token'][1])
                    
                    elif qe_method == 'multi_query_expansion' and 'query_expansion_temperature' in search_space:
                        if isinstance(search_space['query_expansion_temperature'], list):
                            trial_config['query_expansion_temperature'] = trial.suggest_categorical('query_expansion_temperature', search_space['query_expansion_temperature'])
                        elif isinstance(search_space['query_expansion_temperature'], tuple):
                            trial_config['query_expansion_temperature'] = trial.suggest_float('query_expansion_temperature', 
                                                                                            search_space['query_expansion_temperature'][0], 
                                                                                            search_space['query_expansion_temperature'][1])

                    for key, value in search_space.items():
                        if (key.startswith('query_expansion_') and 
                            key not in ['query_expansion_method', 'query_expansion_model', 
                                    'query_expansion_temperature', 'query_expansion_max_token'] and 
                            key not in trial_config):
                            if isinstance(value, list):
                                trial_config[key] = trial.suggest_categorical(key, value)
                            elif isinstance(value, tuple) and len(value) == 2:
                                if isinstance(value[0], float):
                                    trial_config[key] = trial.suggest_float(key, value[0], value[1])
                                else:
                                    trial_config[key] = trial.suggest_int(key, value[0], value[1])
            
            if 'retriever_top_k' in search_space:
                trial_config['retriever_top_k'] = trial.suggest_categorical('retriever_top_k', search_space['retriever_top_k'])
            
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
                trial_config['retriever_top_k'] = trial.suggest_categorical('retriever_top_k', search_space['retriever_top_k'])
        
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
        
    
    def _extract_query_expansion_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'query_expansion_method' in full_space and 'query_expansion_method' not in fixed_config:
            space['query_expansion_method'] = full_space['query_expansion_method']
        
        if 'retriever_top_k' in full_space and 'retriever_top_k' not in fixed_config:
            top_k_values = full_space['retriever_top_k']
            # top_k is ALWAYS expanded for both BO and Grid
            if (isinstance(top_k_values, list) and 
                len(top_k_values) == 2 and 
                all(isinstance(x, int) for x in top_k_values) and
                top_k_values[1] > top_k_values[0]):
                min_val, max_val = min(top_k_values), max(top_k_values)
                space['retriever_top_k'] = list(range(min_val, max_val + 1))
            else:
                space['retriever_top_k'] = top_k_values
        
        qe_config = self.config_generator.extract_node_config("query_expansion")
        if qe_config:
            retrieval_methods = []
            bm25_tokenizers = []
            vectordb_names = []
            
            retrieval_config = self.config_generator.extract_node_config("retrieval")
            if retrieval_config:
                for module in retrieval_config.get("modules", []):
                    module_type = module.get("module_type")
                    if module_type == "bm25":
                        retrieval_methods.append("bm25")
                        tokenizers = module.get("bm25_tokenizer", ["porter_stemmer"])
                        if not isinstance(tokenizers, list):
                            tokenizers = [tokenizers]
                        bm25_tokenizers.extend(tokenizers)
                    elif module_type == "vectordb":
                        retrieval_methods.append("vectordb")
                        vdbs = module.get("vectordb", ["default"])
                        if not isinstance(vdbs, list):
                            vdbs = [vdbs]
                        vectordb_names.extend(vdbs)
            
            if retrieval_methods and 'retrieval_method' not in fixed_config:
                space['retrieval_method'] = list(set(retrieval_methods))
            
            if bm25_tokenizers and 'bm25_tokenizer' not in fixed_config:
                space['bm25_tokenizer'] = list(set(bm25_tokenizers))
            
            if vectordb_names and 'vectordb_name' not in fixed_config:
                space['vectordb_name'] = list(set(vectordb_names))
        
        for key, value in full_space.items():
            if key.startswith('query_expansion_') and key not in fixed_config and key not in space:
                if key not in ['query_expansion_retrieval_method', 'query_expansion_bm25_tokenizer', 'query_expansion_vectordb_name']:
                    # For Grid Search: keep lists as discrete values
                    # For BO: can expand integer ranges but not float ranges
                    if (isinstance(value, list) and 
                        len(value) == 2 and 
                        all(isinstance(x, int) for x in value) and
                        value[1] > value[0]):
                        # Only expand integer ranges for certain parameters and only for BO
                        if ('max_token' in key) and self.search_type != 'grid':
                            min_val, max_val = min(value), max(value)
                            space[key] = list(range(min_val, max_val + 1))
                        else:
                            space[key] = value
                    else:
                        space[key] = value
        
        return space
    
    def _extract_retrieval_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'retrieval_method' in full_space and 'retrieval_method' not in fixed_config:
            space['retrieval_method'] = full_space['retrieval_method']
        
        if 'retriever_top_k' in full_space and 'retriever_top_k' not in fixed_config:
            top_k_values = full_space['retriever_top_k']
            if (isinstance(top_k_values, list) and 
                len(top_k_values) == 2 and 
                all(isinstance(x, int) for x in top_k_values) and
                top_k_values[1] > top_k_values[0]):
                min_val, max_val = min(top_k_values), max(top_k_values)
                space['retriever_top_k'] = list(range(min_val, max_val + 1))
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
            reranker_values = full_space['reranker_top_k']

            if (isinstance(reranker_values, list) and 
                len(reranker_values) == 2 and 
                all(isinstance(x, int) for x in reranker_values) and
                reranker_values[1] > reranker_values[0]):
                min_val, max_val = min(reranker_values), max(reranker_values)
                expanded_values = [k for k in range(min_val, max_val + 1) if k <= prev_top_k]
                if expanded_values:
                    space['reranker_top_k'] = expanded_values
            elif isinstance(reranker_values, list):
                valid_values = [k for k in reranker_values if k <= prev_top_k]
                if valid_values:
                    space['reranker_top_k'] = valid_values
            elif isinstance(reranker_values, tuple):
                min_k, max_k = reranker_values
                max_k = min(max_k, prev_top_k)
                if min_k <= max_k:
                    space['reranker_top_k'] = (min_k, max_k)

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
                            if self.search_type == 'grid':
                                space[param_name] = threshold_vals
                            else:
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
                            if self.search_type == 'grid':
                                space[param_name] = percentile_vals
                            else:
                                space[param_name] = (percentile_vals[0], percentile_vals[1])
                        elif isinstance(percentile_vals, list):
                            space[param_name] = percentile_vals
                        else:
                            space[param_name] = [percentile_vals]
        
        return space
    
    def _extract_compressor_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'passage_compressor_method' in full_space and 'passage_compressor_method' not in fixed_config:
            space['passage_compressor_method'] = full_space['passage_compressor_method']
        
        if 'compressor_llm' in full_space and 'compressor_llm' not in fixed_config:
            space['compressor_llm'] = full_space['compressor_llm']
        
        if 'compressor_model' in full_space and 'compressor_model' not in fixed_config:
            space['compressor_model'] = full_space['compressor_model']
        
        if 'compressor_batch' in full_space and 'compressor_batch' not in fixed_config:
            space['compressor_batch'] = full_space['compressor_batch']

        if 'compressor_temperature' in full_space and 'compressor_temperature' not in fixed_config:
            temp_values = full_space['compressor_temperature']
            space['compressor_temperature'] = temp_values
        
        if 'compressor_max_tokens' in full_space and 'compressor_max_tokens' not in fixed_config:
            token_values = full_space['compressor_max_tokens']
            
            if (isinstance(token_values, list) and 
                len(token_values) == 2 and 
                all(isinstance(x, int) for x in token_values) and
                token_values[1] > token_values[0] and
                self.search_type != 'grid'):
                min_val, max_val = min(token_values), max(token_values)
                space['compressor_max_tokens'] = list(range(min_val, max_val + 1))
            else:
                space['compressor_max_tokens'] = token_values

        for param in ['compressor_compression_ratio', 'compressor_threshold', 'compressor_damping', 'compressor_max_iterations']:
            if param in full_space and param not in fixed_config:
                space[param] = full_space[param]
        
        if 'compressor_spacy_model' in full_space and 'compressor_spacy_model' not in fixed_config:
            space['compressor_spacy_model'] = full_space['compressor_spacy_model']
        
        compressor_config = self.config_generator.extract_node_config("passage_compressor")
        if compressor_config and compressor_config.get("modules"):
            methods = []
            for module in compressor_config["modules"]:
                method = module.get("module_type")
                if method:
                    methods.append(method)
            
            if methods and 'passage_compressor_method' not in space and 'passage_compressor_method' not in fixed_config:
                space['passage_compressor_method'] = methods
                
            for module in compressor_config["modules"]:
                method = module.get("module_type")
                if not method or method == "pass_compressor":
                    continue
                
                if method in ['tree_summarize', 'refine']:
                    if 'temperature' in module and 'compressor_temperature' not in fixed_config and 'compressor_temperature' not in space:
                        temps = module['temperature']
                        if isinstance(temps, list):
                            space['compressor_temperature'] = temps
                        else:
                            space['compressor_temperature'] = [temps]
                    
                    if 'max_tokens' in module and 'compressor_max_tokens' not in fixed_config and 'compressor_max_tokens' not in space:
                        tokens = module.get('max_tokens', module.get('max_token'))
                        if tokens:
                            if isinstance(tokens, list) and len(tokens) == 2 and self.search_type != 'grid':
                                if all(isinstance(x, int) for x in tokens) and tokens[1] > tokens[0]:
                                    space['compressor_max_tokens'] = list(range(tokens[0], tokens[1] + 1))
                                else:
                                    space['compressor_max_tokens'] = tokens
                            elif isinstance(tokens, list):
                                space['compressor_max_tokens'] = tokens
                            else:
                                space['compressor_max_tokens'] = [tokens]
                
                elif method == 'lexrank':
                    for param_key, module_key in [
                        ('compressor_compression_ratio', 'compression_ratio'),
                        ('compressor_threshold', 'threshold'),
                        ('compressor_damping', 'damping'),
                        ('compressor_max_iterations', 'max_iterations')
                    ]:
                        if module_key in module and param_key not in fixed_config and param_key not in space:
                            vals = module[module_key]
                            if isinstance(vals, list) and len(vals) == 2:
                                if self.search_type == 'grid':
                                    space[param_key] = vals  
                                else:
                                    space[param_key] = (vals[0], vals[1])  
                            elif isinstance(vals, list):
                                space[param_key] = vals
                            else:
                                space[param_key] = [vals]
                
                elif method == 'spacy':
                    if 'compression_ratio' in module and 'compressor_compression_ratio' not in fixed_config and 'compressor_compression_ratio' not in space:
                        ratios = module['compression_ratio']
                        if isinstance(ratios, list) and len(ratios) == 2:
                            if self.search_type == 'grid':
                                space['compressor_compression_ratio'] = ratios
                            else:
                                space['compressor_compression_ratio'] = (ratios[0], ratios[1])
                        elif isinstance(ratios, list):
                            space['compressor_compression_ratio'] = ratios
                        else:
                            space['compressor_compression_ratio'] = [ratios]
                    
                    if 'spacy_model' in module and 'compressor_spacy_model' not in fixed_config and 'compressor_spacy_model' not in space:
                        models = module['spacy_model']
                        if isinstance(models, list):
                            space['compressor_spacy_model'] = models
                        else:
                            space['compressor_spacy_model'] = [models]
        
        return space
    
    def _extract_prompt_generator_search_space(self, full_space: Dict[str, Any], fixed_config: Dict[str, Any]) -> Dict[str, Any]:
        space = {}
        
        if 'prompt_maker_method' in full_space and 'prompt_maker_method' not in fixed_config:
            space['prompt_maker_method'] = full_space['prompt_maker_method']
        
        if 'prompt_template_idx' in full_space and 'prompt_template_idx' not in fixed_config:
            space['prompt_template_idx'] = full_space['prompt_template_idx']
        
        if 'generator_model' in full_space and 'generator_model' not in fixed_config:
            space['generator_model'] = full_space['generator_model']
        
        if 'generator_temperature' in full_space and 'generator_temperature' not in fixed_config:
            temp_values = full_space['generator_temperature']

            if (isinstance(temp_values, list) and 
                len(temp_values) == 2 and 
                all(isinstance(x, (int, float)) for x in temp_values) and
                self.search_type != 'grid'):
                space['generator_temperature'] = temp_values
            else:
                space['generator_temperature'] = temp_values
        
        if 'prompt_maker_generator_model' in full_space and 'prompt_maker_generator_model' not in fixed_config:
            space['prompt_maker_generator_model'] = full_space['prompt_maker_generator_model']
        
        if 'prompt_maker_temperature' in full_space and 'prompt_maker_temperature' not in fixed_config:
            temp_values = full_space['prompt_maker_temperature']
            if (isinstance(temp_values, list) and 
                len(temp_values) == 2 and 
                all(isinstance(x, (int, float)) for x in temp_values) and
                self.search_type != 'grid'):
                space['prompt_maker_temperature'] = temp_values
            else:
                space['prompt_maker_temperature'] = temp_values
        
        return space