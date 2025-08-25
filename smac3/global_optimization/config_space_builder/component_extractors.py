from typing import Dict, Any


class ComponentExtractors:
    
    def __init__(self, config_generator):
        self.config_generator = config_generator
        self._unified_space = None
    
    def get_unified_space(self) -> Dict[str, Any]:
        if self._unified_space is None:
            self._unified_space = self._extract_all_hyperparameters()
        return self._unified_space
    
    def _extract_all_hyperparameters(self) -> Dict[str, Any]:
        params = {}
        
        if self.config_generator.node_exists("query_expansion"):
            params.update(self._extract_query_expansion_params())
        
        if self.config_generator.node_exists("retrieval"):
            params.update(self._extract_retrieval_params())
        
        if self.config_generator.node_exists("passage_reranker"):
            params.update(self._extract_reranker_params())
        
        if self.config_generator.node_exists("passage_filter"):
            params.update(self._extract_filter_params())
        
        if self.config_generator.node_exists("passage_compressor"):
            params.update(self._extract_compressor_params())
        
        if self.config_generator.node_exists("prompt_maker"):
            params.update(self._extract_prompt_maker_params())
        
        if self.config_generator.node_exists("generator"):
            params.update(self._extract_generator_params())
        
        return params
    
    def _extract_query_expansion_params(self) -> Dict[str, Any]:
        params = {}
        qe_config = self.config_generator.extract_node_config("query_expansion")
        
        if not qe_config or not qe_config.get("modules"):
            return params
        
        methods = []
        method_specific_params = {}
        
        for module in qe_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method == "query_decompose":
                    models = module.get("model", [])
                    if not isinstance(models, list):
                        models = [models]
                    
                    if models: 
                        method_specific_params[method] = {}
                        if models:
                            method_specific_params[method]["models"] = models
                
                elif method == "hyde":
                    models = module.get("model", [])
                    if not isinstance(models, list):
                        models = [models]
                    max_tokens = module.get("max_token", [64])
                    if not isinstance(max_tokens, list):
                        max_tokens = [max_tokens]
                    
                    if models: 
                        method_specific_params[method] = {
                            "max_tokens": max_tokens
                        }
                        if models:
                            method_specific_params[method]["models"] = models
                
                elif method == "multi_query_expansion":
                    models = module.get("model", [])
                    if not isinstance(models, list):
                        models = [models]
                    temps = module.get("temperature", [0.7])
                    if not isinstance(temps, list):
                        temps = [temps]
                    
                    method_specific_params[method] = {"temperatures": temps}
                    if models:
                        method_specific_params[method]["models"] = models
        
        if methods:
            params['query_expansion_method'] = {
                'type': 'categorical',
                'values': methods
            }
            
            all_models = set()
            methods_with_model = []
            
            for method, method_params in method_specific_params.items():
                if "models" in method_params:
                    all_models.update(method_params["models"])
                    methods_with_model.append(method)
            
            if all_models:
                params['query_expansion_model'] = {
                    'type': 'categorical',
                    'values': list(all_models),
                    'condition': ('query_expansion_method', methods_with_model)
                }
            
            if "hyde" in method_specific_params and "max_tokens" in method_specific_params["hyde"]:
                params['query_expansion_max_token'] = {
                    'type': 'categorical',
                    'values': method_specific_params["hyde"]["max_tokens"],
                    'condition': ('query_expansion_method', ['hyde'])
                }
            
            if "multi_query_expansion" in method_specific_params and "temperatures" in method_specific_params["multi_query_expansion"]:
                params['query_expansion_temperature'] = {
                    'type': 'float',
                    'values': method_specific_params["multi_query_expansion"]["temperatures"],
                    'condition': ('query_expansion_method', ['multi_query_expansion'])
                }
        
        qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
        if qe_retrieval_options.get('methods'):
            params['query_expansion_retrieval_method'] = {
                'type': 'categorical',
                'values': qe_retrieval_options['methods'],
                'condition': ('query_expansion_method', [m for m in methods if m != 'pass_query_expansion'])
            }
            
            if 'vectordb' in qe_retrieval_options['methods'] and qe_retrieval_options.get('vectordb_names'):
                params['query_expansion_vectordb_name'] = {
                    'type': 'categorical',
                    'values': qe_retrieval_options['vectordb_names'],
                    'condition': [
                        ('query_expansion_method', [m for m in methods if m != 'pass_query_expansion']),
                        ('query_expansion_retrieval_method', ['vectordb'])
                    ]
                }
        
        return params
    
    def _extract_retrieval_params(self) -> Dict[str, Any]:
        params = {}
        retrieval_config = self.config_generator.extract_node_config("retrieval")
        
        if not retrieval_config:
            return params
        
        top_k_values = retrieval_config.get('top_k', [5])
        if isinstance(top_k_values, list) and len(top_k_values) > 0:
            params['retriever_top_k'] = {
                'type': 'int' if len(top_k_values) == 2 else 'categorical',
                'values': top_k_values
            }
        
        methods = []
        bm25_tokenizers = []
        vectordb_names = []
        
        for module in retrieval_config.get("modules", []):
            module_type = module.get("module_type")
            if module_type == "bm25":
                methods.append("bm25")
                tokenizers = module.get("bm25_tokenizer", ["porter_stemmer"])
                if not isinstance(tokenizers, list):
                    tokenizers = [tokenizers]
                bm25_tokenizers.extend(tokenizers)
            elif module_type == "vectordb":
                methods.append("vectordb")
                vdbs = module.get("vectordb", ["default"])
                if not isinstance(vdbs, list):
                    vdbs = [vdbs]
                vectordb_names.extend(vdbs)
        
        if methods:
            params['retrieval_method'] = {
                'type': 'categorical',
                'values': list(set(methods))
            }
        
        if bm25_tokenizers:
            params['bm25_tokenizer'] = {
                'type': 'categorical',
                'values': list(set(bm25_tokenizers)),
                'condition': ('retrieval_method', ['bm25'])
            }
        
        if vectordb_names:
            params['vectordb_name'] = {
                'type': 'categorical',
                'values': list(set(vectordb_names)),
                'condition': ('retrieval_method', ['vectordb'])
            }
        
        return params
    
    def _extract_reranker_params(self) -> Dict[str, Any]:
        params = {}
        reranker_config = self.config_generator.extract_node_config("passage_reranker")
        
        if not reranker_config or not reranker_config.get("modules"):
            return params
        
        methods = []
        
        for module in reranker_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
        
        if methods:
            params['passage_reranker_method'] = {
                'type': 'categorical',
                'values': methods
            }

        top_k_values = reranker_config.get('top_k', [5])
        if isinstance(top_k_values, list) and len(top_k_values) > 0:
            non_pass_methods = [m for m in methods if m != 'pass_reranker']
            if non_pass_methods: 
                params['reranker_top_k'] = {
                    'type': 'int' if len(top_k_values) == 2 else 'categorical',
                    'values': top_k_values,
                    'condition': ('passage_reranker_method', non_pass_methods)
                }

        reranker_configs = []
        for module in reranker_config.get("modules", []):
            method = module.get("module_type")
            if method and method != 'pass_reranker':
                if method in ['colbert_reranker', 'sentence_transformer_reranker', 
                            'flag_embedding_reranker', 'flag_embedding_llm_reranker',
                            'openvino_reranker', 'flashrank_reranker', 'monot5']:

                    if 'model_name' in module:
                        models = module.get("model_name", [])
                        if not isinstance(models, list):
                            models = [models]
                        for model in models:
                            reranker_configs.append(f"{method}_{model}")

                    elif 'model' in module and method == 'flashrank_reranker':
                        models = module.get("model", [])
                        if not isinstance(models, list):
                            models = [models]
                        for model in models:
                            reranker_configs.append(f"{method}_{model}")
                    else:
                        reranker_configs.append(method)
                else:
                    reranker_configs.append(method)
        
        if reranker_configs:
            params['reranker_config'] = {
                'type': 'categorical',
                'values': reranker_configs,
                'condition': ('passage_reranker_method', [m for m in methods if m != 'pass_reranker'])
            }
        
        return params
    
    def _extract_filter_params(self) -> Dict[str, Any]:
        params = {}
        filter_config = self.config_generator.extract_node_config("passage_filter")
        
        if not filter_config or not filter_config.get("modules"):
            return params
        
        methods = []
        threshold_values = {}
        percentile_values = {}
        
        for module in filter_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method == "threshold_cutoff":
                    if "threshold" in module:
                        threshold_values[method] = module["threshold"]
                elif method == "percentile_cutoff":
                    if "percentile" in module:
                        percentile_values[method] = module["percentile"]
                elif method == "similarity_threshold_cutoff":
                    if "threshold" in module:
                        threshold_values[method] = module["threshold"]
                elif method == "similarity_percentile_cutoff":
                    if "percentile" in module:
                        percentile_values[method] = module["percentile"]
        
        if methods:
            params['passage_filter_method'] = {
                'type': 'categorical',
                'values': methods
            }
        
        if threshold_values:
            params['threshold'] = {
                'type': 'float',
                'method_values': threshold_values,
                'condition': ('passage_filter_method', list(threshold_values.keys()))
            }
        
        if percentile_values:
            params['percentile'] = {
                'type': 'float',
                'method_values': percentile_values,
                'condition': ('passage_filter_method', list(percentile_values.keys()))
            }
        
        return params
    
    def _extract_compressor_params(self) -> Dict[str, Any]:
        params = {}
        compressor_config = self.config_generator.extract_node_config("passage_compressor")
        
        if not compressor_config or not compressor_config.get("modules"):
            return params
        
        methods = []
        llm_models = {}
        compression_ratios = {}
        thresholds = {}
        dampings = {}
        max_iterations = {}
        spacy_models = {}
        
        for module in compressor_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method in ["tree_summarize", "refine"]:
                    llm = module.get("llm")
                    model = module.get("model")
                    if llm and model:
                        if method not in llm_models:
                            llm_models[method] = []
                        llm_models[method].append(f"{llm}_{model}")
                
                elif method == "lexrank":
                    if "compression_ratio" in module:
                        compression_ratios[method] = module["compression_ratio"]
                    if "threshold" in module:
                        thresholds[method] = module["threshold"]
                    if "damping" in module:
                        dampings[method] = module["damping"]
                    if "max_iterations" in module:
                        max_iterations[method] = module["max_iterations"]
                
                elif method == "spacy":
                    if "compression_ratio" in module:
                        compression_ratios[method] = module["compression_ratio"]
                    if "spacy_model" in module:
                        spacy_models[method] = module["spacy_model"]
        
        if methods:
            params['passage_compressor_method'] = {
                'type': 'categorical',
                'values': methods
            }

        all_llm_models = []
        methods_with_llm = []
        for method, models in llm_models.items():
            all_llm_models.extend(models)
            methods_with_llm.append(method)
        
        if all_llm_models:
            params['compressor_llm_model'] = {
                'type': 'categorical',
                'values': list(set(all_llm_models)),
                'condition': ('passage_compressor_method', methods_with_llm)
            }

        if compression_ratios:
            all_are_ranges = True
            all_min_values = []
            all_max_values = []
            
            for method, values in compression_ratios.items():
                if isinstance(values, list) and len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                    all_min_values.append(min(values))
                    all_max_values.append(max(values))
                else:
                    all_are_ranges = False
                    break

            if all_are_ranges and all_min_values and all_max_values:
                params['compressor_compression_ratio'] = {
                    'type': 'float',
                    'values': [min(all_min_values), max(all_max_values)],
                    'condition': ('passage_compressor_method', list(compression_ratios.keys()))
                }
            else:
                params['compressor_compression_ratio'] = {
                    'type': 'float',
                    'method_values': compression_ratios,
                    'condition': ('passage_compressor_method', list(compression_ratios.keys()))
                }

        self._add_compressor_float_param(params, thresholds, 'compressor_threshold', ['lexrank'])
        self._add_compressor_float_param(params, dampings, 'compressor_damping', ['lexrank'])
        self._add_compressor_int_param(params, max_iterations, 'compressor_max_iterations', ['lexrank'])

        if spacy_models:
            all_spacy_models = []
            for models in spacy_models.values():
                if isinstance(models, list):
                    all_spacy_models.extend(models)
                else:
                    all_spacy_models.append(models)
            
            params['compressor_spacy_model'] = {
                'type': 'categorical',
                'values': list(set(all_spacy_models)),
                'condition': ('passage_compressor_method', ['spacy'])
            }
        
        return params
    
    def _add_compressor_float_param(self, params, values_dict, param_name, condition_methods):
        if values_dict:
            all_values = []
            is_range = False
            
            for method, values in values_dict.items():
                if isinstance(values, list):
                    if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                        is_range = True
                        all_values.extend(values)
                    else:
                        all_values.extend(values)
                else:
                    all_values.append(values)
            
            if is_range and len(set(all_values)) == 2:
                params[param_name] = {
                    'type': 'float',
                    'values': [min(all_values), max(all_values)],
                    'condition': ('passage_compressor_method', condition_methods)
                }
            else:
                params[param_name] = {
                    'type': 'float',
                    'method_values': values_dict,
                    'condition': ('passage_compressor_method', condition_methods)
                }
    
    def _add_compressor_int_param(self, params, values_dict, param_name, condition_methods):
        if values_dict:
            all_values = []
            is_range = False
            
            for method, values in values_dict.items():
                if isinstance(values, list):
                    if len(values) == 2 and all(isinstance(v, int) for v in values):
                        is_range = True
                        all_values.extend(values)
                    else:
                        all_values.extend(values)
                else:
                    all_values.append(values)
            
            if is_range and len(set(all_values)) == 2:
                params[param_name] = {
                    'type': 'int',
                    'values': [min(all_values), max(all_values)],
                    'condition': ('passage_compressor_method', condition_methods)
                }
            else:
                params[param_name] = {
                    'type': 'int',
                    'method_values': values_dict,
                    'condition': ('passage_compressor_method', condition_methods)
                }
    
    def _extract_prompt_maker_params(self) -> Dict[str, Any]:
        params = {}
        prompt_methods, prompt_indices = self.config_generator.extract_prompt_maker_options()
        
        if prompt_methods:
            params['prompt_maker_method'] = {
                'type': 'categorical',
                'values': prompt_methods
            }
            
            if prompt_indices:
                params['prompt_template_idx'] = {
                    'type': 'categorical',
                    'values': prompt_indices,
                    'condition': ('prompt_maker_method', [m for m in prompt_methods if m != 'pass_prompt_maker'])
                }
        
        return params
    
    def _extract_generator_params(self) -> Dict[str, Any]:
        params = {}
        gen_params = self.config_generator.extract_generator_parameters()
        
        if gen_params.get('models'):
            params['generator_model'] = {
                'type': 'categorical',
                'values': gen_params['models']
            }
        
        if gen_params.get('temperatures'):
            temps = gen_params['temperatures']
            if len(temps) == 2 and isinstance(temps[0], (int, float)):
                params['generator_temperature'] = {
                    'type': 'float',
                    'values': temps
                }
            else:
                params['generator_temperature'] = {
                    'type': 'categorical',
                    'values': temps
                }
        
        return params