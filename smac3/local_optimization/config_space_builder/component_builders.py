from typing import Dict, Any
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, InCondition, EqualsCondition, AndConjunction


class ComponentSpaceBuilder:
    def __init__(self, config_generator, parent, seed: int = 42):
        self.config_generator = config_generator
        self.parent = parent
        self.seed = seed
    
    def build_component_space(self, component: str, fixed_params: Dict[str, Any] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        
        if fixed_params is None:
            fixed_params = {}
        
        if component == 'passage_compressor':
            return self._build_compressor_space(cs, fixed_params)
        elif component == 'retrieval':
            return self._build_retrieval_space(cs, fixed_params)
        elif component == 'passage_filter':
            return self._build_filter_space(cs, fixed_params)
        elif component == 'passage_reranker':
            return self._build_reranker_space(cs, fixed_params)
        else:
            return self._build_generic_space(cs, component, fixed_params)
    
    def _build_compressor_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        compressor_config = self.config_generator.extract_node_config("passage_compressor")
        if not compressor_config:
            return cs
        
        methods = []
        method_params = {}
        
        for module in compressor_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method in ['tree_summarize', 'refine']:
                    gen_type = module.get('generator_module_type', 'llama_index_llm')
                    
                    if gen_type == 'vllm':
                        models = module.get('llm', [])
                    else:
                        models = module.get('model', ['gpt-3.5-turbo'])
                    
                    if not isinstance(models, list):
                        models = [models]
                    
                    llm = module.get('llm', 'openai')
                    
                    if method not in method_params:
                        method_params[method] = {'models': []}
                    
                    for model in models:
                        method_params[method]['models'].append(f"{llm}_{model}")
                
                elif method == 'lexrank':
                    method_params[method] = {}
                    
                    compression_ratios = module.get('compression_ratio', [0.5])
                    if isinstance(compression_ratios, list) and len(compression_ratios) == 2:
                        method_params[method]['compression_ratio'] = tuple(compression_ratios)
                    else:
                        method_params[method]['compression_ratio'] = compression_ratios if isinstance(compression_ratios, list) else [compression_ratios]
                    
                    thresholds = module.get('threshold', [0.1])
                    if isinstance(thresholds, list) and len(thresholds) == 2:
                        method_params[method]['threshold'] = tuple(thresholds)
                    else:
                        method_params[method]['threshold'] = thresholds if isinstance(thresholds, list) else [thresholds]
                    
                    dampings = module.get('damping', [0.85])
                    if isinstance(dampings, list) and len(dampings) == 2:
                        method_params[method]['damping'] = tuple(dampings)
                    else:
                        method_params[method]['damping'] = dampings if isinstance(dampings, list) else [dampings]
                    
                    iterations = module.get('max_iterations', [30])
                    if isinstance(iterations, list) and len(iterations) == 2:
                        method_params[method]['max_iterations'] = tuple(iterations)
                    else:
                        method_params[method]['max_iterations'] = iterations if isinstance(iterations, list) else [iterations]
                
                elif method == 'spacy':
                    method_params[method] = {}
                    
                    compression_ratios = module.get('compression_ratio', [0.5])
                    if isinstance(compression_ratios, list) and len(compression_ratios) == 2:
                        method_params[method]['compression_ratio'] = tuple(compression_ratios)
                    else:
                        method_params[method]['compression_ratio'] = compression_ratios if isinstance(compression_ratios, list) else [compression_ratios]
                    
                    spacy_models = module.get('spacy_model', ['en_core_web_sm'])
                    if not isinstance(spacy_models, list):
                        spacy_models = [spacy_models]
                    method_params[method]['spacy_model'] = spacy_models
        
        if methods:
            method_param = Categorical('passage_compressor_method', 
                                    items=methods,
                                    default=self.parent._get_default_value('passage_compressor_method', methods))
            cs.add(method_param)
        
        if 'tree_summarize' in method_params or 'refine' in method_params:
            all_models = set()
            if 'tree_summarize' in method_params:
                all_models.update(method_params['tree_summarize'].get('models', []))
            if 'refine' in method_params:
                all_models.update(method_params['refine'].get('models', []))
            
            if all_models:
                model_param = Categorical('compressor_llm_model',
                                        items=list(all_models),
                                        default=list(all_models)[0])
                cs.add(model_param)
                
                cs.add(InCondition(cs['compressor_llm_model'],
                                cs['passage_compressor_method'],
                                ['tree_summarize', 'refine']))
        
        compression_ratio_added = False
        
        if 'lexrank' in method_params:
            params = method_params['lexrank']
            
            if 'compression_ratio' in params and not compression_ratio_added:
                if isinstance(params['compression_ratio'], tuple):
                    comp_ratio_param = Float('compressor_compression_ratio',
                                        bounds=params['compression_ratio'],
                                        default=params['compression_ratio'][0])
                else:
                    comp_ratio_param = Categorical('compressor_compression_ratio',
                                                items=params['compression_ratio'],
                                                default=params['compression_ratio'][0])
                cs.add(comp_ratio_param)
                compression_ratio_added = True
            
            if 'threshold' in params:
                if isinstance(params['threshold'], tuple):
                    threshold_param = Float('compressor_threshold',
                                        bounds=params['threshold'],
                                        default=params['threshold'][0])
                else:
                    threshold_param = Categorical('compressor_threshold',
                                                items=params['threshold'],
                                                default=params['threshold'][0])
                cs.add(threshold_param)
                cs.add(EqualsCondition(cs['compressor_threshold'],
                                    cs['passage_compressor_method'],
                                    'lexrank'))
            
            if 'damping' in params:
                if isinstance(params['damping'], tuple):
                    damping_param = Float('compressor_damping',
                                        bounds=params['damping'],
                                        default=params['damping'][0])
                else:
                    damping_param = Categorical('compressor_damping',
                                            items=params['damping'],
                                            default=params['damping'][0])
                cs.add(damping_param)
                cs.add(EqualsCondition(cs['compressor_damping'],
                                    cs['passage_compressor_method'],
                                    'lexrank'))
            
            if 'max_iterations' in params:
                if isinstance(params['max_iterations'], tuple):
                    iterations_param = Integer('compressor_max_iterations',
                                            bounds=params['max_iterations'],
                                            default=params['max_iterations'][0])
                else:
                    iterations_param = Categorical('compressor_max_iterations',
                                                items=params['max_iterations'],
                                                default=params['max_iterations'][0])
                cs.add(iterations_param)
                cs.add(EqualsCondition(cs['compressor_max_iterations'],
                                    cs['passage_compressor_method'],
                                    'lexrank'))
        
        if 'spacy' in method_params:
            params = method_params['spacy']
            
            if 'spacy_model' in params:
                model_param = Categorical('compressor_spacy_model',
                                        items=params['spacy_model'],
                                        default=params['spacy_model'][0])
                cs.add(model_param)
                cs.add(EqualsCondition(cs['compressor_spacy_model'],
                                    cs['passage_compressor_method'],
                                    'spacy'))
        
        if compression_ratio_added:
            methods_with_compression = []
            if 'lexrank' in method_params:
                methods_with_compression.append('lexrank')
            if 'spacy' in method_params:
                methods_with_compression.append('spacy')
            
            if len(methods_with_compression) > 1:
                cs.add(InCondition(cs['compressor_compression_ratio'],
                                cs['passage_compressor_method'],
                                methods_with_compression))
            elif len(methods_with_compression) == 1:
                cs.add(EqualsCondition(cs['compressor_compression_ratio'],
                                    cs['passage_compressor_method'],
                                    methods_with_compression[0]))
        
        return cs
    
    def _build_retrieval_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        retrieval_config = self.config_generator.extract_node_config("retrieval")
        top_k_values = retrieval_config.get('top_k', [5])
        if isinstance(top_k_values, list) and len(top_k_values) > 0:
            if len(top_k_values) == 2:
                top_k_param = Integer('retriever_top_k', 
                                    bounds=(min(top_k_values), max(top_k_values)),
                                    default=min(top_k_values))
            else:
                top_k_param = Categorical('retriever_top_k', 
                                        items=top_k_values,
                                        default=top_k_values[0])
        else:
            top_k_param = Integer('retriever_top_k', bounds=(1, 10), default=5)
        
        cs.add(top_k_param)
        
        retrieval_configs = []
        for module in retrieval_config.get("modules", []):
            module_type = module.get("module_type")
            if module_type == "bm25":
                tokenizers = module.get("bm25_tokenizer", ["porter_stemmer"])
                if not isinstance(tokenizers, list):
                    tokenizers = [tokenizers]
                for tokenizer in tokenizers:
                    retrieval_configs.append(f"bm25_{tokenizer}")
            elif module_type == "vectordb":
                vdbs = module.get("vectordb", ["default"])
                if not isinstance(vdbs, list):
                    vdbs = [vdbs]
                for vdb in vdbs:
                    retrieval_configs.append(f"vectordb_{vdb}")
        
        if retrieval_configs:
            retrieval_config_param = Categorical('retrieval_config', 
                                            items=retrieval_configs,
                                            default=retrieval_configs[0])
            cs.add(retrieval_config_param)
            print(f"[DEBUG] Added retrieval_config with choices: {retrieval_configs}")
        
        return cs
    
    def _build_filter_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        filter_config = self.config_generator.extract_node_config("passage_filter")
        if not filter_config:
            return cs
        
        filter_methods = []
        threshold_values = {}
        percentile_values = {}
        
        for module in filter_config.get("modules", []):
            method = module.get("module_type")
            if method:
                filter_methods.append(method)
                
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
        
        if filter_methods:
            method_param = Categorical('passage_filter_method', 
                                    items=filter_methods,
                                    default=filter_methods[0])
            cs.add(method_param)
            print(f"[DEBUG] Added passage_filter_method with choices: {filter_methods}")
        
        if threshold_values:
            all_threshold_values = []
            for method, values in threshold_values.items():
                if isinstance(values, list):
                    all_threshold_values.extend(values)
                else:
                    all_threshold_values.append(values)
            
            all_threshold_values = sorted(list(set(all_threshold_values)))
            
            if len(all_threshold_values) == 2:
                threshold_param = Float('threshold', 
                                    bounds=(min(all_threshold_values), max(all_threshold_values)),
                                    default=min(all_threshold_values))
            elif len(all_threshold_values) > 2:
                threshold_param = Categorical('threshold', 
                                            items=all_threshold_values,
                                            default=all_threshold_values[0])
            else:
                threshold_param = Float('threshold', 
                                    bounds=(0.0, 1.0),
                                    default=0.75)
            
            cs.add(threshold_param)
            
            threshold_methods = [m for m in filter_methods if m in threshold_values]
            if threshold_methods and 'passage_filter_method' in cs:
                cs.add(InCondition(cs['threshold'], 
                                cs['passage_filter_method'], 
                                threshold_methods))
            print(f"[DEBUG] Added threshold parameter with values: {all_threshold_values}")
        
        if percentile_values:
            all_percentile_values = []
            for method, values in percentile_values.items():
                if isinstance(values, list):
                    all_percentile_values.extend(values)
                else:
                    all_percentile_values.append(values)
            
            all_percentile_values = sorted(list(set(all_percentile_values)))
            
            if len(all_percentile_values) == 2:
                percentile_param = Float('percentile', 
                                    bounds=(min(all_percentile_values), max(all_percentile_values)),
                                    default=min(all_percentile_values))
            elif len(all_percentile_values) > 2:
                percentile_param = Categorical('percentile', 
                                            items=all_percentile_values,
                                            default=all_percentile_values[0])
            else:
                percentile_param = Float('percentile', 
                                    bounds=(0.0, 1.0),
                                    default=0.6)
            
            cs.add(percentile_param)
            
            percentile_methods = [m for m in filter_methods if m in percentile_values]
            if percentile_methods and 'passage_filter_method' in cs:
                cs.add(InCondition(cs['percentile'], 
                                cs['passage_filter_method'], 
                                percentile_methods))
            print(f"[DEBUG] Added percentile parameter with values: {all_percentile_values}")
            
        return cs
            
    def _build_reranker_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        reranker_config = self.config_generator.extract_node_config("passage_reranker")
        if not reranker_config:
            return cs

        top_k_values = reranker_config.get('top_k', [5])
        prev_top_k = fixed_params.get('retriever_top_k', 10)
        
        if isinstance(top_k_values, list) and len(top_k_values) > 0:
            if len(top_k_values) == 2:
                min_k, max_k = min(top_k_values), max(top_k_values)
                max_k = min(max_k, prev_top_k)
                if min_k <= max_k:
                    top_k_param = Integer('reranker_top_k', 
                                        bounds=(min_k, max_k),
                                        default=min_k)
                else:
                    top_k_param = Integer('reranker_top_k', 
                                        bounds=(min_k, min_k),
                                        default=min_k)
            else:
                valid_values = [k for k in top_k_values if k <= prev_top_k]
                if valid_values:
                    top_k_param = Categorical('reranker_top_k', 
                                            items=valid_values,
                                            default=valid_values[0])
                else:
                    top_k_param = Integer('reranker_top_k', 
                                        bounds=(1, 1),
                                        default=1)
        else:
            top_k_param = Integer('reranker_top_k', bounds=(1, prev_top_k), default=min(5, prev_top_k))
        
        cs.add(top_k_param)
        
        reranker_configs = []
        for module in reranker_config.get("modules", []):
            method = module.get("module_type")
            if method:
                if method in ['pass_reranker', 'upr', 'colbert_reranker']:
                    reranker_configs.append(method)
                elif method in ['sentence_transformer_reranker', 'flag_embedding_reranker', 
                            'flag_embedding_llm_reranker', 'openvino_reranker', 'monot5']:
                    models = module.get("model_name", [])
                    if not isinstance(models, list):
                        models = [models]
                    if models:
                        for model in models:
                            reranker_configs.append(f"{method}_{model}")
                    else:
                        reranker_configs.append(method)
                elif method == 'flashrank_reranker':
                    models = module.get("model", [])
                    if not isinstance(models, list):
                        models = [models]
                    if models:
                        for model in models:
                            reranker_configs.append(f"{method}_{model}")
                    else:
                        reranker_configs.append(method)
        
        if reranker_configs:
            reranker_config_param = Categorical('reranker_config', 
                                            items=reranker_configs,
                                            default=reranker_configs[0])
            cs.add(reranker_config_param)
        
        return cs
    
    def _build_generic_space(self, cs: ConfigurationSpace, component: str, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()
        
        if component == 'query_expansion':
            self._add_query_expansion_specific_params(cs, unified_space, fixed_params)
        else:
            component_params = self._get_component_parameters(component, unified_space)
            
            for param_name, param_info in component_params.items():
                if param_name in fixed_params:
                    continue
                    
                param = self.parent._create_parameter(param_name, param_info['type'], param_info)
                if param:
                    cs.add(param)
                    
                    condition = param_info.get('condition')
                    if condition and not self._condition_uses_fixed_param(condition, fixed_params):
                        self.parent._add_single_condition(cs, param_name, condition)
        
        self._add_component_method_specific_params(cs, component, unified_space, fixed_params)
        
        print(f"[DEBUG] Final config space parameters: {list(cs.get_hyperparameters_dict().keys())}")
        
        return cs
    
    def _get_component_parameters(self, component: str, unified_space: Dict[str, Any]) -> Dict[str, Any]:
        component_params = {}
        
        component_prefixes = {
            'query_expansion': ['query_expansion_'],
            'retrieval': ['retrieval_', 'retriever_', 'bm25_', 'vectordb_'],
            'passage_reranker': ['passage_reranker_', 'reranker_'],
            'passage_filter': ['passage_filter_'],
            'passage_compressor': ['passage_compressor_', 'compressor_'],
            'prompt_maker_generator': ['prompt_maker_', 'prompt_', 'generator_'],  
        }
                
        prefixes = component_prefixes.get(component, [])
        
        for param_name, param_info in unified_space.items():
            if param_name == 'retriever_top_k' and component in ['query_expansion', 'retrieval']:
                component_params[param_name] = param_info
            elif any(param_name.startswith(prefix) for prefix in prefixes):
                if component == 'retrieval' and param_name.startswith('query_expansion_'):
                    continue
                component_params[param_name] = param_info
        
        return component_params
    
    def _add_query_expansion_specific_params(self, cs: ConfigurationSpace, 
                                       unified_space: Dict[str, Any], 
                                       fixed_params: Dict[str, Any]):
        if 'query_expansion_method' in unified_space and 'query_expansion_method' not in fixed_params:
            qe_param_info = unified_space['query_expansion_method']
            qe_param = Categorical('query_expansion_method', qe_param_info['values'], 
                                default=self.parent._get_default_value('query_expansion_method', qe_param_info['values']))
            cs.add(qe_param)
        
        if 'retriever_top_k' in unified_space and 'retriever_top_k' not in fixed_params:
            top_k_info = unified_space['retriever_top_k']
            top_k_param = self.parent._create_parameter('retriever_top_k', top_k_info['type'], top_k_info)
            if top_k_param:
                cs.add(top_k_param)

        retrieval_config = self.config_generator.extract_node_config("retrieval")
        if retrieval_config:
            retrieval_methods = []
            bm25_tokenizers = []
            vectordb_names = []
            
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

            if retrieval_methods and 'retrieval_method' not in fixed_params and 'query_expansion_method' in cs:
                retrieval_method_param = Categorical('retrieval_method', 
                                                list(set(retrieval_methods)),
                                                default=retrieval_methods[0])
                cs.add(retrieval_method_param)

                cs.add(EqualsCondition(cs['retrieval_method'], 
                                    cs['query_expansion_method'], 
                                    'pass_query_expansion'))

            if bm25_tokenizers and 'bm25_tokenizer' not in fixed_params and 'retrieval_method' in cs:
                bm25_param = Categorical('bm25_tokenizer', 
                                    list(set(bm25_tokenizers)),
                                    default=bm25_tokenizers[0])
                cs.add(bm25_param)

                cs.add(AndConjunction(
                    EqualsCondition(cs['bm25_tokenizer'], cs['query_expansion_method'], 'pass_query_expansion'),
                    EqualsCondition(cs['bm25_tokenizer'], cs['retrieval_method'], 'bm25')
                ))

            if vectordb_names and 'vectordb_name' not in fixed_params and 'retrieval_method' in cs:
                vdb_param = Categorical('vectordb_name', 
                                    list(set(vectordb_names)),
                                    default=vectordb_names[0])
                cs.add(vdb_param)
                
                cs.add(AndConjunction(
                    EqualsCondition(cs['vectordb_name'], cs['query_expansion_method'], 'pass_query_expansion'),
                    EqualsCondition(cs['vectordb_name'], cs['retrieval_method'], 'vectordb')
                ))

        qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
        if qe_retrieval_options and qe_retrieval_options.get('methods'):
            if 'query_expansion_retrieval_method' not in cs and 'query_expansion_retrieval_method' not in fixed_params:
                qe_retrieval_param = Categorical('query_expansion_retrieval_method', 
                                            qe_retrieval_options['methods'],
                                            default=qe_retrieval_options['methods'][0])
                cs.add(qe_retrieval_param)

                if 'query_expansion_method' in cs:
                    active_qe_methods = [m for m in cs['query_expansion_method'].choices if m != 'pass_query_expansion']
                    if active_qe_methods:
                        cs.add(InCondition(cs['query_expansion_retrieval_method'], 
                                        cs['query_expansion_method'], 
                                        active_qe_methods))

            if 'vectordb' in qe_retrieval_options['methods'] and qe_retrieval_options.get('vectordb_names'):
                if 'query_expansion_vectordb_name' not in cs and 'query_expansion_vectordb_name' not in fixed_params:
                    qe_vdb_param = Categorical('query_expansion_vectordb_name',
                                            qe_retrieval_options['vectordb_names'],
                                            default=qe_retrieval_options['vectordb_names'][0])
                    cs.add(qe_vdb_param)
                    
                    if 'query_expansion_retrieval_method' in cs:
                        cs.add(EqualsCondition(cs['query_expansion_vectordb_name'],
                                            cs['query_expansion_retrieval_method'],
                                            'vectordb'))

            if 'bm25' in qe_retrieval_options['methods'] and qe_retrieval_options.get('bm25_tokenizers'):
                if 'query_expansion_bm25_tokenizer' not in cs and 'query_expansion_bm25_tokenizer' not in fixed_params:
                    qe_bm25_param = Categorical('query_expansion_bm25_tokenizer',
                                            qe_retrieval_options['bm25_tokenizers'],
                                            default=qe_retrieval_options['bm25_tokenizers'][0])
                    cs.add(qe_bm25_param)
                    
                    if 'query_expansion_retrieval_method' in cs:
                        cs.add(EqualsCondition(cs['query_expansion_bm25_tokenizer'],
                                            cs['query_expansion_retrieval_method'],
                                            'bm25'))

        query_expansion_params = {k: v for k, v in unified_space.items() 
                                if k.startswith('query_expansion_') and k not in 
                                ['query_expansion_method', 'query_expansion_retrieval_method', 
                                'query_expansion_vectordb_name', 'query_expansion_bm25_tokenizer']}
        
        for param_name, param_info in query_expansion_params.items():
            if param_name not in fixed_params:
                param = self.parent._create_parameter(param_name, param_info['type'], param_info)
                if param:
                    cs.add(param)
                    
                    condition = param_info.get('condition')
                    if condition and not self._condition_uses_fixed_param(condition, fixed_params):
                        self.parent._add_single_condition(cs, param_name, condition)
    
    def _add_component_method_specific_params(self, cs: ConfigurationSpace, component: str,
                                            unified_space: Dict[str, Any], fixed_params: Dict[str, Any]):
        if component == 'passage_filter':
            if 'threshold' in unified_space and 'threshold' not in fixed_params:
                threshold_info = unified_space['threshold']
                if 'method_values' in threshold_info:
                    all_values = self._extract_all_values(threshold_info['method_values'])
                    if all_values:
                        param = self._create_parameter_from_values('threshold', all_values, threshold_info['type'])
                        cs.add(param)
                        
                        if threshold_info.get('condition'):
                            self.parent._add_single_condition(cs, 'threshold', threshold_info['condition'])
            
            if 'percentile' in unified_space and 'percentile' not in fixed_params:
                percentile_info = unified_space['percentile']
                if 'method_values' in percentile_info:
                    all_values = self._extract_all_values(percentile_info['method_values'])
                    if all_values:
                        param = self._create_parameter_from_values('percentile', all_values, percentile_info['type'])
                        cs.add(param)
                        
                        if percentile_info.get('condition'):
                            self.parent._add_single_condition(cs, 'percentile', percentile_info['condition'])
    
    def _condition_uses_fixed_param(self, condition, fixed_params: Dict[str, Any]) -> bool:
        if isinstance(condition, list):
            return any(self._condition_uses_fixed_param(c, fixed_params) for c in condition)
        else:
            parent_param = condition[0] if condition else None
            return parent_param in fixed_params
    
    def _extract_all_values(self, method_values: Dict[str, Any]) -> list:
        all_values = []
        for method, values in method_values.items():
            if isinstance(values, list):
                all_values.extend(values)
            else:
                all_values.append(values)
        return sorted(list(set(all_values)))
    
    def _create_parameter_from_values(self, name: str, values: list, param_type: str):
        if not values:
            return None
        
        if param_type == 'float':
            if len(values) == 2:
                return Float(name, bounds=(min(values), max(values)), default=min(values))
            else:
                return Categorical(name, values, default=values[0])
        else:
            return Categorical(name, values, default=values[0])