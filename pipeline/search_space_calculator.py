from typing import Dict, Any, List, Tuple, Optional
import math


class SearchSpaceCalculator:
    
    def __init__(self, config_generator):
        self.config_generator = config_generator
        self.search_type = 'grid'
    
    def calculate_component_combinations(self, component: str, search_space: Dict[str, Any] = None, 
                                        fixed_config: Dict[str, Any] = None, 
                                        best_configs: Dict[str, Any] = None) -> Tuple[int, str]:
        if not search_space:
            search_space = self._extract_component_search_space(component)
        
        if component == 'query_expansion':
            return self._calculate_qe_combinations(search_space, best_configs)
        elif component == 'retrieval':
            return self._calculate_retrieval_combinations(search_space, best_configs)
        elif component == 'passage_reranker':
            return self._calculate_reranker_combinations(search_space, fixed_config, best_configs)
        elif component == 'passage_filter':
            return self._calculate_filter_combinations(search_space)
        elif component == 'passage_compressor':
            return self._calculate_compressor_combinations(search_space)
        elif component == 'prompt_maker_generator':
            return self._calculate_prompt_gen_combinations(search_space)
        
        return 1, "Unknown component"
    
    def _calculate_qe_combinations(self, search_space: Dict[str, Any], best_configs: Dict[str, Any] = None) -> Tuple[int, str]:
        if 'query_expansion_config' in search_space:
            configs = search_space['query_expansion_config']
            retriever_top_k_values = self._get_top_k_values(search_space.get('retriever_top_k', [10]))
            retrieval_combinations = self._get_qe_retrieval_combinations(search_space)
            
            total = 0
            for config in configs:
                if config == 'pass_query_expansion':
                    total += len(retriever_top_k_values) * retrieval_combinations
                else:
                    parts = config.split('::')
                    if len(parts) >= 3:
                        method = parts[0]
                        base_combos = len(retriever_top_k_values) * retrieval_combinations
                        
                        if method == 'hyde' and 'query_expansion_max_token' in search_space:
                            total += base_combos * len(search_space['query_expansion_max_token'])
                        elif method == 'multi_query_expansion' and 'query_expansion_temperature' in search_space:
                            total += base_combos * len(search_space['query_expansion_temperature'])
                        else:
                            total += base_combos
            
            return total, "Query expansion config-based combinations"
        
        return self._calculate_qe_standard_combinations(search_space)
    
    def _calculate_qe_standard_combinations(self, search_space: Dict[str, Any]) -> Tuple[int, str]:
        methods = search_space.get('query_expansion_method', ['pass_query_expansion'])
        top_k_values = self._get_top_k_values(search_space.get('retriever_top_k', [10]))
        retrieval_combinations = self._get_standard_retrieval_combinations(search_space)
        
        total = 0
        for method in methods:
            base_combos = len(top_k_values) * retrieval_combinations
            
            if method == 'pass_query_expansion':
                total += base_combos
            elif method == 'query_decompose':
                models = search_space.get('query_expansion_model', [])
                total += len(models) * base_combos if models else base_combos
            elif method == 'hyde':
                models = search_space.get('query_expansion_model', [])
                max_tokens = search_space.get('query_expansion_max_token', [])
                model_count = len(models) if models else 1
                token_count = len(max_tokens) if max_tokens else 1
                total += model_count * token_count * base_combos
            elif method == 'multi_query_expansion':
                models = search_space.get('query_expansion_model', [])
                temps = search_space.get('query_expansion_temperature', [])
                model_count = len(models) if models else 1
                temp_count = len(temps) if temps else 1
                total += model_count * temp_count * base_combos
        
        return total, "Standard query expansion combinations"
    
    def _calculate_retrieval_combinations(self, search_space: Dict[str, Any], best_configs: Dict[str, Any] = None) -> Tuple[int, str]:
        if best_configs and 'query_expansion' in best_configs:
            qe_method = best_configs['query_expansion'].get('query_expansion_method')
            if qe_method and qe_method != 'pass_query_expansion':
                return 0, "Skipped due to active query expansion"
        
        methods = search_space.get('retrieval_method', [])
        top_k_values = self._get_top_k_values(search_space.get('retriever_top_k', [10]))
        
        total = 0
        for method in methods:
            if method == 'bm25':
                tokenizers = search_space.get('bm25_tokenizer', [])
                total += len(top_k_values) * (len(tokenizers) if tokenizers else 1)
            elif method == 'vectordb':
                vdb_names = search_space.get('vectordb_name', ['default'])
                total += len(top_k_values) * len(vdb_names)
            else:
                total += len(top_k_values)
        
        return total, "Retrieval combinations"
    
    def _calculate_reranker_combinations(self, search_space: Dict[str, Any], fixed_config: Dict[str, Any] = None, 
                                        best_configs: Dict[str, Any] = None) -> Tuple[int, str]:
        methods = search_space.get('passage_reranker_method', [])
        top_k_values = search_space.get('reranker_top_k', [])
        
        if fixed_config and 'retriever_top_k' in fixed_config:
            retriever_top_k = fixed_config['retriever_top_k']
            top_k_values = [k for k in top_k_values if k <= retriever_top_k]
        
        if len(top_k_values) == 0:
            return 0, "No valid reranker_top_k values"
        
        total = 0
        for method in methods:
            if method == 'pass_reranker':
                total += 1
            elif method in ['upr', 'colbert_reranker']:
                total += len(top_k_values)
            else:
                model_key = f"{method}_models"
                models = search_space.get(model_key, [])
                if models:
                    total += len(models) * len(top_k_values)
                else:
                    total += len(top_k_values)
        
        return total, "Reranker combinations"
    
    def _calculate_filter_combinations(self, search_space: Dict[str, Any]) -> Tuple[int, str]:
        methods = search_space.get('passage_filter_method', [])
        
        total = 0
        for method in methods:
            if method == 'pass_passage_filter':
                total += 1
            elif 'threshold' in method:
                param_key = f"{method}_threshold"
                values = search_space.get(param_key, [])
                total += len(values) if values else 1
            elif 'percentile' in method:
                param_key = f"{method}_percentile"
                values = search_space.get(param_key, [])
                total += len(values) if values else 1
        
        return total, "Filter combinations"
    
    def _calculate_compressor_combinations(self, search_space: Dict[str, Any]) -> Tuple[int, str]:
        if 'passage_compressor_config' in search_space:
            configs = search_space['passage_compressor_config']
            
            total = 0
            for config in configs:
                if config == 'pass_compressor':
                    total += 1
                elif config == 'lexrank':
                    thresholds = search_space.get('lexrank_threshold', [0.05, 0.3])
                    dampings = search_space.get('lexrank_damping', [0.75, 0.9])
                    max_iters = search_space.get('lexrank_max_iterations', [15, 40])
                    comp_ratios = search_space.get('lexrank_compression_ratio', [0.3, 0.7])
                    
                    threshold_count = len(thresholds) if isinstance(thresholds, list) else 1
                    damping_count = len(dampings) if isinstance(dampings, list) else 1
                    iter_count = len(max_iters) if isinstance(max_iters, list) else 1
                    ratio_count = len(comp_ratios) if isinstance(comp_ratios, list) else 1
                    
                    total += threshold_count * damping_count * iter_count * ratio_count
                elif config.startswith('spacy::'):
                    comp_ratios = search_space.get('spacy_compression_ratio', [0.3, 0.5])
                    ratio_count = len(comp_ratios) if isinstance(comp_ratios, list) else 1
                    total += ratio_count
                else:
                    total += 1
            
            return total, "Compressor config-based combinations"
        
        return self._calculate_compressor_standard(search_space)

    def _calculate_compressor_standard(self, search_space: Dict[str, Any]) -> Tuple[int, str]:
        methods = search_space.get('passage_compressor_method', [])
        
        total = 0
        for method in methods:
            if method == 'pass_compressor':
                total += 1
            elif method in ['tree_summarize', 'refine']:
                models = search_space.get('compressor_model', [])
                llms = search_space.get('compressor_llm', [])
                model_count = len(models) if models else 1
                llm_count = len(llms) if llms else 1
                total += model_count * llm_count
            elif method == 'lexrank':
                comp_ratios = search_space.get('compressor_compression_ratio', [0.5])
                thresholds = search_space.get('compressor_threshold', [0.1])
                dampings = search_space.get('compressor_damping', [0.85])
                max_iters = search_space.get('compressor_max_iterations', [30])
                
                ratio_count = len(comp_ratios) if isinstance(comp_ratios, list) else 1
                threshold_count = len(thresholds) if isinstance(thresholds, list) else 1
                damping_count = len(dampings) if isinstance(dampings, list) else 1
                iter_count = len(max_iters) if isinstance(max_iters, list) else 1
                
                total += ratio_count * threshold_count * damping_count * iter_count
            elif method == 'spacy':
                comp_ratios = search_space.get('compressor_compression_ratio', [0.5])
                spacy_models = search_space.get('compressor_spacy_model', ['en_core_web_sm'])
                
                ratio_count = len(comp_ratios) if isinstance(comp_ratios, list) else 1
                model_count = len(spacy_models) if isinstance(spacy_models, list) else 1
                
                total += ratio_count * model_count
        
        return total, "Standard compressor combinations"
    
    def _calculate_prompt_gen_combinations(self, search_space: Dict[str, Any]) -> Tuple[int, str]:
        if 'generator_config' in search_space:
            generator_configs = search_space['generator_config']
            temperatures = search_space.get('generator_temperature', [])
            prompt_methods = search_space.get('prompt_maker_method', ['fstring'])
            prompt_indices = search_space.get('prompt_template_idx', [0])
            
            prompt_combinations = len(prompt_methods) * len(prompt_indices)
            gen_combinations = len(generator_configs) * len(temperatures) if temperatures else len(generator_configs)
            
            return prompt_combinations * gen_combinations, "Generator config-based combinations"
        
        prompt_methods = search_space.get('prompt_maker_method', ['fstring'])
        prompt_indices = search_space.get('prompt_template_idx', [0])
        generator_models = search_space.get('generator_model', [])
        temperatures = search_space.get('generator_temperature', [])
        
        prompt_combinations = len(prompt_methods) * len(prompt_indices)
        gen_combinations = len(generator_models) * len(temperatures) if generator_models and temperatures else 1
        
        return prompt_combinations * gen_combinations, "Standard prompt/generator combinations"
    
    def _get_top_k_values(self, top_k_config: Any) -> List[int]:
        if not isinstance(top_k_config, list):
            return [top_k_config]
        
        if len(top_k_config) == 2 and all(isinstance(x, int) for x in top_k_config):
            if top_k_config[1] > top_k_config[0]:
                return list(range(top_k_config[0], top_k_config[1] + 1))
        
        return top_k_config
    
    def _get_list_count(self, value: Any) -> int:
        if not isinstance(value, list):
            return 1
        return len(value)
    
    def _get_qe_retrieval_combinations(self, search_space: Dict[str, Any]) -> int:
        retrieval_methods = search_space.get('query_expansion_retrieval_method', [])
        if not retrieval_methods:
            retrieval_methods = search_space.get('retrieval_method', [])
        
        total = 0
        for method in retrieval_methods:
            if method == 'bm25':
                tokenizers = search_space.get('query_expansion_bm25_tokenizer', 
                                            search_space.get('bm25_tokenizer', []))
                total += len(tokenizers) if tokenizers else 1
            elif method == 'vectordb':
                vdb_names = search_space.get('query_expansion_vectordb_name',
                                           search_space.get('vectordb_name', []))
                total += len(vdb_names) if vdb_names else 1
        
        return total if total > 0 else 1
    
    def _get_standard_retrieval_combinations(self, search_space: Dict[str, Any]) -> int:
        retrieval_methods = search_space.get('retrieval_method', ['bm25'])
        
        total = 0
        for method in retrieval_methods:
            if method == 'bm25':
                tokenizers = search_space.get('bm25_tokenizer', [])
                total += len(tokenizers) if tokenizers else 1
            elif method == 'vectordb':
                vdb_names = search_space.get('vectordb_name', [])
                total += len(vdb_names) if vdb_names else 1
        
        return total if total > 0 else 1
    
    def _extract_component_search_space(self, component: str) -> Dict[str, Any]:
        if component == 'query_expansion':
            return self._extract_qe_search_space_from_config()
        elif component == 'retrieval':
            return self._extract_retrieval_search_space_from_config()
        elif component == 'passage_reranker':
            return self._extract_reranker_search_space_from_config()
        elif component == 'passage_filter':
            return self._extract_filter_search_space_from_config()
        elif component == 'passage_compressor':
            return self._extract_compressor_search_space_from_config()
        elif component == 'prompt_maker_generator':
            return self._extract_prompt_gen_search_space_from_config()
        
        return {}
    
    def _extract_qe_search_space_from_config(self) -> Dict[str, Any]:
        qe_config = self.config_generator.extract_node_config("query_expansion")
        if not qe_config or not qe_config.get("modules", []):
            return {}
        
        space = {}
        methods = []
        models = []
        max_tokens = []
        temperatures = []
        
        for module in qe_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if 'model' in module:
                    module_models = module['model']
                    if isinstance(module_models, list):
                        models.extend(module_models)
                    else:
                        models.append(module_models)
                
                if method == 'hyde' and 'max_token' in module:
                    tokens = module['max_token']
                    if isinstance(tokens, list):
                        max_tokens.extend(tokens)
                    else:
                        max_tokens.append(tokens)
                
                if method == 'multi_query_expansion' and 'temperature' in module:
                    temps = module['temperature']
                    if isinstance(temps, list):
                        temperatures.extend(temps)
                    else:
                        temperatures.append(temps)
        
        if methods:
            space['query_expansion_method'] = list(set(methods))
        if models:
            space['query_expansion_model'] = list(set(models))
        if max_tokens:
            space['query_expansion_max_token'] = list(set(max_tokens))
        if temperatures:
            space['query_expansion_temperature'] = list(set(temperatures))
        
        retrieval_config = self.config_generator.extract_node_config("retrieval")
        if retrieval_config:
            top_k = retrieval_config.get('top_k', [10])
            space['retriever_top_k'] = top_k if isinstance(top_k, list) else [top_k]
            
            retrieval_methods = []
            bm25_tokenizers = []
            vectordb_names = []
            
            for module in retrieval_config.get("modules", []):
                method = module.get("module_type")
                if method == "bm25":
                    retrieval_methods.append("bm25")
                    tokenizers = module.get("bm25_tokenizer", ["porter_stemmer"])
                    if isinstance(tokenizers, list):
                        bm25_tokenizers.extend(tokenizers)
                    else:
                        bm25_tokenizers.append(tokenizers)
                elif method == "vectordb":
                    retrieval_methods.append("vectordb")
                    vdbs = module.get("vectordb", ["default"])
                    if isinstance(vdbs, list):
                        vectordb_names.extend(vdbs)
                    else:
                        vectordb_names.append(vdbs)
            
            if retrieval_methods:
                space['retrieval_method'] = list(set(retrieval_methods))
            if bm25_tokenizers:
                space['bm25_tokenizer'] = list(set(bm25_tokenizers))
            if vectordb_names:
                space['vectordb_name'] = list(set(vectordb_names))
        
        return space
    
    def _extract_retrieval_search_space_from_config(self) -> Dict[str, Any]:
        retrieval_config = self.config_generator.extract_node_config("retrieval")
        if not retrieval_config:
            return {}
        
        space = {}
        
        top_k = retrieval_config.get('top_k', [10])
        space['retriever_top_k'] = top_k if isinstance(top_k, list) else [top_k]
        
        methods = []
        bm25_tokenizers = []
        vectordb_names = []
        
        for module in retrieval_config.get("modules", []):
            method = module.get("module_type")
            if method == "bm25":
                methods.append("bm25")
                tokenizers = module.get("bm25_tokenizer", ["porter_stemmer"])
                if isinstance(tokenizers, list):
                    bm25_tokenizers.extend(tokenizers)
                else:
                    bm25_tokenizers.append(tokenizers)
            elif method == "vectordb":
                methods.append("vectordb")
                vdbs = module.get("vectordb", ["default"])
                if isinstance(vdbs, list):
                    vectordb_names.extend(vdbs)
                else:
                    vectordb_names.append(vdbs)
        
        if methods:
            space['retrieval_method'] = list(set(methods))
        if bm25_tokenizers:
            space['bm25_tokenizer'] = list(set(bm25_tokenizers))
        if vectordb_names:
            space['vectordb_name'] = list(set(vectordb_names))
        
        return space
    
    def _extract_reranker_search_space_from_config(self) -> Dict[str, Any]:
        reranker_config = self.config_generator.extract_node_config("passage_reranker")
        if not reranker_config or not reranker_config.get("modules", []):
            return {}
        
        space = {}
        
        top_k = reranker_config.get('top_k', [5])
        space['reranker_top_k'] = top_k if isinstance(top_k, list) else [top_k]
        
        methods = []
        for module in reranker_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method not in ["pass_reranker", "upr", "colbert_reranker"]:
                    model_field = 'model' if method == 'flashrank_reranker' else 'model_name'
                    if model_field in module:
                        models = module[model_field]
                        model_key = f"{method}_models"
                        space[model_key] = models if isinstance(models, list) else [models]
        
        if methods:
            space['passage_reranker_method'] = methods
        
        return space
    
    def _extract_filter_search_space_from_config(self) -> Dict[str, Any]:
        filter_config = self.config_generator.extract_node_config("passage_filter")
        if not filter_config or not filter_config.get("modules", []):
            return {}
        
        space = {}
        methods = []
        
        for module in filter_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method in ["threshold_cutoff", "similarity_threshold_cutoff"]:
                    param_key = f"{method}_threshold"
                    if "threshold" in module:
                        thresholds = module["threshold"]
                        space[param_key] = thresholds if isinstance(thresholds, list) else [thresholds]
                
                elif method in ["percentile_cutoff", "similarity_percentile_cutoff"]:
                    param_key = f"{method}_percentile"
                    if "percentile" in module:
                        percentiles = module["percentile"]
                        space[param_key] = percentiles if isinstance(percentiles, list) else [percentiles]
        
        if methods:
            space['passage_filter_method'] = methods
        
        return space
    
    def _extract_compressor_search_space_from_config(self) -> Dict[str, Any]:
        compressor_config = self.config_generator.extract_node_config("passage_compressor")
        if not compressor_config or not compressor_config.get("modules", []):
            return {}
        
        space = {}
        methods = []
        
        for module in compressor_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method in ['tree_summarize', 'refine']:
                    if 'model' in module:
                        models = module['model']
                        space['compressor_model'] = models if isinstance(models, list) else [models]
                    if 'llm' in module:
                        llms = module['llm']
                        space['compressor_llm'] = llms if isinstance(llms, list) else [llms]
                
                elif method == 'lexrank':
                    if 'compression_ratio' in module:
                        space['compressor_compression_ratio'] = module['compression_ratio']
                    if 'threshold' in module:
                        space['compressor_threshold'] = module['threshold']
                    if 'damping' in module:
                        space['compressor_damping'] = module['damping']
                    if 'max_iterations' in module:
                        space['compressor_max_iterations'] = module['max_iterations']
                
                elif method == 'spacy':
                    if 'compression_ratio' in module:
                        space['compressor_compression_ratio'] = module['compression_ratio']
                    if 'spacy_model' in module:
                        models = module['spacy_model']
                        space['compressor_spacy_model'] = models if isinstance(models, list) else [models]
        
        if methods:
            space['passage_compressor_method'] = methods
        
        return space
    
    def _extract_prompt_gen_search_space_from_config(self) -> Dict[str, Any]:
        space = {}
        
        prompt_config = self.config_generator.extract_node_config("prompt_maker")
        if prompt_config and prompt_config.get("modules"):
            methods = []
            prompts = []
            
            for module in prompt_config.get("modules", []):
                method = module.get("module_type")
                if method:
                    methods.append(method)
                    if 'prompt' in module:
                        module_prompts = module['prompt']
                        if isinstance(module_prompts, list):
                            prompts.extend(range(len(module_prompts)))
                        else:
                            prompts.append(0)
            
            if methods:
                space['prompt_maker_method'] = methods
            if prompts:
                space['prompt_template_idx'] = list(set(prompts))
        
        gen_config = self.config_generator.extract_node_config("generator")
        if gen_config and gen_config.get("modules"):
            models = []
            temperatures = []
            
            for module in gen_config.get("modules", []):
                if 'model' in module:
                    module_models = module['model']
                    if isinstance(module_models, list):
                        models.extend(module_models)
                    else:
                        models.append(module_models)
                
                if 'temperature' in module:
                    module_temps = module['temperature']
                    if isinstance(module_temps, list):
                        temperatures.extend(module_temps)
                    else:
                        temperatures.append(module_temps)
            
            if models:
                space['generator_model'] = list(set(models))
            if temperatures:
                space['generator_temperature'] = list(set(temperatures))
        
        return space
    
    def calculate_total_combinations(self, max_combinations: int = 10000) -> int:
        total = 1
        components = ['query_expansion', 'retrieval', 'passage_filter', 'passage_reranker', 
                     'passage_compressor', 'prompt_maker_generator']
        
        for component in components:
            combinations, _ = self.calculate_component_combinations(component)
            if component == 'retrieval' and combinations == 0:
                continue
            total *= combinations if combinations > 0 else 1
        
        return min(total, max_combinations)
    
    def suggest_num_samples(self, sample_percentage: float = 0.1, min_samples: int = 10, 
                           max_samples: int = 50, max_combinations: int = 500) -> Dict[str, Any]:
        search_space_size = self.calculate_total_combinations(max_combinations)
        
        if search_space_size > max_combinations:
            log_combinations = math.log10(search_space_size)
            log_max = math.log10(max_combinations)
            
            suggested_samples = int(min_samples + (max_samples - min_samples) * min(log_combinations / log_max, 1.0))
            
            return {
                "num_samples": max_samples,
                "search_space_size": search_space_size,
                "sample_percentage": max_samples / search_space_size if search_space_size > 0 else 0,
                "reasoning": f"Large search space detected ({search_space_size:,} combinations), using max samples ({max_samples}) for better coverage"
            }
        
        suggested_samples = max(min_samples, int(search_space_size * sample_percentage))
        suggested_samples = min(suggested_samples, max_samples)
        
        return {
            "num_samples": suggested_samples,
            "search_space_size": search_space_size,
            "sample_percentage": sample_percentage,
            "reasoning": f"Auto-calculated based on {sample_percentage*100}% of {search_space_size} total combinations"
        }