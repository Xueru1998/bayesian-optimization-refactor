from typing import Dict, Any, List, Tuple, Optional
import itertools


class CombinationCalculator:
    
    def __init__(self, config_generator, search_type: str = 'bo'):
        self.config_generator = config_generator
        self.search_type = search_type
    
    def calculate_component_combinations(
        self, 
        component: str, 
        search_space: Dict[str, Any] = None,
        fixed_config: Dict[str, Any] = None,
        best_configs: Dict[str, Any] = None
    ) -> Tuple[int, str]:
        
        if self.search_type == 'grid':
            combinations = self._calculate_grid_combinations(component, search_space, fixed_config, best_configs)
            note = f"Exact number of combinations for grid search"
        else:
            combinations = self._calculate_bo_combinations(component, search_space, fixed_config, best_configs)
            note = (
                f"Estimated combinations based on discrete values and range endpoints. "
                f"Actual BO exploration may sample more points within continuous ranges."
            )
        
        return combinations, note
    
    def _calculate_grid_combinations(
        self,
        component: str,
        search_space: Dict[str, Any] = None,
        fixed_config: Dict[str, Any] = None,
        best_configs: Dict[str, Any] = None
    ) -> int:
        
        if component == 'query_expansion':
            return self._calculate_qe_grid_combinations(search_space, best_configs)
        elif component == 'retrieval':
            return self._calculate_retrieval_grid_combinations(search_space, best_configs)
        elif component == 'passage_reranker':
            return self._calculate_reranker_grid_combinations(search_space, fixed_config, best_configs)
        elif component == 'passage_filter':
            return self._calculate_filter_grid_combinations(search_space)
        elif component == 'passage_compressor':
            return self._calculate_compressor_grid_combinations(search_space)
        elif component == 'prompt_maker_generator':
            return self._calculate_prompt_gen_grid_combinations(search_space)
        
        return 1
    
    def _calculate_bo_combinations(
        self,
        component: str,
        search_space: Dict[str, Any] = None,
        fixed_config: Dict[str, Any] = None,
        best_configs: Dict[str, Any] = None
    ) -> int:
        
        return self._calculate_grid_combinations(component, search_space, fixed_config, best_configs)
    
    def _calculate_qe_grid_combinations(
        self, 
        search_space: Dict[str, Any] = None,
        best_configs: Dict[str, Any] = None
    ) -> int:
        
        if not search_space:
            qe_config = self.config_generator.extract_node_config("query_expansion")
            if not qe_config or not qe_config.get("modules", []):
                return 0
            
            return self._calculate_qe_from_config(qe_config)
        
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
            
            return total
        
        return self._calculate_qe_standard_combinations(search_space)
    
    def _calculate_qe_from_config(self, qe_config: Dict[str, Any]) -> int:
        
        qe_strategy = qe_config.get('strategy', {})
        qe_retrieval_modules = qe_strategy.get('retrieval_modules', [])
        
        retrieval_combinations = 0
        for module in qe_retrieval_modules:
            method = module.get("module_type")
            if method == "bm25":
                tokenizers = module.get("bm25_tokenizer", ["porter_stemmer"])
                retrieval_combinations += len(tokenizers) if isinstance(tokenizers, list) else 1
            elif method == "vectordb":
                vdb_names = module.get("vectordb", ["default"])
                retrieval_combinations += len(vdb_names) if isinstance(vdb_names, list) else 1
        
        if retrieval_combinations == 0:
            retrieval_combinations = 1
        
        top_k_values = qe_strategy.get('top_k', [10])
        top_k_count = self._get_list_count(top_k_values)
        
        base_combinations = top_k_count * retrieval_combinations
        total_combinations = 0
        
        for module in qe_config.get("modules", []):
            method = module.get("module_type")
            
            if method == "pass_query_expansion":
                total_combinations += base_combinations
            elif method == "query_decompose":
                models = module.get("model", [])
                model_count = self._get_list_count(models)
                total_combinations += model_count * base_combinations
            elif method == "hyde":
                models = module.get("model", [])
                max_tokens = module.get("max_token", [64])
                model_count = self._get_list_count(models)
                token_count = self._get_list_count(max_tokens)
                total_combinations += model_count * token_count * base_combinations
            elif method == "multi_query_expansion":
                models = module.get("model", ["gpt-3.5-turbo"])
                temperatures = module.get("temperature", [0.7])
                model_count = self._get_list_count(models)
                temp_count = self._get_list_count(temperatures)
                total_combinations += model_count * temp_count * base_combinations
        
        return total_combinations
    
    def _calculate_qe_standard_combinations(self, search_space: Dict[str, Any]) -> int:
        
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
        
        return total
    
    def _calculate_retrieval_grid_combinations(
        self,
        search_space: Dict[str, Any] = None,
        best_configs: Dict[str, Any] = None
    ) -> int:
        
        if best_configs and 'query_expansion' in best_configs:
            qe_method = best_configs['query_expansion'].get('query_expansion_method')
            if qe_method and qe_method != 'pass_query_expansion':
                return 0
        
        if not search_space:
            return self._calculate_retrieval_from_config()
        
        return self._calculate_retrieval_from_search_space(search_space)
    
    def _calculate_retrieval_from_config(self) -> int:
        retrieval_config = self.config_generator.extract_node_config("retrieval")
        if not retrieval_config:
            return 1
        
        retrieval_options = self.config_generator.extract_retrieval_options()
        methods = retrieval_options.get('methods', [])
        top_k_values = retrieval_options.get('retriever_top_k_values', [10])
        
        if isinstance(top_k_values, list) and len(top_k_values) == 2:
            if all(isinstance(x, int) for x in top_k_values):
                top_k_count = top_k_values[1] - top_k_values[0] + 1
            else:
                top_k_count = len(top_k_values)
        else:
            top_k_count = len(top_k_values) if isinstance(top_k_values, list) else 1
        
        total = 0
        for method in methods:
            if method == 'bm25':
                tokenizers = retrieval_options.get('bm25_tokenizers', [])
                total += top_k_count * (len(tokenizers) if tokenizers else 1)
            elif method == 'vectordb':
                vectordb_names = retrieval_options.get('vectordb_names', ['default'])
                total += top_k_count * len(vectordb_names)
            else:
                total += top_k_count
        
        return total if total > 0 else 1
    
    def _calculate_retrieval_from_search_space(self, search_space: Dict[str, Any]) -> int:
        methods = search_space.get('retrieval_method', [])
        top_k_values = self._get_top_k_values(search_space.get('retriever_top_k', [10]))
        
        total = 0
        for method in methods:
            if method == 'bm25':
                tokenizers = search_space.get('bm25_tokenizer', [])
                total += len(top_k_values) * len(tokenizers) if tokenizers else len(top_k_values)
            elif method == 'vectordb':
                vdb_names = search_space.get('vectordb_name', ['default'])
                total += len(top_k_values) * len(vdb_names)
            else:
                total += len(top_k_values)
        
        return total
    
    def _calculate_reranker_grid_combinations(
        self,
        search_space: Dict[str, Any] = None,
        fixed_config: Dict[str, Any] = None,
        best_configs: Dict[str, Any] = None
    ) -> int:
        
        if not search_space:
            reranker_config = self.config_generator.extract_node_config("passage_reranker")
            if not reranker_config or not reranker_config.get("modules", []):
                return 0
            
            return self._calculate_reranker_from_config(reranker_config, fixed_config, best_configs)
        
        return self._calculate_reranker_from_search_space(search_space, fixed_config)
    
    def _calculate_reranker_from_config(
        self,
        reranker_config: Dict[str, Any],
        fixed_config: Dict[str, Any] = None,
        best_configs: Dict[str, Any] = None
    ) -> int:
        
        prev_top_k = 10
        if best_configs:
            if 'query_expansion' in best_configs:
                prev_top_k = best_configs['query_expansion'].get('retriever_top_k', 10)
            elif 'retrieval' in best_configs:
                prev_top_k = best_configs['retrieval'].get('retriever_top_k', 10)
        elif fixed_config:
            prev_top_k = fixed_config.get('retriever_top_k', 10)
        
        reranker_top_k_config = reranker_config.get('top_k', [5])
        valid_top_k_values = self._get_valid_reranker_top_k(reranker_top_k_config, prev_top_k)
        
        if len(valid_top_k_values) == 0:
            return 0
        
        total_combinations = 0
        
        for module in reranker_config.get("modules", []):
            method = module.get("module_type")
            
            if method == "pass_reranker":
                total_combinations += 1
            elif method in ["upr", "colbert_reranker"]:
                total_combinations += len(valid_top_k_values)
            else:
                model_field = 'model' if method == 'flashrank_reranker' else 'model_name'
                if model_field in module:
                    models = module[model_field]
                    model_count = self._get_list_count(models)
                    total_combinations += model_count * len(valid_top_k_values)
                else:
                    total_combinations += len(valid_top_k_values)
        
        return total_combinations
    
    def _calculate_reranker_from_search_space(
        self,
        search_space: Dict[str, Any],
        fixed_config: Dict[str, Any] = None
    ) -> int:
        
        methods = search_space.get('passage_reranker_method', [])
        top_k_values = search_space.get('reranker_top_k', [])
        
        if fixed_config and 'retriever_top_k' in fixed_config:
            retriever_top_k = fixed_config['retriever_top_k']
            top_k_values = [k for k in top_k_values if k <= retriever_top_k]
        
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
        
        return total
    
    def _calculate_filter_grid_combinations(self, search_space: Dict[str, Any] = None) -> int:
        
        if not search_space:
            filter_config = self.config_generator.extract_node_config("passage_filter")
            if not filter_config or not filter_config.get("modules", []):
                return 0
            
            return self._calculate_filter_from_config(filter_config)
        
        return self._calculate_filter_from_search_space(search_space)
    
    def _calculate_filter_from_config(self, filter_config: Dict[str, Any]) -> int:
        
        total = 0
        
        for module in filter_config.get("modules", []):
            method = module.get("module_type")
            
            if method == "pass_passage_filter":
                total += 1
            elif method in ["threshold_cutoff", "similarity_threshold_cutoff"]:
                thresholds = module.get("threshold", [0.75])
                total += self._get_list_count(thresholds)
            elif method in ["percentile_cutoff", "similarity_percentile_cutoff"]:
                percentiles = module.get("percentile", [0.6])
                total += self._get_list_count(percentiles)
        
        return total
    
    def _calculate_filter_from_search_space(self, search_space: Dict[str, Any]) -> int:
        
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
        
        return total
    
    def _calculate_compressor_grid_combinations(self, search_space: Dict[str, Any] = None) -> int:
        
        if not search_space:
            compressor_config = self.config_generator.extract_node_config("passage_compressor")
            if not compressor_config or not compressor_config.get("modules", []):
                return 0
            
            return self._calculate_compressor_from_config(compressor_config)
        
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
                    
                    total += len(thresholds) * len(dampings) * len(max_iters) * len(comp_ratios)
                elif config.startswith('spacy::'):
                    comp_ratios = search_space.get('spacy_compression_ratio', [0.3, 0.5])
                    total += len(comp_ratios)
                else:
                    total += 1
            
            return total
        
        return self._calculate_compressor_standard(search_space)
    
    def _calculate_compressor_from_config(self, compressor_config: Dict[str, Any]) -> int:
        
        total = 0
        
        for module in compressor_config.get("modules", []):
            method = module.get("module_type")
            
            if method == 'pass_compressor':
                total += 1
            elif method == 'lexrank':
                comp_ratios = self._get_list_count(module.get('compression_ratio', [0.5]))
                thresholds = self._get_list_count(module.get('threshold', [0.1]))
                dampings = self._get_list_count(module.get('damping', [0.85]))
                max_iters = self._get_list_count(module.get('max_iterations', [30]))
                
                total += comp_ratios * thresholds * dampings * max_iters
            elif method == 'spacy':
                comp_ratios = self._get_list_count(module.get('compression_ratio', [0.5]))
                models = self._get_list_count(module.get('spacy_model', ['en_core_web_sm']))
                
                total += comp_ratios * models
            elif method in ['tree_summarize', 'refine']:
                models = self._get_list_count(module.get('model', ['gpt-3.5-turbo']))
                total += models
        
        return total
    
    def _calculate_compressor_standard(self, search_space: Dict[str, Any]) -> int:
        
        methods = search_space.get('passage_compressor_method', [])
        
        total = 0
        for method in methods:
            if method == 'pass_compressor':
                total += 1
            else:
                total += 1
        
        return total
    
    def _calculate_prompt_gen_grid_combinations(self, search_space: Dict[str, Any] = None) -> int:
        
        if not search_space:
            prompt_config = self.config_generator.extract_node_config("prompt_maker")
            gen_config = self.config_generator.extract_node_config("generator")
            
            prompt_combinations = self._calculate_prompt_from_config(prompt_config) if prompt_config else 1
            gen_combinations = self._calculate_generator_from_config(gen_config) if gen_config else 1
            
            return prompt_combinations * gen_combinations
        
        return self._calculate_prompt_gen_from_search_space(search_space)
    
    def _calculate_prompt_from_config(self, prompt_config: Dict[str, Any]) -> int:
        
        prompt_combinations = 0
        
        for module in prompt_config.get("modules", []):
            method = module.get("module_type")
            if method in ['fstring', 'long_context_reorder', 'window_replacement']:
                prompts = module.get('prompt', [])
                prompt_combinations += self._get_list_count(prompts)
        
        return prompt_combinations if prompt_combinations > 0 else 1
    
    def _calculate_generator_from_config(self, gen_config: Dict[str, Any]) -> int:
        
        generator_combinations = 0
        
        for module in gen_config.get("modules", []):
            models = module.get('model', module.get('llm', []))
            model_count = self._get_list_count(models)
            
            temps = module.get('temperature', [0.7])
            temp_count = self._get_list_count(temps)
            
            generator_combinations += model_count * temp_count
        
        return generator_combinations if generator_combinations > 0 else 1
    
    def _calculate_prompt_gen_from_search_space(self, search_space: Dict[str, Any]) -> int:
        
        prompt_methods = search_space.get('prompt_maker_method', ['fstring'])
        prompt_indices = search_space.get('prompt_template_idx', [0])
        generator_configs = search_space.get('generator_config', [])
        temperatures = search_space.get('generator_temperature', [])
        
        prompt_combinations = len(prompt_methods) * len(prompt_indices)
        gen_combinations = len(generator_configs) * len(temperatures) if generator_configs and temperatures else 1
        
        return prompt_combinations * gen_combinations
    
    def _get_top_k_values(self, top_k_config: Any) -> List[int]:
        
        if not isinstance(top_k_config, list):
            return [top_k_config]
        
        if len(top_k_config) == 2 and all(isinstance(x, int) for x in top_k_config):
            if top_k_config[1] > top_k_config[0]:
                return list(range(top_k_config[0], top_k_config[1] + 1))
        
        return top_k_config
    
    def _get_valid_reranker_top_k(self, reranker_top_k_config: Any, prev_top_k: int) -> List[int]:
        
        if not isinstance(reranker_top_k_config, list):
            reranker_top_k_config = [reranker_top_k_config]
        
        return [k for k in reranker_top_k_config if k <= prev_top_k]
    
    def _get_list_count(self, value: Any) -> int:
        
        if not isinstance(value, list):
            return 1
        
        if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            if value[1] > value[0]:
                if all(isinstance(x, int) for x in value):
                    return value[1] - value[0] + 1
                else:
                    return 2
        
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
    
    def get_detailed_breakdown(
        self,
        component: str,
        search_space: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not search_space:
            return {"error": "No search space provided"}
        
        breakdown = {
            "component": component,
            "search_type": self.search_type,
            "parameters": {}
        }
        
        total = 1
        
        for param_name, param_values in search_space.items():
            param_info = {}
            
            if isinstance(param_values, list):
                param_info["type"] = "categorical"
                param_info["values"] = param_values
                param_info["count"] = len(param_values)
                total *= len(param_values)
                
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                min_val, max_val = param_values
                param_info["type"] = "continuous"
                param_info["range"] = [min_val, max_val]
                
                if self.search_type == 'grid':
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        count = max_val - min_val + 1
                        param_info["count"] = count
                        param_info["note"] = "All integer values in range"
                    else:
                        count = 2  
                        param_info["count"] = count
                        param_info["note"] = "Endpoints only for grid"
                else:
                    if isinstance(min_val, (int, float)):
                        if any(keyword in param_name.lower() for keyword in ['top_k', 'iterations', 'max_token']):
                            if isinstance(min_val, int):
                                count = max_val - min_val + 1
                                param_info["count"] = count
                                param_info["note"] = "Integer range (all values)"
                            else:
                                count = min(10, int(max_val - min_val + 1))
                                param_info["count"] = count
                                param_info["note"] = "Estimated sampling points"
                        elif any(keyword in param_name.lower() for keyword in ['threshold', 'percentile', 'ratio']):
                            count = 10
                            param_info["count"] = count
                            param_info["note"] = "Estimated 10 points for probability/threshold"
                        elif 'temperature' in param_name.lower():
                            count = min(20, int((max_val - min_val) * 10 + 1))
                            param_info["count"] = count
                            param_info["note"] = "Temperature parameter (0.1 increments)"
                        else:
                            count = 10
                            param_info["count"] = count
                            param_info["note"] = "Default 10 points for continuous"
                    else:
                        count = 2
                        param_info["count"] = count
                        param_info["note"] = "Non-numeric range"
                
                total *= count
            
            breakdown["parameters"][param_name] = param_info
        
        breakdown["total_combinations"] = total
        
        if self.search_type == 'bo':
            breakdown["important_note"] = (
                "These are ESTIMATES for BO. Actual Bayesian Optimization will:\n"
                "1. Start with random sampling (n_startup_trials)\n"
                "2. Use acquisition function to explore promising regions\n" 
                "3. May sample any value within continuous ranges\n"
                "4. Total samples determined by n_trials, not combinations"
            )
        
        return breakdown
    
    def print_combination_info(
        self,
        component: str,
        combinations: int,
        note: str,
        search_space: Dict[str, Any] = None
    ) -> None:
        """
        Print detailed information about combination calculation.
        """
        print(f"\n[{component}] Combination Calculation:")
        print(f"  Search Type: {self.search_type.upper()}")
        print(f"  Total Combinations: {combinations:,}")
        
        if self.search_type == 'bo':
            print(f"  Note: {note}")
            
            if search_space:
                breakdown = self.get_detailed_breakdown(component, search_space)

                continuous_params = []
                categorical_params = []
                
                for param_name, param_info in breakdown.get("parameters", {}).items():
                    if param_info["type"] == "continuous":
                        continuous_params.append((param_name, param_info))
                    else:
                        categorical_params.append((param_name, param_info))
                
                if continuous_params:
                    print(f"\n  Continuous parameters (BO will explore within ranges):")
                    for param_name, param_info in continuous_params:
                        range_str = f"[{param_info['range'][0]}, {param_info['range'][1]}]"
                        print(f"    - {param_name}: {range_str}")
                        print(f"      Estimated points: {param_info['count']} ({param_info.get('note', '')})")
                
                if categorical_params:
                    print(f"\n  Categorical parameters (discrete choices):")
                    for param_name, param_info in categorical_params:
                        print(f"    - {param_name}: {param_info['count']} options")
                        if param_info['count'] <= 5:
                            print(f"      Values: {param_info['values']}")
        else:
            print(f"  Note: {note}")
            
            if search_space:
                print(f"\n  Parameters to explore:")
                for param, values in search_space.items():
                    if isinstance(values, list) and len(values) <= 5:
                        print(f"    - {param}: {values}")
                    elif isinstance(values, list):
                        print(f"    - {param}: {len(values)} values")
                    elif isinstance(values, tuple):
                        print(f"    - {param}: range [{values[0]}, {values[1]}]")