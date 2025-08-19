import itertools
import os
import time
from typing import Dict, Any, List, Union, Tuple, Set
import hashlib
import json


class ComponentGridSearchHelper:
    
    def __init__(self):
        self.current_combinations = []
        self.current_index = 0
        self.explored_configs = set()
    
    @staticmethod
    def config_to_hash(config: Dict[str, Any]) -> str:
        sorted_config = dict(sorted(config.items()))
        config_str = json.dumps(sorted_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    
    def get_next_config(self) -> Dict[str, Any]:
        if self.current_index < len(self.current_combinations):
            config = self.current_combinations[self.current_index]
            config_hash = self.config_to_hash(config)
            
            while config_hash in self.explored_configs and self.current_index < len(self.current_combinations) - 1:
                self.current_index += 1
                config = self.current_combinations[self.current_index]
                config_hash = self.config_to_hash(config)
            
            if config_hash not in self.explored_configs:
                self.explored_configs.add(config_hash)
                self.current_index += 1
                return config
        
        return None
    
    @staticmethod
    def convert_to_grid_search_space(search_space: Dict[str, Any], range_params: List[str] = None) -> Dict[str, List[Any]]:
        # Only expand top_k parameters, keep everything else as-is for grid search
        if range_params is None:
            range_params = ['retriever_top_k', 'reranker_top_k']
        
        grid_space = {}
        
        for param_name, param_spec in search_space.items():
            if isinstance(param_spec, list):
                # Only expand ranges for top_k parameters
                should_expand = any(rp in param_name for rp in range_params)
                
                if (should_expand and 
                    len(param_spec) == 2 and 
                    all(isinstance(x, int) for x in param_spec) and
                    param_spec[1] > param_spec[0]):
                    min_val, max_val = min(param_spec), max(param_spec)
                    grid_space[param_name] = list(range(min_val, max_val + 1))
                else:
                    grid_space[param_name] = param_spec
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                min_val, max_val = param_spec
                if isinstance(min_val, int) and any(rp in param_name for rp in range_params):
                    grid_space[param_name] = list(range(min_val, max_val + 1))
                else:
                    # For grid search, just use the two boundary values
                    grid_space[param_name] = [min_val, max_val]
            else:
                grid_space[param_name] = [param_spec]
        
        return grid_space
    
    def save_grid_search_sequence(self, component: str, valid_combinations: List[Dict[str, Any]], 
                              output_dir: str) -> None:
        
        sequence_file = os.path.join(output_dir, f"{component}_grid_search_sequence.txt")
        
        with open(sequence_file, 'w') as f:
            f.write(f"="*80 + "\n")
            f.write(f"GRID SEARCH SEQUENCE FOR: {component.upper()}\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Total configurations to test: {len(valid_combinations)}\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("-"*80 + "\n\n")
            
            for idx, config in enumerate(valid_combinations, 1):
                f.write(f"Configuration #{idx:04d}:\n")

                sorted_params = sorted(config.items())

                for param, value in sorted_params:
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    elif isinstance(value, (list, dict)):
                        value_str = str(value)
                    else:
                        value_str = str(value)
                    
                    f.write(f"  {param:40s} = {value_str}\n")
                
                f.write("\n")

                if idx % 10 == 0 and idx < len(valid_combinations):
                    f.write("-"*40 + f" [{idx}/{len(valid_combinations)}] " + "-"*40 + "\n\n")
            
            f.write("-"*80 + "\n")
            f.write(f"END OF SEQUENCE - {len(valid_combinations)} configurations\n")
            f.write("="*80 + "\n")
        
        print(f"[Grid Search] Saved sequence to: {sequence_file}")
        
    @staticmethod
    def calculate_total_combinations(grid_space: Dict[str, List[Any]]) -> int:
        if not grid_space:
            return 0
        
        total = 1
        for param_values in grid_space.values():
            total *= len(param_values)
        
        return total
    
    @staticmethod
    def _detect_component(search_space: Dict[str, Any]) -> str:
        if 'query_expansion_method' in search_space:
            return 'query_expansion'
        elif 'passage_filter_method' in search_space:
            return 'passage_filter'
        elif 'passage_compressor_method' in search_space:
            return 'passage_compressor'
        elif 'passage_reranker_method' in search_space:
            return 'passage_reranker'
        elif 'retrieval_method' in search_space:
            return 'retrieval'
        elif 'prompt_maker_method' in search_space or 'generator_model' in search_space:
            return 'prompt_maker_generator'
        return None
    
    def get_valid_combinations(self, component: str, search_space: Dict[str, Any], fixed_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if component == 'passage_compressor' and 'passage_compressor_config' in search_space:
            configs = search_space['passage_compressor_config']
            return [{'passage_compressor_config': config} for config in configs]
        
        grid_space = ComponentGridSearchHelper.convert_to_grid_search_space(search_space)
        
        all_combinations = []
        
        if component == 'query_expansion':
            methods = grid_space.get('query_expansion_method', ['pass_query_expansion'])
            top_k_values = grid_space.get('retriever_top_k', [10])
            retrieval_methods = grid_space.get('retrieval_method', ['bm25'])
            bm25_tokenizers = grid_space.get('bm25_tokenizer', [])
            vectordb_names = grid_space.get('vectordb_name', [])
            models = grid_space.get('query_expansion_model', [])
            temperatures = grid_space.get('query_expansion_temperature', [])
            max_tokens = grid_space.get('query_expansion_max_token', [])

            has_models = 'query_expansion_model' in search_space
            has_temperature = 'query_expansion_temperature' in search_space
            has_max_token = 'query_expansion_max_token' in search_space
            
            for method in methods:
                if method == 'pass_query_expansion':
                    for top_k in top_k_values:
                        for retrieval_method in retrieval_methods:
                            if retrieval_method == 'bm25':
                                for tokenizer in bm25_tokenizers:
                                    all_combinations.append({
                                        'query_expansion_method': method,
                                        'retriever_top_k': top_k,
                                        'retrieval_method': retrieval_method,
                                        'bm25_tokenizer': tokenizer
                                    })
                            elif retrieval_method == 'vectordb':
                                for vdb_name in vectordb_names:
                                    all_combinations.append({
                                        'query_expansion_method': method,
                                        'retriever_top_k': top_k,
                                        'retrieval_method': retrieval_method,
                                        'vectordb_name': vdb_name
                                    })
                
                elif method == 'query_decompose':
                    base_combinations = []
                    if has_models and models:
                        for model in models:
                            base_combinations.append({'query_expansion_model': model})
                    else:
                        base_combinations.append({})

                    for base_combo in base_combinations:
                        for top_k in top_k_values:
                            for retrieval_method in retrieval_methods:
                                if retrieval_method == 'bm25':
                                    for tokenizer in bm25_tokenizers:
                                        combo = base_combo.copy()
                                        combo.update({
                                            'query_expansion_method': method,
                                            'retriever_top_k': top_k,
                                            'retrieval_method': retrieval_method,
                                            'bm25_tokenizer': tokenizer
                                        })
                                        all_combinations.append(combo)
                                elif retrieval_method == 'vectordb':
                                    for vdb_name in vectordb_names:
                                        combo = base_combo.copy()
                                        combo.update({
                                            'query_expansion_method': method,
                                            'retriever_top_k': top_k,
                                            'retrieval_method': retrieval_method,
                                            'vectordb_name': vdb_name
                                        })
                                        all_combinations.append(combo)
                
                elif method == 'hyde':
                    base_combinations = []
                    if has_models and models:
                        for model in models:
                            if has_max_token and max_tokens:
                                for token in max_tokens:
                                    base_combinations.append({
                                        'query_expansion_model': model,
                                        'query_expansion_max_token': token
                                    })
                            else:
                                base_combinations.append({'query_expansion_model': model})
                    else:
                        if has_max_token and max_tokens:
                            for token in max_tokens:
                                base_combinations.append({'query_expansion_max_token': token})
                        else:
                            base_combinations.append({})

                    for base_combo in base_combinations:
                        for top_k in top_k_values:
                            for retrieval_method in retrieval_methods:
                                if retrieval_method == 'bm25':
                                    for tokenizer in bm25_tokenizers:
                                        combo = base_combo.copy()
                                        combo.update({
                                            'query_expansion_method': method,
                                            'retriever_top_k': top_k,
                                            'retrieval_method': retrieval_method,
                                            'bm25_tokenizer': tokenizer
                                        })
                                        all_combinations.append(combo)
                                elif retrieval_method == 'vectordb':
                                    for vdb_name in vectordb_names:
                                        combo = base_combo.copy()
                                        combo.update({
                                            'query_expansion_method': method,
                                            'retriever_top_k': top_k,
                                            'retrieval_method': retrieval_method,
                                            'vectordb_name': vdb_name
                                        })
                                        all_combinations.append(combo)
                
                elif method == 'multi_query_expansion':
                    base_combinations = []
                    if has_models and models:
                        for model in models:
                            if has_temperature and temperatures:
                                for temp in temperatures:
                                    base_combinations.append({
                                        'query_expansion_model': model,
                                        'query_expansion_temperature': temp
                                    })
                            else:
                                base_combinations.append({'query_expansion_model': model})
                    else:
                        if has_temperature and temperatures:
                            for temp in temperatures:
                                base_combinations.append({'query_expansion_temperature': temp})
                        else:
                            base_combinations.append({})

                    for base_combo in base_combinations:
                        for top_k in top_k_values:
                            for retrieval_method in retrieval_methods:
                                if retrieval_method == 'bm25':
                                    for tokenizer in bm25_tokenizers:
                                        combo = base_combo.copy()
                                        combo.update({
                                            'query_expansion_method': method,
                                            'retriever_top_k': top_k,
                                            'retrieval_method': retrieval_method,
                                            'bm25_tokenizer': tokenizer
                                        })
                                        all_combinations.append(combo)
                                elif retrieval_method == 'vectordb':
                                    for vdb_name in vectordb_names:
                                        combo = base_combo.copy()
                                        combo.update({
                                            'query_expansion_method': method,
                                            'retriever_top_k': top_k,
                                            'retrieval_method': retrieval_method,
                                            'vectordb_name': vdb_name
                                        })
                                        all_combinations.append(combo)
            
            return all_combinations
        
        elif component == 'passage_filter':
            methods = grid_space.get('passage_filter_method', [])
            
            for method in methods:
                if method == 'pass_passage_filter':
                    all_combinations.append({'passage_filter_method': method})
                elif method == 'threshold_cutoff':
                    param_key = 'threshold_cutoff_threshold'
                    thresholds = search_space.get(param_key, [])
                    if isinstance(thresholds, tuple):
                        thresholds = [thresholds[0], thresholds[1]]
                    elif not isinstance(thresholds, list):
                        thresholds = [thresholds]
                    
                    for threshold in thresholds:
                        all_combinations.append({
                            'passage_filter_method': method,
                            param_key: threshold
                        })
                elif method == 'percentile_cutoff':
                    param_key = 'percentile_cutoff_percentile'
                    percentiles = search_space.get(param_key, [])
                    if isinstance(percentiles, tuple):
                        percentiles = [percentiles[0], percentiles[1]]
                    elif not isinstance(percentiles, list):
                        percentiles = [percentiles]
                    
                    for percentile in percentiles:
                        all_combinations.append({
                            'passage_filter_method': method,
                            param_key: percentile
                        })
                elif method == 'similarity_threshold_cutoff':
                    param_key = 'similarity_threshold_cutoff_threshold'
                    thresholds = search_space.get(param_key, [])
                    if isinstance(thresholds, tuple):
                        thresholds = [thresholds[0], thresholds[1]]
                    elif not isinstance(thresholds, list):
                        thresholds = [thresholds]
                    
                    for threshold in thresholds:
                        all_combinations.append({
                            'passage_filter_method': method,
                            param_key: threshold
                        })
                elif method == 'similarity_percentile_cutoff':
                    param_key = 'similarity_percentile_cutoff_percentile'
                    percentiles = search_space.get(param_key, [])
                    if isinstance(percentiles, tuple):
                        percentiles = [percentiles[0], percentiles[1]]
                    elif not isinstance(percentiles, list):
                        percentiles = [percentiles]
                    
                    for percentile in percentiles:
                        all_combinations.append({
                            'passage_filter_method': method,
                            param_key: percentile
                        })
            
            return all_combinations
        
        elif component == 'passage_reranker':
            methods = grid_space.get('passage_reranker_method', [])
            top_k_values = grid_space.get('reranker_top_k', [])
            
            if fixed_config and 'retriever_top_k' in fixed_config:
                retriever_top_k = fixed_config['retriever_top_k']
                top_k_values = [k for k in top_k_values if k <= retriever_top_k]
            
            for method in methods:
                if method == 'pass_reranker':
                    all_combinations.append({'passage_reranker_method': method})
                elif method in ['upr', 'colbert_reranker']:
                    for top_k in top_k_values:
                        all_combinations.append({
                            'passage_reranker_method': method,
                            'reranker_top_k': top_k
                        })
                else:
                    model_key = f"{method}_models"
                    models = search_space.get(model_key, [])
                    
                    if not models:
                        for top_k in top_k_values:
                            all_combinations.append({
                                'passage_reranker_method': method,
                                'reranker_top_k': top_k
                            })
                    else:
                        for model in models:
                            for top_k in top_k_values:
                                combo = {
                                    'passage_reranker_method': method,
                                    'reranker_top_k': top_k,
                                    model_key: model
                                }
                                all_combinations.append(combo)
            
            return all_combinations
        
        else:
            if not grid_space:
                return [{}]
            
            if component == 'prompt_maker_generator':
                prompt_methods = grid_space.get('prompt_maker_method', [])
                prompt_templates = grid_space.get('prompt_template_idx', [])
                
                if isinstance(search_space.get('prompt_template_idx'), tuple):
                    min_idx, max_idx = search_space['prompt_template_idx']
                    prompt_templates = list(range(min_idx, max_idx + 1))
                
                generator_models = grid_space.get('generator_model', [])
                generator_temps = grid_space.get('generator_temperature', [])
                
                if isinstance(search_space.get('generator_temperature'), tuple):
                    generator_temps = []

                for method in prompt_methods:
                    for template_idx in prompt_templates:
                        for model in generator_models:
                            if generator_temps:
                                for temp in generator_temps:
                                    all_combinations.append({
                                        'prompt_maker_method': method,
                                        'prompt_template_idx': template_idx,
                                        'generator_model': model,
                                        'generator_temperature': temp
                                    })
                            else:
                                all_combinations.append({
                                    'prompt_maker_method': method,
                                    'prompt_template_idx': template_idx,
                                    'generator_model': model
                                })
                
                return all_combinations
            
            elif component == 'passage_compressor':
                all_combinations = []
                
                methods = search_space.get('passage_compressor_method', [])
                
                for method in methods:
                    if method == 'pass_compressor':
                        all_combinations.append({'passage_compressor_method': method})
                    
                    elif method in ['tree_summarize', 'refine']:
                        models = search_space.get('compressor_model', ['gpt-3.5-turbo-16k'])
                        llms = search_space.get('compressor_llm', ['openai'])
                        
                        for model in models if isinstance(models, list) else [models]:
                            for llm in llms if isinstance(llms, list) else [llms]:
                                all_combinations.append({
                                    'passage_compressor_method': method,
                                    'compressor_model': model,
                                    'compressor_llm': llm,
                                    'compressor_generator_module_type': 'llama_index_llm'
                                })
                    
                    elif method == 'lexrank':
                        comp_ratios = search_space.get('compressor_compression_ratio', [0.3, 0.7])
                        thresholds = search_space.get('compressor_threshold', [0.05, 0.3])
                        dampings = search_space.get('compressor_damping', [0.75, 0.9])
                        max_iters = search_space.get('compressor_max_iterations', [15, 40])
                        
                        if isinstance(comp_ratios, tuple):
                            comp_ratios = [comp_ratios[0], comp_ratios[1]]
                        if isinstance(thresholds, tuple):
                            thresholds = [thresholds[0], thresholds[1]]
                        if isinstance(dampings, tuple):
                            dampings = [dampings[0], dampings[1]]
                        if isinstance(max_iters, tuple):
                            max_iters = [max_iters[0], max_iters[1]]
                        
                        for cr in comp_ratios:
                            for th in thresholds:
                                for dp in dampings:
                                    for mi in max_iters:
                                        all_combinations.append({
                                            'passage_compressor_method': method,
                                            'compressor_compression_ratio': cr,
                                            'compressor_threshold': th,
                                            'compressor_damping': dp,
                                            'compressor_max_iterations': mi
                                        })
                    
                    elif method == 'spacy':
                        comp_ratios = search_space.get('compressor_compression_ratio', [0.3, 0.5])
                        spacy_models = search_space.get('compressor_spacy_model', ["en_core_web_sm", "en_core_web_md"])
                        
                        if isinstance(comp_ratios, tuple):
                            comp_ratios = [comp_ratios[0], comp_ratios[1]]
                        
                        for cr in comp_ratios:
                            for sm in spacy_models:
                                all_combinations.append({
                                    'passage_compressor_method': method,
                                    'compressor_compression_ratio': cr,
                                    'compressor_spacy_model': sm
                                })
                
                return all_combinations
            
            else:
                keys = list(grid_space.keys())
                values = [grid_space[key] for key in keys]
                
                all_combinations = []
                for combo in itertools.product(*values):
                    combination = dict(zip(keys, combo))
                    
                    if ComponentGridSearchHelper._is_valid_combination(component, combination, fixed_config):
                        all_combinations.append(combination)
                
                return all_combinations
    
    @staticmethod
    def _is_valid_combination(component: str, combination: Dict[str, Any], fixed_config: Dict[str, Any] = None) -> bool:
        
        if component == 'retrieval':
            method = combination.get('retrieval_method')
            if method == 'bm25':
                if 'vectordb_name' in combination:
                    return False
                if 'bm25_tokenizer' not in combination:
                    return False
            elif method == 'vectordb':
                if 'bm25_tokenizer' in combination:
                    return False
                if 'vectordb_name' not in combination:
                    return False
        
        return True
    
    def calculate_grid_search_combinations(self, component: str, search_space: Dict[str, Any], fixed_config: Dict[str, Any] = None) -> int:
        valid_combinations = self.get_valid_combinations(component, search_space, fixed_config)
        self.current_combinations = valid_combinations
        self.current_index = 0
        self.explored_configs.clear()
        return len(valid_combinations)
        
    def print_grid_search_info(self, component: str, search_space: Dict[str, Any], 
                            total_combinations: int, fixed_config: Dict[str, Any] = None) -> None:
        print(f"\n[{component}] Grid Search Configuration:")
        print(f"  Total valid combinations: {total_combinations}")
        print(f"  Each configuration will be tested exactly once")
        
        if component == 'query_expansion':
            grid_space = ComponentGridSearchHelper.convert_to_grid_search_space(search_space)
            
            methods = grid_space.get('query_expansion_method', [])
            top_k_values = grid_space.get('retriever_top_k', [])
            retrieval_methods = grid_space.get('retrieval_method', [])
            bm25_tokenizers = grid_space.get('bm25_tokenizer', [])
            models = grid_space.get('query_expansion_model', [])
            temperatures = grid_space.get('query_expansion_temperature', [])
            max_tokens = grid_space.get('query_expansion_max_token', [])
            
            top_k_count = len(top_k_values)
            retrieval_count = len(bm25_tokenizers) if 'bm25' in retrieval_methods else 0

            has_models = 'query_expansion_model' in search_space
            has_temperature = 'query_expansion_temperature' in search_space
            has_max_token = 'query_expansion_max_token' in search_space
            
            print(f"\n  Breakdown by method:")
            for method in methods:
                if method == 'pass_query_expansion':
                    count = retrieval_count * top_k_count
                    print(f"    - {method}: {count} combinations (top_k: {top_k_count} × tokenizers: {retrieval_count})")
                else:
                    param_count = 1
                    param_desc = []
                    
                    if has_models and models:
                        param_count *= len(models)
                        param_desc.append(f"models: {len(models)}")

                    if has_temperature and temperatures:
                        if method == 'multi_query_expansion':
                            param_count *= len(temperatures)
                            param_desc.append(f"temps: {len(temperatures)}")

                    if has_max_token and max_tokens:
                        if method == 'hyde':
                            param_count *= len(max_tokens)
                            param_desc.append(f"max_tokens: {len(max_tokens)}")
                    
                    count = param_count * retrieval_count * top_k_count
                    param_desc.extend([f"top_k: {top_k_count}", f"tokenizers: {retrieval_count}"])
                    print(f"    - {method}: {count} combinations ({' × '.join(param_desc)})")
        
        elif component == 'passage_filter':
            print("\n  Parameters in search space:")
            for param, values in search_space.items():
                print(f"    {param}: {values}")
        elif component == 'passage_reranker':
            print("\n  Parameters in search space:")
            for param, values in search_space.items():
                print(f"    {param}: {values}")
        else:
            grid_space = ComponentGridSearchHelper.convert_to_grid_search_space(search_space)
            print(f"\n  Parameters to explore:")
            
            for param, values in grid_space.items():
                if len(values) <= 5:
                    print(f"    {param}: {values}")
                else:
                    print(f"    {param}: {values[:3]} ... {values[-2:]} ({len(values)} values)")
        
        if component == 'passage_filter':
            print("\n  Note: Using method-specific parameter names for grid search")
        elif component == 'query_expansion':
            print("\n  Note: Method-specific parameters will be conditionally applied based on the selected method")
            print("  Note: Invalid parameter combinations will be filtered out")
        elif component == 'passage_compressor':
            if 'passage_compressor_config' in search_space:
                print("\n  Note: Using pre-configured compressor combinations")
        elif component == 'passage_reranker' and fixed_config and 'retriever_top_k' in fixed_config:
            print(f"\n  Note: reranker_top_k constrained to <= {fixed_config['retriever_top_k']}")