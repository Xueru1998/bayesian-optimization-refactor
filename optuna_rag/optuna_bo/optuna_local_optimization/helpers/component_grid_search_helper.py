import itertools
import os
import time
from typing import Dict, Any, List, Union, Tuple, Set
import hashlib
import json
from pipeline.search_space_calculator import CombinationCalculator


class ComponentGridSearchHelper:
    
    def __init__(self):
        self.current_combinations = []
        self.current_index = 0
        self.explored_configs = set()
        self.combination_calculator = None
        
    def initialize_calculator(self, config_generator):
        
        self.combination_calculator = CombinationCalculator(
            config_generator, 
            search_type='grid'
        )
    
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
        if range_params is None:
            range_params = ['retriever_top_k', 'reranker_top_k']
        
        grid_space = {}
        
        for param_name, param_spec in search_space.items():
            if isinstance(param_spec, list):
                if (len(param_spec) == 2 and 
                    all(isinstance(x, int) for x in param_spec) and
                    param_spec[1] > param_spec[0] and
                    any(rp in param_name for rp in range_params)):
                    min_val, max_val = param_spec[0], param_spec[1]
                    grid_space[param_name] = list(range(min_val, max_val + 1))
                else:
                    grid_space[param_name] = param_spec
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                min_val, max_val = param_spec
                if isinstance(min_val, int) and any(rp in param_name for rp in range_params):
                    grid_space[param_name] = list(range(min_val, max_val + 1))
                else:
                    grid_space[param_name] = [min_val, max_val]
            else:
                grid_space[param_name] = [param_spec]
        
        return grid_space
    
    def calculate_total_combinations(self, max_combinations: int = 10000) -> int:
        try:
            total_combinations = 1
            components = [
                'query_expansion', 'retrieval', 'passage_filter', 
                'passage_reranker', 'passage_compressor', 'prompt_maker_generator'
            ]
            
            print(f"\nDetailed combination breakdown:")
            
            for component in components:
                combinations, note = self.combination_calculator.calculate_component_combinations(
                    component
                )
                
                if component == 'retrieval' and combinations == 0:
                    print(f"  {component}: skipped (active query expansion)")
                else:
                    print(f"  {component}: {combinations} combinations")
                    if combinations > 0:
                        total_combinations *= combinations
            
            print(f"  Total combinations: {total_combinations}")
            print(f"  Note: {note}")
            
            return min(total_combinations, max_combinations)
            
        except Exception as e:
            print(f"Error calculating total combinations: {e}")
            import traceback
            traceback.print_exc()
            return 100
    
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
        
    def save_all_components_grid_sequence(self, active_components: List[str], result_dir: str, 
                                     optimizer_instance) -> None:
        
        for idx, component in enumerate(active_components):
            component_dir = os.path.join(result_dir, f"{idx}_{component}")
            os.makedirs(component_dir, exist_ok=True)

            fixed_config = optimizer_instance._get_fixed_config(component, active_components)
            search_space = optimizer_instance.search_space_builder.build_component_search_space(component, fixed_config)
            
            if search_space:
                valid_combinations = self.get_valid_combinations(component, search_space, fixed_config)

                self.save_grid_search_sequence(component, valid_combinations, component_dir)
    
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
        all_combinations = []
        if component == 'retrieval':
            grid_space = ComponentGridSearchHelper.convert_to_grid_search_space(search_space)
            
            retrieval_methods = grid_space.get('retrieval_method', ['bm25'])
            top_k_values = grid_space.get('retriever_top_k', [10])
            bm25_tokenizers = grid_space.get('bm25_tokenizer', ['space'])
            vectordb_names = grid_space.get('vectordb_name', [])
            
            for method in retrieval_methods:
                for top_k in top_k_values:
                    if method == 'bm25':
                        for tokenizer in bm25_tokenizers:
                            all_combinations.append({
                                'retrieval_method': method,
                                'retriever_top_k': top_k,
                                'bm25_tokenizer': tokenizer
                            })
                    elif method == 'vectordb':
                        for vdb_name in vectordb_names:
                            all_combinations.append({
                                'retrieval_method': method,
                                'retriever_top_k': top_k,
                                'vectordb_name': vdb_name
                            })
            
            return all_combinations
        
        if component == 'query_expansion':
            if 'query_expansion_config' in search_space:
                configs = search_space['query_expansion_config']
                
                retriever_top_k_values = search_space.get('retriever_top_k', [10])
                if isinstance(retriever_top_k_values, list) and len(retriever_top_k_values) == 2:
                    if all(isinstance(x, int) for x in retriever_top_k_values) and retriever_top_k_values[1] > retriever_top_k_values[0]:
                        retriever_top_k_values = list(range(retriever_top_k_values[0], retriever_top_k_values[1] + 1))
                elif not isinstance(retriever_top_k_values, list):
                    retriever_top_k_values = [retriever_top_k_values]
                
                for config in configs:
                    if config == 'pass_query_expansion':
                        for top_k in retriever_top_k_values:
                            if 'retrieval_method' in search_space:
                                for retrieval_method in search_space['retrieval_method']:
                                    if retrieval_method == 'bm25' and 'bm25_tokenizer' in search_space:
                                        for tokenizer in search_space['bm25_tokenizer']:
                                            all_combinations.append({
                                                'query_expansion_config': config,
                                                'retriever_top_k': top_k,
                                                'retrieval_method': retrieval_method,
                                                'bm25_tokenizer': tokenizer
                                            })
                                    elif retrieval_method == 'vectordb' and 'vectordb_name' in search_space:
                                        for vdb_name in search_space['vectordb_name']:
                                            all_combinations.append({
                                                'query_expansion_config': config,
                                                'retriever_top_k': top_k,
                                                'retrieval_method': retrieval_method,
                                                'vectordb_name': vdb_name
                                            })
                            else:
                                all_combinations.append({
                                    'query_expansion_config': config,
                                    'retriever_top_k': top_k
                                })
                    else:
                        parts = config.split('::')
                        if len(parts) >= 3:
                            method = parts[0]
                            base_config = {'query_expansion_config': config}
                            
                            if method == 'hyde' and 'query_expansion_max_token' in search_space:
                                for max_token in search_space['query_expansion_max_token']:
                                    for top_k in retriever_top_k_values:
                                        if 'query_expansion_retrieval_method' in search_space:
                                            for retrieval_method in search_space['query_expansion_retrieval_method']:
                                                if retrieval_method == 'bm25' and 'query_expansion_bm25_tokenizer' in search_space:
                                                    for tokenizer in search_space['query_expansion_bm25_tokenizer']:
                                                        combo = base_config.copy()
                                                        combo.update({
                                                            'query_expansion_max_token': max_token,
                                                            'retriever_top_k': top_k,
                                                            'query_expansion_retrieval_method': retrieval_method,
                                                            'query_expansion_bm25_tokenizer': tokenizer
                                                        })
                                                        all_combinations.append(combo)
                                                elif retrieval_method == 'vectordb' and 'query_expansion_vectordb_name' in search_space:
                                                    for vdb_name in search_space['query_expansion_vectordb_name']:
                                                        combo = base_config.copy()
                                                        combo.update({
                                                            'query_expansion_max_token': max_token,
                                                            'retriever_top_k': top_k,
                                                            'query_expansion_retrieval_method': retrieval_method,
                                                            'query_expansion_vectordb_name': vdb_name
                                                        })
                                                        all_combinations.append(combo)
                                        else:
                                            combo = base_config.copy()
                                            combo.update({
                                                'query_expansion_max_token': max_token,
                                                'retriever_top_k': top_k
                                            })
                                            all_combinations.append(combo)
                            
                            elif method == 'query_decompose':
                                for top_k in retriever_top_k_values:
                                    if 'query_expansion_retrieval_method' in search_space:
                                        for retrieval_method in search_space['query_expansion_retrieval_method']:
                                            if retrieval_method == 'bm25' and 'query_expansion_bm25_tokenizer' in search_space:
                                                for tokenizer in search_space['query_expansion_bm25_tokenizer']:
                                                    combo = base_config.copy()
                                                    combo.update({
                                                        'retriever_top_k': top_k,
                                                        'query_expansion_retrieval_method': retrieval_method,
                                                        'query_expansion_bm25_tokenizer': tokenizer
                                                    })
                                                    all_combinations.append(combo)
                                            elif retrieval_method == 'vectordb' and 'query_expansion_vectordb_name' in search_space:
                                                for vdb_name in search_space['query_expansion_vectordb_name']:
                                                    combo = base_config.copy()
                                                    combo.update({
                                                        'retriever_top_k': top_k,
                                                        'query_expansion_retrieval_method': retrieval_method,
                                                        'query_expansion_vectordb_name': vdb_name
                                                    })
                                                    all_combinations.append(combo)
                                    else:
                                        combo = base_config.copy()
                                        combo['retriever_top_k'] = top_k
                                        all_combinations.append(combo)
                            
                            elif method == 'multi_query_expansion' and 'query_expansion_temperature' in search_space:
                                for temperature in search_space['query_expansion_temperature']:
                                    for top_k in retriever_top_k_values:
                                        if 'query_expansion_retrieval_method' in search_space:
                                            for retrieval_method in search_space['query_expansion_retrieval_method']:
                                                if retrieval_method == 'bm25' and 'query_expansion_bm25_tokenizer' in search_space:
                                                    for tokenizer in search_space['query_expansion_bm25_tokenizer']:
                                                        combo = base_config.copy()
                                                        combo.update({
                                                            'query_expansion_temperature': temperature,
                                                            'retriever_top_k': top_k,
                                                            'query_expansion_retrieval_method': retrieval_method,
                                                            'query_expansion_bm25_tokenizer': tokenizer
                                                        })
                                                        all_combinations.append(combo)
                                                elif retrieval_method == 'vectordb' and 'query_expansion_vectordb_name' in search_space:
                                                    for vdb_name in search_space['query_expansion_vectordb_name']:
                                                        combo = base_config.copy()
                                                        combo.update({
                                                            'query_expansion_temperature': temperature,
                                                            'retriever_top_k': top_k,
                                                            'query_expansion_retrieval_method': retrieval_method,
                                                            'query_expansion_vectordb_name': vdb_name
                                                        })
                                                        all_combinations.append(combo)
                                        else:
                                            combo = base_config.copy()
                                            combo.update({
                                                'query_expansion_temperature': temperature,
                                                'retriever_top_k': top_k
                                            })
                                            all_combinations.append(combo)
                
                return all_combinations

            grid_space = ComponentGridSearchHelper.convert_to_grid_search_space(search_space)
            
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
            methods = search_space.get('passage_filter_method', [])
            
            for method in methods:
                if method == 'pass_passage_filter':
                    all_combinations.append({'passage_filter_method': method})
                elif method == 'threshold_cutoff':
                    param_key = 'threshold_cutoff_threshold'
                    thresholds = search_space.get(param_key, [])
                    if isinstance(thresholds, list):
                        for threshold in thresholds:
                            all_combinations.append({
                                'passage_filter_method': method,
                                param_key: threshold
                            })
                elif method == 'percentile_cutoff':
                    param_key = 'percentile_cutoff_percentile'
                    percentiles = search_space.get(param_key, [])
                    if isinstance(percentiles, list):
                        for percentile in percentiles:
                            all_combinations.append({
                                'passage_filter_method': method,
                                param_key: percentile
                            })
                elif method == 'similarity_threshold_cutoff':
                    param_key = 'similarity_threshold_cutoff_threshold'
                    thresholds = search_space.get(param_key, [])
                    if isinstance(thresholds, list):
                        for threshold in thresholds:
                            all_combinations.append({
                                'passage_filter_method': method,
                                param_key: threshold
                            })
                elif method == 'similarity_percentile_cutoff':
                    param_key = 'similarity_percentile_cutoff_percentile'
                    percentiles = search_space.get(param_key, [])
                    if isinstance(percentiles, list):
                        for percentile in percentiles:
                            all_combinations.append({
                                'passage_filter_method': method,
                                param_key: percentile
                            })
            
            return all_combinations
        
        elif component == 'passage_reranker':
            methods = search_space.get('passage_reranker_method', [])
            top_k_values = search_space.get('reranker_top_k', [])
            
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
        
        if component == 'passage_compressor' and 'passage_compressor_config' in search_space:
            configs = search_space['passage_compressor_config']
            all_combinations = []
            
            for config in configs:
                if config == 'pass_compressor':
                    all_combinations.append({'passage_compressor_config': config})
                elif config == 'lexrank':
                    thresholds = search_space.get('lexrank_threshold', [0.05, 0.3])
                    dampings = search_space.get('lexrank_damping', [0.75, 0.9])
                    max_iters = search_space.get('lexrank_max_iterations', [15, 40])
                    comp_ratios = search_space.get('lexrank_compression_ratio', [0.3, 0.7])
                    
                    for threshold in thresholds:
                        for damping in dampings:
                            for max_iter in max_iters:
                                for comp_ratio in comp_ratios:
                                    all_combinations.append({
                                        'passage_compressor_config': config,
                                        'lexrank_threshold': threshold,
                                        'lexrank_damping': damping,
                                        'lexrank_max_iterations': max_iter,
                                        'lexrank_compression_ratio': comp_ratio
                                    })
                elif config.startswith('spacy::'):
                    comp_ratios = search_space.get('spacy_compression_ratio', [0.3, 0.5])
                    spacy_model = config.split('::')[1] if '::' in config else 'en_core_web_sm'
                    
                    for comp_ratio in comp_ratios:
                        all_combinations.append({
                            'passage_compressor_config': config,
                            'spacy_compression_ratio': comp_ratio,
                            'spacy_spacy_model': spacy_model
                        })
                else:
                    all_combinations.append({'passage_compressor_config': config})
            
            return all_combinations
        elif component == 'prompt_maker_generator':
            all_combinations = []
            
            prompt_methods = search_space.get('prompt_maker_method', ['fstring'])
            prompt_indices = search_space.get('prompt_template_idx', [0])
            generator_configs = search_space.get('generator_config', [])
            temperatures = search_space.get('generator_temperature', [])
            
            for method in prompt_methods:
                for prompt_idx in prompt_indices:
                    for gen_config in generator_configs:
                        for temp in temperatures:
                            all_combinations.append({
                                'prompt_maker_method': method,
                                'prompt_template_idx': prompt_idx,
                                'generator_config': gen_config,
                                'generator_temperature': temp
                            })
            
            return all_combinations
        
        else:
            grid_space = ComponentGridSearchHelper.convert_to_grid_search_space(search_space)
            
            if not grid_space:
                return [{}]
            
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