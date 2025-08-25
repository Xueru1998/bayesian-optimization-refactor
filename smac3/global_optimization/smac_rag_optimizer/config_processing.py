import numpy as np
from typing import Dict, Any, List, Optional


class ConfigProcessing:
    
    def _ensure_conditional_parameters(self, config_dict: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        rng = np.random.RandomState(seed)

        reranker_key = None
        for key in ['reranker_topk', 'reranker_top_k']:
            if key in config_dict:
                reranker_key = key
                break
        
        if reranker_key and config_dict[reranker_key] == 1:
            config_dict['passage_filter_method'] = 'pass_passage_filter'
            config_dict['passage_filter_config'] = 'pass_passage_filter'

            filter_params = ['threshold', 'percentile', 'threshold_cutoff_threshold', 
                        'percentile_cutoff_percentile', 'similarity_threshold_cutoff_threshold',
                        'similarity_percentile_cutoff_percentile']
            for param in filter_params:
                if param in config_dict:
                    del config_dict[param]
            
            print(f"[CONSTRAINT] Set passage_filter to 'pass' because {reranker_key}=1")

        if 'query_expansion_method' in config_dict and config_dict['query_expansion_method'] != 'pass_query_expansion':
            for param in ['retrieval_method', 'bm25_tokenizer', 'vectordb_name']:
                if param in config_dict:
                    del config_dict[param]
                    print(f"[DEBUG] Removed {param} since query expansion is active")
            
            qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()

            if 'query_expansion_retrieval_method' not in config_dict:
                if qe_retrieval_options and qe_retrieval_options.get('methods'):
                    config_dict['query_expansion_retrieval_method'] = rng.choice(qe_retrieval_options['methods'])
                    print(f"[DEBUG] Added missing query_expansion_retrieval_method: {config_dict['query_expansion_retrieval_method']}")
                else:
                    config_dict['query_expansion_retrieval_method'] = 'bm25'
                    print(f"[DEBUG] No QE retrieval options available, defaulting to bm25")

            qe_retrieval_method = config_dict.get('query_expansion_retrieval_method')
            
            if qe_retrieval_method == 'bm25':
                if 'query_expansion_vectordb_name' in config_dict:
                    del config_dict['query_expansion_vectordb_name']
                    print(f"[DEBUG] Removed query_expansion_vectordb_name since using BM25")

                if 'query_expansion_bm25_tokenizer' not in config_dict:
                    if qe_retrieval_options and qe_retrieval_options.get('bm25_tokenizers'):
                        config_dict['query_expansion_bm25_tokenizer'] = rng.choice(qe_retrieval_options['bm25_tokenizers'])
                        print(f"[DEBUG] Added missing query_expansion_bm25_tokenizer: {config_dict['query_expansion_bm25_tokenizer']}")
                    else:
                        config_dict['query_expansion_bm25_tokenizer'] = 'space'
                        print(f"[DEBUG] No BM25 tokenizers available, using default: space")
            
            elif qe_retrieval_method == 'vectordb':
                if 'query_expansion_bm25_tokenizer' in config_dict:
                    del config_dict['query_expansion_bm25_tokenizer']
                    print(f"[DEBUG] Removed query_expansion_bm25_tokenizer since using vectordb")

                if 'query_expansion_vectordb_name' not in config_dict:
                    if qe_retrieval_options and qe_retrieval_options.get('vectordb_names'):
                        config_dict['query_expansion_vectordb_name'] = rng.choice(qe_retrieval_options['vectordb_names'])
                        print(f"[DEBUG] Added missing query_expansion_vectordb_name: {config_dict['query_expansion_vectordb_name']}")
                    else:
                        config_dict['query_expansion_vectordb_name'] = 'default'
                        print(f"[DEBUG] No vectordb names available, using default: default")
        
        else:
            for param in ['query_expansion_retrieval_method', 'query_expansion_bm25_tokenizer', 'query_expansion_vectordb_name']:
                if param in config_dict:
                    del config_dict[param]
                    print(f"[DEBUG] Removed {param} since query expansion is not active")

        if 'passage_filter_method' in config_dict and config_dict['passage_filter_method'] != 'pass_passage_filter':
            filter_method = config_dict['passage_filter_method']
            
            unified_space = self.config_space_builder.unified_extractor.extract_search_space('smac')
            
            if filter_method in ['threshold_cutoff', 'similarity_threshold_cutoff']:
                self._add_threshold_if_missing(config_dict, unified_space, filter_method, rng)
            elif filter_method in ['percentile_cutoff', 'similarity_percentile_cutoff']:
                self._add_percentile_if_missing(config_dict, unified_space, filter_method, rng)
        
        config_dict = self._validate_topk_constraints(config_dict)
        
        return config_dict
    
    def _add_threshold_if_missing(self, config_dict, unified_space, filter_method, rng):
        if 'threshold' not in config_dict:
            if 'threshold' in unified_space and 'method_values' in unified_space['threshold']:
                method_values = unified_space['threshold']['method_values']
                if filter_method in method_values:
                    values = method_values[filter_method]
                    config_dict['threshold'] = self._sample_from_values(values, rng, 0.0, 1.0)
                    print(f"[DEBUG] Added threshold: {config_dict['threshold']:.4f}")
                else:
                    config_dict['threshold'] = rng.uniform(0.65, 0.85)
            else:
                config_dict['threshold'] = rng.uniform(0.65, 0.85)
    
    def _add_percentile_if_missing(self, config_dict, unified_space, filter_method, rng):
        if 'percentile' not in config_dict:
            if 'percentile' in unified_space and 'method_values' in unified_space['percentile']:
                method_values = unified_space['percentile']['method_values']
                if filter_method in method_values:
                    values = method_values[filter_method]
                    config_dict['percentile'] = self._sample_from_values(values, rng, 0.0, 1.0)
                    print(f"[DEBUG] Added percentile: {config_dict['percentile']:.4f}")
                else:
                    config_dict['percentile'] = rng.uniform(0.6, 0.8)
            else:
                config_dict['percentile'] = rng.uniform(0.6, 0.8)
    
    def _sample_from_values(self, values: List[float], rng, min_limit: float, max_limit: float) -> float:
        if isinstance(values, list) and len(values) >= 2:
            return rng.uniform(min(values), max(values))
        elif isinstance(values, list) and len(values) == 1:
            val = values[0]
            min_val = max(min_limit, val - 0.2)
            max_val = min(max_limit, val + 0.2)
            return rng.uniform(min_val, max_val)
        else:
            return rng.uniform(0.65, 0.85)
    
    def _validate_topk_constraints(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        retriever_key = None
        reranker_key = None
        
        for key in ['retriever_topk', 'retriever_top_k']:
            if key in config_dict:
                retriever_key = key
                break
        
        for key in ['reranker_topk', 'reranker_top_k']:
            if key in config_dict:
                reranker_key = key
                break
        
        if retriever_key and reranker_key:
            retriever_value = config_dict[retriever_key]
            reranker_value = config_dict[reranker_key]
            
            if isinstance(retriever_value, (int, float, np.integer, np.floating)):
                retriever_value = int(retriever_value)
            if isinstance(reranker_value, (int, float, np.integer, np.floating)):
                reranker_value = int(reranker_value)
            
            if reranker_value > retriever_value:
                original_value = reranker_value
                config_dict[reranker_key] = retriever_value
                print(f"[CONSTRAINT] Adjusted {reranker_key} from {original_value} to {retriever_value} to not exceed {retriever_key}")
        
        return config_dict
    
    def _prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config_dict = config if isinstance(config, dict) else dict(config)

        if 'query_expansion_method' not in config_dict:
            if 'retrieval_method' not in config_dict:
                retrieval_options = self.config_generator.extract_retrieval_options()
                if retrieval_options and retrieval_options.get('methods'):
                    config_dict['retrieval_method'] = retrieval_options['methods'][0]

                    if config_dict['retrieval_method'] == 'bm25' and 'bm25_tokenizer' not in config_dict:
                        if retrieval_options.get('bm25_tokenizers'):
                            config_dict['bm25_tokenizer'] = retrieval_options['bm25_tokenizers'][0]
                    elif config_dict['retrieval_method'] == 'vectordb' and 'vectordb_name' not in config_dict:
                        if retrieval_options.get('vectordb_names'):
                            config_dict['vectordb_name'] = retrieval_options['vectordb_names'][0]

        config_dict = self.config_space_builder.clean_trial_config(config_dict)
        config_dict = self._convert_numpy_types(config_dict)

        for temp_param in ['generator_temperature', 'query_expansion_temperature', 'temperature']:
            if temp_param in config_dict:
                try:
                    config_dict[temp_param] = float(config_dict[temp_param])
                except:
                    config_dict[temp_param] = 0.7
        
        config_dict = self._validate_topk_constraints(config_dict)
        
        return config_dict