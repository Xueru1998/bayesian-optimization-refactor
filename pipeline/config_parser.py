from typing import Dict, Any


class ConfigParser:
    def __init__(self, config_generator):
        self.config_generator = config_generator
    
    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        parsed_config = config.copy()
        
        if 'query_expansion_config' in config:
            qe_parsed = self.parse_query_expansion_config(config['query_expansion_config'])
            parsed_config.update(qe_parsed)
            
        if 'retrieval_config' in config:
            retrieval_parsed = self.parse_retrieval_config(config['retrieval_config'])
            parsed_config.update(retrieval_parsed)
            
        if 'passage_reranker_config' in config:
            reranker_parsed = self.parse_reranker_config(config['passage_reranker_config'])
            parsed_config.update(reranker_parsed)
            
        if 'passage_filter_config' in config:
            filter_parsed = self.parse_filter_config(config['passage_filter_config'])
            parsed_config.update(filter_parsed)
            
        if 'passage_compressor_config' in config:
            compressor_parsed = self.parse_compressor_config(config['passage_compressor_config'])
            parsed_config.update(compressor_parsed)
            
        if 'prompt_maker_config' in config:
            prompt_parsed = self.parse_prompt_config(config['prompt_maker_config'])
            parsed_config.update(prompt_parsed)
            
        return parsed_config
    
    def parse_query_expansion_config(self, qe_config_str: str) -> Dict[str, Any]:
        if not qe_config_str:
            return {}
        
        if qe_config_str == 'pass_query_expansion':
            return {'query_expansion_method': 'pass_query_expansion'}
        
        parts = qe_config_str.split('::')
        
        if len(parts) >= 3:
            method, gen_type, model = parts[0], parts[1], parts[2]
            config = {
                'query_expansion_method': method,
                'query_expansion_generator_module_type': gen_type,
                'query_expansion_model': model
            }
            
            if method == 'hyde' and len(parts) >= 4:
                config['query_expansion_max_token'] = int(parts[3])
            elif method == 'multi_query_expansion' and len(parts) >= 4:
                config['query_expansion_temperature'] = float(parts[3])
            
            unified_params = self.config_generator.extract_unified_parameters('query_expansion')
            for gen_config in unified_params.get('generator_configs', []):
                if (gen_config['method'] == method and 
                    gen_config['generator_module_type'] == gen_type and 
                    model in gen_config['models']):
                    if gen_type == 'sap_api':
                        config['query_expansion_api_url'] = gen_config.get('api_url')
                        config['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                    else:
                        config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                    break
            
            return config
        
        if qe_config_str.startswith('query_decompose_'):
            model = qe_config_str.replace('query_decompose_', '')
            return {
                'query_expansion_method': 'query_decompose',
                'query_expansion_model': model
            }
        
        if qe_config_str.startswith('hyde_'):
            parts = qe_config_str.replace('hyde_', '').rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                model = parts[0]
                max_token = int(parts[1])
                return {
                    'query_expansion_method': 'hyde',
                    'query_expansion_model': model,
                    'query_expansion_max_token': max_token
                }
            else:
                return {
                    'query_expansion_method': 'hyde',
                    'query_expansion_max_token': int(qe_config_str.split('_')[-1])
                }
        
        if qe_config_str.startswith('multi_query_expansion_'):
            temp = float(qe_config_str.split('_')[-1])
            return {
                'query_expansion_method': 'multi_query_expansion',
                'query_expansion_temperature': temp
            }
        
        if qe_config_str == 'query_decompose':
            return {'query_expansion_method': 'query_decompose'}
        
        return {}
    
    def parse_retrieval_config(self, retrieval_config_str: str) -> Dict[str, Any]:
        if not retrieval_config_str:
            return {}
        
        if retrieval_config_str.startswith('bm25_'):
            tokenizer = retrieval_config_str.replace('bm25_', '')
            return {
                'retrieval_method': 'bm25',
                'bm25_tokenizer': tokenizer
            }
        elif retrieval_config_str.startswith('vectordb_'):
            vdb_name = retrieval_config_str.replace('vectordb_', '')
            return {
                'retrieval_method': 'vectordb',
                'vectordb_name': vdb_name
            }
        
        return {}
    
    def parse_reranker_config(self, reranker_config_str: str) -> Dict[str, Any]:
        if not reranker_config_str:
            return {}
        
        if reranker_config_str == 'pass_reranker':
            return {'passage_reranker_method': 'pass_reranker'}
        
        simple_methods = ['upr']
        if reranker_config_str in simple_methods:
            return {'passage_reranker_method': reranker_config_str}
        
        model_based_methods = [
            'colbert_reranker',
            'sentence_transformer_reranker',
            'flag_embedding_reranker',
            'flag_embedding_llm_reranker',
            'openvino_reranker',
            'flashrank_reranker',
            'monot5'
        ]
        
        for method in model_based_methods:
            if reranker_config_str.startswith(method + '_'):
                model_name = reranker_config_str[len(method) + 1:]
                return {
                    'passage_reranker_method': method,
                    'reranker_model_name': model_name
                }
        
        return {'passage_reranker_method': reranker_config_str}
    
    def parse_filter_config(self, filter_config_str: str) -> Dict[str, Any]:
        if not filter_config_str:
            return {}
        
        parts = filter_config_str.split('_')
        
        if filter_config_str.startswith('threshold_cutoff_'):
            return {
                'passage_filter_method': 'threshold_cutoff',
                'threshold': float(parts[-1])
            }
        elif filter_config_str.startswith('percentile_cutoff_'):
            return {
                'passage_filter_method': 'percentile_cutoff',
                'percentile': float(parts[-1])
            }
        elif filter_config_str.startswith('similarity_threshold_cutoff_'):
            return {
                'passage_filter_method': 'similarity_threshold_cutoff',
                'threshold': float(parts[-1])
            }
        elif filter_config_str.startswith('similarity_percentile_cutoff_'):
            return {
                'passage_filter_method': 'similarity_percentile_cutoff',
                'percentile': float(parts[-1])
            }
        
        return {}
    
    def parse_compressor_config(self, compressor_config_str: str) -> Dict[str, Any]:
        if not compressor_config_str:
            return {}
        
        if compressor_config_str == 'pass_compressor':
            return {'passage_compressor_method': 'pass_compressor'}
        
        parts = compressor_config_str.split('::', 3)
        
        if parts[0] in ['lexrank', 'spacy']:
            method = parts[0]
            config = {'passage_compressor_method': method}
            
            if method == 'spacy' and len(parts) > 1:
                config['spacy_model'] = parts[1]
            
            return config
        
        elif len(parts) >= 3:
            method, gen_type, model = parts[0], parts[1], parts[2]
            
            config = {
                'passage_compressor_method': method,
                'compressor_generator_module_type': gen_type,
                'compressor_model': model
            }
            
            unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
            for comp_config in unified_params.get('compressor_configs', []):
                if (comp_config['method'] == method and 
                    comp_config['generator_module_type'] == gen_type and 
                    model in comp_config['models']):
                    
                    config['compressor_llm'] = comp_config.get('llm', 'openai')
                    
                    if gen_type == 'sap_api':
                        config['compressor_api_url'] = comp_config.get('api_url')
                    elif gen_type == 'vllm':
                        config['compressor_llm'] = model
                        if 'tensor_parallel_size' in comp_config:
                            config['compressor_tensor_parallel_size'] = comp_config['tensor_parallel_size']
                        if 'gpu_memory_utilization' in comp_config:
                            config['compressor_gpu_memory_utilization'] = comp_config['gpu_memory_utilization']
                    
                    if 'batch' in comp_config:
                        config['compressor_batch'] = comp_config['batch']
                    break
            
            return config
        else:
            method = parts[0]
            
            if method in ['lexrank', 'spacy']:
                config = {'passage_compressor_method': method}
                if method == 'spacy' and len(parts) > 1:
                    config['spacy_model'] = parts[1]
                return config
            else:
                llm = parts[1] if len(parts) > 1 else 'openai'
                model = parts[2] if len(parts) > 2 else None
                
                config = {
                    'passage_compressor_method': method,
                    'compressor_llm': llm
                }
                if model:
                    config['compressor_model'] = model
                
                return config
        
        if compressor_config_str.startswith('tree_summarize_') or compressor_config_str.startswith('refine_'):
            parts = compressor_config_str.split('_', 2)
            if len(parts) >= 3:
                method = parts[0] + '_' + parts[1]
                llm_and_model = parts[2]
                
                llm_parts = llm_and_model.split('_', 1)
                if len(llm_parts) == 2:
                    return {
                        'passage_compressor_method': method,
                        'compressor_llm': llm_parts[0],
                        'compressor_model': llm_parts[1]
                    }
        
        return {'passage_compressor_method': compressor_config_str}
    
    def parse_prompt_config(self, prompt_config_str: str) -> Dict[str, Any]:
        if not prompt_config_str:
            return {}
        
        if prompt_config_str == 'pass_prompt_maker':
            return {'prompt_maker_method': 'pass_prompt_maker'}
        
        known_prompt_methods = ['fstring', 'long_context_reorder', 'window_replacement']
        
        parts = prompt_config_str.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            method = parts[0]
            if method in known_prompt_methods:
                return {
                    'prompt_maker_method': method,
                    'prompt_template_idx': int(parts[1])
                }
        
        if prompt_config_str in known_prompt_methods:
            return {'prompt_maker_method': prompt_config_str}
        
        print(f"Warning: Unknown prompt config '{prompt_config_str}'. Using default 'fstring_0'.")
        return {
            'prompt_maker_method': 'fstring',
            'prompt_template_idx': 0
        }
    
    def parse_generator_config(self, gen_config_str: str) -> Dict[str, Any]:
        if not gen_config_str:
            return {}
        
        module_type, model = gen_config_str.split('::', 1)
        
        config = {
            'generator_module_type': module_type,
            'generator_model': model
        }
        
        unified_params = self.config_generator.extract_unified_parameters('generator')
        for module_config in unified_params.get('module_configs', []):
            if module_config['module_type'] == module_type and model in module_config['models']:
                if module_type == 'sap_api':
                    config['generator_api_url'] = module_config.get('api_url')
                    if not config['generator_api_url']:
                        raise ValueError("SAP API URL not found in generator configuration")
                    config['generator_llm'] = module_config.get('llm', 'mistralai')
                elif module_type == 'vllm':
                    config['generator_llm'] = model
                else:
                    config['generator_llm'] = module_config.get('llm', 'openai')
                break
        
        return config
    
    def extract_component_config(self, component: str, config: Dict[str, Any]) -> Dict[str, Any]:
        component_config = {}
        
        if component == 'retrieval':
            relevant_keys = ['retrieval_method', 'retriever_top_k', 'bm25_tokenizer', 'vectordb_name']
        elif component == 'query_expansion':
            relevant_keys = ['query_expansion_method', 'query_expansion_model', 
                            'query_expansion_temperature', 'query_expansion_max_token']
        elif component == 'passage_reranker':
            relevant_keys = ['passage_reranker_method', 'reranker_top_k', 'reranker_model']
        elif component == 'passage_filter':
            relevant_keys = ['passage_filter_method', 'threshold', 'percentile']
        elif component == 'passage_compressor':
            relevant_keys = ['passage_compressor_method', 'compressor_model', 
                            'lexrank_compression_ratio', 'spacy_model']
        elif component == 'prompt_maker':
            relevant_keys = ['prompt_maker_method', 'prompt_template_idx']
        elif component == 'generator':
            relevant_keys = ['generator_module_type', 'generator_model', 'generator_temperature']
        else:
            relevant_keys = []
        
        for key in relevant_keys:
            if key in config:
                component_config[key] = config[key]
        
        return component_config