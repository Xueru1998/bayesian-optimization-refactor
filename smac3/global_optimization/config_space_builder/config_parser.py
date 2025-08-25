from typing import Dict, Any


class ConfigParser:
    
    def __init__(self, config_generator):
        self.config_generator = config_generator
    
    def parse_query_expansion_config(self, qe_config_str: str) -> Dict[str, Any]:
        if not qe_config_str or qe_config_str == 'pass_query_expansion':
            return {'query_expansion_method': 'pass_query_expansion'}
        
        parts = qe_config_str.split('::', 2)
        if len(parts) == 3:
            method, gen_type, model = parts
            config = {
                'query_expansion_method': method,
                'query_expansion_generator_module_type': gen_type,
                'query_expansion_model': model
            }
            
            unified_params = self.config_generator.extract_unified_parameters('query_expansion')
            for gen_config in unified_params.get('generator_configs', []):
                if (gen_config['method'] == method and 
                    gen_config['generator_module_type'] == gen_type and 
                    model in gen_config['models']):
                    config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                    if gen_type == 'sap_api':
                        config['query_expansion_api_url'] = gen_config.get('api_url')
                    break
            
            return config
        
        return {'query_expansion_method': qe_config_str}
    
    def parse_reranker_config(self, reranker_config_str: str) -> Dict[str, Any]:
        print(f"\n[parse_reranker_config DEBUG] Input: {reranker_config_str}")
        
        if not reranker_config_str or reranker_config_str == 'pass_reranker':
            result = {'passage_reranker_method': 'pass_reranker'}
            print(f"  Result: {result}")
            return result

        if reranker_config_str == 'sap_api':
            config = {
                'passage_reranker_method': 'sap_api'
            }

            unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
            models = unified_params.get('models', {})
            api_endpoints = unified_params.get('api_endpoints', {})
            
            if 'sap_api' in models and models['sap_api']:
                config['reranker_model_name'] = models['sap_api'][0]
            else:
                config['reranker_model_name'] = 'cohere-rerank-v3.5'
                
            if 'sap_api' in api_endpoints:
                config['reranker_api_url'] = api_endpoints['sap_api']
            
            print(f"  Result: {config}")
            return config

        parts = reranker_config_str.split('::', 1)
        if len(parts) == 2:
            method, model = parts
            config = {
                'passage_reranker_method': method,
                'reranker_model_name': model
            }
            print(f"  Parsed method: {method}, model: {model}")
            print(f"  Result: {config}")
            return config

        result = {'passage_reranker_method': reranker_config_str}
        print(f"  No :: found, treating as method only")
        print(f"  Result: {result}")
        return result

    def parse_compressor_config(self, comp_config_str: str) -> Dict[str, Any]:
        if not comp_config_str or comp_config_str == 'pass_compressor':
            return {'passage_compressor_method': 'pass_compressor'}
        
        parts = comp_config_str.split('::', 3)
        
        if len(parts) == 3:
            method, gen_type, model = parts
            
            config = {
                'passage_compressor_method': method,
                'compressor_generator_module_type': gen_type,
                'compressor_model': model
            }

            unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
            for comp_config in unified_params.get('compressor_configs', []):
                if (comp_config['method'] == method and 
                    comp_config.get('generator_module_type') == gen_type and 
                    model in comp_config.get('models', [])):

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
            
        elif len(parts) == 2:
            method = parts[0]
            second_part = parts[1]
            
            config = {
                'passage_compressor_method': method,
                'compressor_llm': second_part
            }

            unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
            for comp_config in unified_params.get('compressor_configs', []):
                if comp_config['method'] == method:
                    if 'llm' in comp_config and comp_config['llm'] == second_part:
                        if comp_config.get('models'):
                            config['compressor_model'] = comp_config['models'][0]
                            config['compressor_generator_module_type'] = comp_config.get('generator_module_type', 'llama_index_llm')
                    elif second_part in comp_config.get('models', []):
                        config['compressor_llm'] = comp_config.get('llm', 'openai')
                        config['compressor_model'] = second_part
                        config['compressor_generator_module_type'] = comp_config.get('generator_module_type', 'llama_index_llm')

                    if comp_config.get('generator_module_type') == 'sap_api':
                        config['compressor_api_url'] = comp_config.get('api_url')
                    break
            
            return config
        
        return {'passage_compressor_method': comp_config_str}
    
    def parse_generator_config(self, gen_config_str: str) -> Dict[str, Any]:
        if not gen_config_str:
            return {}
        
        parts = gen_config_str.split('::', 1)
        if len(parts) == 2:
            module_type, model = parts
            config = {
                'generator_module_type': module_type,
                'generator_model': model
            }
            
            unified_params = self.config_generator.extract_unified_parameters('generator')
            for module_config in unified_params.get('module_configs', []):
                if module_config['module_type'] == module_type and model in module_config['models']:
                    if module_type == 'sap_api':
                        config['generator_api_url'] = module_config.get('api_url')
                        config['generator_llm'] = module_config.get('llm', 'mistralai')
                    elif module_type == 'vllm':
                        config['generator_llm'] = model
                    else:
                        config['generator_llm'] = module_config.get('llm', 'openai')
                    break
            
            return config
        
        return {'generator_model': gen_config_str}