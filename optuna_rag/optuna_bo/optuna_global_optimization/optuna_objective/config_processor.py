from typing import Dict, Any


class ConfigProcessor:
    def __init__(self, config_generator):
        self.config_generator = config_generator
    
    def parse_composite_config(self, config_str: str, component: str, 
                            params: Dict[str, Any], config_generator) -> None:
        
        if component == 'passage_compressor' and config_str != 'pass_compressor':
            parts = config_str.split('::', 3)
            if len(parts) == 3:
                method, gen_type, model = parts
                params['passage_compressor_method'] = method
                params['compressor_generator_module_type'] = gen_type
                params['compressor_model'] = model

                unified_params = config_generator.extract_unified_parameters('passage_compressor')
                for comp_config in unified_params.get('compressor_configs', []):
                    if (comp_config['method'] == method and 
                        comp_config['generator_module_type'] == gen_type and 
                        model in comp_config['models']):
                        
                        params['compressor_llm'] = comp_config.get('llm', 'openai')
                        
                        if gen_type == 'sap_api':
                            params['compressor_api_url'] = comp_config.get('api_url')
                            params['compressor_bearer_token'] = comp_config.get('bearer_token')
                        elif gen_type == 'vllm':
                            params['compressor_llm'] = model
                            if 'tensor_parallel_size' in comp_config:
                                params['compressor_tensor_parallel_size'] = comp_config['tensor_parallel_size']
                        
                        if 'batch' in comp_config:
                            params['compressor_batch'] = comp_config['batch']
                        break