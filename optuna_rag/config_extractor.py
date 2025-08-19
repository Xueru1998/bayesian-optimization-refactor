from typing import Dict, Any, List, Union, Tuple
from pipeline.search_space_extractor import UnifiedSearchSpaceExtractor


class OptunaConfigExtractor:
    def __init__(self, config_generator, search_type='grid'):
        self.config_generator = config_generator
        self.search_type = search_type
        self.unified_extractor = UnifiedSearchSpaceExtractor(config_generator)
        
        self.has_query_expansion = self.config_generator.node_exists("query_expansion")
        self.has_retrieval = self.config_generator.node_exists("retrieval")
        self.has_prompt_maker = self.config_generator.node_exists("prompt_maker")
        self.has_generator = self.config_generator.node_exists("generator")
        self.has_reranker = self.config_generator.node_exists("passage_reranker")
        
        self._optimizing_component = None
    
    def set_optimizing_component(self, component: str):
        self._optimizing_component = component
    
    def extract_search_space(self) -> Dict[str, Union[List[Any], Tuple[float, float]]]:
        if self.search_type == 'grid':
            return self.unified_extractor.extract_search_space('optuna_grid')
        else:
            unified_space = self.unified_extractor._extract_unified_space()
            
            has_qe_config = 'query_expansion_config' in unified_space and unified_space['query_expansion_config'].get('values')
            
            if has_qe_config:
                params_to_remove = [
                    'query_expansion_method',
                    'query_expansion_generator_module_type', 
                    'query_expansion_llm',
                    'query_expansion_model'
                ]
                for param in params_to_remove:
                    unified_space.pop(param, None)
            
            bo_space = self._convert_to_optuna_bo_space(unified_space)
            
            if 'retriever_top_k' in unified_space and 'retriever_top_k' not in bo_space:
                top_k_info = unified_space['retriever_top_k']
                values = top_k_info.get('values', [])
                if len(values) >= 2:
                    bo_space['retriever_top_k'] = (min(values), max(values))
                elif len(values) == 1:
                    bo_space['retriever_top_k'] = self._get_bo_range_for_single_value('retriever_top_k', values[0], 'int')
            
            if has_qe_config:
                param_info = unified_space['query_expansion_config']
                if param_info.get('values'):
                    bo_space['query_expansion_config'] = param_info['values']
                    
                    qe_configs = param_info['values']
                    
                    if any('hyde' in config for config in qe_configs if config != 'pass_query_expansion'):
                        if 'query_expansion_max_token' not in bo_space:
                            max_token_values = unified_space.get('query_expansion_max_token', {}).get('values', [])
                            if max_token_values:
                                if len(max_token_values) == 1:
                                    bo_space['query_expansion_max_token'] = self._get_bo_range_for_single_value('query_expansion_max_token', max_token_values[0], 'int')
                                else:
                                    bo_space['query_expansion_max_token'] = (min(max_token_values), max(max_token_values))
                            else:
                                bo_space['query_expansion_max_token'] = (50, 200)
                    
                    if any('multi_query' in config for config in qe_configs if config != 'pass_query_expansion'):
                        if 'query_expansion_temperature' not in bo_space:
                            temp_values = unified_space.get('query_expansion_temperature', {}).get('values', [])
                            if temp_values:
                                if len(temp_values) == 1:
                                    bo_space['query_expansion_temperature'] = self._get_bo_range_for_single_value('query_expansion_temperature', temp_values[0], 'float')
                                else:
                                    bo_space['query_expansion_temperature'] = (min(temp_values), max(temp_values))
                            else:
                                bo_space['query_expansion_temperature'] = (0.0, 1.0)
                    
                    qe_params = self.config_generator._extract_query_expansion_unified()
                    retrieval_options = qe_params.get('retrieval_options', {})
                    
                    methods = retrieval_options.get('methods', [])
                    if methods and 'query_expansion_retrieval_method' not in bo_space:
                        bo_space['query_expansion_retrieval_method'] = methods

                    if 'bm25' in methods and retrieval_options.get('bm25_tokenizers'):
                        if 'query_expansion_bm25_tokenizer' not in bo_space:
                            bo_space['query_expansion_bm25_tokenizer'] = retrieval_options['bm25_tokenizers']
                    if 'vectordb' in methods and retrieval_options.get('vectordb_names'):
                        if 'query_expansion_vectordb_name' not in bo_space:
                            bo_space['query_expansion_vectordb_name'] = retrieval_options['vectordb_names']
            
            if self.has_prompt_maker:
                prompt_methods, prompt_indices = self.config_generator.extract_prompt_maker_options()
                if prompt_methods and 'prompt_maker_method' not in bo_space:
                    bo_space['prompt_maker_method'] = prompt_methods
                if prompt_indices and 'prompt_template_idx' not in bo_space:
                    bo_space['prompt_template_idx'] = (min(prompt_indices), max(prompt_indices)) if len(prompt_indices) > 1 else prompt_indices
            
            if self.has_generator:
                gen_config = self.config_generator.extract_node_config("generator")
                if gen_config and gen_config.get("modules"):
                    all_models = []
                    all_temps = []
                    
                    for module in gen_config.get("modules", []):
                        llms = module.get("llm", [])
                        if isinstance(llms, str):
                            llms = [llms]
                        all_models.extend(llms if isinstance(llms, list) else [])
                        
                        temps = module.get("temperature", [])
                        if isinstance(temps, (int, float)):
                            temps = [temps]
                        all_temps.extend(temps if isinstance(temps, list) else [])
                    
                    if all_models and 'generator_model' not in bo_space:
                        bo_space['generator_model'] = list(set(all_models))
                    if all_temps and 'generator_temperature' not in bo_space:
                        unique_temps = list(set(all_temps))
                        if len(unique_temps) > 1:
                            bo_space['generator_temperature'] = (min(unique_temps), max(unique_temps))
                        else:
                            bo_space['generator_temperature'] = unique_temps
            
            if self.has_reranker:
                self._extract_reranker_module_specific_models(unified_space, bo_space)
            
            return bo_space
    
    def _ensure_list(self, value):
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]
    
    def _get_bo_range_for_single_value(self, param_name: str, value: float, param_type: str) -> Tuple[float, float]:
        if param_type == 'int':
            value = int(value)
            if 'top_k' in param_name or 'max_token' in param_name:
                min_val = max(1, int(value * 0.5))
                max_val = int(value * 1.5)
            elif 'idx' in param_name or 'index' in param_name:
                min_val = max(0, value - 1)
                max_val = value + 1
            else:
                min_val = max(1, int(value * 0.8))
                max_val = int(value * 1.2)
            
            if min_val >= max_val:
                if value == 0:
                    return (0, 1)
                else:
                    return (max(0, value - 1), value + 1)
            
            return (min_val, max_val)
        else:
            if 'temperature' in param_name or 'percentile' in param_name:
                min_val = max(0.0, value - 0.2)
                max_val = min(1.0, value + 0.2)
            elif 'threshold' in param_name:
                min_val = max(0.0, value - 0.2)
                max_val = min(1.0, value + 0.2)
            else:
                min_val = max(0.0, value - 0.1)
                max_val = min(1.0, value + 0.1)
            
            if min_val >= max_val:
                if value == 0.0:
                    return (0.0, 0.1)
                elif value == 1.0:
                    return (0.9, 1.0)
                else:
                    return (max(0.0, value - 0.05), min(1.0, value + 0.05))
            
            return (min_val, max_val)
    
    def _extract_reranker_module_specific_models(self, unified_space: Dict[str, Any], bo_space: Dict[str, Any]):
        reranker_config = self.config_generator.extract_node_config("passage_reranker")
        if not reranker_config or not reranker_config.get("modules"):
            return
        
        for module in reranker_config["modules"]:
            module_type = module.get("module_type")
            if not module_type or module_type == "pass_reranker":
                continue
            
            model_param_key = f"{module_type}_model_name"
            
            if "model_name" in module:
                models = module["model_name"]
                if isinstance(models, str):
                    models = [models]
                if models:
                    bo_space[model_param_key] = models
            elif "model" in module:
                models = module["model"]
                if isinstance(models, str):
                    models = [models]
                if models:
                    bo_space[model_param_key] = models
                    
        
    
    def _convert_to_optuna_bo_space(self, unified_space: Dict[str, Any]) -> Dict[str, Any]:
        bo_space = {}
        
        composite_params = {
            'query_expansion_config': ['query_expansion_method', 'query_expansion_generator_module_type', 
                                    'query_expansion_llm', 'query_expansion_model'],
            'passage_compressor_config': ['passage_compressor_method', 'compressor_generator_module_type',
                                        'compressor_llm', 'compressor_model'],
            'generator_config': ['generator_module_type', 'generator_model', 'generator_llm']
        }
        
        params_to_exclude = set()
        for composite_param, individual_params in composite_params.items():
            if composite_param in unified_space and unified_space[composite_param].get('values'):
                params_to_exclude.update(individual_params)

        for composite_param in composite_params.keys():
            if composite_param in unified_space:
                param_info = unified_space[composite_param]
                if param_info.get('values'):
                    bo_space[composite_param] = param_info['values']

                    if composite_param == 'passage_compressor_config':
                        self._add_compressor_additional_params(bo_space, unified_space, param_info['values'])

        for param_name, param_info in unified_space.items():
            if param_name in params_to_exclude or param_name in composite_params:
                continue
                
            param_type = param_info['type']
            values = param_info.get('values', [])
            
            if param_name in ['threshold', 'percentile', 'reranker_model_name', 'reranker_model']:
                continue
            
            if param_type == 'categorical':
                if param_name == 'passage_reranker_method':
                    if values and len(values) > 0:
                        bo_space[param_name] = values
                elif values and len(values) > 0:
                    bo_space[param_name] = values
                elif 'method_models' in param_info and param_name not in ['reranker_model_name', 'reranker_model']:
                    all_models = []
                    for models in param_info['method_models'].values():
                        all_models.extend(models)
                    if all_models:
                        bo_space[param_name] = list(set(all_models))
            
            elif param_type in ['int', 'float']:
                if len(values) == 0:
                    continue
                elif len(values) == 1:
                    if 'top_k' in param_name:
                        value = values[0]
                        min_val = max(1, int(value * 0.5))
                        max_val = int(value * 1.5)
                        bo_space[param_name] = (min_val, max_val)
                    else:
                        min_val, max_val = self._get_bo_range_for_single_value(param_name, values[0], param_type)
                        bo_space[param_name] = (min_val, max_val)
                elif len(values) == 2:
                    if param_name == 'reranker_top_k':
                        max_allowed = param_info.get('max_value', max(values))
                        bo_space[param_name] = (min(values), min(max(values), max_allowed))
                    else:
                        bo_space[param_name] = (min(values), max(values))
                else:
                    if param_name == 'reranker_top_k':
                        max_allowed = param_info.get('max_value', max(values))
                        bo_space[param_name] = (min(values), min(max(values), max_allowed))
                    else:
                        bo_space[param_name] = (min(values), max(values))
            
            elif 'ranges' in param_info:
                all_ranges = []
                for method_ranges in param_info['ranges'].values():
                    all_ranges.extend(method_ranges)
                if all_ranges:
                    bo_space[param_name] = (min(all_ranges), max(all_ranges))
        
        filter_config = unified_space.get('passage_filter_method')
        if filter_config and filter_config['type'] == 'categorical':
            
            threshold_info = unified_space.get('threshold', {})
            if threshold_info and 'method_values' in threshold_info:
                method_values = threshold_info['method_values']
                for method, values in method_values.items():
                    method_key = f"{method}_threshold"
                    values_list = self._ensure_list(values)
                    
                    if len(values_list) == 0:
                        continue
                    elif len(values_list) == 1:
                        val = values_list[0]
                        min_val = max(0.0, val - 0.2)
                        max_val = min(1.0, val + 0.2)
                        bo_space[method_key] = (min_val, max_val)
                    else:
                        bo_space[method_key] = (min(values_list), max(values_list))

            percentile_info = unified_space.get('percentile', {})
            if percentile_info and 'method_values' in percentile_info:
                method_values = percentile_info['method_values']
                for method, values in method_values.items():
                    method_key = f"{method}_percentile"
                    values_list = self._ensure_list(values)
                    
                    if len(values_list) == 0:
                        continue
                    elif len(values_list) == 1:
                        val = values_list[0]
                        min_val = max(0.0, val - 0.2)
                        max_val = min(1.0, val + 0.2)
                        bo_space[method_key] = (min_val, max_val)
                    else:
                        bo_space[method_key] = (min(values_list), max(values_list))
        
        return bo_space

    
    def _add_compressor_additional_params(self, bo_space: Dict[str, Any], 
                                 unified_space: Dict[str, Any], 
                                 compressor_configs: List[str]):

        active_methods = [config for config in compressor_configs if config != 'pass_compressor']
        if not active_methods:
            return

        comp_params = self.config_generator.extract_unified_parameters('passage_compressor')

        needs_temperature = False
        temp_values = set()
        
        for comp_config in comp_params.get('compressor_configs', []):
            if 'temperatures' in comp_config:
                temp_values.update(comp_config['temperatures'])
                needs_temperature = True
        
        if needs_temperature and temp_values:
            if len(temp_values) == 1:
                temp_val = list(temp_values)[0]
                bo_space['compressor_temperature'] = (max(0.0, temp_val - 0.2), min(1.0, temp_val + 0.2))
            else:
                bo_space['compressor_temperature'] = (min(temp_values), max(temp_values))
        elif needs_temperature:
            bo_space['compressor_temperature'] = (0.0, 1.0)

        needs_max_tokens = any('sap_api' in config for config in active_methods)
        max_token_values = set()
        
        for comp_config in comp_params.get('compressor_configs', []):
            if 'max_tokens' in comp_config:
                max_token_values.update(comp_config['max_tokens'])
                needs_max_tokens = True
        
        if needs_max_tokens and max_token_values:
            if len(max_token_values) == 1:
                token_val = list(max_token_values)[0]
                bo_space['compressor_max_tokens'] = (max(100, token_val - 200), token_val + 200)
            else:
                bo_space['compressor_max_tokens'] = (min(max_token_values), max(max_token_values))
        elif needs_max_tokens:
            bo_space['compressor_max_tokens'] = (100, 1000)

        if 'compressor_batch' not in bo_space:
            batch_values = set()
            for comp_config in comp_params.get('compressor_configs', []):
                if 'batch' in comp_config:
                    batch_values.add(comp_config['batch'])
            
            if batch_values:
                if len(batch_values) == 1:
                    bo_space['compressor_batch'] = list(batch_values)
                else:
                    bo_space['compressor_batch'] = list(batch_values)
            else:
                bo_space['compressor_batch'] = [8, 16, 32]
    
    def _ensure_retrieval_params_in_space(self, bo_space: Dict[str, Any], unified_space: Dict[str, Any]):

        if 'retriever_top_k' in unified_space and 'retriever_top_k' not in bo_space:
            top_k_info = unified_space['retriever_top_k']
            values = top_k_info.get('values', [])
            if len(values) >= 2:
                bo_space['retriever_top_k'] = (min(values), max(values))
            elif len(values) == 1:
                bo_space['retriever_top_k'] = self._get_bo_range_for_single_value('retriever_top_k', values[0], 'int')
        
        if 'query_expansion_config' not in bo_space and 'query_expansion_config' in unified_space:
            param_info = unified_space['query_expansion_config']
            if param_info.get('values'):
                bo_space['query_expansion_config'] = param_info['values']
        
        qe_params = self.config_generator._extract_query_expansion_unified()
        retrieval_options = qe_params.get('retrieval_options', {})
        
        methods = retrieval_options.get('methods', [])
        if methods and 'query_expansion_retrieval_method' not in bo_space:
            bo_space['query_expansion_retrieval_method'] = methods

        if 'bm25' in methods and retrieval_options.get('bm25_tokenizers'):
            if 'query_expansion_bm25_tokenizer' not in bo_space:
                bo_space['query_expansion_bm25_tokenizer'] = retrieval_options['bm25_tokenizers']
        if 'vectordb' in methods and retrieval_options.get('vectordb_names'):
            if 'query_expansion_vectordb_name' not in bo_space:
                bo_space['query_expansion_vectordb_name'] = retrieval_options['vectordb_names']

        if 'retrieval_method' not in bo_space:
            retrieval_params = self.config_generator._extract_retrieval_unified()
            methods = retrieval_params.get('methods', [])
            if methods:
                bo_space['retrieval_method'] = methods

            if retrieval_params.get('bm25_tokenizers'):
                bo_space['bm25_tokenizer'] = retrieval_params['bm25_tokenizers']
            if retrieval_params.get('vectordb_names'):
                bo_space['vectordb_name'] = retrieval_params['vectordb_names']
