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
    
    def extract_search_space(self) -> Dict[str, Union[List[Any], Tuple[float, float]]]:
        if self.search_type == 'grid':
            return self.unified_extractor.extract_search_space('optuna_grid')
        else:
            unified_space = self.unified_extractor._extract_unified_space()
            bo_space = self._convert_to_optuna_bo_space(unified_space)

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

        has_active_query_expansion = False
        if 'query_expansion_method' in unified_space:
            qe_methods = unified_space['query_expansion_method'].get('values', [])
            non_pass_methods = [m for m in qe_methods if m != 'pass_query_expansion']
            if non_pass_methods:
                has_active_query_expansion = True
        
        for param_name, param_info in unified_space.items():
            if has_active_query_expansion and param_name in ['retrieval_method', 'bm25_tokenizer', 'vectordb_name']:
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
                    if self.search_type == 'grid':
                        bo_space[param_name] = values
                    else:
                        if 'top_k' in param_name:
                            value = values[0]
                            min_val = max(1, int(value * 0.5))
                            max_val = int(value * 1.5)
                            bo_space[param_name] = (min_val, max_val)
                        else:
                            min_val, max_val = self._get_bo_range_for_single_value(param_name, values[0], param_type)
                            bo_space[param_name] = (min_val, max_val)
                elif len(values) == 2:
                    if (('top_k' in param_name or 'max_token' in param_name) and 
                        all(isinstance(v, int) for v in values) and 
                        values[1] > values[0]):
                        min_val, max_val = min(values), max(values)
                        
                        if self.search_type == 'grid':
                            bo_space[param_name] = list(range(min_val, max_val + 1))
                        else:

                            bo_space[param_name] = list(range(min_val, max_val + 1))
                    else:
                        if self.search_type == 'grid':
                            bo_space[param_name] = values
                        else:
                            if param_type == 'int':
                                bo_space[param_name] = (int(min(values)), int(max(values)))
                            else:
                                bo_space[param_name] = (float(min(values)), float(max(values)))
                else:
                    if self.search_type == 'grid':
                        bo_space[param_name] = values
                    else:
                        if 'top_k' in param_name or 'max_token' in param_name:
                            bo_space[param_name] = values
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
                    elif self.search_type == 'grid':
                        bo_space[method_key] = values_list
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
                    elif self.search_type == 'grid':
                        bo_space[method_key] = values_list
                    elif len(values_list) == 1:
                        val = values_list[0]
                        min_val = max(0.0, val - 0.2)
                        max_val = min(1.0, val + 0.2)
                        bo_space[method_key] = (min_val, max_val)
                    else:
                        bo_space[method_key] = (min(values_list), max(values_list))
            
        return bo_space