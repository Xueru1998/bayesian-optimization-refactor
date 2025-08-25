from typing import Dict, Any, List
from ConfigSpace import Categorical, Float, Integer


class ParameterFactory:
    
    def create_parameter(self, name: str, param_type: str, param_info: Dict[str, Any]):
        if param_type == 'categorical':
            values = param_info.get('values', [])
            if not values:
                return None
            return Categorical(name, values, default=self._get_default_value(name, values))
        
        elif param_type == 'int':
            values = param_info.get('values', [])
            if len(values) == 2:
                return Integer(name, bounds=(min(values), max(values)), default=min(values))
            elif len(values) > 2:
                return Categorical(name, values, default=values[0])
            else:
                return None
        
        elif param_type == 'float':
            if 'method_values' in param_info:
                all_values = self._extract_all_values(param_info['method_values'])
                if all_values:
                    return self._create_parameter_from_values(name, all_values, param_type)
            else:
                values = param_info.get('values', [])
                if len(values) == 2:
                    return Float(name, bounds=(min(values), max(values)), default=min(values))
                elif len(values) > 2:
                    return Categorical(name, values, default=values[0])
        
        return None
    
    def _extract_all_values(self, method_values: Dict[str, Any]) -> List[Any]:
        all_values = []
        for method, values in method_values.items():
            if isinstance(values, list):
                all_values.extend(values)
            else:
                all_values.append(values)
        return sorted(list(set(all_values)))
    
    def _create_parameter_from_values(self, name: str, values: List[Any], param_type: str):
        if not values:
            return None
        
        if param_type == 'float':
            if len(values) == 2:
                return Float(name, bounds=(min(values), max(values)), default=min(values))
            else:
                return Categorical(name, values, default=values[0])
        else:
            return Categorical(name, values, default=values[0])
    
    def _get_default_value(self, param_name: str, values: List[Any]) -> Any:
        default_map = {
            'retrieval_method': 'bm25',
            'bm25_tokenizer': 'space',
            'passage_filter_method': 'pass_passage_filter',
            'passage_compressor_method': 'pass_compressor',
            'passage_reranker_method': 'pass_reranker',
            'prompt_maker_method': 'fstring',
            'query_expansion_method': 'pass_query_expansion'
        }
        
        if param_name in default_map and default_map[param_name] in values:
            return default_map[param_name]
        
        return values[0] if values else None