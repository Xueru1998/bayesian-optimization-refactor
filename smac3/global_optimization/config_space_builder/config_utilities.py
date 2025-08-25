from typing import Dict, Any


class ConfigUtilities:
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = config.copy()
        
        params_to_remove = [
            'retrieval_config',
            'query_expansion_config', 
            'passage_filter_config',
            'compressor_config',
            'prompt_config'
        ]
        
        for param in params_to_remove:
            cleaned.pop(param, None)

        if 'reranker_config' in cleaned:
            reranker_config_str = cleaned.pop('reranker_config')
            parsed_config = self._parse_reranker_config(reranker_config_str)
            cleaned.update(parsed_config)
        
        if 'compressor_llm_model' in cleaned:
            llm_model = cleaned.pop('compressor_llm_model')
            if '_' in llm_model:
                llm, model = llm_model.split('_', 1)
                cleaned['compressor_llm'] = llm
                cleaned['compressor_model'] = model
                
        if 'passage_compressor_method' in cleaned:
            if cleaned['passage_compressor_method'] not in ['lexrank', 'spacy']:
                cleaned.pop('compression_ratio', None)
                cleaned.pop('compressor_compression_ratio', None)
                
        return cleaned
    
    def _parse_reranker_config(self, reranker_config_str: str) -> Dict[str, Any]:
        if not reranker_config_str:
            return {}

        simple_methods = ['pass_reranker', 'upr', 'colbert_reranker']
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
                result = {'passage_reranker_method': method}

                if method == 'flashrank_reranker':
                    result['reranker_model'] = model_name
                else:
                    result['reranker_model_name'] = model_name
                
                return result

        return {'passage_reranker_method': reranker_config_str}
    
    def get_search_space_info(self, unified_space: Dict[str, Any]) -> Dict[str, Any]:
        n_hyperparameters = len(unified_space)
        n_categorical = 0
        n_continuous = 0
        n_integer = 0
        
        for param_name, param_info in unified_space.items():
            param_type = param_info.get('type', '')
            if param_type == 'categorical':
                n_categorical += 1
            elif param_type == 'float':
                n_continuous += 1
            elif param_type == 'int':
                n_integer += 1
        
        total_combinations = 1
        for param_name, param_info in unified_space.items():
            if param_info.get('type') == 'categorical':
                n_values = len(param_info.get('values', []))
                if n_values > 0:
                    total_combinations *= n_values
            elif param_info.get('type') == 'int':
                values = param_info.get('values', [])
                if len(values) == 2:
                    total_combinations *= (values[1] - values[0] + 1)
                elif len(values) > 2:
                    total_combinations *= len(values)
        
        return {
            'n_hyperparameters': n_hyperparameters,
            'n_categorical': n_categorical,
            'n_continuous': n_continuous,
            'n_integer': n_integer,
            'total_combinations': total_combinations,
            'hyperparameters': list(unified_space.keys())
        }