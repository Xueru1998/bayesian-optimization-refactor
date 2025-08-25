from typing import Dict, Any


class ConfigCleaner:
    
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
            print(f"[DEBUG] Parsed reranker_config '{reranker_config_str}' to: {parsed_config}")
            
        if 'query_expansion_method' in cleaned and cleaned['query_expansion_method'] != 'pass_query_expansion':
            retrieval_params_to_remove = ['retrieval_method', 'bm25_tokenizer', 'vectordb_name']
            for param in retrieval_params_to_remove:
                if param in cleaned:
                    print(f"[DEBUG] Removing {param} because query expansion is active")
                    del cleaned[param]
        
        if 'compressor_llm_model' in cleaned:
            llm_model = cleaned.pop('compressor_llm_model')
            if '_' in llm_model:
                llm, model = llm_model.split('_', 1)
                cleaned['compressor_llm'] = llm
                cleaned['compressor_model'] = model
        
        retriever_key = None
        reranker_key = None
        
        for key in ['retriever_topk', 'retriever_top_k']:
            if key in cleaned:
                retriever_key = key
                break
        
        for key in ['reranker_topk', 'reranker_top_k']:
            if key in cleaned:
                reranker_key = key
                break
        
        if retriever_key and reranker_key:
            if cleaned[reranker_key] > cleaned[retriever_key]:
                original_value = cleaned[reranker_key]
                cleaned[reranker_key] = cleaned[retriever_key]
                print(f"[DEBUG] clean_trial_config: Adjusted {reranker_key} from {original_value} to {cleaned[retriever_key]} to not exceed {retriever_key}")
        
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