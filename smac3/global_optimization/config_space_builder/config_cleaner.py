from typing import Dict, Any


class ConfigCleaner:
    
    def __init__(self, config_generator, query_expansion_retrieval_options):
        self.config_generator = config_generator
        self.query_expansion_retrieval_options = query_expansion_retrieval_options
    
    def clean_trial_config(self, config: Dict[str, Any], config_parser) -> Dict[str, Any]:
        cleaned = config.copy()

        composite_params = {
            'query_expansion_config': config_parser.parse_query_expansion_config,
            'passage_compressor_config': config_parser.parse_compressor_config,
            'generator_config': config_parser.parse_generator_config,
            'reranker_config': config_parser.parse_reranker_config
        }
        
        for param_name, parser_func in composite_params.items():
            if param_name in cleaned:
                parsed_params = parser_func(cleaned[param_name])
                cleaned.update(parsed_params)
                del cleaned[param_name]

        method = cleaned.get('passage_reranker_method')
        if method and method != 'pass_reranker':
            model_param = f'reranker_model_{method}'
            if model_param in cleaned:
                cleaned['reranker_model_name'] = cleaned[model_param]
            else:
                from pipeline.search_space_extractor import OptimizationType
                from .smac_config_space_builder import SMACConfigSpaceBuilder
                builder = SMACConfigSpaceBuilder(self.config_generator)
                unified_params = builder.unified_extractor.extract_search_space(OptimizationType.SMAC)
                if 'reranker_model_name' in unified_params and 'method_models' in unified_params['reranker_model_name']:
                    method_models = unified_params['reranker_model_name']['method_models']
                    if method in method_models and method_models[method]:
                        cleaned['reranker_model_name'] = method_models[method][0]
                        print(f"[DEBUG] Added default reranker model for {method}: {cleaned['reranker_model_name']}")

        for key in list(cleaned.keys()):
            if key.startswith("reranker_model_") and key != "reranker_model_name":
                del cleaned[key]

        if '_metadata' in cleaned:
            del cleaned['_metadata']
        
        if 'passage_compressor_method' in cleaned:
            method = cleaned['passage_compressor_method']
            if method not in ['lexrank', 'spacy']:
                cleaned.pop('compression_ratio', None)
                cleaned.pop('compressor_compression_ratio', None)
                cleaned.pop('spacy_compression_ratio', None)

        cleaned = self._clean_query_expansion_params(cleaned)
        
        return cleaned
    
    def _clean_query_expansion_params(self, cleaned: Dict[str, Any]) -> Dict[str, Any]:
        if 'query_expansion_method' in cleaned:
            if cleaned['query_expansion_method'] == 'pass_query_expansion':
                if 'query_expansion_retrieval_method' in cleaned:
                    cleaned['retrieval_method'] = cleaned['query_expansion_retrieval_method']
                    del cleaned['query_expansion_retrieval_method']
                
                if cleaned.get('retrieval_method') == 'bm25' and 'query_expansion_bm25_tokenizer' in cleaned:
                    cleaned['bm25_tokenizer'] = cleaned['query_expansion_bm25_tokenizer']
                    del cleaned['query_expansion_bm25_tokenizer']
                elif cleaned.get('retrieval_method') == 'vectordb' and 'query_expansion_vectordb_name' in cleaned:
                    cleaned['vectordb_name'] = cleaned['query_expansion_vectordb_name']
                    del cleaned['query_expansion_vectordb_name']

                for param in ['query_expansion_temperature', 'query_expansion_max_token', 
                            'query_expansion_generator_module_type', 'query_expansion_model',
                            'query_expansion_llm', 'query_expansion_api_url', 
                            'query_expansion_bearer_token']:
                    if param in cleaned:
                        del cleaned[param]
            
            else:
                for param in ['retrieval_method', 'bm25_tokenizer', 'vectordb_name']:
                    if param in cleaned:
                        del cleaned[param]

                qe_retrieval_options = self.query_expansion_retrieval_options

                if 'query_expansion_retrieval_method' not in cleaned:
                    if qe_retrieval_options and qe_retrieval_options.get('methods'):
                        cleaned['query_expansion_retrieval_method'] = qe_retrieval_options['methods'][0]
                    else:
                        cleaned['query_expansion_retrieval_method'] = 'bm25'
                
                qe_retrieval_method = cleaned.get('query_expansion_retrieval_method')
                
                if qe_retrieval_method == 'bm25':
                    if 'query_expansion_vectordb_name' in cleaned:
                        del cleaned['query_expansion_vectordb_name']
                    
                    if 'query_expansion_bm25_tokenizer' not in cleaned:
                        if qe_retrieval_options and qe_retrieval_options.get('bm25_tokenizers'):
                            cleaned['query_expansion_bm25_tokenizer'] = qe_retrieval_options['bm25_tokenizers'][0]
                        else:
                            cleaned['query_expansion_bm25_tokenizer'] = 'space'
                
                elif qe_retrieval_method == 'vectordb':
                    if 'query_expansion_bm25_tokenizer' in cleaned:
                        del cleaned['query_expansion_bm25_tokenizer']
                    
                    if 'query_expansion_vectordb_name' not in cleaned:
                        if qe_retrieval_options and qe_retrieval_options.get('vectordb_names'):
                            cleaned['query_expansion_vectordb_name'] = qe_retrieval_options['vectordb_names'][0]
                        else:
                            cleaned['query_expansion_vectordb_name'] = 'default'
        
        return cleaned