from typing import Dict, Any, List


class ParameterValidator:
    def __init__(self, search_space: Dict[str, Any]):
        self.search_space = search_space
        self.valid_param_combinations = []
    
    def _precompute_valid_param_combinations(self):
        self.valid_param_combinations = []
        
        qe_methods = self.search_space.get('query_expansion_method', ['pass_query_expansion'])
        qe_models = self.search_space.get('query_expansion_model', [])
        qe_max_tokens = self.search_space.get('query_expansion_max_token', [])
        qe_temperatures = self.search_space.get('query_expansion_temperature', [])
        
        qe_retrieval_methods = self.search_space.get('query_expansion_retrieval_method', [])
        qe_bm25_tokenizers = self.search_space.get('query_expansion_bm25_tokenizer', [])
        qe_vectordb_names = self.search_space.get('query_expansion_vectordb_name', [])
        
        retrieval_methods = self.search_space.get('retrieval_method', [])
        bm25_tokenizers = self.search_space.get('bm25_tokenizer', [])
        vectordb_names = self.search_space.get('vectordb_name', [])
        
        retriever_top_k_values = self.search_space.get('retriever_top_k', [10])
        
        print(f"[DEBUG] Starting combination generation:")
        print(f"  QE methods: {qe_methods}")
        print(f"  QE retrieval methods: {qe_retrieval_methods}")
        print(f"  Retrieval methods: {retrieval_methods}")
        print(f"  Top-K values: {retriever_top_k_values}")
        
        for top_k in retriever_top_k_values:
            for qe_method in qe_methods:
                if qe_method == 'pass_query_expansion':
                    for retrieval_method in retrieval_methods:
                        base_params = {
                            'query_expansion_method': qe_method,
                            'retrieval_method': retrieval_method,
                            'retriever_top_k': top_k
                        }
                        
                        if retrieval_method == 'bm25':
                            for tokenizer in bm25_tokenizers:
                                params = base_params.copy()
                                params['bm25_tokenizer'] = tokenizer
                                self._add_other_component_combinations(params)
                        elif retrieval_method == 'vectordb':
                            for vdb_name in vectordb_names:
                                params = base_params.copy()
                                params['vectordb_name'] = vdb_name
                                self._add_other_component_combinations(params)
                        else:
                            self._add_other_component_combinations(base_params)
                else:
                    for qe_retrieval_method in qe_retrieval_methods:
                        base_qe_params = {
                            'query_expansion_method': qe_method,
                            'query_expansion_retrieval_method': qe_retrieval_method,
                            'retriever_top_k': top_k
                        }
                        
                        if qe_method in ['query_decompose', 'hyde'] and qe_models:
                            for model in qe_models:
                                model_params = base_qe_params.copy()
                                model_params['query_expansion_model'] = model
                                
                                if qe_method == 'hyde' and qe_max_tokens:
                                    for max_token in qe_max_tokens:
                                        hyde_params = model_params.copy()
                                        hyde_params['query_expansion_max_token'] = max_token
                                        self._add_qe_retrieval_params(hyde_params, qe_retrieval_method, 
                                                                    qe_bm25_tokenizers, qe_vectordb_names)
                                else:
                                    self._add_qe_retrieval_params(model_params, qe_retrieval_method,
                                                                qe_bm25_tokenizers, qe_vectordb_names)
                                    
                        elif qe_method == 'multi_query_expansion':
                            if qe_models and qe_temperatures:
                                for model in qe_models:
                                    for temp in qe_temperatures:
                                        temp_params = base_qe_params.copy()
                                        temp_params['query_expansion_model'] = model
                                        temp_params['query_expansion_temperature'] = temp
                                        self._add_qe_retrieval_params(temp_params, qe_retrieval_method,
                                                                    qe_bm25_tokenizers, qe_vectordb_names)
                            else:
                                self._add_qe_retrieval_params(base_qe_params, qe_retrieval_method,
                                                            qe_bm25_tokenizers, qe_vectordb_names)
                        else:
                            self._add_qe_retrieval_params(base_qe_params, qe_retrieval_method,
                                                        qe_bm25_tokenizers, qe_vectordb_names)
        
        print(f"[DEBUG] Total valid combinations generated: {len(self.valid_param_combinations)}")

    def _add_other_component_combinations(self, base_params):
        filter_configs = self.search_space.get('passage_filter_config', ['pass_passage_filter'])
        reranker_top_k_values = self.search_space.get('reranker_top_k', [])
        prompt_configs = self.search_space.get('prompt_config', self.search_space.get('prompt_maker_config', []))
        generator_models = self.search_space.get('generator_model', [])
        generator_temps = self.search_space.get('generator_temperature', [])
        
        for filter_config in filter_configs:
            filter_params = base_params.copy()
            filter_params['passage_filter_config'] = filter_config

            reranker_configs = self.search_space.get('reranker_config', [])
            
            if reranker_configs:
                for reranker_config in reranker_configs:
                    if reranker_top_k_values:
                        for reranker_top_k in reranker_top_k_values:
                            if reranker_top_k <= filter_params['retriever_top_k']:
                                reranker_params = filter_params.copy()
                                reranker_params['reranker_config'] = reranker_config
                                reranker_params['reranker_top_k'] = reranker_top_k
                                self._add_prompt_and_generator_combinations(reranker_params, prompt_configs, generator_models, generator_temps)
                    else:
                        reranker_params = filter_params.copy()
                        reranker_params['reranker_config'] = reranker_config
                        self._add_prompt_and_generator_combinations(reranker_params, prompt_configs, generator_models, generator_temps)
            else:
                self._add_prompt_and_generator_combinations(filter_params, prompt_configs, generator_models, generator_temps)
                
    def _add_qe_retrieval_params(self, base_params, qe_retrieval_method, 
                            qe_bm25_tokenizers, qe_vectordb_names):
        if qe_retrieval_method == 'bm25' and qe_bm25_tokenizers:
            for tokenizer in qe_bm25_tokenizers:
                params = base_params.copy()
                params['query_expansion_bm25_tokenizer'] = tokenizer
                self._add_other_component_combinations(params)
        elif qe_retrieval_method == 'vectordb' and qe_vectordb_names:
            for vdb_name in qe_vectordb_names:
                params = base_params.copy()
                params['query_expansion_vectordb_name'] = vdb_name
                self._add_other_component_combinations(params)
        else:
            self._add_other_component_combinations(base_params)
                
    def _add_prompt_and_generator_combinations(self, base_params, prompt_configs, generator_models, generator_temps):
        if prompt_configs and generator_models:
            for prompt_config in prompt_configs:
                for generator_model in generator_models:
                    for generator_temp in generator_temps:
                        final_params = base_params.copy()
                        final_params['prompt_config'] = prompt_config
                        final_params['generator_model'] = generator_model
                        final_params['generator_temperature'] = generator_temp
                        self.valid_param_combinations.append(final_params)
        elif generator_models:
            for generator_model in generator_models:
                for generator_temp in generator_temps:
                    final_params = base_params.copy()
                    final_params['generator_model'] = generator_model
                    final_params['generator_temperature'] = generator_temp
                    self.valid_param_combinations.append(final_params)
        elif prompt_configs:
            for prompt_config in prompt_configs:
                final_params = base_params.copy()
                final_params['prompt_config'] = prompt_config
                self.valid_param_combinations.append(final_params)
        else:
            self.valid_param_combinations.append(base_params)
    
    def _add_qe_retrieval_combinations(self, base_params):
        qe_retrieval_methods = self.search_space.get('query_expansion_retrieval_method', ['bm25'])
        
        for method in qe_retrieval_methods:
            method_params = base_params.copy()
            method_params['query_expansion_retrieval_method'] = method
            
            if method == 'bm25':
                tokenizers = self.search_space.get('query_expansion_bm25_tokenizer', [])
                for tokenizer in tokenizers:
                    tokenizer_params = method_params.copy()
                    tokenizer_params['query_expansion_bm25_tokenizer'] = tokenizer
                    self._add_other_component_combinations(tokenizer_params)
            elif method == 'vectordb':
                vdb_names = self.search_space.get('query_expansion_vectordb_name', [])
                for vdb_name in vdb_names:
                    vdb_params = method_params.copy()
                    vdb_params['query_expansion_vectordb_name'] = vdb_name
                    self._add_other_component_combinations(vdb_params)