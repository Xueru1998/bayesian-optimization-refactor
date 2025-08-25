from typing import Dict, Any, List
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, InCondition, Constant, EqualsCondition, AndConjunction


class ComponentBuilders:
    def __init__(self, parent):
        self.parent = parent
    
    def build_retrieval_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()
        
        if 'retriever_top_k' in unified_space and 'retriever_top_k' not in fixed_params:
            top_k_info = unified_space['retriever_top_k']
            top_k_param = self.parent._create_parameter('retriever_top_k', top_k_info['type'], top_k_info)
            if top_k_param:
                cs.add(top_k_param)
        
        if 'retrieval_method' in unified_space and 'retrieval_method' not in fixed_params:
            method_info = unified_space['retrieval_method']
            method_param = Categorical('retrieval_method', method_info['values'], 
                                     default=self.parent._get_default_value('retrieval_method', method_info['values']))
            cs.add(method_param)
        
        if 'bm25_tokenizer' in unified_space and 'bm25_tokenizer' not in fixed_params:
            tokenizer_info = unified_space['bm25_tokenizer']
            tokenizer_param = Categorical('bm25_tokenizer', tokenizer_info['values'],
                                        default=self.parent._get_default_value('bm25_tokenizer', tokenizer_info['values']))
            cs.add(tokenizer_param)
            
            if 'retrieval_method' in cs:
                cs.add(EqualsCondition(cs['bm25_tokenizer'], cs['retrieval_method'], 'bm25'))
        
        if 'vectordb_name' in unified_space and 'vectordb_name' not in fixed_params:
            vdb_info = unified_space['vectordb_name']
            vdb_param = Categorical('vectordb_name', vdb_info['values'],
                                  default=self.parent._get_default_value('vectordb_name', vdb_info['values']))
            cs.add(vdb_param)
            
            if 'retrieval_method' in cs:
                cs.add(EqualsCondition(cs['vectordb_name'], cs['retrieval_method'], 'vectordb'))
        print(f"[DEBUG] All retrieval_method choices in ConfigSpace: {cs['retrieval_method'].choices}")
        
        return cs
    
    def build_filter_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()
        
        if 'passage_filter_method' in unified_space and 'passage_filter_method' not in fixed_params:
            method_info = unified_space['passage_filter_method']
            method_param = Categorical('passage_filter_method', method_info['values'],
                                     default=self.parent._get_default_value('passage_filter_method', method_info['values']))
            cs.add(method_param)
        
        if 'threshold' in unified_space and 'threshold' not in fixed_params:
            threshold_info = unified_space['threshold']
            if 'method_values' in threshold_info:
                all_values = self.parent._extract_all_values(threshold_info['method_values'])
                if all_values:
                    param = self.parent._create_parameter_from_values('threshold', all_values, threshold_info['type'])
                    if param:
                        cs.add(param)
                        if threshold_info.get('condition') and 'passage_filter_method' in cs:
                            self.parent._add_single_condition(cs, 'threshold', threshold_info['condition'])
        
        if 'percentile' in unified_space and 'percentile' not in fixed_params:
            percentile_info = unified_space['percentile']
            if 'method_values' in percentile_info:
                all_values = self.parent._extract_all_values(percentile_info['method_values'])
                if all_values:
                    param = self.parent._create_parameter_from_values('percentile', all_values, percentile_info['type'])
                    if param:
                        cs.add(param)
                        if percentile_info.get('condition') and 'passage_filter_method' in cs:
                            self.parent._add_single_condition(cs, 'percentile', percentile_info['condition'])
        
        return cs
    
    def build_reranker_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()
        
        if 'passage_reranker_method' in unified_space and 'passage_reranker_method' not in fixed_params:
            unified_params = self.parent.config_generator.extract_unified_parameters('passage_reranker')
            models_by_method = unified_params.get('models', {})

            method_model_combinations = []
            
            for method in unified_space['passage_reranker_method']['values']:
                if method == 'pass_reranker':
                    method_model_combinations.append('pass_reranker')
                elif method == 'sap_api':
                    method_model_combinations.append('sap_api')
                elif method in models_by_method and models_by_method[method]:
                    for model in models_by_method[method]:
                        method_model_combinations.append(f"{method}::{model}")
                else:
                    method_model_combinations.append(method)

            reranker_config_param = Categorical('reranker_config', method_model_combinations,
                                            default=method_model_combinations[0])
            cs.add(reranker_config_param)

            if 'reranker_top_k' in unified_space and 'reranker_top_k' not in fixed_params:
                top_k_info = unified_space['reranker_top_k']
                prev_top_k = fixed_params.get('retriever_top_k', top_k_info.get('max_value', 10))
                
                values = top_k_info['values']
                if isinstance(values, list):
                    if len(values) == 2:
                        min_k, max_k = min(values), max(values)
                        max_k = min(max_k, prev_top_k)
                        if min_k <= max_k:
                            top_k_param = Integer('reranker_top_k', bounds=(min_k, max_k), default=min_k)
                        else:
                            top_k_param = Constant('reranker_top_k', min(values[0], prev_top_k))
                    else:
                        valid_values = [v for v in values if v <= prev_top_k]
                        if valid_values:
                            if len(valid_values) == 1:
                                top_k_param = Constant('reranker_top_k', valid_values[0])
                            else:
                                top_k_param = Categorical('reranker_top_k', valid_values, default=valid_values[0])
                        else:
                            top_k_param = Constant('reranker_top_k', min(values[0], prev_top_k))
                else:
                    top_k_param = Integer('reranker_top_k', bounds=(1, prev_top_k), default=min(5, prev_top_k))
                
                cs.add(top_k_param)

                non_pass_configs = [c for c in method_model_combinations if c != 'pass_reranker']
                if non_pass_configs:
                    cs.add(InCondition(cs['reranker_top_k'], cs['reranker_config'], non_pass_configs))
        
        return cs
    
    def build_generator_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()
        
        if 'generator_config' in unified_space and 'generator_config' not in fixed_params:
            config_info = unified_space['generator_config']
            config_param = Categorical('generator_config', config_info['values'],
                                     default=config_info['values'][0])
            cs.add(config_param)
            
            if 'generator_temperature' in unified_space:
                temp_info = unified_space['generator_temperature']
                temp_param = self.parent._create_parameter('generator_temperature', temp_info['type'], temp_info)
                if temp_param:
                    cs.add(temp_param)
            
            if 'generator_max_tokens' in unified_space:
                token_info = unified_space['generator_max_tokens']
                if token_info.get('condition'):
                    token_param = self.parent._create_parameter('generator_max_tokens', token_info['type'], token_info)
                    if token_param:
                        cs.add(token_param)
        else:
            gen_params = ['generator_model', 'generator_temperature', 'generator_module_type', 'generator_llm']
            
            for param_name in gen_params:
                if param_name in unified_space and param_name not in fixed_params:
                    param_info = unified_space[param_name]
                    param = self.parent._create_parameter(param_name, param_info['type'], param_info)
                    if param:
                        cs.add(param)
        
        return cs
    
    def build_prompt_generator_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()
        
        if 'prompt_maker_method' in unified_space and 'prompt_maker_method' not in fixed_params:
            method_info = unified_space['prompt_maker_method']
            method_param = Categorical('prompt_maker_method', method_info['values'],
                                     default=self.parent._get_default_value('prompt_maker_method', method_info['values']))
            cs.add(method_param)
        
        if 'prompt_template_idx' in unified_space and 'prompt_template_idx' not in fixed_params:
            idx_info = unified_space['prompt_template_idx']
            idx_param = self.parent._create_parameter('prompt_template_idx', idx_info['type'], idx_info)
            if idx_param:
                cs.add(idx_param)
                if idx_info.get('condition') and 'prompt_maker_method' in cs:
                    self.parent._add_single_condition(cs, 'prompt_template_idx', idx_info['condition'])
        
        self.build_generator_space(cs, fixed_params)
        
        return cs
    
    # Query Expansion Methods
    def build_query_expansion_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()

        if 'query_expansion_config' in unified_space and 'query_expansion_config' not in fixed_params:
            config_info = unified_space['query_expansion_config']
            config_values = config_info['values']

            pass_configs = [v for v in config_values if v == 'pass_query_expansion']
            active_configs = [v for v in config_values if v != 'pass_query_expansion']
            
            print(f"[DEBUG] Query expansion configs - Pass: {len(pass_configs)}, Active: {len(active_configs)}")
            
            if len(active_configs) == 0:
                print(f"[DEBUG] Only pass_query_expansion available, building retrieval-only space")
                return self._build_pass_query_expansion_space(cs, fixed_params)
            
            elif len(pass_configs) > 0 and len(active_configs) > 0:
                print(f"[DEBUG] Mixed pass/active query expansion, building separate spaces")
                return self._build_mixed_query_expansion_space(cs, fixed_params, pass_configs, active_configs)
            
            else:
                print(f"[DEBUG] Only active query expansion methods, building full space")
                config_param = Categorical('query_expansion_config', config_values,
                                         default=config_values[0])
                cs.add(config_param)

                self._add_active_query_expansion_params(cs, unified_space, fixed_params)
        
        else:
            return self._build_query_expansion_space_legacy(cs, fixed_params)
        
        return cs
    
    def _build_pass_query_expansion_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        cs.add(Constant('query_expansion_config', 'pass_query_expansion'))

        unified_space = self.parent.get_unified_space()

        if 'retriever_top_k' in unified_space and 'retriever_top_k' not in fixed_params:
            top_k_info = unified_space['retriever_top_k']
            top_k_param = self.parent._create_parameter('retriever_top_k', top_k_info['type'], top_k_info)
            if top_k_param:
                cs.add(top_k_param)
  
        if 'retrieval_method' in unified_space and 'retrieval_method' not in fixed_params:
            method_info = unified_space['retrieval_method']
            method_param = Categorical('retrieval_method', method_info['values'], 
                                     default=self.parent._get_default_value('retrieval_method', method_info['values']))
            cs.add(method_param)

        if 'bm25_tokenizer' in unified_space and 'bm25_tokenizer' not in fixed_params:
            tokenizer_info = unified_space['bm25_tokenizer']
            tokenizer_param = Categorical('bm25_tokenizer', tokenizer_info['values'],
                                        default=self.parent._get_default_value('bm25_tokenizer', tokenizer_info['values']))
            cs.add(tokenizer_param)

            if 'retrieval_method' in cs:
                cs.add(EqualsCondition(cs['bm25_tokenizer'], cs['retrieval_method'], 'bm25'))

        if 'vectordb_name' in unified_space and 'vectordb_name' not in fixed_params:
            vdb_info = unified_space['vectordb_name']
            vdb_param = Categorical('vectordb_name', vdb_info['values'],
                                  default=self.parent._get_default_value('vectordb_name', vdb_info['values']))
            cs.add(vdb_param)

            if 'retrieval_method' in cs:
                cs.add(EqualsCondition(cs['vectordb_name'], cs['retrieval_method'], 'vectordb'))
        
        print(f"[DEBUG] Pass query expansion space created with {len(cs.get_hyperparameters())} parameters")
        return cs
    
    def _build_mixed_query_expansion_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any],
                                          pass_configs: List[str], active_configs: List[str]) -> ConfigurationSpace:
        all_configs = pass_configs + active_configs
        config_param = Categorical('query_expansion_config', all_configs, default=all_configs[0])
        cs.add(config_param)
        
        unified_space = self.parent.get_unified_space()

        if 'retriever_top_k' in unified_space and 'retriever_top_k' not in fixed_params:
            top_k_info = unified_space['retriever_top_k']
            top_k_param = self.parent._create_parameter('retriever_top_k', top_k_info['type'], top_k_info)
            if top_k_param:
                cs.add(top_k_param)

        if 'retrieval_method' in unified_space and 'retrieval_method' not in fixed_params:
            method_info = unified_space['retrieval_method']
            method_param = Categorical('retrieval_method', method_info['values'], 
                                     default=self.parent._get_default_value('retrieval_method', method_info['values']))
            cs.add(method_param)
            cs.add(EqualsCondition(cs['retrieval_method'], cs['query_expansion_config'], 'pass_query_expansion'))
        
        if 'bm25_tokenizer' in unified_space and 'bm25_tokenizer' not in fixed_params:
            tokenizer_info = unified_space['bm25_tokenizer']
            tokenizer_param = Categorical('bm25_tokenizer', tokenizer_info['values'],
                                        default=self.parent._get_default_value('bm25_tokenizer', tokenizer_info['values']))
            cs.add(tokenizer_param)

            if 'retrieval_method' in cs:
                cs.add(AndConjunction(
                    EqualsCondition(cs['bm25_tokenizer'], cs['query_expansion_config'], 'pass_query_expansion'),
                    EqualsCondition(cs['bm25_tokenizer'], cs['retrieval_method'], 'bm25')
                ))
        
        if 'vectordb_name' in unified_space and 'vectordb_name' not in fixed_params:
            vdb_info = unified_space['vectordb_name']
            vdb_param = Categorical('vectordb_name', vdb_info['values'],
                                  default=self.parent._get_default_value('vectordb_name', vdb_info['values']))
            cs.add(vdb_param)

            if 'retrieval_method' in cs:
                cs.add(AndConjunction(
                    EqualsCondition(cs['vectordb_name'], cs['query_expansion_config'], 'pass_query_expansion'),
                    EqualsCondition(cs['vectordb_name'], cs['retrieval_method'], 'vectordb')
                ))

        if active_configs:
            if 'query_expansion_retrieval_method' in unified_space:
                qe_method_info = unified_space['query_expansion_retrieval_method']
                qe_method_param = Categorical('query_expansion_retrieval_method', qe_method_info['values'],
                                            default=qe_method_info['values'][0])
                cs.add(qe_method_param)

                cs.add(InCondition(cs['query_expansion_retrieval_method'], 
                                 cs['query_expansion_config'], 
                                 active_configs))

            self._add_active_query_expansion_params(cs, unified_space, fixed_params, active_configs)
        
        print(f"[DEBUG] Mixed query expansion space created with {len(cs.get_hyperparameters())} parameters")
        return cs
    
    def _add_active_query_expansion_params(self, cs: ConfigurationSpace, unified_space: Dict[str, Any], 
                                          fixed_params: Dict[str, Any], active_configs: List[str] = None):
        if 'query_expansion_temperature' in unified_space:
            temp_info = unified_space['query_expansion_temperature']
            temp_param = self.parent._create_parameter('query_expansion_temperature', temp_info['type'], temp_info)
            if temp_param:
                cs.add(temp_param)

                multi_query_configs = [c for c in (active_configs or []) if 'multi_query' in c]
                if multi_query_configs and 'query_expansion_config' in cs:
                    cs.add(InCondition(cs['query_expansion_temperature'], 
                                     cs['query_expansion_config'], 
                                     multi_query_configs))

        if 'query_expansion_max_token' in unified_space:
            token_info = unified_space['query_expansion_max_token']
            token_param = self.parent._create_parameter('query_expansion_max_token', token_info['type'], token_info)
            if token_param:
                cs.add(token_param)

                hyde_configs = [c for c in (active_configs or []) if 'hyde' in c]
                if hyde_configs and 'query_expansion_config' in cs:
                    cs.add(InCondition(cs['query_expansion_max_token'], 
                                     cs['query_expansion_config'], 
                                     hyde_configs))

        if 'query_expansion_bm25_tokenizer' in unified_space and 'query_expansion_bm25_tokenizer' not in fixed_params:
            tokenizer_info = unified_space['query_expansion_bm25_tokenizer']
            tokenizer_param = Categorical('query_expansion_bm25_tokenizer', tokenizer_info['values'],
                                        default=tokenizer_info['values'][0])
            cs.add(tokenizer_param)

            if 'query_expansion_retrieval_method' in cs:
                cs.add(EqualsCondition(cs['query_expansion_bm25_tokenizer'], 
                                     cs['query_expansion_retrieval_method'], 
                                     'bm25'))
        
        if 'query_expansion_vectordb_name' in unified_space and 'query_expansion_vectordb_name' not in fixed_params:
            vdb_info = unified_space['query_expansion_vectordb_name']
            vdb_param = Categorical('query_expansion_vectordb_name', vdb_info['values'],
                                  default=vdb_info['values'][0])
            cs.add(vdb_param)

            if 'query_expansion_retrieval_method' in cs:
                cs.add(EqualsCondition(cs['query_expansion_vectordb_name'], 
                                     cs['query_expansion_retrieval_method'], 
                                     'vectordb'))
    
    def _build_query_expansion_space_legacy(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()
        
        if 'query_expansion_method' in unified_space and 'query_expansion_method' not in fixed_params:
            method_info = unified_space['query_expansion_method']
            method_param = Categorical('query_expansion_method', method_info['values'],
                                     default=self.parent._get_default_value('query_expansion_method', method_info['values']))
            cs.add(method_param)
        
        if 'query_expansion_model' in unified_space and 'query_expansion_model' not in fixed_params:
            model_info = unified_space['query_expansion_model']
            model_param = Categorical('query_expansion_model', model_info['values'],
                                    default=model_info['values'][0])
            cs.add(model_param)
            
            if model_info.get('condition'):
                self.parent._add_single_condition(cs, 'query_expansion_model', model_info['condition'])
        
        self._add_query_expansion_retrieval_params(cs, unified_space, fixed_params)
        
        return cs
    
    def _add_query_expansion_retrieval_params(self, cs: ConfigurationSpace, unified_space: Dict[str, Any], 
                                            fixed_params: Dict[str, Any]):
        if 'query_expansion_retrieval_method' in unified_space and 'query_expansion_retrieval_method' not in fixed_params:
            method_info = unified_space['query_expansion_retrieval_method']
            method_param = Categorical('query_expansion_retrieval_method', method_info['values'],
                                     default=method_info['values'][0])
            print(f"[DEBUG] query_expansion_retrieval_method values: {method_info.get('values', [])}")
            
            cs.add(method_param)
            
            if method_info.get('condition'):
                self.parent._add_single_condition(cs, 'query_expansion_retrieval_method', method_info['condition'])
        
        if 'query_expansion_bm25_tokenizer' in unified_space and 'query_expansion_bm25_tokenizer' not in fixed_params:
            tokenizer_info = unified_space['query_expansion_bm25_tokenizer']
            print(f"[DEBUG] query_expansion_bm25_tokenizer values: {tokenizer_info.get('values', [])}")
            print(f"[DEBUG] query_expansion_bm25_tokenizer condition: {tokenizer_info.get('condition')}")
            tokenizer_param = Categorical('query_expansion_bm25_tokenizer', tokenizer_info['values'],
                                        default=tokenizer_info['values'][0])
            cs.add(tokenizer_param)
            
            if tokenizer_info.get('condition'):
                self.parent._add_single_condition(cs, 'query_expansion_bm25_tokenizer', tokenizer_info['condition'])
        
        if 'query_expansion_vectordb_name' in unified_space and 'query_expansion_vectordb_name' not in fixed_params:
            vdb_info = unified_space['query_expansion_vectordb_name']
            vdb_param = Categorical('query_expansion_vectordb_name', vdb_info['values'],
                                  default=vdb_info['values'][0])
            cs.add(vdb_param)
            
            if vdb_info.get('condition'):
                self.parent._add_single_condition(cs, 'query_expansion_vectordb_name', vdb_info['condition'])
    
    # Compressor Methods
    def build_compressor_space(self, cs: ConfigurationSpace, fixed_params: Dict[str, Any]) -> ConfigurationSpace:
        unified_space = self.parent.get_unified_space()

        if 'passage_compressor_config' in unified_space and 'passage_compressor_config' not in fixed_params:
            config_info = unified_space['passage_compressor_config']
            config_param = Categorical('passage_compressor_config', config_info['values'],
                                    default=config_info['values'][0])
            cs.add(config_param)
            
            print(f"[DEBUG] Compressor config values: {config_info['values'][:5]}...") 
            
            self._add_compression_ratio(cs, unified_space, fixed_params)
            self._add_lexrank_parameters(cs, unified_space, fixed_params)
            
            print(f"[DEBUG] Final ConfigSpace has {len(cs.get_hyperparameters())} parameters")
            for param in cs.get_hyperparameters():
                print(f"[DEBUG]   - {param.name}: {type(param).__name__}")
            
        elif 'passage_compressor_method' in unified_space and 'passage_compressor_method' not in fixed_params:
            self._build_legacy_compressor_space(cs, unified_space, fixed_params)
        
        return cs
    
    def _add_compression_ratio(self, cs: ConfigurationSpace, unified_space: Dict[str, Any], fixed_params: Dict[str, Any]):
        if 'compression_ratio' in unified_space and 'compression_ratio' not in fixed_params:
            comp_ratio_info = unified_space['compression_ratio']
            print(f"[DEBUG] compression_ratio info: {comp_ratio_info}")
            if comp_ratio_info.get('type') == 'float' and 'values' in comp_ratio_info:
                values = comp_ratio_info['values']
                if len(values) == 2:
                    comp_ratio_param = Float('compression_ratio', bounds=(values[0], values[1]), default=values[0])
                else:
                    comp_ratio_param = Categorical('compression_ratio', values, default=values[0])
                cs.add(comp_ratio_param)

                if comp_ratio_info.get('condition'):
                    print(f"[DEBUG] Adding condition for compression_ratio: {comp_ratio_info['condition']}")
                    self.parent._add_single_condition(cs, 'compression_ratio', comp_ratio_info['condition'])
    
    def _add_lexrank_parameters(self, cs: ConfigurationSpace, unified_space: Dict[str, Any], fixed_params: Dict[str, Any]):
        if 'lexrank_threshold' in unified_space and 'lexrank_threshold' not in fixed_params:
            threshold_info = unified_space['lexrank_threshold']
            print(f"[DEBUG] lexrank_threshold info: {threshold_info}")
            if threshold_info.get('type') == 'float' and 'values' in threshold_info:
                values = threshold_info['values']
                if len(values) == 2:
                    threshold_param = Float('lexrank_threshold', bounds=(values[0], values[1]), default=values[0])
                else:
                    threshold_param = Categorical('lexrank_threshold', values, default=values[0])
                cs.add(threshold_param)
                
                if threshold_info.get('condition'):
                    print(f"[DEBUG] Adding condition for lexrank_threshold: {threshold_info['condition']}")
                    self.parent._add_single_condition(cs, 'lexrank_threshold', threshold_info['condition'])
        
        if 'lexrank_damping' in unified_space and 'lexrank_damping' not in fixed_params:
            damping_info = unified_space['lexrank_damping']
            print(f"[DEBUG] lexrank_damping info: {damping_info}")
            if damping_info.get('type') == 'float' and 'values' in damping_info:
                values = damping_info['values']
                if len(values) == 2:
                    damping_param = Float('lexrank_damping', bounds=(values[0], values[1]), default=values[0])
                else:
                    damping_param = Categorical('lexrank_damping', values, default=values[0])
                cs.add(damping_param)
                
                if damping_info.get('condition'):
                    print(f"[DEBUG] Adding condition for lexrank_damping: {damping_info['condition']}")
                    self.parent._add_single_condition(cs, 'lexrank_damping', damping_info['condition'])
        
        if 'lexrank_max_iterations' in unified_space and 'lexrank_max_iterations' not in fixed_params:
            iter_info = unified_space['lexrank_max_iterations']
            print(f"[DEBUG] lexrank_max_iterations info: {iter_info}")
            if iter_info.get('type') == 'int' and 'values' in iter_info:
                values = iter_info['values']
                if len(values) == 2:
                    iter_param = Integer('lexrank_max_iterations', bounds=(values[0], values[1]), default=values[0])
                else:
                    iter_param = Categorical('lexrank_max_iterations', values, default=values[0])
                cs.add(iter_param)
                
                if iter_info.get('condition'):
                    print(f"[DEBUG] Adding condition for lexrank_max_iterations: {iter_info['condition']}")
                    self.parent._add_single_condition(cs, 'lexrank_max_iterations', iter_info['condition'])
    
    def _build_legacy_compressor_space(self, cs: ConfigurationSpace, unified_space: Dict[str, Any], fixed_params: Dict[str, Any]):
        method_info = unified_space['passage_compressor_method']
        method_param = Categorical('passage_compressor_method', method_info['values'],
                                default=self.parent._get_default_value('passage_compressor_method', method_info['values']))
        cs.add(method_param)

        if 'compressor_llm' in unified_space and 'compressor_llm' not in fixed_params:
            llm_info = unified_space['compressor_llm']
            llm_param = Categorical('compressor_llm', llm_info['values'],
                                default=llm_info['values'][0])
            cs.add(llm_param)
            
            if llm_info.get('condition') and 'passage_compressor_method' in cs:
                self.parent._add_single_condition(cs, 'compressor_llm', llm_info['condition'])
        
        if 'compressor_model' in unified_space and 'compressor_model' not in fixed_params:
            model_info = unified_space['compressor_model']
            model_param = Categorical('compressor_model', model_info['values'],
                                    default=model_info['values'][0])
            cs.add(model_param)
            
            if model_info.get('condition') and 'passage_compressor_method' in cs:
                self.parent._add_single_condition(cs, 'compressor_model', model_info['condition'])

        if 'compression_ratio' in unified_space and 'compression_ratio' not in fixed_params:
            comp_ratio_info = unified_space['compression_ratio']
            if comp_ratio_info.get('type') == 'float' and 'values' in comp_ratio_info:
                values = comp_ratio_info['values']
                if len(values) == 2:
                    comp_ratio_param = Float('compression_ratio', bounds=(values[0], values[1]), default=values[0])
                else:
                    comp_ratio_param = Categorical('compression_ratio', values, default=values[0])
                cs.add(comp_ratio_param)
                
                if 'passage_compressor_method' in cs:
                    cs.add(InCondition(cs['compression_ratio'], cs['passage_compressor_method'], ['lexrank', 'spacy']))
        
        self._add_lexrank_parameters(cs, unified_space, fixed_params)
        
        if 'spacy_model' in unified_space and 'spacy_model' not in fixed_params:
            spacy_info = unified_space['spacy_model']
            spacy_param = Categorical('spacy_model', spacy_info['values'], 
                                    default=spacy_info['values'][0])
            cs.add(spacy_param)
            
            if 'passage_compressor_method' in cs:
                cs.add(EqualsCondition(cs['spacy_model'], cs['passage_compressor_method'], 'spacy'))