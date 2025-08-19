from typing import Dict, Any, List, Tuple
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, InCondition, EqualsCondition, AndConjunction, Constant, ForbiddenAndConjunction, ForbiddenEqualsClause
from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_extractor import UnifiedSearchSpaceExtractor, OptimizationType
import numpy as np


class SMACConfigSpaceBuilder:
    def __init__(self, config_generator: ConfigGenerator, seed: int = 42):
        self.config_generator = config_generator
        self.seed = seed
        self.unified_extractor = UnifiedSearchSpaceExtractor(config_generator)
        qe_params = config_generator.extract_unified_parameters('query_expansion')
        self.query_expansion_retrieval_options = qe_params.get('retrieval_options', {})
    
    def build_configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        
        unified_space = self.unified_extractor.extract_search_space(OptimizationType.SMAC)

        added_params = set()

        non_pass_reranker_configs = []

        if 'passage_reranker_method' in unified_space:
            unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
            models_by_method = unified_params.get('models', {})

            reranker_config_values = []
            
            for method in unified_space['passage_reranker_method']['values']:
                if method == 'pass_reranker':
                    reranker_config_values.append('pass_reranker')
                elif method == 'sap_api':
                    reranker_config_values.append('sap_api')
                    non_pass_reranker_configs.append('sap_api')
                elif method in models_by_method and models_by_method[method]:
                    for model in models_by_method[method]:
                        config_str = f"{method}::{model}"
                        reranker_config_values.append(config_str)
                        non_pass_reranker_configs.append(config_str)
                else:
                    reranker_config_values.append(method)
                    non_pass_reranker_configs.append(method)
                    print(f"    No models found for {method}, using method name only")

            unified_space['reranker_config'] = {
                'type': 'categorical',
                'values': reranker_config_values
            }

            del unified_space['passage_reranker_method']

            if 'reranker_top_k' in unified_space and unified_space['reranker_top_k'].get('condition'):
                condition = unified_space['reranker_top_k']['condition']
                if isinstance(condition, tuple) and condition[0] == 'passage_reranker_method':
                    unified_space['reranker_top_k']['condition'] = ('reranker_config', 'not_equals', 'pass_reranker')

        if 'query_expansion_method' in unified_space:
            if 'retrieval_method' in unified_space:
                unified_space['retrieval_method']['condition'] = ('query_expansion_method', ['pass_query_expansion'])
            
            if 'bm25_tokenizer' in unified_space:
                unified_space['bm25_tokenizer']['condition'] = [
                    ('query_expansion_method', ['pass_query_expansion']),
                    ('retrieval_method', ['bm25'])
                ]
            
            if 'vectordb_name' in unified_space:
                unified_space['vectordb_name']['condition'] = [
                    ('query_expansion_method', ['pass_query_expansion']),
                    ('retrieval_method', ['vectordb'])
                ]

        for param_name, param_info in unified_space.items():
            if param_name in added_params:
                print(f"  Skipping {param_name} - already added")
                continue
                
            param = self._create_parameter(param_name, param_info['type'], param_info)
            if param:
                cs.add(param)
                added_params.add(param_name)

        self._add_conditions(cs, unified_space)

        if non_pass_reranker_configs and 'reranker_config' in cs and 'reranker_top_k' in cs:
            cs.add(InCondition(cs['reranker_top_k'], cs['reranker_config'], non_pass_reranker_configs))

        if 'retriever_top_k' in cs and 'reranker_top_k' in cs:
            self._add_forbidden_reranker_retriever_relation(cs)
        
        return cs
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = config.copy()

        composite_params = {
            'query_expansion_config': self._parse_query_expansion_config,
            'passage_compressor_config': self._parse_compressor_config,
            'generator_config': self._parse_generator_config
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
                unified_params = self.unified_extractor.extract_search_space(OptimizationType.SMAC)
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
    
    def _create_parameter(self, name: str, param_type: str, param_info: Dict[str, Any]):
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
    
    def _add_forbidden_reranker_retriever_relation(self, cs: ConfigurationSpace):
        retriever_param = cs['retriever_top_k']
        reranker_param = cs['reranker_top_k']

        retriever_bounds = retriever_param.lower, retriever_param.upper
        reranker_bounds = reranker_param.lower, reranker_param.upper

        for retriever_k in range(retriever_bounds[0], retriever_bounds[1] + 1):
            for reranker_k in range(retriever_k, reranker_bounds[1] + 1):
                clause = ForbiddenAndConjunction(
                    ForbiddenEqualsClause(retriever_param, retriever_k),
                    ForbiddenEqualsClause(reranker_param, reranker_k)
                )
                cs.add_forbidden_clause(clause)
    
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
    
    def _add_conditions(self, cs: ConfigurationSpace, unified_space: Dict[str, Any]):
        for param_name, param_info in unified_space.items():
            if 'condition' in param_info and param_name in cs:
                self._add_single_condition(cs, param_name, param_info['condition'])
    
    def _add_single_condition(self, cs: ConfigurationSpace, child_name: str, condition):
        if isinstance(condition, list) and len(condition) > 0:
            if isinstance(condition[0], tuple):
                conjunctions = []
                for cond_item in condition:
                    if isinstance(cond_item, tuple) and len(cond_item) == 3:
                        parent_name, op, parent_values = cond_item
                        if parent_name in cs:
                            if op == 'equals':
                                conjunctions.append(EqualsCondition(cs[child_name], cs[parent_name], parent_values))
                            elif op == 'in' and isinstance(parent_values, list):
                                if len(parent_values) == 1:
                                    conjunctions.append(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                                else:
                                    conjunctions.append(InCondition(cs[child_name], cs[parent_name], parent_values))
                    elif isinstance(cond_item, tuple) and len(cond_item) == 2:
                        parent_name, parent_values = cond_item
                        if parent_name in cs and isinstance(parent_values, list):
                            if len(parent_values) == 1:
                                conjunctions.append(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                            else:
                                conjunctions.append(InCondition(cs[child_name], cs[parent_name], parent_values))
                
                if len(conjunctions) == 1:
                    cs.add(conjunctions[0])
                elif len(conjunctions) > 1:
                    cs.add(AndConjunction(*conjunctions))
            else:
                parent_name, parent_values = condition
                if parent_name in cs and isinstance(parent_values, list):
                    if len(parent_values) == 1:
                        cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                    else:
                        cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))
        elif isinstance(condition, tuple):
            if len(condition) == 3:
                parent_name, op, parent_values = condition
                if parent_name in cs:
                    if op == 'equals':
                        cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values))
                    elif op == 'in' and isinstance(parent_values, list):
                        if len(parent_values) == 1:
                            cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                        else:
                            cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))
            elif len(condition) == 2:
                parent_name, parent_values = condition
                if parent_name in cs and isinstance(parent_values, list):
                    if len(parent_values) == 1:
                        cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                    else:
                        cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = config.copy()

        composite_params = {
            'query_expansion_config': self._parse_query_expansion_config,
            'passage_compressor_config': self._parse_compressor_config,
            'generator_config': self._parse_generator_config,
            'reranker_config': self._parse_reranker_config  
        }
        
        for param_name, parser_func in composite_params.items():
            if param_name in cleaned:
                parsed_params = parser_func(cleaned[param_name])
                cleaned.update(parsed_params)
                del cleaned[param_name]

        if 'reranker_model_name' in cleaned and 'method_models' in cleaned.get('_metadata', {}):
            method = cleaned.get('passage_reranker_method')
            if method:
                model_mappings = cleaned['_metadata']['method_models']
                if method in model_mappings:
                    valid_models = model_mappings[method]
                    if cleaned['reranker_model_name'] not in valid_models:
                        cleaned['reranker_model_name'] = valid_models[0] if valid_models else cleaned['reranker_model_name']

        if '_metadata' in cleaned:
            del cleaned['_metadata']

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
    
    def _parse_query_expansion_config(self, qe_config_str: str) -> Dict[str, Any]:
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
    
    def _parse_reranker_config(self, reranker_config_str: str) -> Dict[str, Any]:
        print(f"\n[_parse_reranker_config DEBUG] Input: {reranker_config_str}")
        
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

        result = {'passage_reranker_method': reranker_config_str}
        print(f"  No :: found, treating as method only")
        print(f"  Result: {result}")
        return result

    def _parse_compressor_config(self, comp_config_str: str) -> Dict[str, Any]:
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
    
    def _parse_generator_config(self, gen_config_str: str) -> Dict[str, Any]:
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
    
    def get_search_space_info(self) -> Dict[str, Any]:
        unified_space = self.unified_extractor.extract_search_space(OptimizationType.SMAC)
        
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