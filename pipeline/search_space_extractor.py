from typing import Dict, Any, List, Tuple, Union
from enum import Enum
from ray import tune
from pipeline.utils import Utils


class OptimizationType(Enum):
    BOHB = "bohb"
    SMAC = "smac"
    OPTUNA_GRID = "optuna_grid"
    OPTUNA_BO = "optuna_bo"


class UnifiedSearchSpaceExtractor:
    def __init__(self, config_generator):
        self.config_generator = config_generator
        self._cache = {}
    
    def extract_search_space(self, optimizer_type: Union[OptimizationType, str]) -> Dict[str, Any]:
        if isinstance(optimizer_type, str):
            optimizer_type = OptimizationType(optimizer_type.lower())
        
        unified_space = self._extract_unified_space()
        
        if optimizer_type == OptimizationType.BOHB:
            return self._convert_to_bohb_space(unified_space)
        elif optimizer_type == OptimizationType.SMAC:
            return unified_space
        elif optimizer_type == OptimizationType.OPTUNA_GRID:
            return self._convert_to_optuna_grid_space(unified_space)
        elif optimizer_type == OptimizationType.OPTUNA_BO:
            return self._convert_to_optuna_bo_space(unified_space)
    
    def _extract_unified_space(self) -> Dict[str, Any]:
        if 'unified' in self._cache:
            return self._cache['unified']
        
        space = {}
        
        extractors = [
            ('query_expansion', self._extract_query_expansion_params),
            ('retrieval', self._extract_retrieval_params),
            ('passage_reranker', self._extract_reranker_params),
            ('passage_filter', self._extract_filter_params),
            ('passage_compressor', self._extract_compressor_params),
            ('prompt_maker', self._extract_prompt_maker_params),
            ('generator', self._extract_generator_params)
        ]
        
        for node_type, extractor in extractors:
            if self.config_generator.node_exists(node_type):
                extractor(space)
        
        self._cache['unified'] = space
        return space
    
    def _extract_query_expansion_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('query_expansion')
        methods = params.get('methods', [])
        
        if not methods:
            return
        
        space['query_expansion_method'] = {
            'type': 'categorical',
            'values': methods
        }
        
        active_methods = [m for m in methods if m != 'pass_query_expansion']
        
        qe_options = []
        option_metadata = {}
        
        if 'pass_query_expansion' in methods:
            qe_options.append('pass_query_expansion')
            option_metadata['pass_query_expansion'] = {
                'method': 'pass_query_expansion',
                'generator_module_type': None,
                'model': None
            }
        
        for gen_config in params.get('generator_configs', []):
            method = gen_config['method']
            gen_type = gen_config['generator_module_type']
            
            for model in gen_config['models']:
                option_key = f"{method}::{gen_type}::{model}"
                qe_options.append(option_key)
                
                option_metadata[option_key] = {
                    'method': method,
                    'generator_module_type': gen_type,
                    'model': model,
                    'llm': gen_config.get('llm'),
                    'api_url': gen_config.get('api_url')
                }
        
        if qe_options:
            space['query_expansion_config'] = {
                'type': 'categorical',
                'values': qe_options,
                'metadata': option_metadata
            }

        if params.get('all_temperatures'):
            space['query_expansion_temperature'] = {
                'type': 'float',
                'values': params['all_temperatures'],
                'condition': ('query_expansion_method', 'equals', 'multi_query_expansion')
            }
        
        if params.get('all_max_tokens'):
            space['query_expansion_max_token'] = {
                'type': 'int',
                'values': params['all_max_tokens'],
                'condition': ('query_expansion_method', 'equals', 'hyde')
            }
        
        retrieval_options = params.get('retrieval_options', {})
        if retrieval_options.get('methods'):
            space['query_expansion_retrieval_method'] = {
                'type': 'categorical',
                'values': retrieval_options['methods']
            }
            
            if retrieval_options.get('bm25_tokenizers'):
                space['query_expansion_bm25_tokenizer'] = {
                    'type': 'categorical',
                    'values': retrieval_options['bm25_tokenizers'],
                    'condition': ('query_expansion_retrieval_method', 'equals', 'bm25')
                }
            
            if retrieval_options.get('vectordb_names'):
                space['query_expansion_vectordb_name'] = {
                    'type': 'categorical',
                    'values': retrieval_options['vectordb_names'],
                    'condition': ('query_expansion_retrieval_method', 'equals', 'vectordb')
                }

    
    def _extract_retrieval_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('retrieval')
        
        if params.get('retriever_top_k_values'):
            space['retriever_top_k'] = {
                'type': 'int',
                'values': params['retriever_top_k_values']
            }
        
        if params.get('methods'):
            conditions = [('query_expansion_method', 'equals', 'pass_query_expansion')] if 'query_expansion_method' in space else []
            
            space['retrieval_method'] = {
                'type': 'categorical',
                'values': params['methods'],
                'condition': conditions[0] if conditions else None
            }
            
            if params.get('bm25_tokenizers'):
                base_conditions = conditions + [('retrieval_method', 'equals', 'bm25')]
                space['bm25_tokenizer'] = {
                    'type': 'categorical',
                    'values': params['bm25_tokenizers'],
                    'condition': base_conditions if len(base_conditions) > 1 else base_conditions[0] if base_conditions else None
                }
            
            if params.get('vectordb_names'):
                base_conditions = conditions + [('retrieval_method', 'equals', 'vectordb')]
                space['vectordb_name'] = {
                    'type': 'categorical',
                    'values': params['vectordb_names'],
                    'condition': base_conditions if len(base_conditions) > 1 else base_conditions[0] if base_conditions else None
                }
    
    def _extract_reranker_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('passage_reranker')
        
        if not params.get('methods'):
            return
        
        space['passage_reranker_method'] = {
            'type': 'categorical',
            'values': params['methods']
        }
        
        if params.get('models'):
            model_mappings = params['models']
            space['reranker_model_name'] = {
                'type': 'categorical',
                'method_models': model_mappings,
                'condition': ('passage_reranker_method', 'in', list(model_mappings.keys()))
            }
        
        if params.get('api_endpoints'):
            api_mappings = params['api_endpoints']
            space['reranker_api_url'] = {
                'type': 'categorical',
                'method_apis': api_mappings,
                'condition': ('passage_reranker_method', 'in', list(api_mappings.keys()))
            }
        
        if params.get('top_k_values'):
            retriever_max = max(space.get('retriever_top_k', {}).get('values', [10]))
            constrained_values = [min(k, retriever_max) for k in params['top_k_values']]
            
            space['reranker_top_k'] = {
                'type': 'int',
                'values': list(set(constrained_values)),
                'max_value': retriever_max,
                'condition': ('passage_reranker_method', 'not_equals', 'pass_reranker')
            }
    
    def _extract_filter_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('passage_filter')
        
        if not params.get('methods'):
            return
        
        space['passage_filter_method'] = {
            'type': 'categorical',
            'values': params['methods']
        }
        
        if params.get('threshold_values'):
            space['threshold'] = {
                'type': 'float',
                'method_values': params['threshold_values'],
                'condition': ('passage_filter_method', 'in', list(params['threshold_values'].keys()))
            }
        
        if params.get('percentile_values'):
            space['percentile'] = {
                'type': 'float',
                'method_values': params['percentile_values'],
                'condition': ('passage_filter_method', 'in', list(params['percentile_values'].keys()))
            }
    
    def _extract_compressor_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('passage_compressor')
        
        if not params.get('methods'):
            return

        comp_options = []
        option_metadata = {}
        seen_configs = set()
        
        if 'pass_compressor' in params['methods']:
            comp_options.append('pass_compressor')
            option_metadata['pass_compressor'] = {'method': 'pass_compressor'}
            seen_configs.add('pass_compressor')
        
        for comp_config in params.get('compressor_configs', []):
            method = comp_config['method']

            if method in ['tree_summarize', 'refine']:
                gen_type = comp_config['generator_module_type']
                for model in comp_config['models']:
                    option_key = f"{method}::{gen_type}::{model}"
                    if option_key not in seen_configs:
                        seen_configs.add(option_key)
                        comp_options.append(option_key)
                        
                        option_metadata[option_key] = {
                            'method': method,
                            'generator_module_type': gen_type,
                            'llm': comp_config.get('llm'),
                            'model': model,
                            'api_url': comp_config.get('api_url')
                        }

            elif method in ['lexrank', 'spacy', 'sentence_rank', 'keyword_extraction', 'query_focused']:
                if method == 'spacy':
                    for model in comp_config.get('spacy_model', ['en_core_web_sm']):
                        option_key = f"{method}::{model}"
                        if option_key not in seen_configs:
                            seen_configs.add(option_key)
                            comp_options.append(option_key)
                            option_metadata[option_key] = {
                                'method': method,
                                'spacy_model': model
                            }
                else:
                    option_key = method
                    if option_key not in seen_configs:
                        seen_configs.add(option_key)
                        comp_options.append(option_key)
                        option_metadata[option_key] = {'method': method}
        
        if comp_options:
            space['passage_compressor_config'] = {
                'type': 'categorical',
                'values': comp_options,
                'metadata': option_metadata
            }

        for comp_config in params.get('compressor_configs', []):
            method = comp_config['method']

            if method in ['lexrank', 'spacy', 'sentence_rank', 'keyword_extraction', 'query_focused']:
                if 'compression_ratio' in comp_config:
                    space['compression_ratio'] = {
                        'type': 'float',
                        'values': comp_config['compression_ratio'],
                        'condition': ('passage_compressor_config', 'contains', method)
                    }

            if method == 'lexrank':
                if 'threshold' in comp_config:
                    space['lexrank_threshold'] = {
                        'type': 'float',
                        'values': comp_config['threshold'],
                        'condition': ('passage_compressor_config', 'equals', 'lexrank')
                    }
                if 'damping' in comp_config:
                    space['lexrank_damping'] = {
                        'type': 'float',
                        'values': comp_config['damping'],
                        'condition': ('passage_compressor_config', 'equals', 'lexrank')
                    }
                if 'max_iterations' in comp_config:
                    space['lexrank_max_iterations'] = {
                        'type': 'int',
                        'values': comp_config['max_iterations'],
                        'condition': ('passage_compressor_config', 'equals', 'lexrank')
                    }
                    
    def _extract_prompt_maker_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('prompt_maker')
        
        if params.get('methods'):
            space['prompt_maker_method'] = {
                'type': 'categorical',
                'values': params['methods']
            }
            
            if params.get('prompt_indices'):
                space['prompt_template_idx'] = {
                    'type': 'int',
                    'values': params['prompt_indices'],
                    'condition': ('prompt_maker_method', 'in', ['fstring', 'long_context_reorder', 'window_replacement'])
                }
    
    def _extract_generator_params(self, space: Dict[str, Any]):
        params = self.config_generator.extract_unified_parameters('generator')
        
        if not params.get('module_configs'):
            return

        gen_options = []
        option_metadata = {}
        
        for module_config in params['module_configs']:
            module_type = module_config['module_type']
            
            for model in module_config['models']:
                option_key = f"{module_type}::{model}"
                gen_options.append(option_key)
                
                option_metadata[option_key] = {
                    'module_type': module_type,
                    'model': model,
                    'llm': module_config.get('llm'),
                    'api_url': module_config.get('api_url')
                }
        
        if gen_options:
            space['generator_config'] = {
                'type': 'categorical',
                'values': gen_options,
                'metadata': option_metadata
            }

        if params.get('all_temperatures'):
            unique_temps = list(set(params['all_temperatures']))
            space['generator_temperature'] = {
                'type': 'float',
                'values': unique_temps
            }

        if params.get('all_max_tokens'):
            space['generator_max_tokens'] = {
                'type': 'int',
                'values': params['all_max_tokens'],
                'condition': ('generator_module_type', 'equals', 'sap_api')
            }

    
    def _get_bo_range_for_single_value(self, param_name: str, value: float, param_type: str) -> Tuple[float, float]:
        if param_type == 'int':
            value = int(value)
            
            range_configs = {
                'top_k': (0.5, 1.5),
                'max_token': (0.5, 1.5),
                'idx': (-1, 1, True),
                'index': (-1, 1, True),
                'default': (0.8, 1.2)
            }
            
            for key, config in range_configs.items():
                if key in param_name:
                    if len(config) == 3:
                        return max(0, value + config[0]), value + config[1]
                    else:
                        min_factor, max_factor = config
                        return max(1, int(value * min_factor)), int(value * max_factor)
            
            min_factor, max_factor = range_configs['default']
            min_val = max(1, int(value * min_factor))
            max_val = int(value * max_factor)
            
        else:
            range_configs = {
                'temperature': 0.2,
                'percentile': 0.2,
                'threshold': 0.2,
                'default': 0.1
            }
            
            delta = next((v for k, v in range_configs.items() if k in param_name), range_configs['default'])
            min_val = max(0.0, value - delta)
            max_val = min(1.0, value + delta)
        
        if min_val >= max_val:
            if param_type == 'int':
                return (0, 1) if value == 0 else (max(0, value - 1), value + 1)
            else:
                if value == 0.0:
                    return (0.0, 0.1)
                elif value == 1.0:
                    return (0.9, 1.0)
                else:
                    return (max(0.0, value - 0.05), min(1.0, value + 0.05))
        
        return (min_val, max_val)
    
    def _convert_to_bohb_space(self, unified_space: Dict[str, Any]) -> Dict[str, Any]:
        bohb_space = {}
        
        for param_name, param_info in unified_space.items():
            param_type = param_info['type']
            values = param_info.get('values', [])
            condition = param_info.get('condition')
            metadata = param_info.get('metadata')

            if metadata and param_type == 'categorical':
                if condition:
                    bohb_space[param_name] = tune.sample_from(
                        lambda config, vals=values, cond=condition, meta=metadata, name=param_name: 
                        self._evaluate_condition_with_metadata(config, cond, vals, meta, name)
                    )
                else:
                    bohb_space[param_name] = tune.choice(values)
                continue

            if 'method_values' in param_info:
                bohb_space[param_name] = tune.sample_from(
                    lambda config, info=param_info, name=param_name: 
                    self._evaluate_method_specific_parameter(config, info, name)
                )
                continue

            if param_type == 'categorical':
                if condition:
                    bohb_space[param_name] = tune.sample_from(
                        lambda config, vals=values, cond=condition, name=param_name: 
                        self._evaluate_condition(config, cond, vals, name)
                    )
                else:
                    bohb_space[param_name] = tune.choice(values)
            
            elif param_type in ['int', 'float']:
                bohb_space[param_name] = self._convert_numeric_param_to_bohb(
                    param_name, param_type, values, condition
                )
        
        return bohb_space
    
    def _convert_numeric_param_to_bohb(self, param_name: str, param_type: str, 
                                      values: List, condition: Any) -> Any:
        if not values:
            return None
            
        if len(values) == 1:
            return self._handle_single_value_bohb(param_name, param_type, values[0], condition)
        elif len(values) == 2:
            return self._handle_range_values_bohb(param_name, param_type, values, condition)
        else:
            return self._handle_multiple_values_bohb(param_name, param_type, values, condition)
    
    def _handle_single_value_bohb(self, param_name: str, param_type: str, 
                                 value: float, condition: Any) -> Any:
        if param_name in ['threshold', 'percentile']:
            min_val, max_val = self._get_bo_range_for_single_value(param_name, value, param_type)
            if condition:
                return tune.sample_from(
                    lambda config, min_v=min_val, max_v=max_val, cond=condition:
                    tune.uniform(min_v, max_v).sample() if self._check_condition(config, cond) else None
                )
            else:
                return tune.uniform(min_val, max_val)
                
        elif param_name == 'reranker_top_k':
            return tune.sample_from(
                lambda config, val=value, cond=condition:
                self._evaluate_reranker_topk(config, cond, val)
            )
        else:
            min_val, max_val = self._get_bo_range_for_single_value(param_name, value, param_type)
            
            if param_type == 'int':
                values_range = list(range(min_val, max_val + 1))
                if condition:
                    return tune.sample_from(
                        lambda config, vals=values_range, cond=condition:
                        self._evaluate_condition(config, cond, vals, param_name)
                    )
                else:
                    return tune.choice(values_range)
            else:
                if condition:
                    return tune.sample_from(
                        lambda config, min_v=min_val, max_v=max_val, cond=condition:
                        tune.uniform(min_v, max_v).sample() if self._check_condition(config, cond) else None
                    )
                else:
                    return tune.uniform(min_val, max_val)
    
    def _handle_range_values_bohb(self, param_name: str, param_type: str, 
                                 values: List, condition: Any) -> Any:
        min_val, max_val = min(values), max(values)
        
        if min_val < max_val:
            if param_type == 'int':
                if param_name == 'reranker_top_k':
                    return tune.sample_from(
                        lambda config, min_v=min_val, max_v=max_val, cond=condition:
                        self._evaluate_reranker_topk_range(config, cond, min_v, max_v)
                    )
                else:
                    values_range = list(range(min_val, max_val + 1))
                    if condition:
                        return tune.sample_from(
                            lambda config, vals=values_range, cond=condition:
                            self._evaluate_condition(config, cond, vals, param_name)
                        )
                    else:
                        return tune.choice(values_range)
            else:
                if condition:
                    return tune.sample_from(
                        lambda config, min_v=min_val, max_v=max_val, cond=condition:
                        tune.uniform(min_v, max_v).sample() if self._check_condition(config, cond) else None
                    )
                else:
                    return tune.uniform(min_val, max_val)
        else:
            return self._handle_multiple_values_bohb(param_name, param_type, values, condition)
    
    def _handle_multiple_values_bohb(self, param_name: str, param_type: str, 
                                    values: List, condition: Any) -> Any:
        if param_name == 'reranker_top_k' and condition:
            return tune.sample_from(
                lambda config, vals=values, cond=condition:
                self._evaluate_reranker_topk_discrete(config, cond, vals)
            )
        elif condition:
            return tune.sample_from(
                lambda config, vals=values, cond=condition:
                self._evaluate_condition(config, cond, vals, param_name)
            )
        else:
            return tune.choice(values)
    
    def _evaluate_method_specific_parameter(self, config, param_info, param_name):
        condition = param_info.get('condition')
        
        if condition and not self._check_condition(config, condition):
            return None
        
        method_values = param_info.get('method_values', {})
        if param_name in ['threshold', 'percentile']:
            filter_method = config.get('passage_filter_method')
            if filter_method and filter_method in method_values:
                values = Utils.ensure_list(method_values[filter_method])
                if values:
                    if len(values) == 1:
                        min_val, max_val = self._get_bo_range_for_single_value(
                            param_name, values[0], param_info['type']
                        )
                        return tune.uniform(min_val, max_val).sample()
                    elif len(values) == 2:
                        min_val, max_val = min(values), max(values)
                        if min_val < max_val:
                            return tune.uniform(min_val, max_val).sample()
                    
                    return tune.choice(values).sample()
        
        return None
    
    def _evaluate_reranker_topk(self, config, condition, value):
        if condition and not self._check_condition(config, condition):
            return None
        
        retriever_k = config.get('retriever_top_k', 10)
        
        if value > retriever_k:
            min_val, max_val = self._get_bo_range_for_single_value('reranker_top_k', value, 'int')
            actual_min = max(1, min(min_val, retriever_k))
            actual_max = min(max_val, retriever_k)
            
            if actual_min > actual_max:
                return min(value, retriever_k)
            
            valid_values = list(range(actual_min, actual_max + 1))
            return tune.choice(valid_values).sample()
        else:
            return value
    
    def _evaluate_reranker_topk_range(self, config, condition, min_val, max_val):
        if condition and not self._check_condition(config, condition):
            return None
        
        retriever_k = config.get('retriever_top_k', 10)
        actual_min = min_val
        actual_max = min(max_val, retriever_k)
        
        if actual_min > actual_max:
            return min(min_val, retriever_k)
        
        valid_values = list(range(actual_min, actual_max + 1))
        return tune.choice(valid_values).sample()
    
    def _evaluate_reranker_topk_discrete(self, config, condition, values):
        if not self._check_condition(config, condition):
            return None
        
        retriever_k = config.get('retriever_top_k', 10)
        valid_values = [v for v in values if v <= retriever_k]
        
        if not valid_values:
            return min(values[0], retriever_k)
        
        return tune.choice(valid_values).sample()
    
    def _evaluate_condition(self, config, condition, values, param_name=None):
        if not self._check_condition(config, condition):
            return None
        
        if isinstance(values, list):
            return tune.choice(values).sample()
        else:
            return values
    
    def _check_condition(self, config, condition):
        if condition is None:
            return True
        
        if isinstance(condition, list):
            return all(self._check_single_condition(config, c) for c in condition)
        else:
            return self._check_single_condition(config, condition)
    
    def _evaluate_condition_with_metadata(self, config, condition, values, metadata, param_name):
        if not self._check_condition(config, condition):
            return None

        if condition and len(condition) == 3 and condition[1] == 'contains':
            filtered_values = [v for v in values if condition[2] in v]
            if filtered_values:
                return tune.choice(filtered_values).sample()
            return None
        
        return tune.choice(values).sample()
    
    def _check_single_condition(self, config, condition):
        if condition is None:
            return True
        
        param, op, value = condition
        config_value = config.get(param)
        
        condition_checks = {
            'equals': lambda: config_value == value,
            'not_equals': lambda: config_value != value,
            'in': lambda: config_value in value,
            'not_in': lambda: config_value not in value,
            'contains': lambda: value in str(config_value) if config_value else False
        }
        
        return condition_checks.get(op, lambda: True)()
    
    def _convert_to_optuna_grid_space(self, unified_space: Dict[str, Any]) -> Dict[str, Any]:
        grid_space = {}

        composite_params = [
            'query_expansion_config',
            'passage_compressor_config', 
            'generator_config'
        ]
        
        for param_name in composite_params:
            if param_name in unified_space:
                grid_space[param_name] = unified_space[param_name]['values']

        param_groups = [
            ('query_expansion', [
                'query_expansion_method', 'query_expansion_temperature', 'query_expansion_max_token',
                'query_expansion_retrieval_method', 'query_expansion_bm25_tokenizer', 
                'query_expansion_vectordb_name'
            ]),
            ('retrieval', [
                'retrieval_method', 'bm25_tokenizer', 'vectordb_name', 'retriever_top_k'
            ]),
            ('reranker', [
                'passage_reranker_method', 'reranker_model_name', 'reranker_top_k'
            ]),
            ('filter', [
                'passage_filter_method'
            ]),
            ('prompt', [
                'prompt_maker_method', 'prompt_template_idx'
            ]),
            ('generator', [
                'generator_temperature', 'generator_max_tokens'
            ])
        ]
        
        for group_name, param_names in param_groups:
            for param_name in param_names:
                if param_name in unified_space and param_name not in grid_space:
                    self._add_param_to_grid_space(grid_space, param_name, unified_space[param_name])
        
        self._handle_filter_params_grid(grid_space, unified_space)
        
        return grid_space
    
    def _add_param_to_grid_space(self, grid_space, param_name, param_info):
        if param_name == 'reranker_model_name' and 'method_models' in param_info:
            all_models = []
            for models in param_info['method_models'].values():
                all_models.extend(models)
            grid_space[param_name] = list(set(all_models))
        else:
            values = param_info.get('values', [])
            if values:
                grid_space[param_name] = values
    
    def _handle_filter_params_grid(self, grid_space, unified_space):
        if 'threshold' in unified_space and 'method_values' in unified_space['threshold']:
            for method, values in unified_space['threshold']['method_values'].items():
                grid_space[f'{method}_threshold'] = values
        
        if 'percentile' in unified_space and 'method_values' in unified_space['percentile']:
            for method, values in unified_space['percentile']['method_values'].items():
                grid_space[f'{method}_percentile'] = values
    
    def _convert_to_optuna_bo_space(self, unified_space: Dict[str, Any]) -> Dict[str, Any]:
        return self._convert_to_optuna_grid_space(unified_space)