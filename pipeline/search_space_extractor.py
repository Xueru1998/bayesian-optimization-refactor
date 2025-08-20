from typing import Dict, Any, List, Tuple, Union
from enum import Enum
from pipeline.utils import Utils


class OptimizationType(Enum):
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
        
        if optimizer_type == OptimizationType.SMAC:
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