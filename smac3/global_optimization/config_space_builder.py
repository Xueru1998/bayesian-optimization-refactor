from typing import Dict, Any, List, Tuple
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, InCondition, EqualsCondition, AndConjunction, ForbiddenEqualsClause, ForbiddenAndConjunction
from pipeline.config_manager import ConfigGenerator
from pipeline.utils import Utils
import numpy as np


class SMACConfigSpaceBuilder:
    
    def __init__(self, config_generator: ConfigGenerator, seed: int = 42):
        self.config_generator = config_generator
        self.seed = seed
        self._unified_space = None
        
        class UnifiedExtractor:
            def __init__(self, parent):
                self.parent = parent
            
            def extract_search_space(self, format_type='smac'):
                return self.parent.get_unified_space()
        
        self.unified_extractor = UnifiedExtractor(self)
        self.query_expansion_retrieval_options = config_generator.extract_query_expansion_retrieval_options()
    
    def get_unified_space(self) -> Dict[str, Any]:
        if self._unified_space is None:
            self._unified_space = self._extract_all_hyperparameters()
        return self._unified_space
    
    def build_configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        
        unified_space = self.unified_extractor.extract_search_space('smac')
        
        for param_name, param_info in unified_space.items():
            if param_info.get('type') == 'categorical' and 'values' in param_info:
                original_values = param_info['values']
                unique_values = list(dict.fromkeys(original_values))
                if len(unique_values) < len(original_values):
                    print(f"[WARNING] Removed duplicates from {param_name}: {original_values} -> {unique_values}")
                    param_info['values'] = unique_values

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
        
        if 'query_expansion_retrieval_method' in unified_space:
            pass
        else:
            if 'query_expansion_method' in unified_space:
                qe_retrieval_options = self.query_expansion_retrieval_options
                if qe_retrieval_options and qe_retrieval_options.get('methods'):
                    unified_space['query_expansion_retrieval_method'] = {
                        'type': 'categorical',
                        'values': qe_retrieval_options['methods'],
                        'condition': ('query_expansion_method', [m for m in unified_space['query_expansion_method']['values'] if m != 'pass_query_expansion'])
                    }
                    
                    if 'bm25' in qe_retrieval_options['methods'] and qe_retrieval_options.get('bm25_tokenizers'):
                        unified_space['query_expansion_bm25_tokenizer'] = {
                            'type': 'categorical',
                            'values': qe_retrieval_options['bm25_tokenizers'],
                            'condition': [
                                ('query_expansion_method', [m for m in unified_space['query_expansion_method']['values'] if m != 'pass_query_expansion']),
                                ('query_expansion_retrieval_method', ['bm25'])
                            ]
                        }
                    
                    if 'vectordb' in qe_retrieval_options['methods'] and qe_retrieval_options.get('vectordb_names'):
                        unified_space['query_expansion_vectordb_name'] = {
                            'type': 'categorical',
                            'values': qe_retrieval_options['vectordb_names'],
                            'condition': [
                                ('query_expansion_method', [m for m in unified_space['query_expansion_method']['values'] if m != 'pass_query_expansion']),
                                ('query_expansion_retrieval_method', ['vectordb'])
                            ]
                        }
        
        for param_name, param_info in unified_space.items():
            param = self._create_parameter(param_name, param_info['type'], param_info)
            if param:
                cs.add(param)
        
        self._add_conditions(cs, unified_space)
        
        if 'retriever_top_k' in cs and 'reranker_top_k' in cs:
            self._add_forbidden_reranker_retriever_relation(cs)
        
        reranker_top_k_param = None
        if 'reranker_top_k' in cs:
            reranker_top_k_param = 'reranker_top_k'
        elif 'reranker_topk' in cs:
            reranker_top_k_param = 'reranker_topk'
        
        if 'reranker_top_k' in cs and 'passage_filter_method' in cs:
            filter_methods = [m for m in unified_space.get('passage_filter_method', {}).get('values', []) 
                            if m != 'pass_passage_filter']
            
            # When reranker_top_k=1, forbid all non-pass filter methods
            for filter_method in filter_methods:
                cs.add_forbidden_clause(
                    ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs['reranker_top_k'], 1),
                        ForbiddenEqualsClause(cs['passage_filter_method'], filter_method)
                    )
                )
                    
                print(f"[DEBUG] Added constraint: {reranker_top_k_param}=1 forces passage_filter_method='pass_passage_filter'")
        
        return cs
    
    def _extract_all_hyperparameters(self) -> Dict[str, Any]:
        params = {}
        
        if self.config_generator.node_exists("query_expansion"):
            params.update(self._extract_query_expansion_params())
        
        if self.config_generator.node_exists("retrieval"):
            params.update(self._extract_retrieval_params())
        
        if self.config_generator.node_exists("passage_reranker"):
            params.update(self._extract_reranker_params())
        
        if self.config_generator.node_exists("passage_filter"):
            params.update(self._extract_filter_params())
        
        if self.config_generator.node_exists("passage_compressor"):
            params.update(self._extract_compressor_params())
        
        if self.config_generator.node_exists("prompt_maker"):
            params.update(self._extract_prompt_maker_params())
        
        if self.config_generator.node_exists("generator"):
            params.update(self._extract_generator_params())
        
        return params
    
    def _extract_query_expansion_params(self) -> Dict[str, Any]:
        params = {}
        qe_config = self.config_generator.extract_node_config("query_expansion")
        
        if not qe_config or not qe_config.get("modules"):
            return params
        
        methods = []
        method_specific_params = {}
        
        for module in qe_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method == "query_decompose":
                    models = module.get("model", [])
                    if not isinstance(models, list):
                        models = [models]
                    
                    if models: 
                        method_specific_params[method] = {}
                        if models:
                            method_specific_params[method]["models"] = models
                
                elif method == "hyde":
                    models = module.get("model", [])
                    if not isinstance(models, list):
                        models = [models]
                    max_tokens = module.get("max_token", [64])
                    if not isinstance(max_tokens, list):
                        max_tokens = [max_tokens]
                    
                    if models: 
                        method_specific_params[method] = {
                            "max_tokens": max_tokens
                        }
                        if models:
                            method_specific_params[method]["models"] = models
                
                elif method == "multi_query_expansion":
                    models = module.get("model", [])
                    if not isinstance(models, list):
                        models = [models]
                    temps = module.get("temperature", [0.7])
                    if not isinstance(temps, list):
                        temps = [temps]
                    
                    method_specific_params[method] = {"temperatures": temps}
                    if models:
                        method_specific_params[method]["models"] = models
        
        if methods:
            params['query_expansion_method'] = {
                'type': 'categorical',
                'values': methods
            }
            
            all_models = set()
            methods_with_model = []
            
            for method, method_params in method_specific_params.items():
                if "models" in method_params:
                    all_models.update(method_params["models"])
                    methods_with_model.append(method)
            
            if all_models:
                params['query_expansion_model'] = {
                    'type': 'categorical',
                    'values': list(all_models),
                    'condition': ('query_expansion_method', methods_with_model)
                }
            
            if "hyde" in method_specific_params and "max_tokens" in method_specific_params["hyde"]:
                params['query_expansion_max_token'] = {
                    'type': 'categorical',
                    'values': method_specific_params["hyde"]["max_tokens"],
                    'condition': ('query_expansion_method', ['hyde'])
                }
            
            if "multi_query_expansion" in method_specific_params and "temperatures" in method_specific_params["multi_query_expansion"]:
                params['query_expansion_temperature'] = {
                    'type': 'float',
                    'values': method_specific_params["multi_query_expansion"]["temperatures"],
                    'condition': ('query_expansion_method', ['multi_query_expansion'])
                }
        
        qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
        if qe_retrieval_options.get('methods'):
            params['query_expansion_retrieval_method'] = {
                'type': 'categorical',
                'values': qe_retrieval_options['methods'],
                'condition': ('query_expansion_method', [m for m in methods if m != 'pass_query_expansion'])
            }
            
            if 'vectordb' in qe_retrieval_options['methods'] and qe_retrieval_options.get('vectordb_names'):
                params['query_expansion_vectordb_name'] = {
                    'type': 'categorical',
                    'values': qe_retrieval_options['vectordb_names'],
                    'condition': [
                        ('query_expansion_method', [m for m in methods if m != 'pass_query_expansion']),
                        ('query_expansion_retrieval_method', ['vectordb'])
                    ]
                }
        
        return params
    
    def _extract_retrieval_params(self) -> Dict[str, Any]:
        params = {}
        retrieval_config = self.config_generator.extract_node_config("retrieval")
        
        if not retrieval_config:
            return params
        
        top_k_values = retrieval_config.get('top_k', [5])
        if isinstance(top_k_values, list) and len(top_k_values) > 0:
            params['retriever_top_k'] = {
                'type': 'int' if len(top_k_values) == 2 else 'categorical',
                'values': top_k_values
            }
        
        methods = []
        bm25_tokenizers = []
        vectordb_names = []
        
        for module in retrieval_config.get("modules", []):
            module_type = module.get("module_type")
            if module_type == "bm25":
                methods.append("bm25")
                tokenizers = module.get("bm25_tokenizer", ["porter_stemmer"])
                if not isinstance(tokenizers, list):
                    tokenizers = [tokenizers]
                bm25_tokenizers.extend(tokenizers)
            elif module_type == "vectordb":
                methods.append("vectordb")
                vdbs = module.get("vectordb", ["default"])
                if not isinstance(vdbs, list):
                    vdbs = [vdbs]
                vectordb_names.extend(vdbs)
        
        if methods:
            params['retrieval_method'] = {
                'type': 'categorical',
                'values': list(set(methods))
            }
        
        if bm25_tokenizers:
            params['bm25_tokenizer'] = {
                'type': 'categorical',
                'values': list(set(bm25_tokenizers)),
                'condition': ('retrieval_method', ['bm25'])
            }
        
        if vectordb_names:
            params['vectordb_name'] = {
                'type': 'categorical',
                'values': list(set(vectordb_names)),
                'condition': ('retrieval_method', ['vectordb'])
            }
        
        return params
    
    def _extract_reranker_params(self) -> Dict[str, Any]:
        params = {}
        reranker_config = self.config_generator.extract_node_config("passage_reranker")
        
        if not reranker_config or not reranker_config.get("modules"):
            return params
        
        methods = []
        
        for module in reranker_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
        
        if methods:
            params['passage_reranker_method'] = {
                'type': 'categorical',
                'values': methods
            }

        top_k_values = reranker_config.get('top_k', [5])
        if isinstance(top_k_values, list) and len(top_k_values) > 0:
            non_pass_methods = [m for m in methods if m != 'pass_reranker']
            if non_pass_methods: 
                params['reranker_top_k'] = {
                    'type': 'int' if len(top_k_values) == 2 else 'categorical',
                    'values': top_k_values,
                    'condition': ('passage_reranker_method', non_pass_methods)
                }

        reranker_configs = []
        for module in reranker_config.get("modules", []):
            method = module.get("module_type")
            if method and method != 'pass_reranker':
                if method in ['colbert_reranker', 'sentence_transformer_reranker', 
                            'flag_embedding_reranker', 'flag_embedding_llm_reranker',
                            'openvino_reranker', 'flashrank_reranker', 'monot5']:

                    if 'model_name' in module:
                        models = module.get("model_name", [])
                        if not isinstance(models, list):
                            models = [models]
                        for model in models:
                            reranker_configs.append(f"{method}_{model}")

                    elif 'model' in module and method == 'flashrank_reranker':
                        models = module.get("model", [])
                        if not isinstance(models, list):
                            models = [models]
                        for model in models:
                            reranker_configs.append(f"{method}_{model}")
                    else:
                        reranker_configs.append(method)
                else:
                    reranker_configs.append(method)
        
        if reranker_configs:
            params['reranker_config'] = {
                'type': 'categorical',
                'values': reranker_configs,
                'condition': ('passage_reranker_method', [m for m in methods if m != 'pass_reranker'])
            }
        
        return params
    
    def _extract_filter_params(self) -> Dict[str, Any]:
        params = {}
        filter_config = self.config_generator.extract_node_config("passage_filter")
        
        if not filter_config or not filter_config.get("modules"):
            return params
        
        methods = []
        threshold_values = {}
        percentile_values = {}
        
        for module in filter_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method == "threshold_cutoff":
                    if "threshold" in module:
                        threshold_values[method] = module["threshold"]
                elif method == "percentile_cutoff":
                    if "percentile" in module:
                        percentile_values[method] = module["percentile"]
                elif method == "similarity_threshold_cutoff":
                    if "threshold" in module:
                        threshold_values[method] = module["threshold"]
                elif method == "similarity_percentile_cutoff":
                    if "percentile" in module:
                        percentile_values[method] = module["percentile"]
        
        if methods:
            params['passage_filter_method'] = {
                'type': 'categorical',
                'values': methods
            }
        
        if threshold_values:
            params['threshold'] = {
                'type': 'float',
                'method_values': threshold_values,
                'condition': ('passage_filter_method', list(threshold_values.keys()))
            }
        
        if percentile_values:
            params['percentile'] = {
                'type': 'float',
                'method_values': percentile_values,
                'condition': ('passage_filter_method', list(percentile_values.keys()))
            }
        
        return params
    
    def _extract_compressor_params(self) -> Dict[str, Any]:
        params = {}
        compressor_config = self.config_generator.extract_node_config("passage_compressor")
        
        if not compressor_config or not compressor_config.get("modules"):
            return params
        
        methods = []
        llm_models = {}
        compression_ratios = {}
        thresholds = {}
        dampings = {}
        max_iterations = {}
        spacy_models = {}
        
        for module in compressor_config.get("modules", []):
            method = module.get("module_type")
            if method:
                methods.append(method)
                
                if method in ["tree_summarize", "refine"]:
                    llm = module.get("llm")
                    model = module.get("model")
                    if llm and model:
                        if method not in llm_models:
                            llm_models[method] = []
                        llm_models[method].append(f"{llm}_{model}")
                
                elif method == "lexrank":
                    if "compression_ratio" in module:
                        compression_ratios[method] = module["compression_ratio"]
                    if "threshold" in module:
                        thresholds[method] = module["threshold"]
                    if "damping" in module:
                        dampings[method] = module["damping"]
                    if "max_iterations" in module:
                        max_iterations[method] = module["max_iterations"]
                
                elif method == "spacy":
                    if "compression_ratio" in module:
                        compression_ratios[method] = module["compression_ratio"]
                    if "spacy_model" in module:
                        spacy_models[method] = module["spacy_model"]
        
        if methods:
            params['passage_compressor_method'] = {
                'type': 'categorical',
                'values': methods
            }

        all_llm_models = []
        methods_with_llm = []
        for method, models in llm_models.items():
            all_llm_models.extend(models)
            methods_with_llm.append(method)
        
        if all_llm_models:
            params['compressor_llm_model'] = {
                'type': 'categorical',
                'values': list(set(all_llm_models)),
                'condition': ('passage_compressor_method', methods_with_llm)
            }

        if compression_ratios:
            all_are_ranges = True
            all_min_values = []
            all_max_values = []
            
            for method, values in compression_ratios.items():
                if isinstance(values, list) and len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                    all_min_values.append(min(values))
                    all_max_values.append(max(values))
                else:
                    all_are_ranges = False
                    break

            if all_are_ranges and all_min_values and all_max_values:
                params['compressor_compression_ratio'] = {
                    'type': 'float',
                    'values': [min(all_min_values), max(all_max_values)],
                    'condition': ('passage_compressor_method', list(compression_ratios.keys()))
                }
            else:
                params['compressor_compression_ratio'] = {
                    'type': 'float',
                    'method_values': compression_ratios,
                    'condition': ('passage_compressor_method', list(compression_ratios.keys()))
                }

        if thresholds:
            all_threshold_values = []
            is_range = False
            
            for method, values in thresholds.items():
                if isinstance(values, list):
                    if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                        is_range = True
                        all_threshold_values.extend(values)
                    else:
                        all_threshold_values.extend(values)
                else:
                    all_threshold_values.append(values)
            
            if is_range and len(set(all_threshold_values)) == 2:
                params['compressor_threshold'] = {
                    'type': 'float',
                    'values': [min(all_threshold_values), max(all_threshold_values)],
                    'condition': ('passage_compressor_method', ['lexrank'])
                }
            else:
                params['compressor_threshold'] = {
                    'type': 'float',
                    'method_values': thresholds,
                    'condition': ('passage_compressor_method', ['lexrank'])
                }

        if dampings:
            all_damping_values = []
            is_range = False
            
            for method, values in dampings.items():
                if isinstance(values, list):
                    if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                        is_range = True
                        all_damping_values.extend(values)
                    else:
                        all_damping_values.extend(values)
                else:
                    all_damping_values.append(values)
            
            if is_range and len(set(all_damping_values)) == 2:
                params['compressor_damping'] = {
                    'type': 'float',
                    'values': [min(all_damping_values), max(all_damping_values)],
                    'condition': ('passage_compressor_method', ['lexrank'])
                }
            else:
                params['compressor_damping'] = {
                    'type': 'float',
                    'method_values': dampings,
                    'condition': ('passage_compressor_method', ['lexrank'])
                }

        if max_iterations:
            all_iteration_values = []
            is_range = False
            
            for method, values in max_iterations.items():
                if isinstance(values, list):
                    if len(values) == 2 and all(isinstance(v, int) for v in values):
                        is_range = True
                        all_iteration_values.extend(values)
                    else:
                        all_iteration_values.extend(values)
                else:
                    all_iteration_values.append(values)
            
            if is_range and len(set(all_iteration_values)) == 2:
                params['compressor_max_iterations'] = {
                    'type': 'int',
                    'values': [min(all_iteration_values), max(all_iteration_values)],
                    'condition': ('passage_compressor_method', ['lexrank'])
                }
            else:
                params['compressor_max_iterations'] = {
                    'type': 'int',
                    'method_values': max_iterations,
                    'condition': ('passage_compressor_method', ['lexrank'])
                }

        if spacy_models:
            all_spacy_models = []
            for models in spacy_models.values():
                if isinstance(models, list):
                    all_spacy_models.extend(models)
                else:
                    all_spacy_models.append(models)
            
            params['compressor_spacy_model'] = {
                'type': 'categorical',
                'values': list(set(all_spacy_models)),
                'condition': ('passage_compressor_method', ['spacy'])
            }
        
        return params
        
    def _extract_prompt_maker_params(self) -> Dict[str, Any]:
        params = {}
        prompt_methods, prompt_indices = self.config_generator.extract_prompt_maker_options()
        
        if prompt_methods:
            params['prompt_maker_method'] = {
                'type': 'categorical',
                'values': prompt_methods
            }
            
            if prompt_indices:
                params['prompt_template_idx'] = {
                    'type': 'categorical',
                    'values': prompt_indices,
                    'condition': ('prompt_maker_method', [m for m in prompt_methods if m != 'pass_prompt_maker'])
                }
        
        return params
    
    def _extract_generator_params(self) -> Dict[str, Any]:
        params = {}
        gen_params = self.config_generator.extract_generator_parameters()
        
        if gen_params.get('models'):
            params['generator_model'] = {
                'type': 'categorical',
                'values': gen_params['models']
            }
        
        if gen_params.get('temperatures'):
            temps = gen_params['temperatures']
            if len(temps) == 2 and isinstance(temps[0], (int, float)):
                params['generator_temperature'] = {
                    'type': 'float',
                    'values': temps
                }
            else:
                params['generator_temperature'] = {
                    'type': 'categorical',
                    'values': temps
                }
        
        return params
    
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
                for parent_name, parent_values in condition:
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
            parent_name, parent_values = condition
            if parent_name in cs and isinstance(parent_values, list):
                if len(parent_values) == 1:
                    cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                else:
                    cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))
    
    def _add_forbidden_reranker_retriever_relation(self, cs: ConfigurationSpace):
        retriever_param = cs['retriever_top_k']
        reranker_param = cs['reranker_top_k']

        # Forbid reranker_top_k > retriever_top_k
        retriever_bounds = retriever_param.lower, retriever_param.upper
        reranker_bounds = reranker_param.lower, reranker_param.upper

        for retriever_k in range(retriever_bounds[0], retriever_bounds[1] + 1):
            for reranker_k in range(retriever_k, reranker_bounds[1] + 1):
                clause = ForbiddenAndConjunction(
                    ForbiddenEqualsClause(retriever_param, retriever_k),
                    ForbiddenEqualsClause(reranker_param, reranker_k)
                )
                cs.add_forbidden_clause(clause)
    
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
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = config.copy()
        
        # Remove meta parameters that aren't used by the pipeline
        params_to_remove = [
            'retrieval_config',
            'query_expansion_config', 
            'passage_filter_config',
            'compressor_config',
            'prompt_config'
        ]
        
        for param in params_to_remove:
            cleaned.pop(param, None)

        # Parse composite configurations (data transformation, not constraint)
        if 'reranker_config' in cleaned:
            reranker_config_str = cleaned.pop('reranker_config')
            parsed_config = self._parse_reranker_config(reranker_config_str)
            cleaned.update(parsed_config)
        
        # Split composite parameters (data transformation, not constraint)
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
    
    def get_search_space_info(self) -> Dict[str, Any]:
        unified_space = self.get_unified_space()
        
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