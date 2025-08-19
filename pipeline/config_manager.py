import copy
from typing import Dict, Any, List, Tuple, Optional
from pipeline.utils import Utils


class NodeDefaults:
    QUERY_EXPANSION = {
        'methods': ['pass_query_expansion'],
        'params': {
            'generator_module_type': 'llama_index_llm',
            'llm': 'openai',
            'model': 'gpt-3.5-turbo-16k',
            'max_token': 64
        }
    }
    
    PASSAGE_RERANKER = {
        'methods': ['pass_reranker'],
        'models': {
            'colbert_reranker': ["colbert-ir/colbertv2.0"],
            'monot5': ["castorini/monot5-3b-msmarco-10k"],
            'sentence_transformer_reranker': ["cross-encoder/ms-marco-MiniLM-L-2-v2"],
            'flag_embedding_reranker': ["BAAI/bge-reranker-large"],
            'flag_embedding_llm_reranker': ["BAAI/bge-reranker-v2-gemma"],
            'openvino_reranker': ["BAAI/bge-reranker-large"],
            'flashrank_reranker': ["ms-marco-MiniLM-L-12-v2"]
        }
    }
    
    PASSAGE_FILTER = {
        'methods': ['pass_passage_filter'],
        'params': {
            'threshold_cutoff': {'threshold': 0.85, 'reverse': False},
            'percentile_cutoff': {'percentile': 0.65, 'reverse': False},
            'similarity_threshold_cutoff': {
                'threshold': 0.7, 'batch': 128, 'embedding_model': 'openai', 'reverse': False
            },
            'similarity_percentile_cutoff': {
                'percentile': 0.7, 'batch': 128, 'embedding_model': 'openai', 'reverse': False
            }
        }
    }
    
    PASSAGE_COMPRESSOR = {
    'methods': ['pass_compressor', 'tree_summarize', 'refine', 'lexrank', 'spacy'],
    'params': {
        'lexrank': {
            'compression_ratio': 0.5,
            'threshold': 0.1,
            'damping': 0.85,
            'max_iterations': 30
        },
        'spacy': {
            'compression_ratio': 0.5,
            'spacy_model': 'en_core_web_sm'
        }
    }
}
    
    PROMPT_TEMPLATES = {
        'fstring': ["Answer this question based on the given context: {query}\n\nContext: {retrieved_contents}\n\nAnswer:"],
        'long_context_reorder': ["Answer this question based on the given context: {query}\n\nContext: {retrieved_contents}\n\nAnswer:"],
        'window_replacement': ["Answer this question based on the given context: {query}\n\nContext: {retrieved_contents}\n\nAnswer:"]
    }
    
    METRICS = {
        'retrieval': ["retrieval_f1", "retrieval_recall", "retrieval_precision", "retrieval_ndcg"],
        'passage_filter': ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
        'passage_reranker': ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
        'passage_compressor': ["retrieval_f1", "retrieval_recall", "retrieval_precision", "retrieval_ndcg"],
        'query_expansion': ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
        'prompt_maker': [{"metric_name": "bleu"}, {"metric_name": "rouge"}, {"metric_name": "meteor"}, 
                        {"metric_name": "sem_score", "embedding_model": "openai"}, {"metric_name": "g_eval"}],
        'generator': [{"metric_name": "bleu"}, {"metric_name": "rouge"}, {"metric_name": "meteor"}]
    }


class ConfigGenerator:
    def __init__(self, config_template):
        self.config_template = config_template
    
    def generate_trial_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(self.config_template)
        
        node_handlers = {
            'query_expansion': self._handle_query_expansion_node,
            'retrieval': self._handle_retrieval_node,
            'passage_reranker': self._handle_passage_reranker_node,
            'passage_filter': self._handle_passage_filter_node,
            'passage_compressor': self._handle_passage_compressor_node,
            'prompt_maker': self._handle_prompt_maker_node,
            'generator': self._handle_generator_node
        }

        for node_line in config.get('node_lines', []):
            for node in node_line.get('nodes', []):
                node_type = node.get('node_type')
                if node_type in node_handlers:
                    node_handlers[node_type](node, params)

        return config
    
    def extract_node_config(self, node_type: str) -> Optional[Dict[str, Any]]:
        for node_line in self.config_template.get('node_lines', []):
            for node in node_line.get('nodes', []):
                if node.get('node_type') == node_type:
                    return node
        return None
    
    def node_exists(self, node_type: str) -> bool:
        return self.extract_node_config(node_type) is not None
    
    def extract_metrics_from_config(self, node_type: str = 'retrieval') -> List:
        node = self.extract_node_config(node_type)
        if node and 'strategy' in node and 'metrics' in node['strategy']:
            return node['strategy']['metrics']
        return NodeDefaults.METRICS.get(node_type, []) if self.node_exists(node_type) else []
    
    def extract_unified_parameters(self, node_type: str) -> Dict[str, Any]:
        extractors = {
            'query_expansion': self._extract_query_expansion_unified,
            'retrieval': self._extract_retrieval_unified,
            'passage_reranker': self._extract_reranker_unified,
            'passage_filter': self._extract_filter_unified,
            'passage_compressor': self._extract_compressor_unified,
            'prompt_maker': self._extract_prompt_maker_unified,
            'generator': self._extract_generator_unified
        }
        
        extractor = extractors.get(node_type)
        return extractor() if extractor else {}
    
    def _extract_query_expansion_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("query_expansion")
        if not node:
            return {}
        
        result = {
            'methods': [],
            'models': {},
            'temperatures': {},
            'max_tokens': {}
        }
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if not method:
                continue
                
            result['methods'].append(method)
            
            if method in ['query_decompose', 'hyde', 'multi_query_expansion']:
                if 'model' in module:
                    result['models'][method] = Utils.ensure_list(module['model'])
                    
                if method == 'hyde' and 'max_token' in module:
                    result['max_tokens'][method] = Utils.ensure_list(module['max_token'])
                    
                elif method == 'multi_query_expansion' and 'temperature' in module:
                    result['temperatures'][method] = Utils.ensure_list(module['temperature'])
        
        strategy = node.get('strategy', {})
        retrieval_modules = strategy.get('retrieval_modules', [])
        if retrieval_modules:
            result['retrieval_options'] = self._extract_retrieval_modules_info(retrieval_modules)
        
        return result
    
    def _extract_retrieval_modules_info(self, modules: List[Dict]) -> Dict[str, List]:
        result = {
            'methods': [],
            'vectordb_names': [],
            'bm25_tokenizers': []
        }
        
        for module in modules:
            module_type = module.get('module_type')
            if not module_type:
                continue
                
            result['methods'].append(module_type)
            
            if module_type == 'vectordb' and 'vectordb' in module:
                Utils.add_to_result_list(module['vectordb'], result['vectordb_names'])
            elif module_type == 'bm25' and 'bm25_tokenizer' in module:
                Utils.add_to_result_list(module['bm25_tokenizer'], result['bm25_tokenizers'])
        
        return result
    
    def _extract_retrieval_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("retrieval")
        if not node:
            return {}
        
        result = {
            'retriever_top_k_values': Utils.ensure_list(node.get('top_k', [5])),
            'methods': [],
            'bm25_tokenizers': [],
            'vectordb_names': []
        }
        
        for module in node.get('modules', []):
            module_type = module.get('module_type')
            if not module_type:
                continue
                
            result['methods'].append(module_type)
            
            if module_type == 'bm25' and 'bm25_tokenizer' in module:
                Utils.add_to_result_list(module['bm25_tokenizer'], result['bm25_tokenizers'])
            elif module_type == 'vectordb':
                vdb_names = module.get('vectordb', ['default'])
                Utils.add_to_result_list(vdb_names, result['vectordb_names'])
        
        return result
    
    def _extract_reranker_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("passage_reranker")
        if not node:
            return {}
        
        result = {
            'top_k_values': Utils.ensure_list(node.get('top_k', [5])),
            'methods': [],
            'models': {}
        }
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if not method:
                continue
                
            result['methods'].append(method)
            
            model_key = 'model' if method in ['flashrank_reranker', 'openvino_reranker'] else 'model_name'
            if model_key in module:
                result['models'][method] = Utils.ensure_list(module[model_key])
        
        return result
    
    def _extract_filter_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("passage_filter")
        if not node:
            return {}
        
        result = {
            'methods': [],
            'threshold_values': {},
            'percentile_values': {}
        }
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if not method:
                continue
                
            result['methods'].append(method)
            
            if method in ["threshold_cutoff", "similarity_threshold_cutoff"] and "threshold" in module:
                result['threshold_values'][method] = Utils.ensure_list(module["threshold"])
            elif method in ["percentile_cutoff", "similarity_percentile_cutoff"] and "percentile" in module:
                result['percentile_values'][method] = Utils.ensure_list(module["percentile"])
        
        return result
    
    def _extract_compressor_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("passage_compressor")
        if not node:
            return {}
        
        result = {
            'methods': [],
            'llms': [],
            'models': [],
            'compression_ratios': {},
            'thresholds': {},
            'dampings': {},
            'max_iterations': {},
            'spacy_models': {}
        }
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if method:
                result['methods'].append(method)
                
                if method in ['tree_summarize', 'refine']:
                    if 'llm' in module:
                        Utils.add_to_result_list(module['llm'], result['llms'])
                    if 'model' in module:
                        Utils.add_to_result_list(module['model'], result['models'])
                
                elif method == 'lexrank':
                    if 'compression_ratio' in module:
                        result['compression_ratios'][method] = Utils.ensure_list(module['compression_ratio'])
                    if 'threshold' in module:
                        result['thresholds'][method] = Utils.ensure_list(module['threshold'])
                    if 'damping' in module:
                        result['dampings'][method] = Utils.ensure_list(module['damping'])
                    if 'max_iterations' in module:
                        result['max_iterations'][method] = Utils.ensure_list(module['max_iterations'])
                
                elif method == 'spacy':
                    if 'compression_ratio' in module:
                        result['compression_ratios'][method] = Utils.ensure_list(module['compression_ratio'])
                    if 'spacy_model' in module:
                        result['spacy_models'][method] = Utils.ensure_list(module['spacy_model'])
        
        return result
        
    def _extract_prompt_maker_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("prompt_maker")
        if not node:
            return {'methods': list(NodeDefaults.PROMPT_TEMPLATES.keys()), 'prompt_indices': [0]}
        
        result = {
            'methods': [],
            'prompt_indices': []
        }
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if method:
                result['methods'].append(method)
                
                if 'prompt' in module:
                    prompts = module['prompt']
                    result['prompt_indices'].extend(list(range(len(prompts))))
        
        result['prompt_indices'] = list(set(result['prompt_indices'])) or [0]
        
        return result
    
    def _extract_generator_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("generator")
        if not node:
            return {}
        
        result = {
            'models': [],
            'temperatures': [],
            'module_types': [],
            'llms': []
        }
        
        for module in node.get('modules', []):
            if 'model' in module:
                Utils.add_to_result_list(module['model'], result['models'])
            if 'llm' in module:
                Utils.add_to_result_list(module['llm'], result['models'])
                Utils.add_to_result_list(module['llm'], result['llms'])
            if 'temperature' in module:
                Utils.add_to_result_list(module['temperature'], result['temperatures'])
            if 'module_type' in module:
                Utils.add_to_result_list(module['module_type'], result['module_types'])
        
        return result
    
    def get_prompt_templates_from_config(self, config, module_type):
        prompt_node = self.extract_node_config("prompt_maker")
        if prompt_node:
            for module in prompt_node.get('modules', []):
                if module.get('module_type') == module_type:
                    return module.get('prompt', [])
        
        return NodeDefaults.PROMPT_TEMPLATES.get(module_type, [])
    
    def extract_query_expansion_retrieval_options(self) -> Dict[str, Any]:
        qe_params = self.extract_unified_parameters('query_expansion')
        return qe_params.get('retrieval_options', {})
    
    def extract_retrieval_options(self) -> Dict[str, List]:
        return self.extract_unified_parameters('retrieval')
    
    def extract_query_expansion_options(self) -> Dict[str, List]:
        qe_params = self.extract_unified_parameters('query_expansion')
        
        result = {
            'methods': qe_params.get('methods', []),
            'models': [],
            'temperatures': [],
            'max_tokens': []
        }
        
        for method_models in qe_params.get('models', {}).values():
            result['models'].extend(method_models)
        
        for method_temps in qe_params.get('temperatures', {}).values():
            result['temperatures'].extend(method_temps)
            
        for method_tokens in qe_params.get('max_tokens', {}).values():
            result['max_tokens'].extend(method_tokens)
        
        return result
    
    def extract_passage_reranker_options(self) -> Dict[str, List]:
        reranker_params = self.extract_unified_parameters('passage_reranker')
        
        result = {
            'methods': reranker_params.get('methods', []),
            'top_k_values': reranker_params.get('top_k_values', []),
            'models': []
        }
        
        for method_models in reranker_params.get('models', {}).values():
            result['models'].extend(method_models)
            
        for method in result['methods']:
            if method in NodeDefaults.PASSAGE_RERANKER['models']:
                Utils.add_to_result_list(NodeDefaults.PASSAGE_RERANKER['models'][method], result['models'])
        
        return result
    
    def extract_passage_compressor_options(self) -> Dict[str, List]:
        return self.extract_unified_parameters('passage_compressor')
    
    def extract_prompt_maker_options(self) -> Tuple[List[str], List[int]]:
        params = self.extract_unified_parameters('prompt_maker')
        return params.get('methods', []), params.get('prompt_indices', [])
    
    def extract_generator_parameters(self) -> Dict[str, List]:
        return self.extract_unified_parameters('generator')
    
    def extract_passage_filter_metrics_from_config(self) -> List[str]:
        return self.extract_metrics_from_config(node_type='passage_filter')
    
    def extract_passage_compressor_metrics_from_config(self) -> List[str]:
        return self.extract_metrics_from_config(node_type='passage_compressor')
    
    def extract_passage_reranker_metrics_from_config(self) -> List[str]:
        return self.extract_metrics_from_config(node_type='passage_reranker')
    
    def extract_query_expansion_metrics_from_config(self) -> List[str]:
        return self.extract_metrics_from_config(node_type='query_expansion')
    
    def extract_retrieval_metrics_from_config(self) -> List[str]:
        return self.extract_metrics_from_config(node_type='retrieval')
    
    def extract_generation_metrics_from_config(self, node_type: str = 'generator') -> List[Dict[str, str]]:
        return self.extract_metrics_from_config(node_type=node_type)
    
    def extract_generator_models_from_node(self, node_type: str) -> List[str]:
        models = []
        node_config = self.extract_node_config(node_type)
        if node_config and 'strategy' in node_config:
            for gen_module in node_config['strategy'].get('generator_modules', []):
                if 'model' in gen_module:
                    Utils.add_to_result_list(gen_module['model'], models)
        return models
    
    def extract_generic_options(self, node_type: str) -> Dict[str, List]:
        result = {
            'methods': [],
            'top_k_values': []
        }
        
        node_config = self.extract_node_config(node_type)
        if not node_config:
            return result
        
        top_k = node_config.get('top_k', [10])
        result['top_k_values'] = [top_k] if not isinstance(top_k, list) else top_k
        
        for module in node_config.get('modules', []):
            module_type = module.get('module_type')
            if module_type and module_type not in result['methods']:
                result['methods'].append(module_type)
        
        if not result['methods'] and node_type == 'query_expansion':
            result['methods'] = NodeDefaults.QUERY_EXPANSION['methods']
        elif not result['methods'] and node_type == 'passage_reranker':
            result['methods'] = NodeDefaults.PASSAGE_RERANKER['methods']
        elif not result['methods'] and node_type == 'passage_filter':
            result['methods'] = NodeDefaults.PASSAGE_FILTER['methods']
        elif not result['methods'] and node_type == 'passage_compressor':
            result['methods'] = NodeDefaults.PASSAGE_COMPRESSOR['methods']
        
        return result
    
    def extract_llm_based_options(self, node_type: str) -> Dict[str, List]:
        result = self.extract_generic_options(node_type)
        result.update({
            'generator_module_types': [],
            'llms': [],
            'models': [],
            'temperatures': [],
            'max_tokens': []
        })
        
        node_config = self.extract_node_config(node_type)
        if not node_config:
            return result
        
        for module in node_config.get('modules', []):
            for key in ['generator_module_type', 'llm', 'model', 'temperature', 'max_token']:
                if key in module:
                    result_key = key + 's' if key != 'generator_module_type' else 'generator_module_types'
                    Utils.add_to_result_list(module[key], result[result_key])
        
        if node_type == 'query_expansion' and not result['generator_module_types']:
            result['generator_module_types'] = [NodeDefaults.QUERY_EXPANSION['params']['generator_module_type']]
        if node_type == 'query_expansion' and not result['llms']:
            result['llms'] = [NodeDefaults.QUERY_EXPANSION['params']['llm']]
        if node_type == 'query_expansion' and not result['models']:
            result['models'] = [NodeDefaults.QUERY_EXPANSION['params']['model']]
        if node_type == 'query_expansion' and not result['max_tokens']:
            result['max_tokens'] = [NodeDefaults.QUERY_EXPANSION['params']['max_token']]
        
        return result
    
    def get_query_expansion_retrieval_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        retrieval_config = {
            'retrieval_method': params.get('query_expansion_retrieval_method', 'bm25'),
            'top_k': params.get('query_expansion_top_k', 10)
        }
        
        if retrieval_config['retrieval_method'] == 'bm25':
            retrieval_config['bm25_tokenizer'] = params.get('query_expansion_bm25_tokenizer', 'space')
        elif retrieval_config['retrieval_method'] == 'vectordb':
            retrieval_config['vectordb_name'] = params.get('query_expansion_vectordb_name', 'default')
        
        return retrieval_config
    
    def _handle_query_expansion_node(self, node, params):
        if 'query_expansion_method' not in params:
            return
            
        node['modules'] = []
        method = params['query_expansion_method']
        
        if method == 'pass_query_expansion':
            node['modules'].append({'module_type': 'pass_query_expansion'})
        elif method in ['query_decompose', 'hyde', 'multi_query_expansion']:
            module_config = {
                'module_type': method,
                'generator_module_type': params.get('query_expansion_generator_module_type', 
                                                   NodeDefaults.QUERY_EXPANSION['params']['generator_module_type']),
                'llm': params.get('query_expansion_llm', NodeDefaults.QUERY_EXPANSION['params']['llm']),
                'model': params.get('query_expansion_model', NodeDefaults.QUERY_EXPANSION['params']['model'])
            }
            
            if method == 'hyde':
                module_config['max_token'] = params.get('query_expansion_max_token', 
                                                       NodeDefaults.QUERY_EXPANSION['params']['max_token'])
            elif method == 'multi_query_expansion' and 'query_expansion_temperature' in params:
                module_config['temperature'] = params['query_expansion_temperature']
            
            node['modules'].append(module_config)
    
    def _handle_retrieval_node(self, node, params):
        if 'query_expansion_method' in params and params['query_expansion_method'] != 'pass_query_expansion':
            return
        
        node['top_k'] = params['retriever_top_k']
        node['modules'] = []
        
        if params['retrieval_method'] == 'bm25':
            node['modules'].append({
                'module_type': 'bm25',
                'bm25_tokenizer': params['bm25_tokenizer']
            })
        elif params['retrieval_method'] == 'vectordb':
            node['modules'].append({
                'module_type': 'vectordb',
                'vectordb': params.get('vectordb_name', 'default')
            })
    
    def _handle_passage_reranker_node(self, node, params):
        if 'passage_reranker_method' not in params:
            return
            
        node['modules'] = []
        node['top_k'] = params.get('reranker_top_k', params['retriever_top_k'])
        
        reranker_config = {
            'module_type': params['passage_reranker_method']
        }
        
        model_key = 'model' if params['passage_reranker_method'] in ['openvino_reranker', 'flashrank_reranker'] else 'model_name'
        param_key = 'reranker_model' if model_key == 'model' else 'reranker_model_name'
        
        if param_key in params:
            reranker_config[model_key] = params[param_key]
        
        if 'reranker_batch' in params:
            reranker_config['batch'] = params['reranker_batch']
        
        node['modules'].append(reranker_config)
    
    def _handle_passage_filter_node(self, node, params):
        if 'passage_filter_method' not in params:
            return
            
        node['modules'] = []
        method = params['passage_filter_method']
        
        filter_configs = {
            'threshold_cutoff': {
                'module_type': 'threshold_cutoff',
                'threshold': params.get('threshold', NodeDefaults.PASSAGE_FILTER['params']['threshold_cutoff']['threshold']),
                'reverse': False
            },
            'percentile_cutoff': {
                'module_type': 'percentile_cutoff',
                'percentile': params.get('percentile', NodeDefaults.PASSAGE_FILTER['params']['percentile_cutoff']['percentile']),
                'reverse': False
            },
            'similarity_threshold_cutoff': {
                'module_type': 'similarity_threshold_cutoff',
                'threshold': params.get('threshold', NodeDefaults.PASSAGE_FILTER['params']['similarity_threshold_cutoff']['threshold']),
                'batch': params.get('batch', NodeDefaults.PASSAGE_FILTER['params']['similarity_threshold_cutoff']['batch']),
                'embedding_model': params.get('embedding_model', NodeDefaults.PASSAGE_FILTER['params']['similarity_threshold_cutoff']['embedding_model'])
            },
            'similarity_percentile_cutoff': {
                'module_type': 'similarity_percentile_cutoff',
                'percentile': params.get('percentile', NodeDefaults.PASSAGE_FILTER['params']['similarity_percentile_cutoff']['percentile']),
                'batch': params.get('batch', NodeDefaults.PASSAGE_FILTER['params']['similarity_percentile_cutoff']['batch']),
                'embedding_model': params.get('embedding_model', NodeDefaults.PASSAGE_FILTER['params']['similarity_percentile_cutoff']['embedding_model'])
            }
        }
        
        if method in filter_configs:
            node['modules'].append(filter_configs[method])
        else:
            node['modules'].append({'module_type': 'pass_passage_filter'})
    
    def _handle_passage_compressor_node(self, node, params):
        if 'passage_compressor_method' not in params:
            return
            
        node['modules'] = []
        method = params['passage_compressor_method']
        
        if method == 'pass_compressor':
            node['modules'].append({'module_type': 'pass_compressor'})
        
        elif method in ['tree_summarize', 'refine']:
            node['modules'].append({
                'module_type': method,
                'llm': params.get('compressor_llm', 'openai'),
                'model': params.get('compressor_model', 'gpt-4o-mini'),
                'batch': params.get('compressor_batch', 16)
            })
        
        elif method == 'lexrank':
            config = {
                'module_type': method,
                'compression_ratio': params.get('compressor_compression_ratio', 0.5),
                'threshold': params.get('compressor_threshold', 0.1),
                'damping': params.get('compressor_damping', 0.85),
                'max_iterations': params.get('compressor_max_iterations', 30)
            }
            node['modules'].append(config)
        
        elif method == 'spacy':
            config = {
                'module_type': method,
                'compression_ratio': params.get('compressor_compression_ratio', 0.5),
                'spacy_model': params.get('compressor_spacy_model', 'en_core_web_sm')
            }
            node['modules'].append(config)
    
    def _handle_prompt_maker_node(self, node, params):
        if 'prompt_maker_method' not in params:
            return
            
        node['modules'] = []
        method = params['prompt_maker_method']
        prompt_template_idx = params.get('prompt_template_idx', 0)
        
        prompt_templates = self.get_prompt_templates_from_config(self.config_template, method)
        if prompt_template_idx >= len(prompt_templates):
            prompt_template_idx = 0
            
        node['modules'].append({
            'module_type': method,
            'prompt': [prompt_templates[prompt_template_idx]]
        })
    
    def _handle_generator_node(self, node, params):
        modules_data = node.get('modules', [])
        modules_format = 'dict' if isinstance(modules_data, dict) and 'modules' in modules_data else 'list'
        
        if modules_format == 'dict':
            node['modules'] = {'modules': []}
        else:
            node['modules'] = []
        
        original_config = self._get_original_generator_config()
        
        generator_config = {
            'module_type': params.get('generator_module_type', original_config['module_type'])
        }
        
        if generator_config['module_type'] == 'vllm_api':
            if original_config.get('uri'):
                generator_config['uri'] = original_config['uri']
            if 'generator_model' in params:
                generator_config['llm'] = params['generator_model']
            generator_config['max_tokens'] = original_config.get('max_tokens', 400)
        else:
            generator_config['llm'] = params.get('generator_llm', params.get('llm', 'openai'))
            if 'generator_model' in params:
                generator_config['model'] = params['generator_model']
        
        if 'generator_temperature' in params:
            generator_config['temperature'] = params['generator_temperature']
        elif 'temperature' in params:
            generator_config['temperature'] = params['temperature']
        
        if modules_format == 'dict':
            node['modules']['modules'].append(generator_config)
        else:
            node['modules'].append(generator_config)
    
    def _get_original_generator_config(self) -> Dict[str, Any]:
        gen_node = self.extract_node_config('generator')
        if gen_node and gen_node.get('modules'):
            first_module = gen_node['modules'][0] if isinstance(gen_node['modules'], list) else gen_node['modules'].get('modules', [{}])[0]
            return {
                'module_type': first_module.get('module_type', 'llama_index_llm'),
                'uri': first_module.get('uri'),
                'max_tokens': first_module.get('max_tokens', 400)
            }
        return {'module_type': 'llama_index_llm', 'max_tokens': 400}
    
    def _get_modules_data(self, node):
        modules_data = node.get('modules', [])
        if isinstance(modules_data, dict) and 'modules' in modules_data:
            return modules_data['modules'], 'dict'
        return modules_data, 'list'
    
    def _set_modules_data(self, node, modules, modules_format):
        if modules_format == 'dict':
            if 'modules' not in node:
                node['modules'] = {}
            node['modules']['modules'] = modules
        else:
            node['modules'] = modules
    
    def _add_module_to_node(self, node, module_config, modules_format):
        if modules_format == 'dict':
            if 'modules' not in node:
                node['modules'] = {'modules': []}
            node['modules']['modules'].append(module_config)
        else:
            if 'modules' not in node:
                node['modules'] = []
            node['modules'].append(module_config)
    
    def _add_to_result_list(self, values, result_list):
        Utils.add_to_result_list(values, result_list)
    
    def _get_default_metrics(self, node_type):
        return NodeDefaults.METRICS.get(node_type, [])