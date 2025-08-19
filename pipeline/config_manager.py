import copy
from typing import Dict, Any, List, Tuple, Optional
from pipeline.utils import Utils
import re
import os


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
            'flashrank_reranker': ["ms-marco-MiniLM-L-12-v2"],
            'sap_api': ["cohere-rerank-v3.5"]
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
        'methods': ['pass_compressor']
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
    
    def resolve_env_vars(self, value: str) -> str:
        if not isinstance(value, str):
            return value
        
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(pattern, replacer, value)
    
    def _extract_query_expansion_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("query_expansion")
        if not node:
            return {}
        
        result = {
            'methods': [],
            'generator_configs': [],
            'all_models': [],
            'all_temperatures': [],
            'all_max_tokens': [],
            'all_generator_types': []
        }
        
        method_groups = {}
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if not method:
                continue
                
            if method not in method_groups:
                method_groups[method] = []
                if method not in result['methods']:
                    result['methods'].append(method)
            
            if method == 'pass_query_expansion':
                continue
            
            gen_config = {
                'method': method,
                'generator_module_type': module.get('generator_module_type', 'llama_index_llm'),
                'models': Utils.ensure_list(module.get('model', [])),
                'llm': module.get('llm', 'openai')
            }
            
            result['all_generator_types'].append(gen_config['generator_module_type'])
            
            if method == 'hyde' and 'max_token' in module:
                gen_config['max_tokens'] = Utils.ensure_list(module['max_token'])
                result['all_max_tokens'].extend(gen_config['max_tokens'])
                
            elif method == 'multi_query_expansion' and 'temperature' in module:
                gen_config['temperatures'] = Utils.ensure_list(module['temperature'])
                result['all_temperatures'].extend(gen_config['temperatures'])
            
            if gen_config['generator_module_type'] == 'sap_api':
                gen_config['api_url'] = module.get('api_url', '${SAP_API_URL}')
                gen_config['bearer_token'] = module.get('bearer_token', '${SAP_BEARER_TOKEN}')
            
            method_groups[method].append(gen_config)
            result['generator_configs'].append(gen_config)
            result['all_models'].extend(gen_config['models'])
        
        result['all_generator_types'] = list(set(result['all_generator_types']))
        result['all_models'] = list(set(result['all_models']))
        result['all_temperatures'] = list(set(result['all_temperatures']))
        result['all_max_tokens'] = list(set(result['all_max_tokens']))
        
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
    
    def extract_retrieval_options(self) -> Dict[str, List]:
        retrieval_params = self._extract_retrieval_unified()
        
        return {
            'methods': retrieval_params.get('methods', []),
            'bm25_tokenizers': retrieval_params.get('bm25_tokenizers', []),
            'vectordb_names': retrieval_params.get('vectordb_names', []),
            'retriever_top_k_values': retrieval_params.get('retriever_top_k_values', [])
        }

    def extract_query_expansion_retrieval_options(self) -> Dict[str, List]:
        qe_params = self._extract_query_expansion_unified()
        retrieval_options = qe_params.get('retrieval_options', {})

        return {
            'methods': retrieval_options.get('methods', []),
            'bm25_tokenizers': retrieval_options.get('bm25_tokenizers', []),
            'vectordb_names': retrieval_options.get('vectordb_names', [])
        }
        
    def _extract_reranker_unified(self) -> Dict[str, Any]:
        node = self.extract_node_config("passage_reranker")
        if not node:
            return {}
        
        result = {
            'top_k_values': Utils.ensure_list(node.get('top_k', [5])),
            'methods': [],
            'models': {},
            'api_endpoints': {}
        }
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if not method:
                continue
                
            result['methods'].append(method)
            
            if method == 'sap_api':
                result['models'][method] = Utils.ensure_list(module.get('model_name', ['cohere-rerank-v3.5']))
                api_url = module.get('api-url') or module.get('api_url') or module.get('api_endpoint')
                if api_url:
                    result['api_endpoints'][method] = api_url
            elif method != 'pass_reranker':
                model_key = 'model' if method in ['flashrank_reranker', 'openvino_reranker'] else 'model_name'
                if model_key in module:
                    result['models'][method] = Utils.ensure_list(module[model_key])
                elif method == 'colbert_reranker':
                    result['models'][method] = ['colbert-ir/colbertv2.0']
        
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
            'compressor_configs': [],
            'all_llms': [],
            'all_models': [],
            'all_generator_types': []
        }

        seen_configs = set()
        
        for module in node.get('modules', []):
            method = module.get('module_type')
            if not method:
                continue
                
            if method not in result['methods']:
                result['methods'].append(method)
            
            if method == 'pass_compressor':
                continue

            if method in ['tree_summarize', 'refine']:
                generator_type = module.get('generator_module_type', 'llama_index_llm')
                config_key = f"{method}::{generator_type}::{module.get('llm', 'openai')}"

                if config_key in seen_configs:
                    continue
                
                seen_configs.add(config_key)
                
                comp_config = {
                    'method': method,
                    'generator_module_type': generator_type,
                    'llm': module.get('llm', 'openai'),
                    'models': Utils.ensure_list(module.get('model', []))
                }
                
                result['all_generator_types'].append(generator_type)
                
                if generator_type == 'sap_api':
                    comp_config['api_url'] = module.get('api_url')
                    if not comp_config['api_url']:
                        raise ValueError(f"SAP API URL is required for {method} compressor module but not found in config")
                
                elif generator_type == 'vllm':
                    comp_config['models'] = Utils.ensure_list(module.get('llm', []))
                    if 'tensor_parallel_size' in module:
                        comp_config['tensor_parallel_size'] = module['tensor_parallel_size']
                    if 'gpu_memory_utilization' in module:
                        comp_config['gpu_memory_utilization'] = module['gpu_memory_utilization']
                
                elif generator_type == 'openai':
                    comp_config['models'] = Utils.ensure_list(module.get('model', module.get('llm', [])))
                
                if 'batch' in module:
                    comp_config['batch'] = module['batch']

                comp_config['models'] = list(dict.fromkeys(comp_config['models']))
                
                result['compressor_configs'].append(comp_config)
                result['all_llms'].append(comp_config['llm'])
                result['all_models'].extend(comp_config['models'])

            else:
                comp_config = {
                    'method': method,
                    'compression_ratio': Utils.ensure_list(module.get('compression_ratio', [0.5]))
                }

                if method == 'lexrank':
                    comp_config['threshold'] = Utils.ensure_list(module.get('threshold', [0.1]))
                    comp_config['damping'] = Utils.ensure_list(module.get('damping', [0.85]))
                    comp_config['max_iterations'] = Utils.ensure_list(module.get('max_iterations', [30]))
                elif method == 'spacy':
                    comp_config['spacy_model'] = Utils.ensure_list(module.get('spacy_model', ['en_core_web_sm']))
                
                result['compressor_configs'].append(comp_config)
        
        result['all_generator_types'] = list(set(result['all_generator_types']))
        result['all_llms'] = list(set(result['all_llms']))
        result['all_models'] = list(set(result['all_models']))
        
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
            'module_configs': [],
            'all_models': [],
            'all_temperatures': [],
            'all_module_types': [],
            'all_max_tokens': []
        }
        
        for module in node.get('modules', []):
            module_type = module.get('module_type')
            if not module_type:
                continue
            
            if module_type == 'vllm':
                models = Utils.ensure_list(module.get('llm', []))
            else:
                models = Utils.ensure_list(module.get('model', []))
            
            module_config = {
                'module_type': module_type,
                'models': models,
                'temperatures': Utils.ensure_list(module.get('temperature', [0.7])),
                'llm': module.get('llm', 'openai')
            }
            
            if module_type == 'sap_api':
                module_config['max_tokens'] = Utils.ensure_list(module.get('max_tokens', [500]))
                module_config['api_url'] = module.get('api_url', '${SAP_API_URL}')
                module_config['bearer_token'] = module.get('bearer_token', '${SAP_BEARER_TOKEN}')
                result['all_max_tokens'].extend(module_config['max_tokens'])
            elif module_type == 'vllm':
                module_config['max_tokens'] = module.get('max_tokens', 512)
            
            result['module_configs'].append(module_config)
            result['all_models'].extend(module_config['models'])
            result['all_temperatures'].extend(module_config['temperatures'])
            result['all_module_types'].append(module_type)
        
        result['all_module_types'] = list(set(result['all_module_types']))
        result['all_models'] = list(set(result['all_models']))
        result['all_temperatures'] = list(set(result['all_temperatures']))
        result['all_max_tokens'] = list(set(result['all_max_tokens']))
        
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
    
    def _ensure_list(self, value):
        return Utils.ensure_list(value)
    
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
        if 'query_expansion_config' in params:
            self._handle_query_expansion_node_composite(node, params)
        elif 'query_expansion_method' in params:
            self._handle_query_expansion_node_legacy(node, params)
        else:
            return
    
    def _handle_query_expansion_node_composite(self, node, params):
        qe_config_str = params['query_expansion_config']
        parts = qe_config_str.split('::', 2)
        
        if len(parts) != 3:
            return
            
        method, gen_type, model = parts
        
        unified_params = self.extract_unified_parameters('query_expansion')
        gen_configs = unified_params.get('generator_configs', [])
        
        target_config = None
        for config in gen_configs:
            if (config['method'] == method and 
                config['generator_module_type'] == gen_type and 
                model in config['models']):
                target_config = config
                break
        
        if not target_config:
            return
        
        module_config = {
            'module_type': method,
            'generator_module_type': gen_type,
            'model': model
        }
        
        if gen_type == 'sap_api':
            module_config['llm'] = target_config.get('llm', 'mistralai')
            module_config['api_url'] = self.resolve_env_vars(target_config.get('api_url', '${SAP_API_URL}'))
            module_config['bearer_token'] = self.resolve_env_vars(target_config.get('bearer_token', '${SAP_BEARER_TOKEN}'))
        elif gen_type == 'llama_index_llm':
            module_config['llm'] = target_config.get('llm', 'openai')
        
        if method == 'hyde' and 'query_expansion_max_token' in params:
            module_config['max_token'] = params['query_expansion_max_token']
        elif method == 'multi_query_expansion' and 'query_expansion_temperature' in params:
            module_config['temperature'] = params['query_expansion_temperature']
        
        node['modules'] = [module_config]
        
    def _handle_query_expansion_node_legacy(self, node, params):
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
        
        if params['passage_reranker_method'] == 'sap_api':
            unified_params = self.extract_unified_parameters('passage_reranker')
            api_endpoints = unified_params.get('api_endpoints', {})
            models = unified_params.get('models', {})

            if 'sap_api' in models and models['sap_api']:
                reranker_config['model_name'] = models['sap_api'][0]
            elif 'reranker_model_name' in params:
                reranker_config['model_name'] = params['reranker_model_name']
            else:
                reranker_config['model_name'] = 'cohere-rerank-v3.5'
 
            if 'sap_api' in api_endpoints:
                reranker_config['api-url'] = api_endpoints['sap_api']
            elif 'reranker_api_url' in params:
                reranker_config['api-url'] = params['reranker_api_url']
        else:
            model_key = 'model' if params['passage_reranker_method'] in ['openvino_reranker', 'flashrank_reranker'] else 'model_name'
            param_key = 'reranker_model' if model_key == 'model' else 'reranker_model_name'
            
            if param_key in params:
                reranker_config[model_key] = params[param_key]
        
        if 'reranker_batch' in params and params['passage_reranker_method'] != 'sap_api':
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
        if 'passage_compressor_config' in params:
            self._handle_passage_compressor_node_composite(node, params)
        elif 'passage_compressor_method' in params:
            self._handle_passage_compressor_node_legacy(node, params)
        else:
            return
    
    def _handle_passage_compressor_node_composite(self, node, params):
        comp_config_str = params['passage_compressor_config']
        
        if comp_config_str == 'pass_compressor':
            node['modules'] = [{'module_type': 'pass_compressor'}]
            return
        
        # Handle lexrank (no :: separator)
        if comp_config_str == 'lexrank':
            module_config = {
                'module_type': 'lexrank',
                'compression_ratio': params.get('compression_ratio', 0.5),
                'threshold': params.get('threshold', 0.1),
                'damping': params.get('damping', 0.85),
                'max_iterations': params.get('max_iterations', 30)
            }
            node['modules'] = [module_config]
            return
        
        # Handle spacy::model_name format
        if comp_config_str.startswith('spacy::'):
            parts = comp_config_str.split('::', 1)
            module_config = {
                'module_type': 'spacy',
                'compression_ratio': params.get('compression_ratio', 0.5),
                'spacy_model': parts[1] if len(parts) > 1 else 'en_core_web_sm'
            }
            node['modules'] = [module_config]
            return
        
        # Handle other non-LLM methods
        if comp_config_str in ['sentence_rank', 'keyword_extraction', 'query_focused']:
            module_config = {
                'module_type': comp_config_str,
                'compression_ratio': params.get('compression_ratio', 0.5)
            }
            node['modules'] = [module_config]
            return
        
        # Handle LLM-based compressors
        parts = comp_config_str.split('::', 3)
        
        if len(parts) < 2:
            return
        
        if len(parts) == 3:
            method, gen_type, model = parts
        else:
            method, llm = parts[0], parts[1]
            gen_type = 'sap_api' if llm == 'mistralai' else 'llama_index_llm'
            model = parts[2] if len(parts) > 2 else None
        
        unified_params = self.extract_unified_parameters('passage_compressor')
        comp_configs = unified_params.get('compressor_configs', [])
        
        target_config = None
        for config in comp_configs:
            if len(parts) == 3:
                if (config['method'] == method and 
                    config['generator_module_type'] == gen_type and 
                    model in config['models']):
                    target_config = config
                    break
            else:
                if (config['method'] == method and 
                    config['llm'] == llm and 
                    (model is None or model in config['models'])):
                    target_config = config
                    break
        
        if not target_config:
            return
        
        module_config = {
            'module_type': method,
            'generator_module_type': target_config.get('generator_module_type', 'llama_index_llm')
        }
        
        if target_config['generator_module_type'] == 'sap_api':
            module_config['llm'] = target_config.get('llm', 'mistralai')
            module_config['model'] = model or target_config['models'][0]
            module_config['api_url'] = self.resolve_env_vars(target_config.get('api_url', '${SAP_API_URL}'))
            module_config['bearer_token'] = self.resolve_env_vars(target_config.get('bearer_token', '${SAP_BEARER_TOKEN}'))
        
        elif target_config['generator_module_type'] == 'vllm':
            module_config['llm'] = model or target_config['models'][0]
            module_config['model'] = model
            if 'tensor_parallel_size' in target_config:
                module_config['tensor_parallel_size'] = target_config['tensor_parallel_size']
            if 'gpu_memory_utilization' in target_config:
                module_config['gpu_memory_utilization'] = target_config['gpu_memory_utilization']
        
        elif target_config['generator_module_type'] == 'openai':
            module_config['llm'] = model or target_config['models'][0]
            module_config['model'] = model or target_config['models'][0]
        
        else:
            module_config['llm'] = target_config.get('llm', 'openai')
            module_config['model'] = model or target_config['models'][0]

        if 'temperature' in params:
            module_config['temperature'] = params['temperature']
        elif 'temperatures' in target_config:
            module_config['temperature'] = target_config['temperatures'][0]
        
        if 'max_tokens' in params:
            module_config['max_tokens'] = params['max_tokens']
        elif 'max_tokens' in target_config:
            module_config['max_tokens'] = target_config['max_tokens'][0]
        
        if 'batch' in target_config:
            module_config['batch'] = target_config['batch']
        
        node['modules'] = [module_config]

    def _handle_passage_compressor_node_legacy(self, node, params):
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
            node['modules'].append({
                'module_type': 'lexrank',
                'compression_ratio': params.get('compression_ratio', 0.5),
                'threshold': params.get('threshold', 0.1),
                'damping': params.get('damping', 0.85),
                'max_iterations': params.get('max_iterations', 30)
            })
        elif method == 'spacy':
            node['modules'].append({
                'module_type': 'spacy',
                'compression_ratio': params.get('compression_ratio', 0.5),
                'spacy_model': params.get('spacy_model', 'en_core_web_sm')
            })
    
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
        if 'generator_config' in params:
            self._handle_generator_node_composite(node, params)
        else:
            self._handle_generator_node_legacy(node, params)
    
    def _handle_generator_node_composite(self, node, params):
        gen_config_str = params['generator_config']
        module_type, model = gen_config_str.split('::', 1)
        
        unified_params = self.extract_unified_parameters('generator')
        module_configs = unified_params.get('module_configs', [])
        
        target_config = None
        for config in module_configs:
            if config['module_type'] == module_type and model in config['models']:
                target_config = config
                break
        
        if not target_config:
            return
        
        generator_config = {
            'module_type': module_type
        }
        
        if module_type == 'sap_api':
            generator_config['llm'] = target_config.get('llm', 'mistralai')
            generator_config['model'] = model
            generator_config['api_url'] = self.resolve_env_vars(target_config.get('api_url', '${SAP_API_URL}'))
            generator_config['bearer_token'] = self.resolve_env_vars(target_config.get('bearer_token', '${SAP_BEARER_TOKEN}'))
            
            if 'generator_max_tokens' in params:
                generator_config['max_tokens'] = params['generator_max_tokens']
            elif target_config.get('max_tokens'):
                generator_config['max_tokens'] = target_config['max_tokens'][0]
        
        elif module_type == 'vllm':
            generator_config['llm'] = model
            generator_config['max_tokens'] = target_config.get('max_tokens', 512)
        
        else:
            generator_config['llm'] = target_config.get('llm', 'openai')
            generator_config['model'] = model
        
        if 'generator_temperature' in params:
            generator_config['temperature'] = params['generator_temperature']
        
        node['modules'] = [generator_config]
    
    def _handle_generator_node_legacy(self, node, params):
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
    
    
    def get_prompt_templates_from_config(self, config: Dict[str, Any], method: str) -> List[str]:
        prompt_maker_node = None
        for node_line in config.get('node_lines', []):
            for node in node_line.get('nodes', []):
                if node.get('node_type') == 'prompt_maker':
                    prompt_maker_node = node
                    break
            if prompt_maker_node:
                break
        
        if not prompt_maker_node:
            return NodeDefaults.PROMPT_TEMPLATES.get(method, [])
        
        for module in prompt_maker_node.get('modules', []):
            if module.get('module_type') == method:
                prompts = module.get('prompt', [])
                if prompts:
                    return prompts
        
        return NodeDefaults.PROMPT_TEMPLATES.get(method, [])
    
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