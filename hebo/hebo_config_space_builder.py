import os
import json
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union

from hebo.design_space.design_space import DesignSpace

from pipeline.config_manager import ConfigGenerator



class HEBOConfigSpaceBuilder:
    
    def __init__(self, config_generator: ConfigGenerator):
        self.config_generator = config_generator
        self.unified_extractor = config_generator
        self._extract_options()
    
    def _extract_options(self):
        self.retrieval_options = self.config_generator.extract_retrieval_options()
        self.query_expansion_options = self.config_generator.extract_query_expansion_options()
        self.query_expansion_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
        self.reranker_options = self.config_generator.extract_passage_reranker_options()
        self.filter_options = self.config_generator.extract_generic_options('passage_filter')
        self.compressor_options = self.config_generator.extract_passage_compressor_options()
        self.prompt_options = self.config_generator.extract_prompt_maker_options()
        self.generator_options = self.config_generator.extract_generator_parameters()
    
    def build_design_space(self) -> Tuple[DesignSpace, Dict[str, Any]]:
        params = []
        param_info = {}
        
        self._add_query_expansion_params(params, param_info)
        self._add_retrieval_params(params, param_info)
        self._add_reranker_params(params, param_info)
        self._add_filter_params(params, param_info)
        self._add_compressor_params(params, param_info)
        self._add_prompt_params(params, param_info)
        self._add_generator_params(params, param_info)
        
        if not params:
            params.append({'name': 'dummy', 'type': 'num', 'lb': 0, 'ub': 1})
        
        design_space = DesignSpace().parse(params)
        
        param_info['_n_params'] = len(params)
        
        print(f"\n[DEBUG] HEBO Design Space built with {len(params)} parameters")
        for param in params:
            print(f"  - {param['name']}: {param['type']}")
        
        return design_space, param_info
    
    def _add_query_expansion_params(self, params: List[Dict], param_info: Dict):
        if not self.config_generator.node_exists("query_expansion"):
            return
        
        methods = self.query_expansion_options.get('methods', [])
        if not methods:
            return
            
        params.append({
            'name': 'query_expansion_method',
            'type': 'cat',
            'categories': methods
        })
        param_info['query_expansion_method'] = {'type': 'categorical', 'values': methods}
        
        active_methods = [m for m in methods if m != 'pass_query_expansion']
        if not active_methods:
            return
        
        qe_retrieval_methods = self.query_expansion_retrieval_options.get('methods', [])
        if qe_retrieval_methods:
            params.append({
                'name': 'query_expansion_retrieval_method',
                'type': 'cat',
                'categories': qe_retrieval_methods
            })
            param_info['query_expansion_retrieval_method'] = {'type': 'categorical', 'values': qe_retrieval_methods}
        
        if self.query_expansion_retrieval_options.get('bm25_tokenizers'):
            params.append({
                'name': 'query_expansion_bm25_tokenizer',
                'type': 'cat',
                'categories': self.query_expansion_retrieval_options['bm25_tokenizers']
            })
        
        if self.query_expansion_options.get('models'):
            params.append({
                'name': 'query_expansion_model',
                'type': 'cat',
                'categories': self.query_expansion_options['models']
            })
        
        if self.query_expansion_options.get('temperatures'):
            temps = self.query_expansion_options['temperatures']
            if isinstance(temps, list) and len(temps) >= 2:
                params.append({
                    'name': 'query_expansion_temperature',
                    'type': 'num',
                    'lb': min(temps),
                    'ub': max(temps)
                })
            elif isinstance(temps, list) and len(temps) == 1:
                params.append({
                    'name': 'query_expansion_temperature',
                    'type': 'num',
                    'lb': max(0.0, temps[0] - 0.2),
                    'ub': min(1.0, temps[0] + 0.2)
                })
        
        if self.query_expansion_options.get('max_tokens'):
            tokens = self.query_expansion_options['max_tokens']
            if isinstance(tokens, list) and len(tokens) >= 2:
                params.append({
                    'name': 'query_expansion_max_token',
                    'type': 'int',
                    'lb': min(tokens),
                    'ub': max(tokens)
                })
            elif isinstance(tokens, list) and len(tokens) == 1:
                params.append({
                    'name': 'query_expansion_max_token',
                    'type': 'int',
                    'lb': max(1, int(tokens[0] * 0.5)),
                    'ub': int(tokens[0] * 1.5)
                })
    
    def _add_retrieval_params(self, params: List[Dict], param_info: Dict):
        if not self.config_generator.node_exists("retrieval"):
            return
        
        top_k_values = self.retrieval_options.get('retriever_top_k_values', [10])
        if isinstance(top_k_values, list) and len(top_k_values) >= 2:
            params.append({
                'name': 'retriever_top_k',
                'type': 'int',
                'lb': min(top_k_values),
                'ub': max(top_k_values)
            })
        elif isinstance(top_k_values, list) and len(top_k_values) == 1:
            params.append({
                'name': 'retriever_top_k',
                'type': 'int',
                'lb': max(1, int(top_k_values[0] * 0.5)),
                'ub': int(top_k_values[0] * 1.5)
            })
        else:
            params.append({
                'name': 'retriever_top_k',
                'type': 'int',
                'lb': 5,
                'ub': 20
            })
        
        methods = self.retrieval_options.get('methods', ['bm25'])
        if methods:
            params.append({
                'name': 'retrieval_method',
                'type': 'cat',
                'categories': methods
            })
            param_info['retrieval_method'] = {'type': 'categorical', 'values': methods}
        
        if 'bm25' in methods and self.retrieval_options.get('bm25_tokenizers'):
            params.append({
                'name': 'bm25_tokenizer',
                'type': 'cat',
                'categories': self.retrieval_options['bm25_tokenizers']
            })
        
        if 'vectordb' in methods and self.retrieval_options.get('vectordb_names'):
            params.append({
                'name': 'vectordb_name',
                'type': 'cat',
                'categories': self.retrieval_options['vectordb_names']
            })
    
    def _add_reranker_params(self, params: List[Dict], param_info: Dict):
        if not self.config_generator.node_exists("passage_reranker"):
            return
        
        methods = self.reranker_options.get('methods', [])
        if methods:
            params.append({
                'name': 'passage_reranker_method',
                'type': 'cat',
                'categories': methods
            })
            param_info['passage_reranker_method'] = {'type': 'categorical', 'values': methods}
        
        if self.reranker_options.get('models'):
            all_models = []
            for model_list in self.reranker_options['models']:
                if isinstance(model_list, list):
                    all_models.extend(model_list)
                else:
                    all_models.append(model_list)
            
            unique_models = []
            seen = set()
            for model in all_models:
                if model not in seen:
                    seen.add(model)
                    unique_models.append(model)
            
            if unique_models:
                params.append({
                    'name': 'reranker_model_name',
                    'type': 'cat',
                    'categories': unique_models
                })
        
        top_k_values = self.reranker_options.get('top_k_values', [5])
        if isinstance(top_k_values, list) and len(top_k_values) >= 2:
            params.append({
                'name': 'reranker_top_k',
                'type': 'int',
                'lb': min(top_k_values),
                'ub': max(top_k_values)
            })
        elif isinstance(top_k_values, list) and len(top_k_values) == 1:
            params.append({
                'name': 'reranker_top_k',
                'type': 'int',
                'lb': max(1, int(top_k_values[0] * 0.5)),
                'ub': int(top_k_values[0] * 1.5)
            })
    
    def _add_filter_params(self, params: List[Dict], param_info: Dict):
        if not self.config_generator.node_exists("passage_filter"):
            return
        
        filter_config = self.config_generator.extract_node_config("passage_filter")
        if not filter_config or not filter_config.get("modules", []):
            return
        
        filter_methods = []
        threshold_ranges = {}
        percentile_ranges = {}
        
        for module in filter_config.get("modules", []):
            module_type = module.get("module_type")
            if module_type:
                filter_methods.append(module_type)
                
                if module_type in ["threshold_cutoff", "similarity_threshold_cutoff"] and "threshold" in module:
                    thresholds = module["threshold"] if isinstance(module["threshold"], list) else [module["threshold"]]
                    if len(thresholds) == 1:
                        threshold_ranges[module_type] = (max(0.0, thresholds[0] - 0.2), min(1.0, thresholds[0] + 0.2))
                    else:
                        threshold_ranges[module_type] = (min(thresholds), max(thresholds))
                
                elif module_type in ["percentile_cutoff", "similarity_percentile_cutoff"] and "percentile" in module:
                    percentiles = module["percentile"] if isinstance(module["percentile"], list) else [module["percentile"]]
                    if len(percentiles) == 1:
                        percentile_ranges[module_type] = (max(0.0, percentiles[0] - 0.2), min(1.0, percentiles[0] + 0.2))
                    else:
                        percentile_ranges[module_type] = (min(percentiles), max(percentiles))
        
        if filter_methods:
            params.append({
                'name': 'passage_filter_method',
                'type': 'cat',
                'categories': filter_methods
            })
            param_info['passage_filter_method'] = {
                'type': 'categorical',
                'values': filter_methods,
                'threshold_ranges': threshold_ranges,
                'percentile_ranges': percentile_ranges
            }
    
    def _add_compressor_params(self, params: List[Dict], param_info: Dict):
        if not self.config_generator.node_exists("passage_compressor"):
            return
        
        methods = self.compressor_options.get('methods', [])
        if methods:
            params.append({
                'name': 'passage_compressor_method',
                'type': 'cat',
                'categories': methods
            })
            param_info['passage_compressor_method'] = {'type': 'categorical', 'values': methods}
        
        if self.compressor_options.get('llms'):
            params.append({
                'name': 'compressor_llm',
                'type': 'cat',
                'categories': self.compressor_options['llms']
            })
        
        if self.compressor_options.get('models'):
            params.append({
                'name': 'compressor_model',
                'type': 'cat',
                'categories': self.compressor_options['models']
            })
    
    #REMOBVED evaluation of prompt maker here!! needs to consider conditional use of evaluation here for local optimization    
    def _add_prompt_params(self, params: List[Dict], param_info: Dict):
        if not self.config_generator.node_exists("prompt_maker"):
            return
        
        prompt_methods, prompt_indices = self.prompt_options
        
        if prompt_methods:
            params.append({
                'name': 'prompt_maker_method',
                'type': 'cat',
                'categories': prompt_methods
            })
        
        if prompt_indices and len(prompt_indices) > 1:
            params.append({
                'name': 'prompt_template_idx',
                'type': 'int',
                'lb': min(prompt_indices),
                'ub': max(prompt_indices)
            })
        
    
    def _add_generator_params(self, params: List[Dict], param_info: Dict):
        if not self.config_generator.node_exists("generator"):
            return
        
        if self.generator_options.get('models'):
            params.append({
                'name': 'generator_model',
                'type': 'cat',
                'categories': self.generator_options['models']
            })
        
        if self.generator_options.get('temperatures'):
            temps = self.generator_options['temperatures']
            if isinstance(temps, list) and len(temps) >= 2:
                params.append({
                    'name': 'generator_temperature',
                    'type': 'num',
                    'lb': min(temps),
                    'ub': max(temps)
                })
            elif isinstance(temps, list) and len(temps) == 1:
                params.append({
                    'name': 'generator_temperature',
                    'type': 'num',
                    'lb': max(0.0, temps[0] - 0.2),
                    'ub': min(1.0, temps[0] + 0.2)
                })
        
        if self.generator_options.get('module_types') and len(self.generator_options['module_types']) > 1:
            params.append({
                'name': 'generator_module_type',
                'type': 'cat',
                'categories': self.generator_options['module_types']
            })
    
    def clean_config(self, config: Dict[str, Any], param_info: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = config.copy()
        
        if 'query_expansion_method' in cleaned:
            qe_method = cleaned.get('query_expansion_method')
            
            if qe_method != 'pass_query_expansion':
                cleaned.pop('retrieval_method', None)
                cleaned.pop('bm25_tokenizer', None)
                cleaned.pop('vectordb_name', None)
                
                if qe_method not in ['query_decompose', 'hyde', 'multi_query_expansion']:
                    cleaned.pop('query_expansion_llm', None)
                    cleaned.pop('query_expansion_model', None)
                if qe_method != 'multi_query_expansion':
                    cleaned.pop('query_expansion_temperature', None)
                if qe_method != 'hyde':
                    cleaned.pop('query_expansion_max_token', None)
                    
                if cleaned.get('query_expansion_retrieval_method') == 'vectordb':
                    if 'query_expansion_vectordb_name' not in cleaned:
                        qe_retrieval_options = self.query_expansion_retrieval_options
                        vdb_names = qe_retrieval_options.get('vectordb_names', ['default'])
                        if vdb_names:
                            import random
                            cleaned['query_expansion_vectordb_name'] = random.choice(vdb_names) if len(vdb_names) > 1 else vdb_names[0]
                        else:
                            cleaned['query_expansion_vectordb_name'] = 'default'
                            
                elif cleaned.get('query_expansion_retrieval_method') == 'bm25':
                    if 'query_expansion_bm25_tokenizer' not in cleaned:
                        qe_retrieval_options = self.query_expansion_retrieval_options
                        tokenizers = qe_retrieval_options.get('bm25_tokenizers', ['space'])
                        if tokenizers:
                            import random
                            cleaned['query_expansion_bm25_tokenizer'] = random.choice(tokenizers) if len(tokenizers) > 1 else tokenizers[0]
                        else:
                            cleaned['query_expansion_bm25_tokenizer'] = 'space'
                            
            else:
                cleaned.pop('query_expansion_retrieval_method', None)
                cleaned.pop('query_expansion_bm25_tokenizer', None)
                cleaned.pop('query_expansion_vectordb_name', None)
                cleaned.pop('query_expansion_llm', None)
                cleaned.pop('query_expansion_model', None)
                cleaned.pop('query_expansion_temperature', None)
                cleaned.pop('query_expansion_max_token', None)
                
                if 'retrieval_method' not in cleaned:
                    retrieval_methods = param_info.get('retrieval_method', {}).get('values', ['bm25'])
                    cleaned['retrieval_method'] = retrieval_methods[0] if retrieval_methods else 'bm25'
        
        if cleaned.get('retrieval_method') != 'bm25':
            cleaned.pop('bm25_tokenizer', None)
        elif cleaned.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' not in cleaned:
            tokenizers = self.retrieval_options.get('bm25_tokenizers', ['space'])
            cleaned['bm25_tokenizer'] = tokenizers[0] if tokenizers else 'space'
            
        if cleaned.get('query_expansion_retrieval_method') != 'bm25':
            cleaned.pop('query_expansion_bm25_tokenizer', None)
        
        if cleaned.get('retrieval_method') != 'vectordb':
            cleaned.pop('vectordb_name', None)
        elif cleaned.get('retrieval_method') == 'vectordb' and 'vectordb_name' not in cleaned:
            vdb_names = self.retrieval_options.get('vectordb_names', ['default'])
            cleaned['vectordb_name'] = vdb_names[0] if vdb_names else 'default'
            
        if cleaned.get('query_expansion_retrieval_method') != 'vectordb':
            cleaned.pop('query_expansion_vectordb_name', None)
        
        filter_method = config.get('passage_filter_method')
        if filter_method:
            filter_info = param_info.get('passage_filter_method', {})
            
            if filter_method in ["threshold_cutoff", "similarity_threshold_cutoff"]:
                if 'threshold' not in cleaned:
                    threshold_ranges = filter_info.get('threshold_ranges', {})
                    if filter_method in threshold_ranges:
                        range_vals = threshold_ranges[filter_method]
                        cleaned['threshold'] = np.random.uniform(range_vals[0], range_vals[1])
                    else:
                        cleaned['threshold'] = np.random.uniform(0.65, 0.85)
                        
            elif filter_method in ["percentile_cutoff", "similarity_percentile_cutoff"]:
                if 'percentile' not in cleaned:
                    percentile_ranges = filter_info.get('percentile_ranges', {})
                    if filter_method in percentile_ranges:
                        range_vals = percentile_ranges[filter_method]
                        cleaned['percentile'] = np.random.uniform(range_vals[0], range_vals[1])
                    else:
                        cleaned['percentile'] = np.random.uniform(0.6, 0.8)
        
        if cleaned.get('passage_reranker_method') == 'pass_reranker':
            cleaned.pop('reranker_top_k', None)
            cleaned.pop('reranker_model_name', None)
        elif cleaned.get('passage_reranker_method') not in [
            'colbert_reranker', 'monot5', 'sentence_transformer_reranker',
            'flag_embedding_reranker', 'flag_embedding_llm_reranker',
            'openvino_reranker', 'flashrank_reranker'
        ]:
            cleaned.pop('reranker_model_name', None)
        else:
            reranker_method = cleaned.get('passage_reranker_method')
            model_name = cleaned.get('reranker_model_name')
            
            if reranker_method == 'flag_embedding_llm_reranker' and model_name:
                valid_models = ['BAAI/bge-reranker-v2-gemma', 'BAAI/bge-reranker-v2-minicpm-layerwise']
                if model_name not in valid_models:
                    cleaned['reranker_model_name'] = 'BAAI/bge-reranker-v2-gemma'
            elif reranker_method == 'flashrank_reranker' and model_name:
                valid_models = ['ms-marco-TinyBERT-L-2-v2', 'ms-marco-MiniLM-L-12-v2', 
                            'ms-marco-MultiBERT-L-12', 'rank-T5-flan', 'ce-esci-MiniLM-L12-v2']
                if model_name not in valid_models:
                    cleaned['reranker_model_name'] = 'ms-marco-MiniLM-L-12-v2'
        
        if 'reranker_top_k' in cleaned and 'retriever_top_k' in cleaned:
            cleaned['reranker_top_k'] = min(cleaned['reranker_top_k'], cleaned['retriever_top_k'])
        
        if filter_method not in ['threshold_cutoff', 'similarity_threshold_cutoff']:
            cleaned.pop('threshold', None)
        if filter_method not in ['percentile_cutoff', 'similarity_percentile_cutoff']:
            cleaned.pop('percentile', None)
        
        compressor_method = cleaned.get('passage_compressor_method')
        if compressor_method not in ['tree_summarize', 'refine']:
            cleaned.pop('compressor_llm', None)
            cleaned.pop('compressor_model', None)
        
        prompt_method = cleaned.get('prompt_maker_method')
        if prompt_method not in ['fstring', 'long_context_reorder']:
            cleaned.pop('prompt_template_idx', None)
        
        return cleaned