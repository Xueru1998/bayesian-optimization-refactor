from datetime import datetime
import os
import json
import hashlib
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class ComponentCacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.component_cache_dirs = {}
        
        for component in ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor', 'prompt_maker', 'generator']:
            component_dir = os.path.join(cache_dir, f"{component}_cache")
            os.makedirs(component_dir, exist_ok=True)
            self.component_cache_dirs[component] = component_dir
        
        self.cache_index_file = os.path.join(cache_dir, 'component_cache_index.json')
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Dict]:
        if os.path.exists(self.cache_index_file):
            with open(self.cache_index_file, 'r') as f:
                return json.load(f)
        return {component: {} for component in self.component_cache_dirs.keys()}
    
    def _save_cache_index(self):
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _parse_unified_config(self, config_str: str, config_type: str) -> Dict[str, Any]:
        parsed_params = {}
        
        if config_type == 'retrieval':
            if config_str.startswith('bm25_'):
                parsed_params['retrieval_method'] = 'bm25'
                parsed_params['bm25_tokenizer'] = config_str.replace('bm25_', '')
            elif config_str.startswith('vectordb_'):
                parsed_params['retrieval_method'] = 'vectordb'
                parsed_params['vectordb_name'] = config_str.replace('vectordb_', '')
                
        elif config_type == 'filter':
            if config_str.startswith('threshold_cutoff_'):
                parsed_params['passage_filter_method'] = 'threshold_cutoff'
                parsed_params['threshold'] = float(config_str.split('_')[-1])
            elif config_str.startswith('percentile_cutoff_'):
                parsed_params['passage_filter_method'] = 'percentile_cutoff'
                parsed_params['percentile'] = float(config_str.split('_')[-1])
            elif config_str.startswith('similarity_threshold_cutoff_'):
                parsed_params['passage_filter_method'] = 'similarity_threshold_cutoff'
                parsed_params['threshold'] = float(config_str.split('_')[-1])
            elif config_str.startswith('similarity_percentile_cutoff_'):
                parsed_params['passage_filter_method'] = 'similarity_percentile_cutoff'
                parsed_params['percentile'] = float(config_str.split('_')[-1])
                
        elif config_type == 'reranker':
            if config_str == 'pass_reranker':
                parsed_params['passage_reranker_method'] = 'pass_reranker'
            else:
                model_based_methods = ['colbert_reranker', 'sentence_transformer_reranker', 
                                    'flag_embedding_reranker', 'flag_embedding_llm_reranker']
                for method in model_based_methods:
                    if config_str.startswith(method + '_'):
                        parsed_params['passage_reranker_method'] = method
                        parsed_params['reranker_model_name'] = config_str[len(method) + 1:]
                        break
                else:
                    parsed_params['passage_reranker_method'] = config_str
                    
        elif config_type == 'compressor':
            if config_str == 'pass_compressor':
                parsed_params['passage_compressor_method'] = 'pass_compressor'
            elif config_str.startswith('tree_summarize_') or config_str.startswith('refine_'):
                parts = config_str.split('_', 2)
                if len(parts) >= 3:
                    method = parts[0] + '_' + parts[1]
                    llm_and_model = parts[2]
                    llm_parts = llm_and_model.split('_', 1)
                    if len(llm_parts) == 2:
                        parsed_params['passage_compressor_method'] = method
                        parsed_params['compressor_llm'] = llm_parts[0]
                        parsed_params['compressor_model'] = llm_parts[1]
                        
        elif config_type == 'prompt':
            if config_str == 'pass_prompt_maker':
                parsed_params['prompt_maker_method'] = 'pass_prompt_maker'
            else:
                parts = config_str.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    parsed_params['prompt_maker_method'] = parts[0]
                    parsed_params['prompt_template_idx'] = int(parts[1])
                else:
                    parsed_params['prompt_maker_method'] = config_str
                    
        elif config_type == 'query_expansion':
            if config_str == 'pass_query_expansion':
                parsed_params['query_expansion_method'] = 'pass_query_expansion'

            elif config_str.startswith('query_decompose_'):
                model = config_str.replace('query_decompose_', '')
                parsed_params['query_expansion_method'] = 'query_decompose'
                parsed_params['query_expansion_model'] = model

            elif config_str.startswith('hyde_'):
                parts = config_str.replace('hyde_', '').rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    model = parts[0]
                    max_token = int(parts[1])
                    parsed_params['query_expansion_method'] = 'hyde'
                    parsed_params['query_expansion_model'] = model
                    parsed_params['query_expansion_max_token'] = max_token
                else:
                    parsed_params['query_expansion_method'] = 'hyde'
                    parsed_params['query_expansion_max_token'] = int(config_str.split('_')[-1])

            elif config_str.startswith('multi_query_expansion_'):
                temp = float(config_str.split('_')[-1])
                parsed_params['query_expansion_method'] = 'multi_query_expansion'
                parsed_params['query_expansion_temperature'] = temp
            
            else:
                parsed_params['query_expansion_method'] = config_str
                
        return parsed_params
    
    def get_component_hash(self, component: str, params: Dict[str, Any]) -> str:
        param_dependencies = {
            'query_expansion': [],
            'retrieval': ['query_expansion'],
            'reranker': ['query_expansion', 'retrieval'],
            'filter': ['query_expansion', 'retrieval', 'reranker'],
            'compressor': ['query_expansion', 'retrieval', 'reranker', 'filter'],
            'prompt_maker': ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor'],
            'generator': ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor', 'prompt_maker']
        }
        
        relevant_params = {}
        
        for dep_component in param_dependencies.get(component, []) + [component]:
            dep_params = self._extract_relevant_params(dep_component, params)

            if component == 'retrieval' and dep_component == 'retrieval':
                qe_method = params.get('query_expansion_method')
                if qe_method and qe_method != 'pass_query_expansion':
                    dep_params['retrieval_skipped_due_to_qe'] = True
            
            for key, value in dep_params.items():
                relevant_params[f"{dep_component}_{key}"] = value
        
        param_str = json.dumps(relevant_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _extract_relevant_params(self, component: str, params: Dict[str, Any]) -> Dict[str, Any]:
        unified_configs = {
            'query_expansion': 'query_expansion_config',
            'retrieval': 'retrieval_config',
            'filter': 'passage_filter_config',
            'reranker': 'reranker_config',
            'compressor': 'compressor_config',
            'prompt_maker': 'prompt_config'
        }
        
        if component in unified_configs and unified_configs[component] in params:
            config_str = params[unified_configs[component]]
            parsed_params = self._parse_unified_config(config_str, component)

            if component == 'retrieval' and 'retriever_top_k' in params:
                parsed_params['retriever_top_k'] = params['retriever_top_k']
            elif component == 'reranker' and 'reranker_top_k' in params:
                parsed_params['reranker_top_k'] = params['reranker_top_k']
            elif component == 'compressor' and 'compressor_batch' in params:
                parsed_params['compressor_batch'] = params['compressor_batch']
            elif component == 'prompt_maker' and 'prompt_maker_generator_model' in params:
                parsed_params['prompt_maker_generator_model'] = params['prompt_maker_generator_model']

            if component == 'query_expansion':
                if 'retriever_top_k' in params:
                    parsed_params['query_expansion_top_k'] = params['retriever_top_k']
                
                if 'query_expansion_retrieval_method' in params:
                    parsed_params['query_expansion_retrieval_method'] = params['query_expansion_retrieval_method']
                if 'query_expansion_bm25_tokenizer' in params:
                    parsed_params['query_expansion_bm25_tokenizer'] = params['query_expansion_bm25_tokenizer']
                if 'query_expansion_vectordb_name' in params:
                    parsed_params['query_expansion_vectordb_name'] = params['query_expansion_vectordb_name']
            
            return parsed_params
        
        param_mappings = {
            'query_expansion': [
                'query_expansion_method', 
                'query_expansion_max_token', 
                'query_expansion_temperature',
                'query_expansion_generator_module_type',
                'query_expansion_llm',
                'query_expansion_model',
                'query_expansion_retrieval_method',
                'query_expansion_bm25_tokenizer',
                'query_expansion_vectordb_name',
                'retriever_top_k'  
            ],
            'retrieval': [
                'retriever_top_k', 
                'retrieval_method', 
                'bm25_tokenizer',
                'vectordb_name',
                'retrieval_skipped_due_to_qe'
            ],
            'reranker': [
                'passage_reranker_method', 
                'reranker_top_k', 
                'reranker_model_name', 
                'reranker_model'
            ],
            'filter': [
                'passage_filter_method', 
                'threshold',
                'percentile'
            ],
            'compressor': [
                'passage_compressor_method', 
                'compressor_llm', 
                'compressor_model', 
                'compressor_batch'
            ],
            'prompt_maker': [
                'prompt_maker_method', 
                'prompt_template_idx', 
                'prompt_maker_generator_model'
            ],
            'generator': [
                'generator_model', 
                'generator_llm',
                'generator_temperature', 
                'generator_module_type'
            ]
        }
        
        relevant_params = {}
        for param in param_mappings.get(component, []):
            if param in params:
                if param == 'retriever_top_k' and component == 'query_expansion':
                    qe_method = params.get('query_expansion_method')
                    if qe_method and qe_method != 'pass_query_expansion':
                        relevant_params['query_expansion_top_k'] = params[param]
                    else:
                        continue
                else:
                    relevant_params[param] = params[param]
        
        return relevant_params
    
    def check_cache(self, component: str, params: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        config_hash = self.get_component_hash(component, params)
        if component == 'query_expansion':
            relevant_params = self._extract_relevant_params(component, params)
            if relevant_params.get('query_expansion_method') and relevant_params['query_expansion_method'] != 'pass_query_expansion':
                print(f"[Cache Debug] Query expansion cache lookup with params: {relevant_params}")
                print(f"[Cache Debug] Hash: {config_hash}")
        
        if component in self.cache_index and config_hash in self.cache_index[component]:
            cache_info = self.cache_index[component][config_hash]
            df_path = cache_info['df_path']
            
            if os.path.exists(df_path):
                try:
                    df = pd.read_parquet(df_path)

                    if component == 'query_expansion':
                        cached_params = cache_info.get('params', {})
                        print(f"[Cache Hit] Found cached query expansion with top_k: {cached_params.get('query_expansion_top_k', 'not recorded')}")
                    
                    return df, cache_info['metrics']
                except Exception as e:
                    print(f"Error loading cached data for {component}: {e}")
                    return None, None

        if component == 'query_expansion' and params.get('query_expansion_method') and params.get('query_expansion_method') != 'pass_query_expansion':
            print(f"[Cache Miss] No cached query expansion found for hash: {config_hash}")
        
        return None, None
    
    def save_to_cache(self, component: str, params: Dict[str, Any], 
                  df: pd.DataFrame, metrics: Dict[str, Any], execution_time: float = None):
        config_hash = self.get_component_hash(component, params)
        
        df_filename = f"{component}_{config_hash}.parquet"
        df_path = os.path.join(self.component_cache_dirs[component], df_filename)
        
        try:
            df.to_parquet(df_path)
            
            if component not in self.cache_index:
                self.cache_index[component] = {}
            
            all_params = {}
            param_dependencies = {
                'query_expansion': [],
                'retrieval': ['query_expansion'],
                'reranker': ['query_expansion', 'retrieval'],
                'filter': ['query_expansion', 'retrieval', 'reranker'],
                'compressor': ['query_expansion', 'retrieval', 'reranker', 'filter'],
                'prompt_maker': ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor'],
                'generator': ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor', 'prompt_maker']
            }
            
            for dep_component in param_dependencies.get(component, []) + [component]:
                dep_params = self._extract_relevant_params(dep_component, params)
                all_params.update(dep_params)
            
            self.cache_index[component][config_hash] = {
                'df_path': df_path,
                'metrics': metrics,
                'params': all_params,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self._save_cache_index()
            
        except Exception as e:
            print(f"Error saving cache for {component}: {e}")