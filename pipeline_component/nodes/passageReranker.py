import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.nodes.passagereranker.colbert import ColbertReranker
from autorag.nodes.passagereranker.flag_embedding import FlagEmbeddingReranker
from autorag.nodes.passagereranker.flag_embedding_llm import FlagEmbeddingLLMReranker
from autorag.nodes.passagereranker.flashrank import FlashRankReranker
from autorag.nodes.passagereranker.monot5 import MonoT5
from autorag.nodes.passagereranker.openvino import OpenVINOReranker
from autorag.nodes.passagereranker.pass_reranker import PassReranker
from autorag.nodes.passagereranker.sentence_transformer import SentenceTransformerReranker
from autorag.nodes.passagereranker.upr import Upr
from autorag.utils.util import empty_cuda_cache

import logging
logger = logging.getLogger("AutoRAG")

class PassageRerankerModule:
    def __init__(self, project_dir: str = ""):
        self.project_dir = project_dir
        
        self.rerankers = {
            "pass_reranker": PassReranker, 
            "colbert_reranker": ColbertReranker,
            "flag_embedding_reranker": FlagEmbeddingReranker,
            "flag_embedding_llm_reranker": FlagEmbeddingLLMReranker,
            "flashrank_reranker": FlashRankReranker,
            "monot5": MonoT5,
            "openvino_reranker": OpenVINOReranker,
            "sentence_transformer_reranker": SentenceTransformerReranker,
            "upr": Upr  
        }
    
    def create_reranker(self, method: str, **kwargs) -> BasePassageReranker:
        if method not in self.rerankers:
            raise ValueError(f"Unknown reranking method: {method}. Available methods: {list(self.rerankers.keys())}")
        
        reranker_class = self.rerankers[method]
        return reranker_class(project_dir=self.project_dir, **kwargs)
    
    def rerank_passages(self, 
                   df: pd.DataFrame, 
                   method: str = "colbert_reranker", 
                   top_k: int = 5,
                   batch: int = 64,
                   **kwargs) -> pd.DataFrame:
        required_columns = ["query", "retrieved_contents", "retrieved_ids", "retrieve_scores"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must have '{col}' column")

        init_params = {}
        operation_params = {'top_k': top_k}
        
        # Handle model-specific parameters for initialization
        if 'model_name' in kwargs:
            init_params['model_name'] = kwargs.pop('model_name')
        if 'model' in kwargs:
            init_params['model'] = kwargs.pop('model')
        if 'use_bf16' in kwargs:
            init_params['use_bf16'] = kwargs.pop('use_bf16')
        if 'prefix_prompt' in kwargs:
            init_params['prefix_prompt'] = kwargs.pop('prefix_prompt')
        if 'suffix_prompt' in kwargs:
            init_params['suffix_prompt'] = kwargs.pop('suffix_prompt')
        if 'cache_dir' in kwargs:
            init_params['cache_dir'] = kwargs.pop('cache_dir')
        if 'max_length' in kwargs:
            init_params['max_length'] = kwargs.pop('max_length')
        
        operation_params['batch'] = batch
        
        operation_params.update(kwargs)
        
        queries = df['query'].tolist()
        contents_list = df['retrieved_contents'].tolist()
        ids_list = df['retrieved_ids'].tolist()

        reranker = self.create_reranker(method, **init_params)
        
        if method == 'upr':
            if 'batch' in operation_params:
                del operation_params['batch']

        reranked_contents, reranked_ids, reranked_scores = reranker._pure(
            queries=queries,
            contents_list=contents_list,
            ids_list=ids_list,
            **operation_params
        )

        result_df = df.copy()
        result_df['retrieved_contents'] = reranked_contents
        result_df['retrieved_ids'] = reranked_ids
        result_df['retrieve_scores'] = reranked_scores

        empty_cuda_cache()
        
        return result_df
    
    def apply_reranking(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        method = config.get('module_type', 'pass_reranker')
        top_k = config.get('top_k', 5)
        batch_size = config.get('batch', 64)
        defaults = {
            'colbert_reranker': ('model_name', 'colbert-ir/colbertv2.0'),
            'flag_embedding_reranker': ('model_name', 'BAAI/bge-reranker-large'),
            'flag_embedding_llm_reranker': ('model_name', 'BAAI/bge-reranker-v2-gemma'),
            'monot5': ('model_name', 'castorini/monot5-base-msmarco-10k'),
            'openvino_reranker': ('model', 'BAAI/bge-reranker-large'),
            'sentence_transformer_reranker': ('model_name', 'cross-encoder/ms-marco-MiniLM-L-2-v2'),
            'flashrank_reranker': ('model', 'ms-marco-MiniLM-L-12-v2'),
        }
        model_params = {}
        model_source = "NONE" 

        # 1. Method-specific models from Optuna (e.g., flashrank_reranker_models)
        method_model_key = f"{method}_models"
        if method_model_key in config:
            if method == 'flashrank_reranker':
                model_params['model'] = config[method_model_key]
                model_source = f"CONFIG ({method_model_key})"
            else:
                model_params['model_name'] = config[method_model_key]
                model_source = f"CONFIG ({method_model_key})"
        
        # 2. Generic reranker_model_name (from SMAC )
        elif 'reranker_model_name' in config:
            if method == 'flashrank_reranker':
                model_params['model'] = config['reranker_model_name']
                model_source = "CONFIG (reranker_model_name->model)"
            else:
                model_params['model_name'] = config['reranker_model_name']
                model_source = "CONFIG (reranker_model_name)"
        
        # 3. Generic reranker_model (alternative format)
        elif 'reranker_model' in config:
            if method == 'flashrank_reranker':
                model_params['model'] = config['reranker_model']
                model_source = "CONFIG (reranker_model)"
            else:
                model_params['model_name'] = config['reranker_model']
                model_source = "CONFIG (reranker_model->model_name)"
        
        # 4. Direct model/model_name in config
        elif method == 'flashrank_reranker' and 'model' in config:
            model_params['model'] = config['model']
            model_source = "CONFIG (model)"
        elif method != 'flashrank_reranker' and 'model_name' in config:
            model_params['model_name'] = config['model_name']
            model_source = "CONFIG (model_name)"
        
        # 5. Fall back to defaults only if no model was set above
        elif method in defaults:
            key, default = defaults[method]
            model_params[key] = default
            model_source = "DEFAULT"
        
        # Add method-specific parameters
        if method == 'flashrank_reranker':
            model_params['cache_dir'] = config.get('cache_dir', '/tmp')
            model_params['max_length'] = config.get('max_length', 512)
        if method == 'upr':
            model_params['use_bf16'] = config.get('use_bf16', False)
            model_params['prefix_prompt'] = config.get('prefix_prompt', 'Passage: ')
            model_params['suffix_prompt'] = config.get('suffix_prompt', 'Please write a question based on this passage.')

        model_value = model_params.get('model') or model_params.get('model_name', 'N/A')
        print(f"[RERANKER] Method: {method} | Model: {model_value} | Source: {model_source}")
        print(f"[RERANKER] Applying with top_k={top_k} batch={batch_size} params={model_params}")
        
        if method in ('pass_reranker', 'pass'):
            return df
        safe_df = df.copy()
        if 'retrieved_ids' in safe_df.columns:
            safe_df['retrieved_ids'] = safe_df['retrieved_ids'].apply(
                lambda x: x.tolist() if hasattr(x, 'tolist') else
                        (list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x)
            )
        return self.rerank_passages(safe_df, method=method, top_k=top_k, batch=batch_size, **model_params)