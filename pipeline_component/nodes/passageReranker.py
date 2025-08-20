import gc
import os
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime, timedelta
import threading

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

import torch
logger = logging.getLogger("AutoRAG")


class SAPReranker(BasePassageReranker):
    _shared_token = None
    _shared_token_expiry = None
    _token_lock = None
    
    def __init__(
        self,
        project_dir: str,
        api_endpoint: str,
        model_name: str = "cohere-rerank-v3.5",
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(project_dir)
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.timeout = timeout
        
        self.auth_url = os.getenv('SAP_RERANKER_AUTH_URL')
        self.client_id = os.getenv('SAP_RERANKER_CLIENT_ID')
        self.client_secret = os.getenv('SAP_RERANKER_CLIENT_SECRET')
        
        if not self.auth_url:
            raise ValueError("SAP_RERANKER_AUTH_URL not found in .env file. Please set SAP_RERANKER_AUTH_URL in your .env file.")
        if not self.client_id:
            raise ValueError("SAP_RERANKER_CLIENT_ID not found in .env file. Please set SAP_RERANKER_CLIENT_ID in your .env file.")
        if not self.client_secret:
            raise ValueError("SAP_RERANKER_CLIENT_SECRET not found in .env file. Please set SAP_RERANKER_CLIENT_SECRET in your .env file.")
        
        if SAPReranker._token_lock is None:
            SAPReranker._token_lock = threading.Lock()
        
        self._ensure_valid_token()
        
    def __del__(self):
        pass
    
    def _refresh_token(self):
        with SAPReranker._token_lock:
            if SAPReranker._shared_token and SAPReranker._shared_token_expiry and datetime.now() < SAPReranker._shared_token_expiry:
                logger.debug("Using existing valid reranker token from cache")
                return
            
            try:
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
                
                data = {
                    'grant_type': 'client_credentials',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret
                }
                
                logger.info("Refreshing SAP reranker bearer token...")
                response = requests.post(
                    self.auth_url,
                    headers=headers,
                    data=data,
                    timeout=30
                )
                response.raise_for_status()
                
                token_data = response.json()
                SAPReranker._shared_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 43199)
                
                SAPReranker._shared_token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                
                logger.info(f"Reranker token refreshed successfully. Expires in {expires_in} seconds")
                
            except Exception as e:
                logger.error(f"Failed to refresh reranker token: {e}")
                raise ValueError(f"Failed to obtain SAP reranker bearer token: {e}")
    
    def _ensure_valid_token(self):
        if not SAPReranker._shared_token or (SAPReranker._shared_token_expiry and datetime.now() >= SAPReranker._shared_token_expiry):
            self._refresh_token()
        else:
            logger.debug("Reranker token is still valid, no refresh needed")

    def pure(
        self,
        queries: List[str],
        contents_list: List[List[str]],
        ids_list: List[List[str]],
        top_k: int,
        **kwargs
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        return self._pure(queries, contents_list, ids_list, top_k, **kwargs)

    def _make_api_request(self, request_body: Dict, retry_count: int = 0) -> Dict:
        self._ensure_valid_token()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SAPReranker._shared_token}",
            "AI-Resource-Group": "enterpriseAIgroup"
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=request_body,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 401 and retry_count < 2:
                logger.warning("Received 401 Unauthorized on reranker. Refreshing token and retrying...")
                self._refresh_token()
                return self._make_api_request(request_body, retry_count + 1)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401 and retry_count < 2:
                logger.warning("Received 401 Unauthorized on reranker. Refreshing token and retrying...")
                self._refresh_token()
                return self._make_api_request(request_body, retry_count + 1)
            else:
                raise

    def _pure(
        self,
        queries: List[str],
        contents_list: List[List[str]],
        ids_list: List[List[str]],
        top_k: int,
        **kwargs
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        
        reranked_contents = []
        reranked_ids = []
        reranked_scores = []
        
        for query, contents, ids in zip(queries, contents_list, ids_list):
            if isinstance(contents, np.ndarray):
                contents = contents.tolist()
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            
            if not contents or len(contents) == 0:
                reranked_contents.append([])
                reranked_ids.append([])
                reranked_scores.append([])
                continue
            
            request_body = {
                "model": self.model_name,
                "query": query,
                "top_n": min(top_k, len(contents)),
                "documents": contents
            }
            
            try:
                result = self._make_api_request(request_body)
                
                indices_scores = [(r["index"], r["relevance_score"]) for r in result["results"]]
                indices_scores.sort(key=lambda x: x[1], reverse=True)
                
                batch_contents = []
                batch_ids = []
                batch_scores = []
                
                for idx, score in indices_scores[:top_k]:
                    if idx < len(contents):
                        batch_contents.append(contents[idx])
                        batch_ids.append(ids[idx])
                        batch_scores.append(score)
                
                reranked_contents.append(batch_contents)
                reranked_ids.append(batch_ids)
                reranked_scores.append(batch_scores)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"SAP reranker API request failed: {str(e)}")
                reranked_contents.append(contents[:top_k])
                reranked_ids.append(ids[:top_k])
                reranked_scores.append([0.0] * min(top_k, len(contents)))
            except (KeyError, ValueError) as e:
                logger.error(f"Failed to parse SAP reranker API response: {str(e)}")
                reranked_contents.append(contents[:top_k])
                reranked_ids.append(ids[:top_k])
                reranked_scores.append([0.0] * min(top_k, len(contents)))
        
        return reranked_contents, reranked_ids, reranked_scores

class PassageRerankerModule:
    def __init__(self, project_dir: str = ""):
        self.project_dir = project_dir
        self._loaded_models = {}
        
        self.rerankers = {
            "pass_reranker": PassReranker, 
            "colbert_reranker": ColbertReranker,
            "flag_embedding_reranker": FlagEmbeddingReranker,
            "flag_embedding_llm_reranker": FlagEmbeddingLLMReranker,
            "flashrank_reranker": FlashRankReranker,
            "monot5": MonoT5,
            "openvino_reranker": OpenVINOReranker,
            "sentence_transformer_reranker": SentenceTransformerReranker,
            "upr": Upr,
            "sap_reranker": SAPReranker,
            "sap_api": SAPReranker
        }
        
    def cleanup_models(self):
       if hasattr(self, '_loaded_models'):
           for model_name in list(self._loaded_models.keys()):
               try:
                   model = self._loaded_models[model_name]
                   del model
               except:
                   pass
           self._loaded_models.clear()
       
       try:
           from autorag.nodes.passagereranker import flag_embedding, flag_embedding_llm, monot5
           
           if hasattr(flag_embedding, 'model_instance'):
               del flag_embedding.model_instance
               flag_embedding.model_instance = None
           
           if hasattr(flag_embedding_llm, 'model_instance'):
               del flag_embedding_llm.model_instance
               flag_embedding_llm.model_instance = None
               
           if hasattr(monot5, 'model_instance'):
               del monot5.model_instance
               monot5.model_instance = None
               
           for attr in ['_model_cache', '_reranker_cache', 'cached_models']:
               if hasattr(flag_embedding, attr):
                   cache = getattr(flag_embedding, attr)
                   if isinstance(cache, dict):
                       cache.clear()
               if hasattr(flag_embedding_llm, attr):
                   cache = getattr(flag_embedding_llm, attr)
                   if isinstance(cache, dict):
                       cache.clear()
       except ImportError:
           pass
       
       try:
           from sentence_transformers import SentenceTransformer
           if hasattr(SentenceTransformer, '_cache'):
               SentenceTransformer._cache.clear()
       except ImportError:
           pass
       
       gc.collect()
       
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           torch.cuda.synchronize()
       
       print(f"[CLEANUP] Model memory cleared, GPU memory available: {self._get_gpu_memory_info()}")
   
    def _get_gpu_memory_info(self):
       try:
           if torch.cuda.is_available():
               allocated = torch.cuda.memory_allocated() / 1024**3
               reserved = torch.cuda.memory_reserved() / 1024**3
               return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
       except:
           pass
       return "N/A"
    
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
        
        if method in ('sap_reranker', 'sap_api'):
            if 'api_endpoint' in kwargs:
                init_params['api_endpoint'] = kwargs.pop('api_endpoint')
            elif 'api_url' in kwargs:
                init_params['api_endpoint'] = kwargs.pop('api_url')
            elif 'api-url' in kwargs:
                init_params['api_endpoint'] = kwargs.pop('api-url')
            elif 'reranker_api_url' in kwargs:
                init_params['api_endpoint'] = kwargs.pop('reranker_api_url')
            if 'timeout' in kwargs:
                init_params['timeout'] = kwargs.pop('timeout')
        
        operation_params['batch'] = batch
        
        operation_params.update(kwargs)
        
        queries = df['query'].tolist()
        contents_list = df['retrieved_contents'].tolist()
        ids_list = df['retrieved_ids'].tolist()

        reranker = self.create_reranker(method, **init_params)
        
        if method in ('upr', 'sap_reranker', 'sap_api'):
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
       
       memory_intensive_models = ['flag_embedding_llm_reranker', 'monot5', 'colbert_reranker']
       if method in memory_intensive_models:
           batch_size = min(batch_size, 8)
           print(f"[RERANKER] Reduced batch size to {batch_size} for {method}")
       
       defaults = {
           'colbert_reranker': ('model_name', 'colbert-ir/colbertv2.0'),
           'flag_embedding_reranker': ('model_name', 'BAAI/bge-reranker-large'),
           'flag_embedding_llm_reranker': ('model_name', 'BAAI/bge-reranker-v2-gemma'),
           'monot5': ('model_name', 'castorini/monot5-base-msmarco-10k'),
           'openvino_reranker': ('model', 'BAAI/bge-reranker-large'),
           'sentence_transformer_reranker': ('model_name', 'cross-encoder/ms-marco-MiniLM-L-2-v2'),
           'flashrank_reranker': ('model', 'ms-marco-MiniLM-L-12-v2'),
           'sap_reranker': ('model_name', 'cohere-rerank-v3.5'),
           'sap_api': ('model_name', 'cohere-rerank-v3.5'),
       }
       
       model_params = {}
       model_source = "NONE"
       
       method_model_key = f"{method}_models"
       if method_model_key in config:
           if method == 'flashrank_reranker':
               model_params['model'] = config[method_model_key]
               model_source = f"CONFIG ({method_model_key})"
           else:
               model_params['model_name'] = config[method_model_key]
               model_source = f"CONFIG ({method_model_key})"
       
       elif 'reranker_model_name' in config:
           if method == 'flashrank_reranker':
               model_params['model'] = config['reranker_model_name']
               model_source = "CONFIG (reranker_model_name->model)"
           else:
               model_params['model_name'] = config['reranker_model_name']
               model_source = "CONFIG (reranker_model_name)"
       
       elif 'reranker_model' in config:
           if method == 'flashrank_reranker':
               model_params['model'] = config['reranker_model']
               model_source = "CONFIG (reranker_model)"
           else:
               model_params['model_name'] = config['reranker_model']
               model_source = "CONFIG (reranker_model->model_name)"
       
       elif method == 'flashrank_reranker' and 'model' in config:
           model_params['model'] = config['model']
           model_source = "CONFIG (model)"
       elif method != 'flashrank_reranker' and 'model_name' in config:
           model_params['model_name'] = config['model_name']
           model_source = "CONFIG (model_name)"
       
       elif method in defaults:
           key, default = defaults[method]
           model_params[key] = default
           model_source = "DEFAULT"
       
       if method == 'flashrank_reranker':
           model_params['cache_dir'] = config.get('cache_dir', '/tmp')
           model_params['max_length'] = config.get('max_length', 512)
       if method == 'upr':
           model_params['use_bf16'] = config.get('use_bf16', False)
           model_params['prefix_prompt'] = config.get('prefix_prompt', 'Passage: ')
           model_params['suffix_prompt'] = config.get('suffix_prompt', 'Please write a question based on this passage.')
       if method in ('sap_reranker', 'sap_api'):
           model_params['api_endpoint'] = (config.get('api_endpoint') or 
                                          config.get('api_url') or 
                                          config.get('api-url') or 
                                          config.get('reranker_api_url'))
           model_params['timeout'] = config.get('timeout', 30)
       
       model_value = model_params.get('model') or model_params.get('model_name', 'N/A')
       print(f"[RERANKER] Method: {method} | Model: {model_value} | Source: {model_source}")
       print(f"[RERANKER] GPU memory before: {self._get_gpu_memory_info()}")
       print(f"[RERANKER] Applying with top_k={top_k} batch={batch_size} params={model_params}")
       
       if method in ('pass_reranker', 'pass'):
           return df
       
       safe_df = df.copy()
       if 'retrieved_ids' in safe_df.columns:
           safe_df['retrieved_ids'] = safe_df['retrieved_ids'].apply(
               lambda x: x.tolist() if hasattr(x, 'tolist') else
                       (list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x)
           )
       
       try:
           model_key = f"{method}_{model_value}"
           self._loaded_models[model_key] = True
           
           result = self.rerank_passages(safe_df, method=method, top_k=top_k, batch=batch_size, **model_params)
           
           print(f"[RERANKER] GPU memory after reranking: {self._get_gpu_memory_info()}")
           return result
           
       except torch.cuda.OutOfMemoryError as e:
           print(f"[ERROR] CUDA OOM during reranking: {e}")
           self.cleanup_models()
           
           if batch_size > 1:
               print(f"[RERANKER] Retrying with batch_size=1")
               try:
                   result = self.rerank_passages(safe_df, method=method, top_k=top_k, batch=1, **model_params)
                   return result
               except Exception as retry_error:
                   print(f"[ERROR] Retry failed: {retry_error}")
                   raise
           else:
               raise
               
       except Exception as e:
           print(f"[ERROR] Reranking failed: {e}")
           raise
           
       finally:
           self.cleanup_models()