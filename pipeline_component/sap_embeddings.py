import os
import time
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

from autorag.vectordb.base import BaseVectorStore
from autorag.vectordb.chroma import Chroma
import yaml
import chromadb

logger = logging.getLogger(__name__)


class SAPEmbedding:
    _shared_token = None
    _shared_token_expiry = None
    _token_lock = None
    _rate_limit_lock = None
    _last_429_time = None
    _consecutive_429_count = 0
    
    def __init__(self, api_url: str, model_type: str = "openai"):
        self.api_url = api_url
        self.model_type = model_type
        
        self.auth_url = os.getenv('SAP_AUTH_URL')
        self.client_id = os.getenv('SAP_CLIENT_ID')
        self.client_secret = os.getenv('SAP_CLIENT_SECRET')
        
        if not self.auth_url:
            raise ValueError("SAP_AUTH_URL not found in .env file")
        if not self.client_id:
            raise ValueError("SAP_CLIENT_ID not found in .env file")
        if not self.client_secret:
            raise ValueError("SAP_CLIENT_SECRET not found in .env file")
        
        self.max_retries = 5
        self.initial_wait_time = 60
        self.max_wait_time = 300
        
        if SAPEmbedding._token_lock is None:
            SAPEmbedding._token_lock = threading.Lock()
        
        if SAPEmbedding._rate_limit_lock is None:
            SAPEmbedding._rate_limit_lock = threading.Lock()
        
        self._ensure_valid_token()
    
    def _refresh_token(self):
        with SAPEmbedding._token_lock:
            if (SAPEmbedding._shared_token and 
                SAPEmbedding._shared_token_expiry and 
                datetime.now() < SAPEmbedding._shared_token_expiry):
                logger.debug("Using existing valid token from cache")
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
                
                logger.info("Refreshing SAP bearer token...")
                response = requests.post(
                    self.auth_url,
                    headers=headers,
                    data=data,
                    timeout=30
                )
                response.raise_for_status()
                
                token_data = response.json()
                SAPEmbedding._shared_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 43199)
                
                SAPEmbedding._shared_token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                
                logger.info(f"Token refreshed successfully. Expires in {expires_in} seconds")
                
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                raise ValueError(f"Failed to obtain SAP bearer token: {e}")
    
    def _ensure_valid_token(self):
        if not SAPEmbedding._shared_token or (
            SAPEmbedding._shared_token_expiry and 
            datetime.now() >= SAPEmbedding._shared_token_expiry):
            self._refresh_token()
    
    def _handle_rate_limit(self, retry_attempt: int):
        with SAPEmbedding._rate_limit_lock:
            SAPEmbedding._last_429_time = datetime.now()
            SAPEmbedding._consecutive_429_count += 1
            
            wait_time = min(
                self.initial_wait_time * (2 ** retry_attempt),
                self.max_wait_time
            )
            
            if SAPEmbedding._consecutive_429_count > 3:
                wait_time = self.max_wait_time
            
            logger.warning(f"Rate limit hit (429). Waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
    
    def _reset_rate_limit_counter(self):
        with SAPEmbedding._rate_limit_lock:
            if SAPEmbedding._last_429_time and (
                datetime.now() - SAPEmbedding._last_429_time).seconds > 300:
                SAPEmbedding._consecutive_429_count = 0
    
    def _is_gemini(self) -> bool:
        return 'gemini' in self.api_url.lower() or self.model_type == 'gemini'
    
    def _build_request_body(self, texts: List[str], task_type: str = None) -> Dict:
        if self._is_gemini():
            if task_type is None:
                task_type = "RETRIEVAL_DOCUMENT"
            
            instances = []
            for text in texts:
                instance = {
                    "task_type": task_type,
                    "content": text
                }
                if task_type == "RETRIEVAL_DOCUMENT":
                    instance["title"] = ""
                instances.append(instance)
            
            return {"instances": instances}
        else:
            if len(texts) == 1:
                return {"input": texts[0]}
            else:
                return {"input": texts}
    
    def _extract_embeddings(self, response_data: Dict, num_texts: int) -> List[List[float]]:
        embeddings = []
        
        if self._is_gemini():
            predictions = response_data.get('predictions', [])
            for prediction in predictions:
                embedding_data = prediction.get('embeddings', {})
                values = embedding_data.get('values', [])
                embeddings.append(values)
        else:
            data = response_data.get('data', [])
            if isinstance(data, list):
                for item in data:
                    embedding = item.get('embedding', [])
                    embeddings.append(embedding)
            else:
                embedding = response_data.get('embedding', [])
                if embedding:
                    embeddings.append(embedding)
        
        return embeddings
    
    def _make_embedding_request(self, texts: List[str], task_type: str = None, 
                               retry_count: int = 0) -> List[List[float]]:
        self._ensure_valid_token()
        self._reset_rate_limit_counter()

        cleaned_texts = []
        for text in texts:
            if not text or not text.strip():
                cleaned_texts.append(" ") 
            else:
                max_chars = 30000  
                if len(text) > max_chars:
                    text = text[:max_chars]
                text = text.replace('\x00', '').strip()
                if not text:
                    text = " "
                cleaned_texts.append(text)
        
        headers = {
            "ai-resource-group": "enterpriseAIgroup",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SAPEmbedding._shared_token}"
        }
        
        request_body = self._build_request_body(cleaned_texts, task_type)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=request_body,
                    timeout=60
                )
                
                if response.status_code == 400:
                    error_msg = f"Bad Request: {response.text}"
                    logger.error(f"SAP API 400 error: {error_msg}")
 
                    if len(texts) > 1:
                        logger.info(f"Retrying with smaller batch size (current: {len(texts)})")
                        mid = len(texts) // 2
                        first_half = self._make_embedding_request(texts[:mid], task_type, retry_count)
                        second_half = self._make_embedding_request(texts[mid:], task_type, retry_count)
                        return first_half + second_half
                    else:
                        logger.error(f"Single text embedding failed: {error_msg}")
                        return [[0.0] * 1536]  
                
                if response.status_code == 429:
                    self._handle_rate_limit(attempt)
                    continue
                
                if response.status_code == 401 and retry_count < 2:
                    logger.warning("Received 401 Unauthorized. Refreshing token and retrying...")
                    self._refresh_token()
                    return self._make_embedding_request(texts, task_type, retry_count + 1)
                
                response.raise_for_status()
                
                with SAPEmbedding._rate_limit_lock:
                    SAPEmbedding._consecutive_429_count = 0
                
                data = response.json()
                embeddings = self._extract_embeddings(data, len(texts))
                
                return embeddings
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    self._handle_rate_limit(attempt)
                    continue
                elif e.response.status_code == 401 and retry_count < 2:
                    logger.warning("Received 401 Unauthorized. Refreshing token and retrying...")
                    self._refresh_token()
                    return self._make_embedding_request(texts, task_type, retry_count + 1)
                elif e.response.status_code == 400:
                    pass
                else:
                    logger.error(f"SAP API HTTP error: {e}")
                    if attempt < self.max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.info(f"Retrying after {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return []
                    
            except Exception as e:
                logger.error(f"SAP API error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Retrying after error in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return []
        
        logger.error(f"Failed to get embeddings after {self.max_retries} attempts")
        return []
    
    def get_text_embedding(self, text: str, task_type: str = None) -> List[float]:
        embeddings = self._make_embedding_request([text], task_type)
        return embeddings[0] if embeddings else []
    
    def get_text_embedding_batch(self, texts: List[str], task_type: str = None) -> List[List[float]]:
        return self._make_embedding_request(texts, task_type)
    
    def get_query_embedding(self, text: str) -> List[float]:
        task_type = "RETRIEVAL_QUERY" if self._is_gemini() else None
        return self.get_text_embedding(text, task_type)
    
    def get_agg_embedding_from_queries(self, queries: List[str]) -> List[float]:
        task_type = "RETRIEVAL_QUERY" if self._is_gemini() else None
        embeddings = self.get_text_embedding_batch(queries, task_type)
        if not embeddings:
            return []
        return np.mean(embeddings, axis=0).tolist()


class SAPChroma(Chroma):
    def __init__(
        self,
        embedding_model: str,
        collection_name: str,
        client_type: str = "persistent",
        similarity_metric: str = "cosine",
        path: str = None,
        host: str = "localhost",
        port: int = 8000,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        tenant: str = "default_tenant",
        database: str = "default_database",
        embedding_batch: int = 100,
        add_batch: int = 100,
    ):
        self.embedding_model_url = embedding_model
        self.add_batch = add_batch  
        
        if 'gemini' in embedding_model.lower():
            model_type = 'gemini'
        else:
            model_type = 'openai'

        self.sap_embedding = SAPEmbedding(
            api_url=embedding_model,
            model_type=model_type
        )
        
        
        if client_type == "ephemeral":
            self.client = chromadb.EphemeralClient()
        elif client_type == "persistent":
            self.client = chromadb.PersistentClient(path=path)
        elif client_type == "http":
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers,
            )
        else:
            raise ValueError(
                f"client_type {client_type} is not supported.\n"
                "Please use one of the following: ephemeral, persistent, http"
            )

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": similarity_metric},
            )
            print(f"[SAPChroma] Collection '{collection_name}' initialized with {self.collection.count()} documents")
        except Exception as e:
            print(f"[SAPChroma] Error creating collection '{collection_name}': {e}")
            raise

        self.embedding = self.sap_embedding
        self.similarity_metric = similarity_metric
        self.embedding_batch = embedding_batch
    
    async def add(self, ids: List[str], texts: List[str]):
        if hasattr(self, 'truncated_inputs'):
            texts = self.truncated_inputs(texts)
        else:
            max_length = 8000 
            texts = [text[:max_length] if len(text) > max_length else text for text in texts]
        
        task_type = "RETRIEVAL_DOCUMENT" if self.sap_embedding._is_gemini() else None
        
        embeddings = self.sap_embedding.get_text_embedding_batch(texts, task_type)
        
        if not embeddings:
            logger.error("Failed to get embeddings from SAP API")
            return
        
        added_count = 0
        for i in range(0, len(ids), self.add_batch):
            batch_ids = ids[i : i + self.add_batch]
            batch_embeddings = embeddings[i : i + self.add_batch]
            batch_texts = texts[i : i + self.add_batch]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                )
                added_count += len(batch_ids)
            except Exception as e:
                print(f"[SAPChroma] Error adding batch: {e}")
                logger.error(f"Error adding documents to collection: {e}")
    
    async def query(
        self, queries: List[str], top_k: int, **kwargs
    ) -> Tuple[List[List[str]], List[List[float]]]:

        if hasattr(self, 'truncated_inputs'):
            queries = self.truncated_inputs(queries)
        else:
            max_length = 8000
            queries = [q[:max_length] if len(q) > max_length else q for q in queries]
        
        task_type = "RETRIEVAL_QUERY" if self.sap_embedding._is_gemini() else None
        
        query_embeddings = self.sap_embedding.get_text_embedding_batch(queries, task_type)
        
        if not query_embeddings:
            logger.error("Failed to get query embeddings from SAP API")
            return [], []
        
        id_result = []
        score_result = []
        
        for idx, query_embedding in enumerate(query_embeddings):
            try:
                result = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.collection.count()) if self.collection.count() > 0 else top_k,
                    **kwargs,
                )
                
                ids = result["ids"][0] if result["ids"] else []
                distances = result["distances"][0] if result["distances"] else []
                
                if self.similarity_metric == "cosine":
                    scores = [1 - dist for dist in distances]
                elif self.similarity_metric == "ip":
                    scores = distances
                elif self.similarity_metric == "l2":
                    scores = [1 / (1 + dist) for dist in distances]
                else:
                    scores = distances
                
                id_result.append(ids)
                score_result.append(scores)
            except Exception as e:
                print(f"[SAPChroma] Error querying for query {idx}: {e}")
                id_result.append([])
                score_result.append([])
        return id_result, score_result
    
    async def fetch(self, ids: List[str]) -> List[List[float]]:
        result = self.collection.get(ids=ids, include=["embeddings"])
        embeddings = result.get("embeddings", [])
        return embeddings if embeddings else []
    
    async def is_exist(self, ids: List[str]) -> List[bool]:
        try:
            result = self.collection.get(ids=ids)
            existing_ids = result.get("ids", [])
            exists = [doc_id in existing_ids for doc_id in ids]
            print(f"[SAPChroma] Checked existence for {len(ids)} IDs: {sum(exists)} exist")
            return exists
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return [False] * len(ids)


def load_sap_vectordb_from_yaml(
    yaml_path: str, vectordb_name: str, project_dir: str = None
) -> BaseVectorStore:
    with open(yaml_path, "r", encoding="utf-8") as f:
        vectordb_config = yaml.safe_load(f)
    
    vectordb_list = vectordb_config.get("vectordb", [])
    
    target_config = None
    for config in vectordb_list:
        if config.get("name") == vectordb_name:
            target_config = config
            break
    
    if not target_config:
        available_names = [config.get("name") for config in vectordb_list]
        raise ValueError(
            f"Vectordb name {vectordb_name} not found in config. "
            f"Available names: {available_names}"
        )
    
    db_type = target_config.get("db_type", "chroma")
    
    if db_type != "chroma":
        raise ValueError(f"Only chroma is supported for SAP embeddings, got {db_type}")
    
    embedding_model = target_config.get("embedding_model")
    
    if not embedding_model or not embedding_model.startswith("http"):
        from autorag.vectordb import load_vectordb_from_yaml
        return load_vectordb_from_yaml(yaml_path, vectordb_name, project_dir)
    
    path = target_config.get("path", "./chroma")
    if "${PROJECT_DIR}" in path and project_dir:
        path = path.replace("${PROJECT_DIR}", project_dir)
    
    collection_name_mapping = {
        'text-embedding-3-large': 'text_embedding_3_large',
        'text-embedding-3-small': 'text_embedding_3_small', 
        'text-embedding-ada-002': 'text_embedding_ada_002',
        'gemini': 'gemini_embedding'
    }
    
    collection_name = collection_name_mapping.get(
        vectordb_name, 
        target_config.get("collection_name", vectordb_name.replace("-", "_"))
    )
    
    return SAPChroma(
        embedding_model=embedding_model,
        collection_name=collection_name,
        client_type=target_config.get("client_type", "persistent"),
        similarity_metric=target_config.get("similarity_metric", "cosine"),
        path=path,
        embedding_batch=target_config.get("embedding_batch", 100),
        add_batch=target_config.get("add_batch", 100),
    )


class SAPVectorDB:
    def __init__(self, project_dir: str, vectordb: str = "default", **kwargs):
        self.project_dir = project_dir
        self.resources_dir = os.path.join(project_dir, "resources")
        
        vectordb_config_path = os.path.join(self.resources_dir, "vectordb.yaml")
        print(f"[SAPVectorDB] Loading config from: {vectordb_config_path}")
        print(f"[SAPVectorDB] Initializing vectordb: {vectordb}")
        
        self.vector_store = load_sap_vectordb_from_yaml(
            vectordb_config_path, vectordb, project_dir
        )
        
        self.embedding_model = self.vector_store.embedding
        self.corpus_df = None
        
        corpus_path = os.path.join(project_dir, "data", "corpus.parquet")
        if os.path.exists(corpus_path):
            import pandas as pd
            self.corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
            print(f"[SAPVectorDB] Loaded corpus with {len(self.corpus_df)} documents")
    
    async def query(self, queries: List[List[str]], top_k: int) -> Tuple[List[List[str]], List[List[float]]]:
        all_ids = []
        all_scores = []
        
        for query_list in queries:
            ids, scores = await self.vector_store.query(query_list, top_k)

            combined_ids = []
            combined_scores = []
            
            for id_list, score_list in zip(ids, scores):
                combined_ids.extend(id_list)
                combined_scores.extend(score_list)

            if combined_ids:
                sorted_results = sorted(
                    zip(combined_scores, combined_ids), 
                    key=lambda x: x[0], 
                    reverse=True
                )[:top_k]
                
                final_scores, final_ids = zip(*sorted_results)
                all_ids.append(list(final_ids))
                all_scores.append(list(final_scores))
            else:
                all_ids.append([])
                all_scores.append([])
                print(f"[SAPVectorDB] No results found for query group")

        return all_ids, all_scores
    
    async def add(self, ids: List[str], texts: List[str]):
        await self.vector_store.add(ids, texts)
    
    async def is_exist(self, ids: List[str]) -> List[bool]:
        return await self.vector_store.is_exist(ids)
    
    async def fetch(self, ids: List[str]) -> List[List[float]]:
        return await self.vector_store.fetch(ids)