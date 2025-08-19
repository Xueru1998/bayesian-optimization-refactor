import os
import shutil
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from autorag.nodes.retrieval.bm25 import BM25
from autorag.nodes.retrieval.vectordb import VectorDB
import yaml

from .embedding_manager import EmbeddingManager
from pipeline.utils import Utils


class RetrievalModule:
    def __init__(self, base_project_dir, use_pregenerated_embeddings=True, centralized_project_dir=None):
        self.base_project_dir = base_project_dir
        self.use_pregenerated_embeddings = use_pregenerated_embeddings

        print(f"  base_project_dir: {base_project_dir}")
        print(f"  centralized_project_dir: {centralized_project_dir}")
        
        if centralized_project_dir:
            print(f"Using provided centralized_project_dir: {centralized_project_dir}")
            self.centralized_project_dir = centralized_project_dir
        else:
            project_root = Utils.find_project_root()
            self.centralized_project_dir = os.path.join(project_root, "autorag_project")
            print(f"RetrievalModule final project root: {project_root}")

        self.embeddings_dir = os.path.join(self.centralized_project_dir, "resources")
        os.makedirs(self.centralized_project_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.embedding_manager = EmbeddingManager(self.embeddings_dir)

    def prepare_project_dir(self, trial_dir, corpus_df, qa_df):
        resources_dir = os.path.join(trial_dir, "resources")
        data_dir = os.path.join(trial_dir, "data")
        os.makedirs(resources_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        corpus_df.to_parquet(os.path.join(data_dir, "corpus.parquet"), index=False)
        qa_df.to_parquet(os.path.join(data_dir, "qa.parquet"), index=False)
        
        if self.use_pregenerated_embeddings and os.path.exists(self.embeddings_dir):
            self.embedding_manager.copy_embeddings_to_trial(trial_dir)
        
        return trial_dir
    
    def run_embedding_script(self, tokenizer=None, vectordb_name=None):
        self.embedding_manager.run_embedding_script(tokenizer, vectordb_name)
    
    def setup_vectordb_config(self, trial_dir):
        return self.embedding_manager.setup_vectordb_config(trial_dir, self.use_pregenerated_embeddings)
    
    def prepare_embeddings(self, trial_dir, corpus_df, config=None, specific_vectordb=None):
        resources_dir = os.path.join(trial_dir, "resources")
        
        if specific_vectordb:
            qa_path = os.path.join(trial_dir, "data", "qa.parquet")
            if not os.path.exists(qa_path):
                qa_path = os.path.join(self.centralized_project_dir, "data", "qa.parquet")
            qa_df = pd.read_parquet(qa_path, engine="pyarrow") if os.path.exists(qa_path) else None
            
            asyncio.run(self.embedding_manager.generate_vectordb_embeddings(
                corpus_df, qa_df, specific_vectordb, self.centralized_project_dir
            ))
            return
        
        if not config:
            return
            
        retrieval_node = None
        for node_line in config.get('node_lines', []):
            for node in node_line.get('nodes', []):
                if node.get('node_type') == 'retrieval':
                    retrieval_node = node
                    break
            if retrieval_node:
                break
                
        if not retrieval_node:
            raise ValueError("No retrieval node found in the configuration")
        
        for module in retrieval_node.get('modules', []):
            if module.get('module_type') == 'bm25':
                tokenizer = module.get('bm25_tokenizer')
                if tokenizer:
                    self.embedding_manager.generate_bm25_embeddings(corpus_df, tokenizer)
            
            elif module.get('module_type') == 'vectordb':
                vectordb_names = module.get('vectordb', 'default')
                if not isinstance(vectordb_names, list):
                    vectordb_names = [vectordb_names]
                
                for vectordb_name in vectordb_names:
                    qa_path = os.path.join(trial_dir, "data", "qa.parquet")
                    if not os.path.exists(qa_path):
                        qa_path = os.path.join(self.centralized_project_dir, "data", "qa.parquet")
                    qa_df = pd.read_parquet(qa_path, engine="pyarrow") if os.path.exists(qa_path) else None
                    
                    asyncio.run(self.embedding_manager.generate_vectordb_embeddings(
                        corpus_df, qa_df, vectordb_name, self.centralized_project_dir
                    ))
    
    def cast_queries(self, queries: Union[str, List[str]]) -> List[str]:
        if isinstance(queries, str):
            return [queries]
        elif isinstance(queries, List):
            return queries
        else:
            raise ValueError(f"queries must be str or list, but got {type(queries)}")
    
    def _perform_bm25_retrieval(self, trial_dir, queries, bm25_tokenizer, top_k=10):
        top_k = int(top_k)
        
        trial_bm25_path = self.embedding_manager.ensure_bm25_exists(trial_dir, bm25_tokenizer)
        
        if not os.path.exists(trial_bm25_path):
            print(f"Failed to create BM25 index, returning empty results")
            return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
        
        bm25_module = BM25(project_dir=trial_dir, bm25_tokenizer=bm25_tokenizer)
        
        try:
            id_results, score_results = bm25_module._pure(queries=queries, top_k=top_k)
            retrieved_ids = id_results
            retrieved_scores = score_results

            corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
            corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
            retrieved_contents = [
                [
                    corpus_df.loc[corpus_df['doc_id'] == doc_id, 'contents'].values[0]
                    if not corpus_df.loc[corpus_df['doc_id'] == doc_id, 'contents'].empty else ""
                    for doc_id in ids_list
                ]
                for ids_list in retrieved_ids
            ]
            return retrieved_contents, retrieved_ids, retrieved_scores
        except Exception as e:
            print(f"Error in BM25 search: {e}")
            return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]

    def _perform_vectordb_retrieval(self, trial_dir, queries, vectordb_name='default', top_k=10):
        top_k = int(top_k)
        
        centralized_vectordb_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
        
        if not os.path.exists(centralized_vectordb_path):
            print(f"Centralized vectordb.yaml not found. Extracting from main config...")

            main_config_path = os.path.join(Utils.find_project_root(), "config.yaml")
            if os.path.exists(main_config_path):
                with open(main_config_path, 'r') as f:
                    main_config = yaml.safe_load(f)
                
                if 'vectordb' in main_config:
                    vectordb_list = main_config['vectordb']
                    
                    for vdb in vectordb_list:
                        if 'path' in vdb and '${PROJECT_DIR}' in vdb['path']:
                            vdb['path'] = vdb['path'].replace('${PROJECT_DIR}', self.centralized_project_dir)
                    
                    vectordb_config = {'vectordb': vectordb_list}
                    
                    os.makedirs(os.path.dirname(centralized_vectordb_path), exist_ok=True)
                    with open(centralized_vectordb_path, 'w') as f:
                        yaml.dump(vectordb_config, f, default_flow_style=False, sort_keys=False)
                    print(f"Extracted and saved vectordb config to {centralized_vectordb_path}")
        
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if os.path.exists(centralized_vectordb_path):
            print(f"Copying vectordb config from centralized location to trial...")
            os.makedirs(os.path.dirname(vectordb_yaml_path), exist_ok=True)
            shutil.copy(centralized_vectordb_path, vectordb_yaml_path)
        else:
            print(f"No vectordb configurations found. Creating default configuration.")
            default_config = {
                'vectordb': [{
                    'name': 'default',
                    'db_type': 'chroma',
                    'client_type': 'persistent',
                    'embedding_model': 'openai',
                    'collection_name': 'openai',
                    'path': os.path.join(self.embeddings_dir, "chroma")
                }]
            }
            
            os.makedirs(os.path.dirname(vectordb_yaml_path), exist_ok=True)
            with open(vectordb_yaml_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        vectordb_configs = self.embedding_manager.get_vectordb_configs(trial_dir)
        
        print(f"Available vectordb configs: {list(vectordb_configs.keys())}")
        
        if vectordb_name not in vectordb_configs:
            available_names = list(vectordb_configs.keys())
            print(f"VectorDB configuration '{vectordb_name}' not found. Available: {available_names}")
            if 'openai' in vectordb_configs:
                print(f"Using 'openai' configuration instead of default.")
                vectordb_name = 'openai'
            elif 'default' in vectordb_configs:
                print(f"Using 'default' configuration instead.")
                vectordb_name = 'default'
            elif available_names:
                print(f"Using first available configuration: '{available_names[0]}'")
                vectordb_name = available_names[0]
            else:
                print(f"No vectordb configurations available.")
                return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
        
        vdb_config = vectordb_configs.get(vectordb_name, {})
        
        vdb_config['path'] = os.path.join(self.embeddings_dir, "chroma")
        
        print(f"Using vectordb config: {vectordb_name} with embedding model: {vdb_config.get('embedding_model')}, collection: {vdb_config.get('collection_name')}")
        
        corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if not os.path.exists(corpus_path):
            corpus_path = os.path.join(self.centralized_project_dir, "data", "corpus.parquet")
        
        corpus_df = None
        if os.path.exists(corpus_path):
            corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
        else:
            print(f"Error: Corpus data not found at {corpus_path}")
            return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
        
        qa_path = os.path.join(trial_dir, "data", "qa.parquet")
        if not os.path.exists(qa_path):
            qa_path = os.path.join(self.centralized_project_dir, "data", "qa.parquet")
        qa_df = pd.read_parquet(qa_path, engine="pyarrow") if os.path.exists(qa_path) else None
        
        if not os.path.exists(centralized_vectordb_path) and os.path.exists(vectordb_yaml_path):
            print(f"Copying vectordb config to centralized location for embedding generation...")
            shutil.copy(vectordb_yaml_path, centralized_vectordb_path)
        
        embeddings_exist = self.embedding_manager.ensure_vectordb_exists(
            self.centralized_project_dir, vectordb_name, corpus_df, qa_df
        )
        
        if not embeddings_exist:
            print(f"Failed to ensure embeddings exist for {vectordb_name}")
            print(f"Attempting to generate embeddings using prepare_embeddings...")
            try:
                self.prepare_embeddings(self.centralized_project_dir, corpus_df, specific_vectordb=vectordb_name)
            except Exception as e:
                print(f"Error generating embeddings with prepare_embeddings: {e}")
                return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
        
        try:
            vectordb_module = VectorDB(project_dir=self.centralized_project_dir, vectordb=vectordb_name)
            
            id_results, score_results = vectordb_module._pure(queries=queries, top_k=top_k)
            retrieved_ids = id_results
            retrieved_scores = score_results

            retrieved_contents = [
                [
                    corpus_df.loc[corpus_df['doc_id'] == doc_id, 'contents'].values[0]
                    if not corpus_df.loc[corpus_df['doc_id'] == doc_id, 'contents'].empty else ""
                    for doc_id in ids_list
                ]
                for ids_list in retrieved_ids
            ]
            return retrieved_contents, retrieved_ids, retrieved_scores
        except Exception as e:
            print(f"CRITICAL ERROR in VectorDB search: {e}")
            import traceback
            traceback.print_exc()

            import sys
            print("\n Terminating program due to critical VectorDB error.")
            sys.exit(1)
    
    def perform_retrieval(self, trial_dir, qa_df, retrieval_method, bm25_tokenizer=None, vectordb_name='default', top_k=10):
        top_k = int(top_k)
        
        if "queries" in qa_df.columns:
            queries_list = qa_df["queries"].tolist()
            queries_2d = []
            for queries in queries_list:
                if isinstance(queries, np.ndarray):
                    queries = queries.tolist()
                casted_queries = self.cast_queries(queries)
                queries_2d.append(casted_queries)
        else:
            queries = qa_df['query'].tolist()
            queries_2d = [[query] for query in queries]

        if retrieval_method == 'bm25':
            return self._perform_bm25_retrieval(trial_dir, queries_2d, bm25_tokenizer, top_k)
        elif retrieval_method == 'vectordb':
            return self._perform_vectordb_retrieval(trial_dir, queries_2d, vectordb_name, top_k)
        else:
            print(f"Unknown retrieval method: {retrieval_method}")
            return ([[] for _ in range(len(queries_2d))], 
                [[] for _ in range(len(queries_2d))],
                [[] for _ in range(len(queries_2d))])