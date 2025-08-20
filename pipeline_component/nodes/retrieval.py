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

from ..embedding.embedding_manager import EmbeddingManager
from pipeline.utils import Utils
from ..embedding.sap_embeddings import SAPVectorDB, load_sap_vectordb_from_yaml


class RetrievalModule:
    def __init__(self, base_project_dir, use_pregenerated_embeddings=True, centralized_project_dir=None):
        self.base_project_dir = base_project_dir
        self.use_pregenerated_embeddings = use_pregenerated_embeddings
        
        if centralized_project_dir:
            print(f"Using provided project_dir: {centralized_project_dir}")
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
    
    def _is_sap_embedding_url(self, embedding_model: str) -> bool:
        if not embedding_model:
            return False
        return embedding_model.startswith("http") and "api.ai.internalprod" in embedding_model
    
    def _get_vectordb_module(self, trial_dir: str, vectordb_name: str = 'default'):
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            vectordb_yaml_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            print(f"VectorDB YAML not found")
            return None
        
        with open(vectordb_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vectordb_list = config.get('vectordb', [])
        target_config = None
        
        for vdb in vectordb_list:
            if vdb.get('name') == vectordb_name:
                target_config = vdb
                break
        
        if not target_config:
            print(f"VectorDB configuration '{vectordb_name}' not found")
            return None
        
        embedding_model = target_config.get('embedding_model', '')
        
        if self._is_sap_embedding_url(embedding_model):
            return SAPVectorDB(project_dir=self.centralized_project_dir, vectordb=vectordb_name)
        else:
            print(f"Using standard AutoRAG VectorDB: {vectordb_name}")
            return VectorDB(project_dir=self.centralized_project_dir, vectordb=vectordb_name)
    
    async def generate_vectordb_embeddings_sap(self, corpus_df: pd.DataFrame, vectordb_name: str, trial_dir: str):
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            vectordb_yaml_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            print(f"VectorDB YAML not found")
            return
        
        print(f"Generating SAP embeddings for {vectordb_name}...")
        
        try:
            vectordb = load_sap_vectordb_from_yaml(vectordb_yaml_path, vectordb_name, self.centralized_project_dir)
            
            corpus_data = corpus_df[['doc_id', 'contents']]
            
            existing_ids = await vectordb.is_exist(corpus_data['doc_id'].tolist())
            new_data = corpus_data[~pd.Series(existing_ids)]
            
            if not new_data.empty:
                print(f"Adding {len(new_data)} new documents to {vectordb_name}")
                
                batch_size = 100
                for i in range(0, len(new_data), batch_size):
                    batch = new_data.iloc[i:i+batch_size]
                    await vectordb.add(
                        ids=batch['doc_id'].tolist(),
                        texts=batch['contents'].tolist()
                    )
                    print(f"Processed {min(i+batch_size, len(new_data))}/{len(new_data)} documents")
            else:
                print(f"All documents already exist in {vectordb_name}")
            
            print(f"SAP embeddings generation completed for {vectordb_name}")
            
        except Exception as e:
            print(f"Error generating SAP embeddings: {e}")
            raise
        
    def check_and_generate_embeddings(trial_dir, corpus_df):
        retrieval_module = RetrievalModule(
            base_project_dir=trial_dir,
            use_pregenerated_embeddings=True,
            centralized_project_dir=Utils.get_centralized_project_dir()
        )
        
        vectordb_configs = retrieval_module.embedding_manager.get_vectordb_configs(trial_dir)
        
        missing_embeddings = []
        for vectordb_name, vdb_config in vectordb_configs.items():
            embedding_model = vdb_config.get('embedding_model', '')
            
            if embedding_model.startswith('http') and 'api.ai.internalprod' in embedding_model:
                if not retrieval_module._check_embeddings_exist(trial_dir, vectordb_name, corpus_df):
                    missing_embeddings.append(vectordb_name)
                    print(f"[Pre-check] Missing embeddings for {vectordb_name}")
                else:
                    print(f"[Pre-check] Embeddings exist for {vectordb_name}")
        
        if missing_embeddings:
            print(f"\n[Pre-generation] Need to generate embeddings for: {missing_embeddings}")
            for vectordb_name in missing_embeddings:
                print(f"\n[Pre-generation] Generating embeddings for {vectordb_name}...")
                try:
                    retrieval_module.prepare_embeddings(trial_dir, corpus_df, specific_vectordb=vectordb_name)
                    print(f"[Pre-generation] Successfully generated embeddings for {vectordb_name}")
                except Exception as e:
                    print(f"[Pre-generation] Failed to generate embeddings for {vectordb_name}: {e}")
                    import traceback
                    traceback.print_exc()
            print("\n[Pre-generation] Embedding generation complete")
        else:
            print("[Pre-check] All required embeddings already exist")
    
    def run_pipeline_with_embedding_check(self, config: Dict[str, Any], trial_dir: str, qa_subset: pd.DataFrame, 
                                     is_local_optimization: bool = False, current_component: str = None) -> Dict[str, Any]:
   
        if config.get('retrieval_method') == 'vectordb':
            corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
            if not os.path.exists(corpus_path):
                corpus_path = os.path.join(Utils.get_centralized_project_dir(), "data", "corpus.parquet")
            
            if os.path.exists(corpus_path):
                corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
                check_and_generate_embeddings(trial_dir, corpus_df)
        
        return self.run_pipeline(config, trial_dir, qa_subset, is_local_optimization, current_component)

    
    def prepare_embeddings(self, trial_dir, corpus_df, config=None, specific_vectordb=None):
        resources_dir = os.path.join(trial_dir, "resources")
        
        if specific_vectordb:
            vectordb_configs = self.embedding_manager.get_vectordb_configs(trial_dir)
            
            if specific_vectordb in vectordb_configs:
                vdb_config = vectordb_configs[specific_vectordb]
                embedding_model = vdb_config.get('embedding_model', '')
                
                if self._is_sap_embedding_url(embedding_model):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(self.generate_vectordb_embeddings_sap(
                            corpus_df, specific_vectordb, trial_dir
                        ))
                    except RuntimeError as e:
                        if "This event loop is already running" in str(e):
                            import nest_asyncio
                            nest_asyncio.apply()
                            task = self.generate_vectordb_embeddings_sap(corpus_df, specific_vectordb, trial_dir)
                            loop.run_until_complete(task)
                        else:
                            raise
                else:
                    qa_path = os.path.join(trial_dir, "data", "qa.parquet")
                    if not os.path.exists(qa_path):
                        qa_path = os.path.join(self.centralized_project_dir, "data", "qa.parquet")
                    qa_df = pd.read_parquet(qa_path, engine="pyarrow") if os.path.exists(qa_path) else None
                    
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(self.embedding_manager.generate_vectordb_embeddings(
                            corpus_df, qa_df, specific_vectordb, self.centralized_project_dir
                        ))
                    except RuntimeError as e:
                        if "This event loop is already running" in str(e):
                            import nest_asyncio
                            nest_asyncio.apply()
                            task = self.embedding_manager.generate_vectordb_embeddings(
                                corpus_df, qa_df, specific_vectordb, self.centralized_project_dir
                            )
                            loop.run_until_complete(task)
                        else:
                            raise
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
                
                vectordb_configs = self.embedding_manager.get_vectordb_configs(trial_dir)
                
                for vectordb_name in vectordb_names:
                    if vectordb_name in vectordb_configs:
                        vdb_config = vectordb_configs[vectordb_name]
                        embedding_model = vdb_config.get('embedding_model', '')
                        
                        if self._is_sap_embedding_url(embedding_model):
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_closed():
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            try:
                                loop.run_until_complete(self.generate_vectordb_embeddings_sap(
                                    corpus_df, vectordb_name, trial_dir
                                ))
                            except RuntimeError as e:
                                if "This event loop is already running" in str(e):
                                    import nest_asyncio
                                    nest_asyncio.apply()
                                    task = self.generate_vectordb_embeddings_sap(corpus_df, vectordb_name, trial_dir)
                                    loop.run_until_complete(task)
                                else:
                                    raise
                        else:
                            qa_path = os.path.join(trial_dir, "data", "qa.parquet")
                            if not os.path.exists(qa_path):
                                qa_path = os.path.join(self.centralized_project_dir, "data", "qa.parquet")
                            qa_df = pd.read_parquet(qa_path, engine="pyarrow") if os.path.exists(qa_path) else None
                            
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_closed():
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            try:
                                loop.run_until_complete(self.embedding_manager.generate_vectordb_embeddings(
                                    corpus_df, qa_df, vectordb_name, self.centralized_project_dir
                                ))
                            except RuntimeError as e:
                                if "This event loop is already running" in str(e):
                                    import nest_asyncio
                                    nest_asyncio.apply()
                                    task = self.embedding_manager.generate_vectordb_embeddings(
                                        corpus_df, qa_df, vectordb_name, self.centralized_project_dir
                                    )
                                    loop.run_until_complete(task)
                                else:
                                    raise
    
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
        
        self._ensure_vectordb_config(trial_dir)
        
        corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if not os.path.exists(corpus_path):
            corpus_path = os.path.join(self.centralized_project_dir, "data", "corpus.parquet")
        
        if not os.path.exists(corpus_path):
            print(f"Error: Corpus data not found")
            return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
        
        corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
        
        vectordb_configs = self.embedding_manager.get_vectordb_configs(trial_dir)
        if vectordb_name in vectordb_configs:
            vdb_config = vectordb_configs[vectordb_name]
            embedding_model = vdb_config.get('embedding_model', '')
            
            if self._is_sap_embedding_url(embedding_model):
                if not self._check_embeddings_exist(trial_dir, vectordb_name, corpus_df):
                    print(f"SAP embeddings don't exist for {vectordb_name}, generating now...")
                    try:
                        self.prepare_embeddings(trial_dir, corpus_df, specific_vectordb=vectordb_name)
                    except Exception as e:
                        print(f"Error generating SAP embeddings: {e}")
                        return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
                
                vectordb_module = self._get_vectordb_module(trial_dir, vectordb_name)
                
                if not vectordb_module:
                    print(f"Failed to initialize SAP VectorDB module for {vectordb_name}")
                    return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
                
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    id_results, score_results = loop.run_until_complete(
                        vectordb_module.query(queries, top_k)
                    )
                    loop.close()
                    
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
                    print(f"Error in SAP VectorDB search: {e}")
                    import traceback
                    traceback.print_exc()
                    return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
            else:
                qa_path = os.path.join(trial_dir, "data", "qa.parquet")
                if not os.path.exists(qa_path):
                    qa_path = os.path.join(self.centralized_project_dir, "data", "qa.parquet")
                qa_df = pd.read_parquet(qa_path, engine="pyarrow") if os.path.exists(qa_path) else None
                
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
                
                vectordb_module = self._get_vectordb_module(trial_dir, vectordb_name)
                
                if not vectordb_module:
                    print(f"Failed to initialize VectorDB module for {vectordb_name}")
                    return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
                
                try:
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
                    
                except ValueError as e:
                    if "not enough values to unpack" in str(e):
                        print(f"VectorDB returned empty results for {vectordb_name}. Collection may be empty.")
                        print(f"Returning empty results for {len(queries)} queries")
                        return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
                    else:
                        raise
                except Exception as e:
                    print(f"Error in VectorDB search: {e}")
                    import traceback
                    traceback.print_exc()
                    return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
        
        print(f"VectorDB configuration '{vectordb_name}' not found")
        return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
    
    def _ensure_vectordb_config(self, trial_dir):
        centralized_vectordb_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
        
        if not os.path.exists(centralized_vectordb_path):
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
        
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if os.path.exists(centralized_vectordb_path):
            os.makedirs(os.path.dirname(vectordb_yaml_path), exist_ok=True)
            shutil.copy(centralized_vectordb_path, vectordb_yaml_path)
    
    def _check_embeddings_exist(self, trial_dir, vectordb_name, corpus_df):
        vectordb_configs = self.embedding_manager.get_vectordb_configs(trial_dir)
        
        if vectordb_name not in vectordb_configs:
            print(f"[Retrieval] VectorDB config '{vectordb_name}' not found")
            return False
        
        vdb_config = vectordb_configs[vectordb_name]
        embedding_model = vdb_config.get('embedding_model', '')
        
        if self._is_sap_embedding_url(embedding_model):
            try:
                from pipeline_component.embedding.sap_embeddings import load_sap_vectordb_from_yaml
                
                vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
                if not os.path.exists(vectordb_yaml_path):
                    vectordb_yaml_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
                
                print(f"[Retrieval] Checking SAP embeddings for '{vectordb_name}'...")
                
                vector_store = load_sap_vectordb_from_yaml(
                    vectordb_yaml_path, vectordb_name, self.centralized_project_dir
                )
                
                doc_count = vector_store.collection.count()
                print(f"[Retrieval] SAP collection '{vectordb_name}' has {doc_count} documents")
                
                if corpus_df is not None:
                    expected_count = len(corpus_df)
                    if doc_count < expected_count * 0.9:
                        print(f"[Retrieval] Collection has {doc_count} docs but corpus has {expected_count} - need to regenerate")
                        return False
                elif doc_count == 0:
                    print(f"[Retrieval] Collection is empty, need to generate embeddings")
                    return False
                
                print(f"[Retrieval] SAP embeddings exist with {doc_count} documents")
                return True
                
            except Exception as e:
                print(f"[Retrieval] Error checking SAP embeddings: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # For standard embeddings, get qa_df
            qa_path = os.path.join(trial_dir, "data", "qa.parquet")
            if not os.path.exists(qa_path):
                qa_path = os.path.join(self.centralized_project_dir, "data", "qa.parquet")
            qa_df = pd.read_parquet(qa_path, engine="pyarrow") if os.path.exists(qa_path) else None
            
            return self.embedding_manager.ensure_vectordb_exists(
                self.centralized_project_dir, vectordb_name, corpus_df, qa_df 
            )

    async def generate_vectordb_embeddings_sap(self, corpus_df: pd.DataFrame, vectordb_name: str, trial_dir: str):
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            vectordb_yaml_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            print(f"VectorDB YAML not found")
            return
        
        print(f"[Retrieval] Generating SAP embeddings for {vectordb_name}...")
        print(f"[Retrieval] Corpus has {len(corpus_df)} documents")
        
        try:
            from pipeline_component.embedding.sap_embeddings import load_sap_vectordb_from_yaml
            
            vectordb = load_sap_vectordb_from_yaml(
                vectordb_yaml_path, vectordb_name, self.centralized_project_dir
            )
            
            existing_count = vectordb.collection.count()
            print(f"[Retrieval] Collection currently has {existing_count} documents")
            
            corpus_data = corpus_df[['doc_id', 'contents']]
            
            if existing_count > 0:
                existing_ids = await vectordb.is_exist(corpus_data['doc_id'].tolist())
                new_data = corpus_data[~pd.Series(existing_ids)]
                print(f"[Retrieval] Found {len(new_data)} new documents to add")
            else:
                new_data = corpus_data
                print(f"[Retrieval] Adding all {len(new_data)} documents")
            
            if not new_data.empty:
                batch_size = 100
                total_added = 0
                
                for i in range(0, len(new_data), batch_size):
                    batch = new_data.iloc[i:i+batch_size]
                    
                    await vectordb.add(
                        ids=batch['doc_id'].tolist(),
                        texts=batch['contents'].tolist()
                    )
                    
                    total_added += len(batch)
                    print(f"[Retrieval] Progress: {total_added}/{len(new_data)} documents processed")
                
                final_count = vectordb.collection.count()
                print(f"[Retrieval] SAP embeddings generation completed. Collection now has {final_count} documents")
            else:
                print(f"[Retrieval] All documents already exist in collection")
            
        except Exception as e:
            print(f"[Retrieval] Error generating SAP embeddings: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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