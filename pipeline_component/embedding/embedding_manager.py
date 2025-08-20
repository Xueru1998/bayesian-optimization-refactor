import os
import shutil
import pandas as pd
import yaml
import asyncio
import subprocess
from typing import Dict, Any, List, Union, Optional

from autorag.nodes.retrieval.bm25 import bm25_ingest, get_bm25_pkl_name
from autorag.vectordb import load_vectordb_from_yaml, load_all_vectordb_from_yaml
from autorag.nodes.retrieval.vectordb import vectordb_ingest, filter_exist_ids_from_retrieval_gt


class EmbeddingManager:
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = embeddings_dir
        os.makedirs(self.embeddings_dir, exist_ok=True)
    
    def get_vectordb_configs(self, trial_dir: str) -> Dict[str, Dict[str, Any]]:
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            vectordb_yaml_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            return {}
        
        with open(vectordb_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vectordb_configs = {}
        for vdb in config.get('vectordb', []):
            name = vdb.get('name')
            if name:
                vectordb_configs[name] = vdb
        
        return vectordb_configs
    
    def copy_embeddings_to_trial(self, trial_dir: str) -> bool:
        resources_dir = os.path.join(trial_dir, "resources")
        os.makedirs(resources_dir, exist_ok=True)
        
        success = True
        
        centralized_vectordb = os.path.join(self.embeddings_dir, "vectordb.yaml")
        if os.path.exists(centralized_vectordb):
            shutil.copy(centralized_vectordb, os.path.join(resources_dir, "vectordb.yaml"))
        
        for tokenizer in ['porter_stemmer', 'space', 'gpt2']:
            bm25_file = get_bm25_pkl_name(tokenizer)
            source_path = os.path.join(self.embeddings_dir, bm25_file)
            target_path = os.path.join(resources_dir, bm25_file)
            
            if os.path.exists(source_path):
                print(f"Copying BM25 file {bm25_file} from centralized location")
                shutil.copy(source_path, target_path)
                
                if not os.path.exists(target_path):
                    print(f"Error: Failed to copy {bm25_file} to trial directory")
                    success = False
            else:
                print(f"Warning: BM25 file {bm25_file} not found in centralized location at {source_path}")
        
        vectordb_configs = self.get_vectordb_configs(trial_dir)
        for vdb_name, vdb_config in vectordb_configs.items():
            db_type = vdb_config.get('db_type', 'chroma')
            
            if db_type == 'chroma':
                collection_name = vdb_config.get('collection_name')
                if collection_name:
                    centralized_chroma = os.path.join(self.embeddings_dir, "chroma")
                    if os.path.exists(centralized_chroma):
                        target_chroma = os.path.join(resources_dir, "chroma")
                        if not os.path.exists(target_chroma):
                            os.makedirs(target_chroma, exist_ok=True)
                        print(f"Chroma directory exists for {collection_name}")
        
        if not success:
            print("Warning: Some required embedding files couldn't be copied from centralized location.")
            print("Make sure to run the embedding.py script first to generate all needed embeddings.")
        
        return success
    
    def setup_vectordb_config(self, trial_dir: str, use_pregenerated: bool = True) -> str:
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if use_pregenerated and os.path.exists(vectordb_yaml_path):
            with open(vectordb_yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config and 'vectordb' in config and isinstance(config['vectordb'], list) and len(config['vectordb']) > 0:
                return vectordb_yaml_path

        chroma_base_path = os.path.join(self.embeddings_dir, "chroma")
        os.makedirs(chroma_base_path, exist_ok=True)
        
        default_configs = [{
            'name': 'default',
            'db_type': 'chroma',
            'client_type': 'persistent',
            'embedding_model': 'openai',
            'collection_name': 'openai',
            'path': chroma_base_path
        }]
        
        vectordb_yaml_content = {'vectordb': default_configs}
        
        os.makedirs(os.path.dirname(vectordb_yaml_path), exist_ok=True)
        with open(vectordb_yaml_path, "w") as f:
            yaml.dump(vectordb_yaml_content, f)
        
        return vectordb_yaml_path
    
    def generate_bm25_embeddings(self, corpus_df: pd.DataFrame, bm25_tokenizer: str, force_regenerate: bool = False) -> str:
        bm25_filename = get_bm25_pkl_name(bm25_tokenizer)
        centralized_bm25_path = os.path.join(self.embeddings_dir, bm25_filename)
        
        if not os.path.exists(centralized_bm25_path) or force_regenerate:
            print(f"Generating BM25 index for tokenizer '{bm25_tokenizer}'...")
            try:
                bm25_ingest(centralized_bm25_path, corpus_df, bm25_tokenizer=bm25_tokenizer)
                print(f"BM25 index saved to: {centralized_bm25_path}")
            except Exception as e:
                print(f"Error generating BM25 index: {e}")
                raise
        
        return centralized_bm25_path
    
    async def generate_vectordb_embeddings(self, corpus_df: pd.DataFrame, qa_df: Optional[pd.DataFrame], 
                                     vectordb_name: str, trial_dir: str, force_regenerate: bool = False):
        vectordb_yaml_path = os.path.join(trial_dir, "resources", "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            vectordb_yaml_path = os.path.join(self.embeddings_dir, "vectordb.yaml")
        
        if not os.path.exists(vectordb_yaml_path):
            print(f"VectorDB YAML not found in trial or centralized location.")
            return
        
        vectordb_configs = self.get_vectordb_configs(trial_dir)
        
        if vectordb_name not in vectordb_configs:
            print(f"VectorDB configuration '{vectordb_name}' not found.")
            return
        
        vdb_config = vectordb_configs[vectordb_name]
        collection_name = vdb_config.get('collection_name')
        
        print(f"Generating vectordb embeddings for {vectordb_name} (collection: {collection_name})...")
        
        try:
            vectordb = load_vectordb_from_yaml(vectordb_yaml_path, vectordb_name, trial_dir)
            
            if qa_df is not None:
                new_passages = await filter_exist_ids_from_retrieval_gt(vectordb, qa_df, corpus_df)
                if not new_passages.empty:
                    await vectordb_ingest(vectordb, new_passages)
                else:
                    await vectordb_ingest(vectordb, corpus_df)
            else:
                await vectordb_ingest(vectordb, corpus_df)
            
            print(f"Vectordb embeddings saved for {vectordb_name}")
        except Exception as e:
            print(f"Error generating vectordb embeddings: {e}")
            raise
    
    def ensure_bm25_exists(self, trial_dir: str, bm25_tokenizer: str) -> str:
        bm25_filename = get_bm25_pkl_name(bm25_tokenizer)
        centralized_bm25_path = os.path.join(self.embeddings_dir, bm25_filename)
        trial_bm25_path = os.path.join(trial_dir, "resources", bm25_filename)
        
        if not os.path.exists(centralized_bm25_path):
            corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
            if os.path.exists(corpus_path):
                corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
                self.generate_bm25_embeddings(corpus_df, bm25_tokenizer)
        
        if os.path.exists(centralized_bm25_path) and not os.path.exists(trial_bm25_path):
            print(f"Copying BM25 index from centralized location to trial directory")
            os.makedirs(os.path.dirname(trial_bm25_path), exist_ok=True)
            shutil.copy(centralized_bm25_path, trial_bm25_path)
        
        return trial_bm25_path
    
    def ensure_vectordb_exists(self, trial_dir: str, vectordb_name: str = 'default', corpus_df: pd.DataFrame = None, qa_df: pd.DataFrame = None) -> bool:
        vectordb_configs = self.get_vectordb_configs(trial_dir)
        
        if not vectordb_configs:
            print(f"No vectordb configurations found in {trial_dir}")
            return False
        
        if vectordb_name not in vectordb_configs:
            print(f"VectorDB configuration '{vectordb_name}' not found in configs. Available: {list(vectordb_configs.keys())}")
            return False
        
        vdb_config = vectordb_configs[vectordb_name]
        db_type = vdb_config.get('db_type', 'chroma')
        embedding_model = vdb_config.get('embedding_model', '')
        
        if embedding_model.startswith("http") and "api.ai.internalprod" in embedding_model:
            return True
        
        collection_name = vdb_config.get('collection_name', 'openai')
        
        print(f"Checking embeddings for {vectordb_name}: collection={collection_name}, type={db_type}")
        
        if db_type == 'chroma':
            centralized_chroma_path = os.path.join(self.embeddings_dir, "chroma")
            
            if os.path.exists(centralized_chroma_path):
                try:
                    from chromadb import PersistentClient
                    client = PersistentClient(path=centralized_chroma_path)
                    
                    collection_exists = False
                    doc_count = 0
                    try:
                        collection = client.get_collection(name=collection_name)
                        collection_exists = True
                        doc_count = collection.count()
                        print(f"Collection '{collection_name}' exists with {doc_count} documents")
                    except ValueError as e:
                        if "does not exist" in str(e):
                            print(f"Collection '{collection_name}' does not exist")
                            collection_exists = False
                        else:
                            raise
                    except Exception as e:
                        print(f"Error checking collection '{collection_name}': {e}")
                        collection_exists = False
                    
                    if collection_exists and doc_count > 0:
                        if corpus_df is not None:
                            expected_count = len(corpus_df)
                            if doc_count < expected_count * 0.9:
                                print(f"Collection has {doc_count} docs but corpus has {expected_count} - need to regenerate")
                                collection_exists = False
                            else:
                                return True
                        else:
                            return True
                    elif collection_exists and doc_count == 0:
                        print(f"Collection exists but is empty - need to generate embeddings")
                        collection_exists = False
                        
                except ImportError as e:
                    print(f"ChromaDB not installed: {e}")
                    return False
                except Exception as e:
                    print(f"Error initializing Chroma client: {e}")
            
            if corpus_df is None:
                corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
                if os.path.exists(corpus_path):
                    corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
                else:
                    print(f"Corpus data not found for embedding generation at {corpus_path}")
                    return False
            
            if qa_df is None:
                qa_path = os.path.join(trial_dir, "data", "qa.parquet")
                if os.path.exists(qa_path):
                    qa_df = pd.read_parquet(qa_path, engine="pyarrow")
            
            try:
                print(f"Generating embeddings for {vectordb_name}...")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.generate_vectordb_embeddings(corpus_df, qa_df, vectordb_name, trial_dir))
                loop.close()
                
                return True
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True


    def check_all_chroma_collections(chroma_path: str) -> list:
        try:
            from chromadb import PersistentClient
            client = PersistentClient(path=chroma_path)
            
            collections = []
            
            metadata_path = os.path.join(chroma_path, 'chroma.sqlite3')
            if os.path.exists(metadata_path):
                import sqlite3
                conn = sqlite3.connect(metadata_path)
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT name FROM collections")
                    collections = [row[0] for row in cursor.fetchall()]
                    print(f"Found collections via SQLite: {collections}")
                except Exception as e:
                    print(f"Error querying SQLite: {e}")
                finally:
                    conn.close()
            
            confirmed_collections = []
            for coll_name in collections:
                try:
                    coll = client.get_collection(name=coll_name)
                    count = coll.count()
                    confirmed_collections.append(coll_name)
                    print(f"Confirmed collection '{coll_name}' with {count} documents")
                except:
                    pass
            
            return confirmed_collections
            
        except Exception as e:
            print(f"Error checking collections: {e}")
            return []
        
    def run_embedding_script(self, tokenizer: Optional[str] = None, vectordb_name: Optional[str] = None):
        try:
            embedding_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding.py")
            
            if not os.path.exists(embedding_script):
                print(f"Warning: embedding.py script not found at {embedding_script}")
                return
            
            cmd = ["python", embedding_script]
            if tokenizer:
                cmd.extend(["--tokenizer", tokenizer])
            if vectordb_name:
                cmd.extend(["--vectordb", vectordb_name])
                
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("Embedding script completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running embedding script: {e}")
        except Exception as e:
            print(f"Unexpected error running embedding script: {e}")