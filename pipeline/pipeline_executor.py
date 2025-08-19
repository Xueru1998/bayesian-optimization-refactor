import os
import time
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from autorag.nodes.retrieval.bm25 import get_bm25_pkl_name, bm25_ingest
from pipeline_component.query_expansion import create_query_expansion
from pipeline_component.retrieval import RetrievalModule
from pipeline_component.passagefilter import PassageFilterModule
from pipeline_component.passagecompressor import PassageCompressorModule
from pipeline_component.promptmaker import PromptMakerModule
from pipeline_component.generator import create_generator
from pipeline_component.passageReranker import PassageRerankerModule
from pipeline.config_manager import ConfigGenerator
from pipeline.config_manager import NodeDefaults
from pipeline.utils import Utils


class RAGPipelineExecutor:
    def __init__(self, config_generator: ConfigGenerator):
        self.config_generator = config_generator
    
    @staticmethod
    def parse_query_expansion_config(qe_config_str: str) -> Dict[str, Any]:
        if not qe_config_str:
            return {}
        
        if qe_config_str == 'pass_query_expansion':
            return {'query_expansion_method': 'pass_query_expansion'}

        if qe_config_str.startswith('query_decompose_'):
            model = qe_config_str.replace('query_decompose_', '')
            return {
                'query_expansion_method': 'query_decompose',
                'query_expansion_model': model
            }

        if qe_config_str.startswith('hyde_'):
            parts = qe_config_str.replace('hyde_', '').rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                model = parts[0]
                max_token = int(parts[1])
                return {
                    'query_expansion_method': 'hyde',
                    'query_expansion_model': model,
                    'query_expansion_max_token': max_token
                }
            else:
                return {
                    'query_expansion_method': 'hyde',
                    'query_expansion_max_token': int(qe_config_str.split('_')[-1])
                }

        if qe_config_str.startswith('multi_query_expansion_'):
            temp = float(qe_config_str.split('_')[-1])
            return {
                'query_expansion_method': 'multi_query_expansion',
                'query_expansion_temperature': temp
            }

        if qe_config_str == 'query_decompose':
            return {'query_expansion_method': 'query_decompose'}
        
        return {}
    
    @staticmethod
    def parse_retrieval_config(retrieval_config_str: str) -> Dict[str, Any]:
        if not retrieval_config_str:
            return {}
        
        if retrieval_config_str.startswith('bm25_'):
            tokenizer = retrieval_config_str.replace('bm25_', '')
            return {
                'retrieval_method': 'bm25',
                'bm25_tokenizer': tokenizer
            }
        elif retrieval_config_str.startswith('vectordb_'):
            vdb_name = retrieval_config_str.replace('vectordb_', '')
            return {
                'retrieval_method': 'vectordb',
                'vectordb_name': vdb_name
            }
        
        return {}
    
    @staticmethod
    def parse_reranker_config(reranker_config_str: str) -> Dict[str, Any]:
        if not reranker_config_str:
            return {}
        
        if reranker_config_str == 'pass_reranker':
            return {'passage_reranker_method': 'pass_reranker'}
        
        simple_methods = ['upr']
        if reranker_config_str in simple_methods:
            return {'passage_reranker_method': reranker_config_str}
        
        model_based_methods = [
            'colbert_reranker',
            'sentence_transformer_reranker',
            'flag_embedding_reranker',
            'flag_embedding_llm_reranker',
            'openvino_reranker',
            'flashrank_reranker',
            'monot5'
        ]
        
        for method in model_based_methods:
            if reranker_config_str.startswith(method + '_'):
                model_name = reranker_config_str[len(method) + 1:]
                return {
                    'passage_reranker_method': method,
                    'reranker_model_name': model_name
                }
        
        return {'passage_reranker_method': reranker_config_str}
    
    @staticmethod
    def parse_filter_config(filter_config_str: str) -> Dict[str, Any]:
        if not filter_config_str:
            return {}
        
        parts = filter_config_str.split('_')
        
        if filter_config_str.startswith('threshold_cutoff_'):
            return {
                'passage_filter_method': 'threshold_cutoff',
                'threshold': float(parts[-1])
            }
        elif filter_config_str.startswith('percentile_cutoff_'):
            return {
                'passage_filter_method': 'percentile_cutoff',
                'percentile': float(parts[-1])
            }
        elif filter_config_str.startswith('similarity_threshold_cutoff_'):
            return {
                'passage_filter_method': 'similarity_threshold_cutoff',
                'threshold': float(parts[-1])
            }
        elif filter_config_str.startswith('similarity_percentile_cutoff_'):
            return {
                'passage_filter_method': 'similarity_percentile_cutoff',
                'percentile': float(parts[-1])
            }
        
        return {}
    
    @staticmethod
    def parse_compressor_config(compressor_config_str: str) -> Dict[str, Any]:
        if not compressor_config_str:
            return {}
        
        if compressor_config_str == 'pass_compressor':
            return {'passage_compressor_method': 'pass_compressor'}
        
        if compressor_config_str.startswith('tree_summarize_') or compressor_config_str.startswith('refine_'):
            parts = compressor_config_str.split('_', 2)
            if len(parts) >= 3:
                method = parts[0] + '_' + parts[1] 
                llm_and_model = parts[2]
                
                llm_parts = llm_and_model.split('_', 1)
                if len(llm_parts) == 2:
                    return {
                        'passage_compressor_method': method,
                        'compressor_llm': llm_parts[0],
                        'compressor_model': llm_parts[1]
                    }
        
        return {'passage_compressor_method': compressor_config_str}
    
    @staticmethod
    def parse_prompt_config(prompt_config_str: str) -> Dict[str, Any]:
        if not prompt_config_str:
            return {}
        
        if prompt_config_str == 'pass_prompt_maker':
            return {'prompt_maker_method': 'pass_prompt_maker'}
        
        known_prompt_methods = ['fstring', 'long_context_reorder', 'window_replacement']
        
        parts = prompt_config_str.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            method = parts[0]
            if method in known_prompt_methods:
                return {
                    'prompt_maker_method': method,
                    'prompt_template_idx': int(parts[1])
                }

        if prompt_config_str in known_prompt_methods:
            return {'prompt_maker_method': prompt_config_str}
        
        print(f"Warning: Unknown prompt config '{prompt_config_str}'. Using default 'fstring_0'.")
        return {
            'prompt_maker_method': 'fstring',
            'prompt_template_idx': 0
        }
        
    def _prepare_embeddings_if_needed(self, config: Dict[str, Any], trial_dir: str):
        if config.get('retrieval_method') == 'vectordb':
            vectordb_name = config.get('vectordb_name', 'default')
            
            retrieval_module = RetrievalModule(
                base_project_dir=trial_dir,
                use_pregenerated_embeddings=True,
                centralized_project_dir=Utils.get_centralized_project_dir()
            )
            
            vectordb_configs = retrieval_module.embedding_manager.get_vectordb_configs(trial_dir)
            
            if vectordb_name in vectordb_configs:
                vdb_config = vectordb_configs[vectordb_name]
                embedding_model = vdb_config.get('embedding_model', '')
                
                if embedding_model.startswith('http') and 'api.ai.internalprod' in embedding_model:
                    print(f"[Trial] Detected SAP API embedding for {vectordb_name}")
                    
                    corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
                    if not os.path.exists(corpus_path):
                        corpus_path = os.path.join(Utils.get_centralized_project_dir(), "data", "corpus.parquet")
                    
                    if os.path.exists(corpus_path):
                        corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
                        
                        if not retrieval_module._check_embeddings_exist(trial_dir, vectordb_name, corpus_df):
                            print(f"[Trial] Generating SAP embeddings for {vectordb_name}...")
                            retrieval_module.prepare_embeddings(
                                trial_dir, corpus_df, specific_vectordb=vectordb_name
                            )
    
        
    def _prepare_bm25_index(self, config: Dict[str, Any], trial_dir: str):        
        bm25_tokenizer = config['bm25_tokenizer']
        centralized_bm25_file = os.path.join(
            Utils.get_centralized_project_dir(), 
            "resources", 
            get_bm25_pkl_name(bm25_tokenizer)
        )
        
        if not os.path.exists(centralized_bm25_file):
            print(f"Generating BM25 index directly for {bm25_tokenizer}")
            corpus_path = os.path.join(Utils.get_centralized_project_dir(), "data", "corpus.parquet")
            if not os.path.exists(corpus_path):
                corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
            
            if os.path.exists(corpus_path):
                corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
                bm25_ingest(centralized_bm25_file, corpus_df, bm25_tokenizer=bm25_tokenizer)
                print(f"BM25 index saved to: {centralized_bm25_file}")
    
    def execute_query_expansion(self, config: Dict[str, Any], trial_dir: str, 
                          working_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[str]]]:
        has_query_expansion = self.config_generator.node_exists("query_expansion")
        
        if not has_query_expansion:
            print("[Trial] Query expansion not defined in config.yaml - skipping")
            return working_df.copy(), None

        if config.get('query_expansion_config') == 'pass_query_expansion':
            print("[Trial] Query expansion config is pass_query_expansion - skipping")
            return working_df.copy(), None

        if 'query_expansion_config' in config:
            qe_config_str = config['query_expansion_config']
            parts = qe_config_str.split('::')
            
            if len(parts) >= 3:
                method, gen_type, model = parts[0], parts[1], parts[2]
                config['query_expansion_method'] = method
                config['query_expansion_generator_module_type'] = gen_type
                config['query_expansion_model'] = model
                
                if method == 'hyde' and len(parts) >= 4:
                    config['query_expansion_max_token'] = int(parts[3])
                elif method == 'multi_query_expansion' and len(parts) >= 4:
                    config['query_expansion_temperature'] = float(parts[3])
                
                unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                for gen_config in unified_params.get('generator_configs', []):
                    if (gen_config['method'] == method and 
                        gen_config['generator_module_type'] == gen_type and 
                        model in gen_config['models']):
                        if gen_type == 'sap_api':
                            config['query_expansion_api_url'] = gen_config.get('api_url')
                            config['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                        else:
                            config['query_expansion_llm'] = gen_config.get('llm', 'openai')
                        break
        
        query_expansion_method = config.get('query_expansion_method')
        
        if not (query_expansion_method and query_expansion_method != 'pass_query_expansion'):
            print("[Trial] No query expansion method specified or pass_query_expansion - skipping")
            return working_df, None
            
        print(f"[Trial] Running query expansion with method: {query_expansion_method}")
        
        try:
            expansion_config = {
                'module_type': query_expansion_method,
            }
            
            if 'query_expansion_generator_module_type' in config:
                expansion_config['generator_module_type'] = config['query_expansion_generator_module_type']
            else:
                qe_node = self.config_generator.extract_node_config("query_expansion")
                if qe_node and 'modules' in qe_node:
                    for module in qe_node['modules']:
                        if module.get('module_type') == query_expansion_method:
                            expansion_config['generator_module_type'] = module.get('generator_module_type', 'llama_index_llm')
                            break
                else:
                    expansion_config['generator_module_type'] = NodeDefaults.QUERY_EXPANSION['params']['generator_module_type']
            
            if expansion_config.get('generator_module_type') == 'sap_api':
                if 'query_expansion_api_url' in config:
                    expansion_config['api_url'] = self.config_generator.resolve_env_vars(config['query_expansion_api_url'])
                expansion_config['llm'] = config.get('query_expansion_llm', 'mistralai')
            else:
                expansion_config['llm'] = config.get('query_expansion_llm', NodeDefaults.QUERY_EXPANSION['params']['llm'])

            if 'query_expansion_model' in config:
                expansion_config['model'] = config['query_expansion_model']
            else:
                expansion_config['model'] = NodeDefaults.QUERY_EXPANSION['params']['model']

            if query_expansion_method == 'hyde':
                expansion_config['max_token'] = config.get('query_expansion_max_token', NodeDefaults.QUERY_EXPANSION['params']['max_token'])
            elif query_expansion_method == 'multi_query_expansion':
                expansion_config['temperature'] = config.get('query_expansion_temperature', 0.7)
            
            print(f"[DEBUG] Query expansion config: {expansion_config}")
            
            if 'query' not in working_df.columns:
                raise ValueError("Query column not found in working_df")
            
            query_expander = create_query_expansion(
                module_type=expansion_config.pop('module_type'),
                project_dir=trial_dir,
                **expansion_config
            )
            
            expanded_df = query_expander.pure(working_df)
            
            if 'queries' not in expanded_df.columns:
                print(f"Warning: Query expansion did not create 'queries' column.")
                return working_df, None
            
            expanded_queries = expanded_df['queries'].tolist()
            
            print(f"Query expansion completed. Expanded {len(working_df)} queries.")
            return expanded_df, expanded_queries
                
        except ImportError as e:
            print(f"Query expansion module not available: {e}")
        except Exception as e:
            print(f"Error in query expansion: {e}, using original queries")
            import traceback
            traceback.print_exc()
            
        return working_df, None
    
    def execute_query_expansion_retrieval(self, config: Dict[str, Any], trial_dir: str,
                                    working_df: pd.DataFrame) -> pd.DataFrame:
        print(f"[Trial] Performing retrieval for query expansion evaluation")
        
        self._prepare_embeddings_if_needed(config, trial_dir)
        
        temp_retrieval_module = RetrievalModule(
            base_project_dir=trial_dir,
            use_pregenerated_embeddings=True,
            centralized_project_dir=Utils.get_centralized_project_dir()
        )
        
        query_expansion_retrieval_config = self.config_generator.get_query_expansion_retrieval_config(config)
        
        retrieval_method = config.get('query_expansion_retrieval_method')
        if not retrieval_method and query_expansion_retrieval_config.get('retrieval_method'):
            retrieval_method = query_expansion_retrieval_config['retrieval_method']
        elif not retrieval_method:
            qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
            if qe_retrieval_options.get('methods'):
                retrieval_method = qe_retrieval_options['methods'][0]
            else:
                print("[Trial] No query expansion retrieval method specified!!")
        
        print(f"[Trial] Query expansion retrieval method: {retrieval_method}")
        
        vectordb_name = 'default'
        bm25_tokenizer = None
        
        if retrieval_method == 'vectordb':
            vectordb_name = config.get('query_expansion_vectordb_name')
            if not vectordb_name:
                qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
                if qe_retrieval_options.get('vectordb_names'):
                    vectordb_name = qe_retrieval_options['vectordb_names'][0]
                else:
                    vectordb_name = query_expansion_retrieval_config.get('vectordb_name', 'default')
            print(f"[Trial] Query expansion using vectordb: {vectordb_name}")
            
            config_for_embeddings = config.copy()
            config_for_embeddings['retrieval_method'] = 'vectordb'
            config_for_embeddings['vectordb_name'] = vectordb_name
            self._prepare_embeddings_if_needed(config_for_embeddings, trial_dir)
            
        elif retrieval_method == 'bm25':
            bm25_tokenizer = config.get('query_expansion_bm25_tokenizer')
            if not bm25_tokenizer:
                qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
                if qe_retrieval_options.get('bm25_tokenizers'):
                    bm25_tokenizer = qe_retrieval_options['bm25_tokenizers'][0]
                else:
                    bm25_tokenizer = query_expansion_retrieval_config.get('bm25_tokenizer', 'porter_stemmer')
            print(f"[Trial] Query expansion using BM25 tokenizer: {bm25_tokenizer}")
            
            self._prepare_bm25_index({'bm25_tokenizer': bm25_tokenizer}, trial_dir)

        top_k = config.get('retriever_top_k', query_expansion_retrieval_config.get('top_k', 10))
        print(f"[Trial] Query expansion using top_k: {top_k}")
        
        temp_df = working_df.copy()
        retrieved_contents, retrieved_ids, retrieved_scores = temp_retrieval_module.perform_retrieval(
            trial_dir,
            temp_df,
            retrieval_method=retrieval_method,
            bm25_tokenizer=bm25_tokenizer,
            vectordb_name=vectordb_name if retrieval_method == 'vectordb' else None,
            top_k=top_k
        )
        
        retrieval_df = pd.DataFrame({
            'query': working_df['query'].values,
            'retrieved_ids': retrieved_ids,
            'retrieved_contents': retrieved_contents,
            'retrieve_scores': retrieved_scores,
            'queries': working_df['queries'].values
        })
        
        return retrieval_df
    
    def execute_retrieval(self, config: Dict[str, Any], trial_dir: str, 
                        working_df: pd.DataFrame) -> pd.DataFrame:
        
        has_retrieval = self.config_generator.node_exists("retrieval")
        if not has_retrieval:
            print("[Trial] Retrieval not defined in config.yaml - skipping")
            empty_df = pd.DataFrame({
                'query': working_df['query'].values,
                'retrieved_ids': [[] for _ in range(len(working_df))],
                'retrieved_contents': [[] for _ in range(len(working_df))],
                'retrieve_scores': [[] for _ in range(len(working_df))]
            })
            return empty_df
        
        self._prepare_embeddings_if_needed(config, trial_dir)
        
        retrieval_info = ""
        if config.get('retrieval_method') == 'bm25':
            retrieval_info = f", bm25_tokenizer: {config.get('bm25_tokenizer', 'default')}"
        elif config.get('retrieval_method') == 'vectordb':
            retrieval_info = f", vectordb: {config.get('vectordb_name', 'default')}"

        print(f"\n[Trial] Running retrieval with method: {config.get('retrieval_method')}"
            f", retriever_top_k: {config.get('retriever_top_k', config.get('top_k'))}{retrieval_info}")
        
        retrieval_module = RetrievalModule(
            base_project_dir=trial_dir,
            use_pregenerated_embeddings=True,
            centralized_project_dir=Utils.get_centralized_project_dir()
        )

        if config.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in config:
            self._prepare_bm25_index(config, trial_dir)

        vectordb_name = 'default'
        if config.get('retrieval_method') == 'vectordb':
            vectordb_name = config.get('vectordb_name', 'default')
            print(f"Using vectordb: {vectordb_name}")
        
        retrieved_contents, retrieved_ids, retrieved_scores = retrieval_module.perform_retrieval(
            trial_dir,
            working_df,
            retrieval_method=config.get('retrieval_method'),
            bm25_tokenizer=config.get('bm25_tokenizer') if config.get('retrieval_method') == 'bm25' else None,
            vectordb_name=vectordb_name, 
            top_k=config.get('retriever_top_k', config.get('top_k', 5))
        )
        
        retrieval_df = pd.DataFrame({
            'query': working_df['query'].values,
            'retrieved_ids': retrieved_ids,
            'retrieved_contents': retrieved_contents,
            'retrieve_scores': retrieved_scores
        })
        
        if 'queries' in working_df.columns:
            retrieval_df['queries'] = working_df['queries'].values
            
        return retrieval_df
    
    def execute_reranker(self, config: Dict[str, Any], trial_dir: str, 
                    working_df: pd.DataFrame) -> pd.DataFrame:
    
        if not self.config_generator.node_exists("passage_reranker"):
            return working_df
        
        method = config.get('passage_reranker_method')
        if not method or method in ('pass_reranker', 'pass'):
            return working_df
        
        print(f"[Trial] Running passage reranker with method: {method}")
        reranker_module = PassageRerankerModule(project_dir=trial_dir)
        
        rerank_cfg = {
            'module_type': method,
            'top_k': config.get('reranker_top_k', config.get('top_k', 5)),
            'batch': config.get('reranker_batch', 16)
        }
        
        method_models_key = f"{method}_models"
        if method_models_key in config:
            if method == 'flashrank_reranker':
                rerank_cfg['model'] = config[method_models_key]
            else:
                rerank_cfg['model_name'] = config[method_models_key]
        
        elif config.get('reranker_model_name'):
            if method == 'flashrank_reranker':
                rerank_cfg['model'] = config['reranker_model_name']
                print(f"[Trial] Using flashrank model: {config['reranker_model_name']}")
            else:
                rerank_cfg['model_name'] = config['reranker_model_name']
                print(f"[Trial] Using model: {config['reranker_model_name']}")
        elif config.get('reranker_model'):
            if method == 'flashrank_reranker':
                rerank_cfg['model'] = config['reranker_model']
            else:
                rerank_cfg['model_name'] = config['reranker_model']

        if method in ('sap_api', 'sap_reranker'):
            api_url = config.get('reranker_api_url')
            
            if api_url:
                rerank_cfg['api-url'] = api_url
                rerank_cfg['api_url'] = api_url  
                rerank_cfg['reranker_api_url'] = api_url
            else:
                print(f"[WARNING] No API URL found for {method} in config keys: {list(config.keys())}")

        if config.get('cache_dir'):
            rerank_cfg['cache_dir'] = config['cache_dir']
        if config.get('max_length'):
            rerank_cfg['max_length'] = config['max_length']
        
        print(f"[DEBUG] Final reranker_config: {rerank_cfg}")
        
        try:
            reranked_df = reranker_module.apply_reranking(working_df, rerank_cfg)
            print(f"Applied {method} reranking with top_k={rerank_cfg['top_k']}")
            if 'model_name' in rerank_cfg:
                print(f"Using model: {rerank_cfg['model_name']}")
            elif 'model' in rerank_cfg:
                print(f"Using model: {rerank_cfg['model']}")
            return reranked_df
            
        except Exception as e:
            print(f"[ERROR] Reranking failed: {e}")
            import traceback
            traceback.print_exc()
            print("[WARNING] Returning original results due to reranking error")
            return working_df
            
        finally:
            if hasattr(reranker_module, 'cleanup_models'):
                reranker_module.cleanup_models()
            
            del reranker_module
            
            import gc
            gc.collect()
    
    def execute_filter(self, config: Dict[str, Any], trial_dir: str, 
                     working_df: pd.DataFrame) -> pd.DataFrame:
        
        has_filter_module = self.config_generator.node_exists("passage_filter")
        
        if not has_filter_module:
            print("[Trial] Passage filter not defined in config.yaml - skipping")
            return working_df
        
        filter_method = config.get('passage_filter_method')
        
        if not filter_method or filter_method == 'pass_passage_filter':
            print("[Trial] No passage filter method specified or pass_passage_filter - skipping")
            return working_df
        
        print(f"[Trial] Running passage filter with method: {filter_method}")
        
        filter_module = PassageFilterModule(project_dir=trial_dir)
        
        if filter_method == 'threshold_cutoff':
            threshold = config.get('threshold', NodeDefaults.PASSAGE_FILTER['params']['threshold_cutoff']['threshold'])
            print(f"[DEBUG] Using threshold: {threshold} (from config: {'threshold' in config})")
            filtered_df = filter_module.apply_filter_directly(
                'threshold_cutoff', working_df, threshold=threshold, reverse=False
            )
            print(f"Applied threshold cutoff with threshold={threshold}")
            
        elif filter_method == 'percentile_cutoff':
            percentile = config.get('percentile', NodeDefaults.PASSAGE_FILTER['params']['percentile_cutoff']['percentile'])
            print(f"[DEBUG] Using percentile: {percentile} (from config: {'percentile' in config})")
            filtered_df = filter_module.apply_filter_directly(
                'percentile_cutoff', working_df, percentile=percentile, reverse=False
            )
            print(f"Applied percentile cutoff with percentile={percentile}")
            
        elif filter_method == 'similarity_percentile_cutoff':
            defaults = NodeDefaults.PASSAGE_FILTER['params']['similarity_percentile_cutoff']
            percentile = config.get('percentile', defaults['percentile'])
            batch = config.get('batch', defaults['batch'])
            embedding_model = config.get('embedding_model', defaults['embedding_model'])
            filtered_df = filter_module.apply_filter_directly(
                'similarity_percentile_cutoff', working_df, 
                percentile=percentile, batch=batch, embedding_model=embedding_model
            )
            print(f"Applied similarity percentile cutoff with percentile={percentile}, embedding model={embedding_model}")
            
        elif filter_method == 'similarity_threshold_cutoff':
            defaults = NodeDefaults.PASSAGE_FILTER['params']['similarity_threshold_cutoff']
            threshold = config.get('threshold', defaults['threshold'])
            batch = config.get('batch', defaults['batch'])
            embedding_model = config.get('embedding_model', defaults['embedding_model'])
            filtered_df = filter_module.apply_filter_directly(
                'similarity_threshold_cutoff', working_df,
                threshold=threshold, batch=batch, embedding_model=embedding_model
            )
            print(f"Applied similarity threshold cutoff with threshold={threshold}, embedding model={embedding_model}")
            
        else:
            filtered_df = working_df
            print(f"Unknown filter method: {filter_method}, using pass-through")
            
        return filtered_df
    
    def execute_compressor(self, config: Dict[str, Any], trial_dir: str, 
             working_df: pd.DataFrame) -> pd.DataFrame:

        has_compressor_module = self.config_generator.node_exists("passage_compressor")
        
        if not has_compressor_module:
            print("[Trial] Passage compressor not defined in config.yaml - skipping")
            return working_df

        if 'passage_compressor_config' in config:
            comp_config_str = config['passage_compressor_config']
            
            if comp_config_str == 'pass_compressor':
                config['passage_compressor_method'] = 'pass_compressor'
            else:
                parts = comp_config_str.split('::', 3)
                
                if parts[0] in ['lexrank', 'spacy']:
                    method = parts[0]
                    config['passage_compressor_method'] = method
                    
                    if method == 'spacy' and len(parts) > 1:
                        config['spacy_model'] = parts[1]
                
                elif len(parts) >= 3:
                    method, gen_type, model = parts[0], parts[1], parts[2]
                    
                    config['passage_compressor_method'] = method
                    config['compressor_generator_module_type'] = gen_type
                    config['compressor_model'] = model

                    unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                    for comp_config in unified_params.get('compressor_configs', []):
                        if (comp_config['method'] == method and 
                            comp_config['generator_module_type'] == gen_type and 
                            model in comp_config['models']):

                            config['compressor_llm'] = comp_config.get('llm', 'openai')

                            if gen_type == 'sap_api':
                                config['compressor_api_url'] = comp_config.get('api_url')
                            elif gen_type == 'vllm':
                                config['compressor_llm'] = model 
                                if 'tensor_parallel_size' in comp_config:
                                    config['compressor_tensor_parallel_size'] = comp_config['tensor_parallel_size']
                                if 'gpu_memory_utilization' in comp_config:
                                    config['compressor_gpu_memory_utilization'] = comp_config['gpu_memory_utilization']

                            if 'batch' in comp_config:
                                config['compressor_batch'] = comp_config['batch']
                            break
                else:
                    method = parts[0]
                    
                    if method in ['lexrank', 'spacy']:
                        config['passage_compressor_method'] = method
                        if method == 'spacy' and len(parts) > 1:
                            config['spacy_model'] = parts[1]
                    else:
                        llm = parts[1] if len(parts) > 1 else 'openai'
                        model = parts[2] if len(parts) > 2 else None
                        
                        config['passage_compressor_method'] = method
                        config['compressor_llm'] = llm
                        if model:
                            config['compressor_model'] = model
        
        compressor_method = config.get('passage_compressor_method')
        
        if not compressor_method or compressor_method == 'pass_compressor':
            print("[Trial] No passage compression method specified or pass_compressor - skipping")
            return working_df
            
        print(f"[Trial] Running passage compressor with method: {compressor_method}")
        compressor_module = PassageCompressorModule(project_dir=trial_dir)

        compression_config = {
            'module_type': compressor_method,
            'batch': config.get('compressor_batch', 16)
        }

        if compressor_method == 'lexrank':
            compression_config['compression_ratio'] = config.get('compression_ratio', 0.5)
            compression_config['threshold'] = config.get('threshold', 0.1)
            compression_config['damping'] = config.get('damping', 0.85)
            compression_config['max_iterations'] = config.get('max_iterations', 30)
            
            print(f"[Trial] LexRank config: compression_ratio={compression_config['compression_ratio']}, "
                f"threshold={compression_config['threshold']}, damping={compression_config['damping']}, "
                f"max_iterations={compression_config['max_iterations']}")
        
        elif compressor_method == 'spacy':
            compression_config['compression_ratio'] = config.get('compression_ratio', 0.5)
            compression_config['spacy_model'] = config.get('spacy_model', 'en_core_web_sm')
            
            print(f"[Trial] SpaCy config: compression_ratio={compression_config['compression_ratio']}, "
                f"model={compression_config['spacy_model']}")
        
        elif compressor_method in ['tree_summarize', 'refine']:
            gen_type = config.get('compressor_generator_module_type')
            if gen_type:
                compression_config['generator_module_type'] = gen_type
                
                if gen_type == 'sap_api':
                    compression_config['llm'] = config.get('compressor_llm', 'mistralai')
                    compression_config['model'] = config.get('compressor_model', 'mistralai-large-instruct')
                    compression_config['api_url'] = self.config_generator.resolve_env_vars(
                        config.get('compressor_api_url', '${SAP_API_URL}')
                    )
                    compression_config['bearer_token'] = self.config_generator.resolve_env_vars(
                        config.get('compressor_bearer_token', '${SAP_BEARER_TOKEN}')
                    )
                elif gen_type == 'vllm':
                    compression_config['llm'] = config.get('compressor_model') 
                    if 'compressor_tensor_parallel_size' in config:
                        compression_config['tensor_parallel_size'] = config['compressor_tensor_parallel_size']
                    if 'compressor_gpu_memory_utilization' in config:
                        compression_config['gpu_memory_utilization'] = config['compressor_gpu_memory_utilization']
                elif gen_type == 'openai':
                    compression_config['model'] = config.get('compressor_model', 'gpt-4o-mini')
                else: 
                    compression_config['llm'] = config.get('compressor_llm', 'openai')
                    compression_config['model'] = config.get('compressor_model', 'gpt-4o-mini')
            else:
                compression_config['llm'] = config.get('compressor_llm', 'openai')
                compression_config['model'] = config.get('compressor_model', 'gpt-4o-mini')

                if compression_config['llm'] == 'mistralai' and 'compressor_api_url' in config:
                    compression_config['generator_module_type'] = 'sap_api'
                    compression_config['api_url'] = self.config_generator.resolve_env_vars(config['compressor_api_url'])
                    compression_config['bearer_token'] = self.config_generator.resolve_env_vars(
                        config.get('compressor_bearer_token', '${SAP_BEARER_TOKEN}')
                    )
        
        compressed_df = compressor_module.apply_compression(working_df, compression_config)
        
        return compressed_df

    
    def execute_prompt_maker(self, config: Dict[str, Any], trial_dir: str, 
                           working_df: pd.DataFrame) -> pd.DataFrame:
        
        prompts_df = None
        
        has_prompt_maker = self.config_generator.node_exists("prompt_maker")
        if not has_prompt_maker:
            print("[Trial] Prompt maker not defined in config.yaml - skipping")
            return prompts_df
        
        if 'prompt_maker_method' not in config:
            print("[Trial] No prompt making method specified - skipping")
            return prompts_df
            
        method_type = config.get('prompt_maker_method', 'fstring')
        template_idx = config.get('prompt_template_idx', 0)
        
        print(f"[Trial] Creating prompts with method: {method_type}, template index: {template_idx}")
        prompt_maker = PromptMakerModule(project_dir=trial_dir, config_manager=self.config_generator)
        
        if method_type == 'window_replacement':
            if 'retrieved_ids' not in working_df.columns:
                print(f"[WARNING] Window replacement requires retrieved_ids column, falling back to fstring")
                method_type = 'fstring'
                template_idx = 0
                if template_idx >= len(NodeDefaults.PROMPT_TEMPLATES.get('fstring', [])):
                    template_idx = 0
            else:
                print(f"[Trial] Using window replacement with corpus metadata")
        
        prompts_df = prompt_maker.create_prompts_from_dataframe(
            working_df, method_type=method_type, template_idx=template_idx
        )
        
        prompts_file = os.path.join(trial_dir, "prompts.json")
        try:
            with open(prompts_file, 'w') as f:
                json.dump({"prompts": prompts_df['prompts'].tolist()}, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save prompts - {e}")
            
        return prompts_df
    
    def execute_generator(self, config: Dict[str, Any], trial_dir: str, 
            prompts_df: pd.DataFrame, working_df: pd.DataFrame, 
            qa_subset: pd.DataFrame) -> pd.DataFrame:
        has_generator = self.config_generator.node_exists("generator")
        if not has_generator:
            print("[Trial] Generator not defined in config.yaml - skipping")
            return None

        if 'generator_config' in config:
            gen_config_str = config['generator_config']
            module_type, model = gen_config_str.split('::', 1)
            
            config['generator_module_type'] = module_type
            config['generator_model'] = model

            unified_params = self.config_generator.extract_unified_parameters('generator')
            for module_config in unified_params.get('module_configs', []):
                if module_config['module_type'] == module_type and model in module_config['models']:
                    if module_type == 'sap_api':
                        config['generator_api_url'] = module_config.get('api_url')
                        if not config['generator_api_url']:
                            raise ValueError("SAP API URL not found in generator configuration")
                        config['generator_llm'] = module_config.get('llm', 'mistralai')
                    elif module_type == 'vllm':
                        config['generator_llm'] = model  
                    else:
                        config['generator_llm'] = module_config.get('llm', 'openai')
                    break
        
        if ('generator_model' not in config and 'generator_llm' not in config) or prompts_df is None:
            print("[Trial] No generation model specified or no prompts available - skipping")
            return None
        
        generator_model = config.get('generator_model', config.get('generator_llm'))
        if not generator_model:
            print("[Trial] No generator model specified - skipping")
            return None
            
        print(f"[Trial] Generating answers with model: {generator_model}")
        
        temperature = round(float(config.get('generator_temperature', 0.7)), 4)
                
        generator_kwargs = {
            'model': generator_model,
            'batch_size': config.get('batch', 8)
        }
        
        module_type = config.get('generator_module_type')
        if not module_type:
            original_config = self.config_generator._get_original_generator_config()
            generator_config = Utils.find_generator_config(
                self.config_generator, 
                "generator", 
                generator_model
            )
            module_type = Utils.detect_module_type(config, generator_config, generator_model)
        
        print(f"[Trial] Using generator module type: {module_type}")
        
        if module_type == 'sap_api':
            generator_kwargs['provider'] = 'sap_api'
            api_url = config.get('generator_api_url')
            if not api_url:
                gen_node = self.config_generator.extract_node_config('generator')
                if gen_node and 'modules' in gen_node:
                    for module in gen_node['modules']:
                        if (module.get('module_type') == 'sap_api' and 
                            generator_model in Utils.ensure_list(module.get('model', []))):
                            api_url = module.get('api_url')
                            break
            
            if not api_url:
                raise ValueError(f"SAP API URL not found for generator model {generator_model}")
            
            generator_kwargs['api_url'] = api_url
            generator_kwargs['llm'] = config.get('generator_llm', 'mistralai')
            if 'generator_max_tokens' in config:
                generator_kwargs['max_tokens'] = config['generator_max_tokens']
        elif module_type == 'vllm':
            generator_kwargs['provider'] = 'vllm'
        elif module_type == 'openai':
            generator_kwargs['provider'] = 'openai'
        elif module_type == 'llama_index' or module_type == 'llama_index_llm':
            generator_kwargs['provider'] = 'llama_index'
            generator_kwargs['llm'] = config.get('generator_llm', 'openai')
        
        try:
            generator = create_generator(**generator_kwargs)
        except Exception as e:
            print(f"[ERROR] Failed to create generator with module_type={module_type}: {e}")
            import traceback
            traceback.print_exc()
            generator = create_generator(model=generator_model)
        
        generated_texts = generator.generate(
            prompts=prompts_df['prompts'].tolist(),
            temperature=temperature,
            max_tokens=config.get('generator_max_tokens', 500)
        )
        
        eval_df = pd.DataFrame()
        eval_df['query'] = qa_subset['query'].values
        eval_df['generated_texts'] = generated_texts
        
        if 'generation_gt' in qa_subset.columns:
            eval_df['generation_gt'] = qa_subset['generation_gt'].values
        elif 'ground_truth' in qa_subset.columns:
            eval_df['generation_gt'] = qa_subset['ground_truth'].values
            
        eval_df['retrieved_contents'] = working_df['retrieved_contents'].values
        
        if 'prompts' in prompts_df.columns:
            eval_df['prompts'] = prompts_df['prompts'].values
        
        eval_file = os.path.join(trial_dir, "evaluation_data.json")
        eval_data = {
            "queries": eval_df['query'].tolist(),
            "generated_texts": eval_df['generated_texts'].tolist(),
            "prompts": eval_df['prompts'].tolist() if 'prompts' in eval_df.columns else [],
            "retrieved_contents": eval_df['retrieved_contents'].tolist(),
            "ground_truths": eval_df['generation_gt'].tolist() if 'generation_gt' in eval_df.columns else []
        }
        Utils.save_json(eval_file, eval_data)
        
        return eval_df