import os
import time
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from autorag.nodes.retrieval.bm25 import get_bm25_pkl_name, bm25_ingest
from pipeline_component.query_expansion import QueryExpansionModule
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
    
    def _parse_query_expansion_config(self, qe_config_str: str) -> Dict[str, Any]:
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
    
    def _parse_retrieval_config(self, retrieval_config_str: str) -> Dict[str, Any]:
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
    
    def _parse_reranker_config(self, reranker_config_str: str) -> Dict[str, Any]:
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
    
    def _parse_filter_config(self, filter_config_str: str) -> Dict[str, Any]:
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
    
    def _parse_compressor_config(self, compressor_config_str: str) -> Dict[str, Any]:
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
    
    def _parse_prompt_config(self, prompt_config_str: str) -> Dict[str, Any]:
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
        
    def _prepare_bm25_index(self, config: Dict[str, Any], trial_dir: str):        
        bm25_tokenizer = config['bm25_tokenizer']
        centralized_bm25_file = os.path.join(
            Utils.get_centralized_project_dir(), 
            "resources", 
            get_bm25_pkl_name(bm25_tokenizer)
        )
        
        if not os.path.exists(centralized_bm25_file):
            print(f"Generating BM25 index directly to {bm25_tokenizer}")
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
        
        query_expansion_method = config.get('query_expansion_method')
        
        if not (query_expansion_method and query_expansion_method != 'pass_query_expansion'):
            print("[Trial] No query expansion method specified or pass_query_expansion - skipping")
            return working_df, None
            
        print(f"[Trial] Running query expansion with method: {query_expansion_method}")
        
        try:                 
            query_expansion_module = QueryExpansionModule(project_dir=trial_dir)

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
                            expansion_config['generator_module_type'] = module.get('generator_module_type', 'vllm')
                            break
                else:
                    expansion_config['generator_module_type'] = NodeDefaults.QUERY_EXPANSION['params']['generator_module_type']

            if 'query_expansion_llm' in config:
                expansion_config['llm'] = config['query_expansion_llm']
            else:
                expansion_config['llm'] = NodeDefaults.QUERY_EXPANSION['params']['llm']

            if 'query_expansion_model' in config:
                expansion_config['model'] = config['query_expansion_model']
                if 'llm' not in expansion_config and expansion_config.get('generator_module_type') == 'vllm':
                    expansion_config['llm'] = config['query_expansion_model']
            else:
                expansion_config['model'] = NodeDefaults.QUERY_EXPANSION['params']['model']

            if query_expansion_method == 'hyde':
                expansion_config['max_tokens'] = config.get('query_expansion_max_token', NodeDefaults.QUERY_EXPANSION['params']['max_token'])
            elif query_expansion_method == 'multi_query_expansion':
                expansion_config['temperature'] = config.get('query_expansion_temperature', 0.7)
            
            print(f"[DEBUG] Query expansion config: {expansion_config}")
            
            if 'query' not in working_df.columns:
                raise ValueError("Query column not found in working_df")
                
            expanded_df, expanded_queries = query_expansion_module.perform_query_expansion(
                working_df, expansion_config
            )
            
            if 'queries' not in expanded_df.columns:
                print(f"Warning: Query expansion did not create 'queries' column.")
                return working_df, None
            
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
        
        temp_retrieval_module = RetrievalModule(
            base_project_dir=trial_dir,
            use_pregenerated_embeddings=True,
            centralized_project_dir=Utils.get_centralized_project_dir()
        )
        
        query_expansion_retrieval_config = self.config_generator.get_query_expansion_retrieval_config(config)
        
        retrieval_method = config.get('query_expansion_retrieval_method')
        if not retrieval_method:
            retrieval_method = config.get('retrieval_method')
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
                vectordb_name = config.get('vectordb_name')
            if not vectordb_name:
                qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
                if qe_retrieval_options.get('vectordb_names'):
                    vectordb_name = qe_retrieval_options['vectordb_names'][0]
                else:
                    vectordb_name = query_expansion_retrieval_config.get('vectordb_name', 'default')
            print(f"[Trial] Query expansion using vectordb: {vectordb_name}")
            
        elif retrieval_method == 'bm25':
            bm25_tokenizer = config.get('query_expansion_bm25_tokenizer')
            if not bm25_tokenizer:
                bm25_tokenizer = config.get('bm25_tokenizer')
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
            
        print(f"\n[Trial] Running retrieval with method: {config.get('retrieval_method')}, retriever_top_k: {config.get('retriever_top_k', config.get('top_k', 5))}")
        
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
            rerank_cfg[method_models_key] = config[method_models_key]

        if config.get('reranker_model_name'):
            if method == 'flashrank_reranker':
                rerank_cfg['model'] = config['reranker_model_name'] 
                print(f"[Trial] Using flashrank model: {config['reranker_model_name']}")
            else:
                rerank_cfg['model_name'] = config['reranker_model_name'] 
                print(f"[Trial] Using model: {config['reranker_model_name']}")
        elif config.get('reranker_model'):
            rerank_cfg['model'] = config['reranker_model']
            print(f"[Trial] Using model: {config['reranker_model']}")

        if config.get('cache_dir'):
            rerank_cfg['cache_dir'] = config['cache_dir']
        if config.get('max_length'):
            rerank_cfg['max_length'] = config['max_length']
        
        print(f"[DEBUG] Final reranker_config: {rerank_cfg}")
        
        try:
            reranked_df = reranker_module.apply_reranking(working_df, rerank_cfg)
            print(f"Applied {method} reranking with top_k={rerank_cfg['top_k']}")
            return reranked_df
        except Exception as e:
            print(f"[ERROR] Reranking failed: {e}")
            import traceback
            traceback.print_exc()
            print("[WARNING] Returning original results due to reranking error")
            return working_df
    
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
        
        compressor_method = config.get('passage_compressor_method')
        
        if not compressor_method or compressor_method == 'pass_compressor':
            print("[Trial] Pass-through compressor - returning original dataframe without evaluation")
            return working_df
            
        print(f"[Trial] Running passage compressor with method: {compressor_method}")
        compressor_module = PassageCompressorModule(project_dir=trial_dir)
        
        compression_config = {
            'module_type': compressor_method,
        }

        if compressor_method in ['tree_summarize', 'refine']:
            compression_config.update({
                'llm': config.get('compressor_llm', 'openai'),
                'model': config.get('compressor_model', 'gpt-4o-mini'),
                'batch': config.get('compressor_batch', 16),
                'generator_module_type': 'llama_index' 
            })

        elif compressor_method == 'lexrank':
            compression_config.update({
                'compression_ratio': config.get('compressor_compression_ratio', 0.5),
                'threshold': config.get('compressor_threshold', 0.1),
                'damping': config.get('compressor_damping', 0.85),
                'max_iterations': config.get('compressor_max_iterations', 30)
            })

        elif compressor_method == 'spacy':
            compression_config.update({
                'compression_ratio': config.get('compressor_compression_ratio', 0.5),
                'spacy_model': config.get('compressor_spacy_model', 'en_core_web_sm')
            })
        
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

        if ('generator_model' not in config and 'generator_llm' not in config) or prompts_df is None:
            print("[Trial] No generation model specified or no prompts available - skipping")
            return None

        generator_model = config.get('generator_model', config.get('generator_llm'))
        if not generator_model:
            print("[Trial] No generator model specified - skipping")
            return None
            
        print(f"[Trial] Generating answers with model: {generator_model}")
        
        temperature = round(float(config.get('generator_temperature', 0.7)), 4)

        original_config = self.config_generator._get_original_generator_config()
        generator_config = Utils.find_generator_config(
            self.config_generator, 
            "generator", 
            generator_model
        )
        
        module_type = Utils.detect_module_type(config, generator_config, generator_model)
        
        print(f"[Trial] Using generator module type: {module_type}")

        if not generator_config:
            generator_config = original_config
        
        try:
            generator = Utils.create_generator_from_config(generator_model, generator_config, module_type)
                
        except Exception as e:
            print(f"[ERROR] Failed to create generator with module_type={module_type}: {e}")
            import traceback
            traceback.print_exc()
            generator = create_generator(model=generator_model)
        
        generated_df = generator.generate_from_dataframe(
            df=prompts_df,
            prompt_column='prompts',
            output_column='generated_texts',
            temperature=float(temperature)
        )
        
        eval_df = pd.DataFrame()
        eval_df['query'] = qa_subset['query'].values
        eval_df['generated_texts'] = generated_df['generated_texts'].values
        
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