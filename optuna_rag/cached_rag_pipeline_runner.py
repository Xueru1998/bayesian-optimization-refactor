from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline_runner import RAGPipelineRunner
from optuna_rag.cache_manager import ComponentCacheManager


class CachedRAGPipelineRunner(RAGPipelineRunner):
    def __init__(self, *args, cache_manager: Optional[ComponentCacheManager] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_manager = cache_manager
        
    def _run_query_expansion_with_retrieval(self, config: Dict[str, Any], trial_dir: str, 
                       qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, bool]:
        if not self.cache_manager:
            return super()._run_query_expansion_with_retrieval(config, trial_dir, qa_subset)

        qe_config_str = config.get('query_expansion_config')
        if qe_config_str:
            parsed_config = self._parse_query_expansion_config(qe_config_str)
            config.update(parsed_config)
        
        query_expansion_method = config.get('query_expansion_method')

        if not query_expansion_method or query_expansion_method == 'pass_query_expansion':
            return qa_subset.copy(), {}, None, False

        cache_config = config.copy()
        
        cached_df, cached_metrics = self.cache_manager.check_cache('query_expansion', cache_config)
        if cached_df is not None and cached_metrics is not None:
            print("[Cached] Using cached query expansion results")

            if 'retriever_top_k' in cache_config:
                print(f"[Cached] Query expansion was performed with top_k: {cache_config['retriever_top_k']}")
            
            working_df = qa_subset.copy()
            if 'queries' in cached_df.columns:
                working_df['queries'] = cached_df['queries'].values
            
            retrieval_df = None
            retrieval_done = False
            
            if 'retrieved_ids' in cached_df.columns and 'retrieved_contents' in cached_df.columns:
                retrieval_df = cached_df
                retrieval_done = True
                print("[Cached] Query expansion includes retrieval results")
            
            return working_df, cached_metrics.get('query_expansion_results', {}), retrieval_df, retrieval_done
        
        working_df, query_expansion_results, retrieval_df, retrieval_done = super()._run_query_expansion_with_retrieval(
            config, trial_dir, qa_subset
        )

        cache_df = working_df.copy()
        if retrieval_df is not None:
            cache_df = retrieval_df

        self.cache_manager.save_to_cache(
            'query_expansion', 
            cache_config,  
            cache_df, 
            {'query_expansion_results': query_expansion_results}
        )
        
        return working_df, query_expansion_results, retrieval_df, retrieval_done
    
    def _run_retrieval(self, config: Dict[str, Any], trial_dir: str, 
                      working_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
        if not self.cache_manager:
            return super()._run_retrieval(config, trial_dir, working_df, qa_subset)

        retrieval_config_str = config.get('retrieval_config')
        if retrieval_config_str:
            parsed_config = self._parse_retrieval_config(retrieval_config_str)
            config.update(parsed_config)

        qe_method = config.get('query_expansion_method')
        if qe_method and qe_method != 'pass_query_expansion':
            cache_key_with_qe = config.copy()
            cache_key_with_qe['retrieval_skipped_due_to_qe'] = True
            cached_df, cached_metrics = self.cache_manager.check_cache('retrieval', cache_key_with_qe)
            if cached_df is not None and cached_metrics is not None:
                print("[Cached] Retrieval was skipped due to query expansion")
                return (
                    working_df,
                    {},
                    0.0,
                    {}
                )

        cached_df, cached_metrics = self.cache_manager.check_cache('retrieval', config)
        if cached_df is not None and cached_metrics is not None:
            print("[Cached] Using cached retrieval results")
            if 'queries' in working_df.columns:
                cached_df['queries'] = working_df['queries'].values
            return (
                cached_df, 
                cached_metrics.get('retrieval_results', {}),
                cached_metrics.get('retrieval_score', 0.0),
                cached_metrics.get('retrieval_results', {})
            )

        retrieval_df, retrieval_results, score, results = super()._run_retrieval(
            config, trial_dir, working_df, qa_subset
        )

        self.cache_manager.save_to_cache(
            'retrieval', 
            config, 
            retrieval_df, 
            {
                'retrieval_results': retrieval_results,
                'retrieval_score': score
            }
        )
        
        return retrieval_df, retrieval_results, score, results
    
    def _run_reranker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                     ground_truths: List) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
        if not self.cache_manager:
            return super()._run_reranker(config, trial_dir, working_df, ground_truths)

        reranker_config_str = config.get('reranker_config')
        if reranker_config_str:
            parsed_config = self._parse_reranker_config(reranker_config_str)
            config.update(parsed_config)
        
        reranker_method = config.get('passage_reranker_method')

        if not reranker_method or reranker_method == 'pass_reranker':
            return working_df, {}, False, 0.0, {}

        cache_config = config.copy()
        
        cached_df, cached_metrics = self.cache_manager.check_cache('reranker', cache_config)
        if cached_df is not None and cached_metrics is not None:
            print("[Cached] Using cached reranker results")
            if 'reranker_top_k' in cache_config:
                print(f"[Cached] Reranker was performed with top_k: {cache_config['reranker_top_k']}")
            return (
                cached_df,
                cached_metrics.get('reranker_results', {}),
                True,
                cached_metrics.get('reranker_score', 0.0),
                cached_metrics.get('reranker_results', {})
            )

        reranked_df, reranker_results, applied, score, results = super()._run_reranker(
            config, trial_dir, working_df, ground_truths
        )

        if applied:
            self.cache_manager.save_to_cache(
                'reranker', 
                cache_config,
                reranked_df, 
                {
                    'reranker_results': reranker_results,
                    'reranker_score': score
                }
            )
        
        return reranked_df, reranker_results, applied, score, results
    
    def _run_filter(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
               ground_truths: List) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
        if not self.cache_manager:
            return super()._run_filter(config, trial_dir, working_df, ground_truths)

        filter_config_str = config.get('passage_filter_config')
        if filter_config_str:
            parsed_config = self._parse_filter_config(filter_config_str)
            filter_method = parsed_config.get('passage_filter_method')
            config['passage_filter_method'] = filter_method
            if 'threshold' in parsed_config:
                config['threshold'] = parsed_config['threshold']
            if 'percentile' in parsed_config:
                config['percentile'] = parsed_config['percentile']
        else:
            filter_method = config.get('passage_filter_method')

        if not filter_method or filter_method == 'pass_passage_filter':
            return working_df, {}, False, 0.0, {}

        cached_df, cached_metrics = self.cache_manager.check_cache('filter', config)
        if cached_df is not None and cached_metrics is not None:
            print("[Cached] Using cached filter results")
            return (
                cached_df,
                cached_metrics.get('filter_results', {}),
                True,
                cached_metrics.get('filter_score', 0.0),
                cached_metrics.get('filter_results', {})
            )

        filtered_df, filter_results, applied, score, results = super()._run_filter(
            config, trial_dir, working_df, ground_truths
        )

        if applied:
            self.cache_manager.save_to_cache(
                'filter', 
                config, 
                filtered_df, 
                {
                    'filter_results': filter_results,
                    'filter_score': score
                }
            )
        
        return filtered_df, filter_results, applied, score, results
    
    def _run_compressor(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                       qa_subset: pd.DataFrame, ground_truths: List) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
        if not self.cache_manager:
            return super()._run_compressor(config, trial_dir, working_df, qa_subset, ground_truths)

        compressor_config_str = config.get('compressor_config')
        if compressor_config_str:
            parsed_config = self._parse_compressor_config(compressor_config_str)
            config.update(parsed_config)
        
        compressor_method = config.get('passage_compressor_method')

        if not compressor_method or compressor_method == 'pass_compressor':
            return working_df, {}, 0.0, {}

        cached_df, cached_metrics = self.cache_manager.check_cache('compressor', config)
        if cached_df is not None and cached_metrics is not None:
            print("[Cached] Using cached compressor results")
            return (
                cached_df,
                cached_metrics.get('token_eval_results', {}),
                cached_metrics.get('compressor_score', 0.0),
                cached_metrics.get('compression_results', {})
            )

        compressed_df, token_eval_results, score, compression_results = super()._run_compressor(
            config, trial_dir, working_df, qa_subset, ground_truths
        )

        self.cache_manager.save_to_cache(
            'compressor', 
            config, 
            compressed_df, 
            {
                'token_eval_results': token_eval_results,
                'compressor_score': score,
                'compression_results': compression_results
            }
        )
        
        return compressed_df, token_eval_results, score, compression_results
    
    def _run_prompt_maker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                         qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not self.cache_manager:
            return super()._run_prompt_maker(config, trial_dir, working_df, qa_subset)

        prompt_config_str = config.get('prompt_config')
        if prompt_config_str:
            parsed_config = self._parse_prompt_config(prompt_config_str)
            config.update(parsed_config)

        if 'prompt_maker_method' not in config and 'generator_model' in config:
            prompts_df = pd.DataFrame()
            prompts_df['prompts'] = [f"Question: {q}\nContext: {c}\nAnswer:"
                                    for q, c in zip(qa_subset['query'].values, working_df['retrieved_contents'].values)]
            return prompts_df, {}

        prompt_method = config.get('prompt_maker_method')
        if prompt_method == 'pass_prompt_maker':
            return None, {}

        cached_df, cached_metrics = self.cache_manager.check_cache('prompt_maker', config)
        if cached_df is not None and cached_metrics is not None:
            print("[Cached] Using cached prompt maker results")
            return cached_df, cached_metrics.get('prompt_results', {})

        prompts_df, prompt_results = super()._run_prompt_maker(config, trial_dir, working_df, qa_subset)

        if prompts_df is not None:
            self.cache_manager.save_to_cache(
                'prompt_maker', 
                config, 
                prompts_df, 
                {'prompt_results': prompt_results}
            )
        
        return prompts_df, prompt_results
    
    def _run_generator(self, config: Dict[str, Any], trial_dir: str, prompts_df: pd.DataFrame,
                      working_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        if not self.cache_manager:
            return super()._run_generator(config, trial_dir, prompts_df, working_df, qa_subset)

        cached_df, cached_metrics = self.cache_manager.check_cache('generator', config)
        if cached_df is not None and cached_metrics is not None:
            print("[Cached] Using cached generator results")
            return cached_metrics.get('generation_results', {})

        generation_results = super()._run_generator(config, trial_dir, prompts_df, working_df, qa_subset)

        if generation_results:
            dummy_df = pd.DataFrame({'completed': [True]})
            self.cache_manager.save_to_cache(
                'generator', 
                config, 
                dummy_df, 
                {'generation_results': generation_results}
            )
        
        return generation_results