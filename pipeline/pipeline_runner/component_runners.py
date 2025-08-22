from typing import Dict, Any, Tuple, List
import pandas as pd
from pipeline.utils import Utils


class ComponentRunners:
    def __init__(self, executor, evaluator, query_expansion_metrics: List,
                 early_stopping_handler=None):
        self.executor = executor
        self.evaluator = evaluator
        self.query_expansion_metrics = query_expansion_metrics
        self.early_stopping_handler = early_stopping_handler
    
    def run_query_expansion_with_retrieval(self, config: Dict[str, Any], trial_dir: str, 
           qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, bool]:
        query_expansion_results = {}
        retrieval_done_in_qe = False
        retrieval_df = None
        
        working_df, expanded_queries = self.executor.execute_query_expansion(config, trial_dir, qa_subset)
        
        qe_method = config.get('query_expansion_method')
        is_pass_qe = (qe_method == 'pass_query_expansion' or expanded_queries is None)
        
        if is_pass_qe:
            print("[Trial] Query expansion is pass, running retrieval directly")
            retrieval_df = self.executor.execute_retrieval(config, trial_dir, qa_subset)
            if retrieval_df is not None:
                retrieval_results = self.evaluator.evaluate_retrieval(retrieval_df, qa_subset)
                retrieval_score = retrieval_results.get('mean_accuracy', 0.0)
                
                if self.early_stopping_handler:
                    self.early_stopping_handler.check_early_stopping('retrieval', retrieval_score, is_local_optimization)
                
                query_expansion_results = {
                    'mean_score': retrieval_score,
                    'metrics': retrieval_results
                }
                working_df = retrieval_df
                retrieval_done_in_qe = True
            return working_df, query_expansion_results, retrieval_df, retrieval_done_in_qe
        
        if self.query_expansion_metrics and 'queries' in working_df.columns:
            query_expansion_results, retrieval_df = self.evaluate_query_expansion_with_retrieval(
                working_df, qa_subset, trial_dir, config
            )
            retrieval_done_in_qe = True
            
            qe_score = query_expansion_results.get('mean_score', 0.0)
            if self.early_stopping_handler:
                self.early_stopping_handler.check_early_stopping('query_expansion', qe_score, is_local_optimization)

            if retrieval_df is not None and isinstance(retrieval_df, pd.DataFrame):
                Utils.update_dataframe_columns(working_df, retrieval_df, 
                    include_cols=['retrieved_ids', 'retrieved_contents', 'retrieve_scores'])

                query_expansion_results['retrieval_df'] = retrieval_df
        
        return working_df, query_expansion_results, retrieval_df, retrieval_done_in_qe
    
    def evaluate_query_expansion_with_retrieval(self, working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                    trial_dir: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        retrieval_df = self.executor.execute_query_expansion_retrieval(config, trial_dir, working_df)
        query_expansion_results = self.evaluator.evaluate_query_expansion(retrieval_df, qa_subset)
        return query_expansion_results, retrieval_df
    
    def run_retrieval(self, config: Dict[str, Any], trial_dir: str, 
      working_df: pd.DataFrame, qa_subset: pd.DataFrame, 
      is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
    
        retrieval_df = self.executor.execute_retrieval(config, trial_dir, working_df)
        retrieval_results = self.evaluator.evaluate_retrieval(retrieval_df, qa_subset)

        if retrieval_df is not None and isinstance(retrieval_df, pd.DataFrame):
            Utils.update_dataframe_columns(working_df, retrieval_df,
                include_cols=['retrieved_ids', 'retrieved_contents', 'retrieve_scores', 'queries'])
        
        retrieval_score = retrieval_results.get('mean_accuracy', 0.0)
        if self.early_stopping_handler:
            self.early_stopping_handler.check_early_stopping('retrieval', retrieval_score, is_local_optimization)
        
        return working_df, retrieval_results, retrieval_score, retrieval_results
    
    def run_reranker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
            qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
    
        reranker_results = {}
        reranker_applied = False
        last_score = 0.0
        last_results = {}
        
        reranked_df = self.executor.execute_reranker(config, trial_dir, working_df)
        
        reranker_results = self.evaluator.evaluate_reranker(reranked_df, qa_subset)
        last_score = reranker_results.get('mean_accuracy', 0.0)
        last_results = reranker_results
        
        if reranked_df is not working_df:
            reranker_applied = True
            print(f"Applied reranking, new mean accuracy: {last_score}")
            if self.early_stopping_handler:
                self.early_stopping_handler.check_early_stopping('reranker', last_score, is_local_optimization)
        else:
            print(f"Pass-through reranking, mean accuracy: {last_score}")
        
        return reranked_df, reranker_results, reranker_applied, last_score, last_results
    
    def run_filter(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
    qa_subset: pd.DataFrame, is_local_optimization: bool = False, 
    component_results: Dict = None) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
        filter_results = {}
        filter_applied = False
        last_score = 0.0
        last_results = {}

        filtered_df = self.executor.execute_filter(config, trial_dir, working_df)

        filter_method = config.get('passage_filter_method')
        is_pass_filter = (filter_method == 'pass_passage_filter' or 
                        filter_method == 'pass' or 
                        filtered_df is working_df)
        
        if is_pass_filter:
            print(f"[Filter] Pass-through filter detected, using previous results")

            if component_results and 'passage_reranker' in component_results:
                last_score = component_results['passage_reranker'].get('best_score', 0.0)
                print(f"[Filter] Using score from passage_reranker: {last_score}")
            else:
                print(f"[Filter] Warning: No reranker results found")
                last_score = 0.0
            
            filter_results = {'mean_accuracy': last_score}
            last_results = filter_results
            
            return working_df, filter_results, filter_applied, last_score, last_results

        filter_applied = True
        
        if 'retrieved_ids' in filtered_df.columns:
            eval_df = filtered_df.copy()
            eval_df['retrieved_ids'] = eval_df['retrieved_ids'].apply(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x
            )
            filter_results = self.evaluator.evaluate_filter(eval_df, qa_subset)
        else:
            filter_results = self.evaluator.evaluate_filter(filtered_df, qa_subset)
        
        last_score = filter_results.get('mean_accuracy', 0.0)
        last_results = filter_results
        
        print(f"[Filter] Applied {filter_method}, new mean accuracy: {last_score}")
        if self.early_stopping_handler:
            self.early_stopping_handler.check_early_stopping('filter', last_score, is_local_optimization)
        
        return filtered_df, filter_results, filter_applied, last_score, last_results
    
    def run_compressor(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
               qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
    
        compressed_df = self.executor.execute_compressor(config, trial_dir, working_df)
        
        token_eval_results = self.evaluator.evaluate_compressor(compressed_df, qa_subset)
        last_score = token_eval_results.get('mean_score', 0.0) if token_eval_results else 0.0
        
        if compressed_df is not working_df:
            print(f"Applied compression, mean score: {last_score}")
            if self.early_stopping_handler:
                self.early_stopping_handler.check_early_stopping('compressor', last_score, is_local_optimization)
        else:
            print(f"Pass-through compression, mean score: {last_score}")
        
        return compressed_df, token_eval_results, last_score, token_eval_results

    def run_prompt_maker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                     qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        prompts_df = self.executor.execute_prompt_maker(config, trial_dir, working_df)
        prompt_results = {}
        print("[Trial] Skipping prompt maker evaluation for optimization runs")
        return prompts_df, prompt_results
    
    def run_generator(self, config: Dict[str, Any], trial_dir: str, prompts_df: pd.DataFrame,
                  working_df: pd.DataFrame, qa_subset: pd.DataFrame) -> pd.DataFrame:
        
        eval_df = self.executor.execute_generator(config, trial_dir, prompts_df, working_df, qa_subset)
        return eval_df