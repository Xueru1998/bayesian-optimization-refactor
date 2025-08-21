import pandas as pd
from typing import Dict, Any
from pipeline.utils import Utils


class LocalOptimizationHandler:
    def __init__(self, executor, evaluator, intermediate_handler, component_runners,
                 retrieval_weight: float = 0.5, generation_weight: float = 0.5):
        self.executor = executor
        self.evaluator = evaluator
        self.intermediate_handler = intermediate_handler
        self.component_runners = component_runners
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.component_results = {}
    
    def handle_local_optimization_component(self, current_component: str, config: Dict[str, Any], 
                                            trial_dir: str, working_df: pd.DataFrame, 
                                            qa_subset: pd.DataFrame, save_intermediate: bool) -> Dict[str, Any]:
        print(f"\n[Local Optimization] Current component: {current_component}")
        print(f"[Local Optimization] Using saved outputs from previous components")
        
        last_retrieval_component = "retrieval"
        last_retrieval_score = 0.0
        
        if hasattr(self, 'component_results'):
            for comp in ['passage_compressor', 'passage_filter', 'passage_reranker', 'retrieval', 'query_expansion']:
                if comp in self.component_results:
                    last_retrieval_component = comp
                    last_retrieval_score = self.component_results[comp].get('best_score', 0.0)
                    print(f"[Local Optimization] Using score from {comp}: {last_retrieval_score}")
                    break
        
        component_handlers = {
            'query_expansion': self._handle_query_expansion_local,
            'retrieval': self._handle_retrieval_local,
            'passage_reranker': self._handle_reranker_local,
            'passage_filter': self._handle_filter_local,
            'passage_compressor': self._handle_compressor_local,
            'prompt_maker_generator': self._handle_prompt_generator_local
        }
        
        handler = component_handlers.get(current_component)
        if handler:
            return handler(config, trial_dir, working_df, qa_subset, 
                         last_retrieval_component, last_retrieval_score, save_intermediate)
        
        return {
            'score': 0.0,
            'combined_score': 0.0,
            'error': f'Unknown component: {current_component}'
        }
    
    def _handle_query_expansion_local(self, config: Dict[str, Any], trial_dir: str, 
                                     working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                                     last_retrieval_component: str, last_retrieval_score: float,
                                     save_intermediate: bool) -> Dict[str, Any]:
        working_df, query_expansion_results, retrieval_df_from_qe, retrieval_done_in_qe = self.component_runners.run_query_expansion_with_retrieval(
            config, trial_dir, working_df, is_local_optimization=True
        )
        
        if retrieval_done_in_qe and retrieval_df_from_qe is not None:
            working_df = retrieval_df_from_qe
            last_retrieval_score = query_expansion_results.get('mean_score', 0.0)
            last_retrieval_component = "query_expansion"
        
        if save_intermediate:
            self.intermediate_handler.save_intermediate_result('query_expansion', working_df, query_expansion_results, trial_dir, config)
        
        return {
            'retrieval_score': 0.0,
            'last_retrieval_component': last_retrieval_component,
            'last_retrieval_score': last_retrieval_score,
            'combined_score': last_retrieval_score,
            'score': last_retrieval_score,
            'query_expansion_score': query_expansion_results.get('mean_score', 0.0),
            'query_expansion_metrics': Utils.json_serializable(query_expansion_results),
            'retrieval_skipped': retrieval_done_in_qe,
            'evaluation_method': 'local_optimization',
            'working_df': working_df
        }
    
    def _handle_retrieval_local(self, config: Dict[str, Any], trial_dir: str, 
                               working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                               last_retrieval_component: str, last_retrieval_score: float,
                               save_intermediate: bool) -> Dict[str, Any]:
        working_df, retrieval_results, last_retrieval_score, last_retrieval_results = self.component_runners.run_retrieval(
            config, trial_dir, working_df, qa_subset, is_local_optimization=True
        )
        
        if save_intermediate:
            self.intermediate_handler.save_intermediate_result('retrieval', working_df, retrieval_results, trial_dir, config)
        
        return {
            'retrieval_score': retrieval_results.get('mean_accuracy', 0.0),
            'last_retrieval_component': 'retrieval',
            'last_retrieval_score': last_retrieval_score,
            'combined_score': last_retrieval_score,
            'score': last_retrieval_score,
            'retrieval_metrics': Utils.json_serializable(retrieval_results),
            'evaluation_method': 'local_optimization',
            'working_df': working_df
        }
    
    def _handle_reranker_local(self, config: Dict[str, Any], trial_dir: str, 
                              working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                              last_retrieval_component: str, last_retrieval_score: float,
                              save_intermediate: bool) -> Dict[str, Any]:
        reranked_df = self.executor.execute_reranker(config, trial_dir, working_df)
        reranker_results = self.evaluator.evaluate_reranker(reranked_df, qa_subset)
        reranker_score = reranker_results.get('mean_accuracy', 0.0)
        
        if save_intermediate:
            self.intermediate_handler.save_intermediate_result('passage_reranker', reranked_df, reranker_results, trial_dir, config)
        
        return {
            'retrieval_score': 0.0,
            'last_retrieval_component': 'passage_reranker',
            'last_retrieval_score': reranker_score,
            'combined_score': reranker_score,
            'score': reranker_score,
            'reranker_score': reranker_score,
            'reranker_metrics': Utils.json_serializable(reranker_results),
            'evaluation_method': 'local_optimization',
            'working_df': reranked_df
        }
    
    def _handle_filter_local(self, config: Dict[str, Any], trial_dir: str, 
                            working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                            last_retrieval_component: str, last_retrieval_score: float,
                            save_intermediate: bool) -> Dict[str, Any]:
        filtered_df = self.executor.execute_filter(config, trial_dir, working_df)
        
        filter_method = config.get('passage_filter_method')
        is_pass_filter = (filter_method == 'pass_passage_filter' or filter_method == 'pass')
        
        if is_pass_filter:
            if hasattr(self, 'component_results') and 'passage_reranker' in self.component_results:
                filter_score = self.component_results['passage_reranker'].get('best_score', 0.0)
            else:
                filter_score = 0.0
            filter_results = {'mean_accuracy': filter_score}
        else:
            filter_results = self.evaluator.evaluate_filter(filtered_df, qa_subset)
            filter_score = filter_results.get('mean_accuracy', 0.0)
        
        if save_intermediate:
            self.intermediate_handler.save_intermediate_result('passage_filter', filtered_df, filter_results, trial_dir, config)
        
        return {
            'retrieval_score': 0.0,
            'last_retrieval_component': 'passage_filter',
            'last_retrieval_score': filter_score,
            'combined_score': filter_score,
            'score': filter_score,
            'filter_score': filter_score,
            'filter_metrics': Utils.json_serializable(filter_results),
            'evaluation_method': 'local_optimization',
            'working_df': filtered_df
        }
    
    def _handle_compressor_local(self, config: Dict[str, Any], trial_dir: str, 
                                working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                                last_retrieval_component: str, last_retrieval_score: float,
                                save_intermediate: bool) -> Dict[str, Any]:
        compressed_df = self.executor.execute_compressor(config, trial_dir, working_df)
        token_eval_results = self.evaluator.evaluate_compressor(compressed_df, qa_subset)
        compressor_score = token_eval_results.get('mean_score', 0.0) if token_eval_results else 0.0
        llm_mean_score = token_eval_results.get('llm_mean_score', 0.0) if token_eval_results else 0.0
        
        final_score = max(compressor_score, llm_mean_score)
        
        if save_intermediate:
            self.intermediate_handler.save_intermediate_result('passage_compressor', compressed_df, token_eval_results, trial_dir, config)
        
        return {
            'retrieval_score': 0.0,
            'last_retrieval_component': 'passage_compressor',
            'last_retrieval_score': final_score,
            'combined_score': final_score,
            'score': final_score,
            'compression_score': final_score,
            'compressor_score': final_score,
            'llm_mean_score': llm_mean_score,
            'compression_metrics': Utils.json_serializable(token_eval_results),
            'evaluation_method': 'local_optimization',
            'working_df': compressed_df
        }
    
    def _handle_prompt_generator_local(self, config: Dict[str, Any], trial_dir: str, 
                                      working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                                      last_retrieval_component: str, last_retrieval_score: float,
                                      save_intermediate: bool) -> Dict[str, Any]:
        if hasattr(self, 'component_results'):
            for comp in ['passage_compressor', 'passage_filter', 'passage_reranker', 'retrieval', 'query_expansion']:
                if comp in self.component_results:
                    last_retrieval_component = comp
                    last_retrieval_score = self.component_results[comp].get('best_score', 0.0)
                    print(f"[Prompt/Generator] Using score from {comp}: {last_retrieval_score}")
                    break

        prompts_df = self.executor.execute_prompt_maker(config, trial_dir, working_df)
        if save_intermediate:
            self.intermediate_handler.save_intermediate_result('prompt_maker', prompts_df, {}, trial_dir, config)

        eval_df = self.executor.execute_generator(config, trial_dir, prompts_df, working_df, qa_subset)
        
        if eval_df is not None and 'generated_texts' in eval_df.columns:
            working_df['generated_texts'] = eval_df['generated_texts']
            working_df['prompts'] = eval_df['prompts'] if 'prompts' in eval_df.columns else prompts_df['prompts']
        
        generation_results = self.evaluator.evaluate_generation(eval_df, qa_subset) if eval_df is not None else {}
        generation_score = generation_results.get('mean_score', 0.0)
        
        combined_score = (self.retrieval_weight * last_retrieval_score + 
                         self.generation_weight * generation_score)
        
        if save_intermediate:
            self.intermediate_handler.save_intermediate_result('generator', eval_df if eval_df is not None else working_df, 
                                        {'generation_score': generation_score, 'generation_metrics': generation_results}, 
                                        trial_dir, config)
        
        return {
            'retrieval_score': 0.0,
            'last_retrieval_component': last_retrieval_component,
            'last_retrieval_score': last_retrieval_score,
            'generation_score': generation_score,
            'generation_metrics': generation_results,
            'combined_score': combined_score,
            'score': combined_score,
            'evaluation_method': 'local_optimization',
            'working_df': working_df,
            'generated_texts': eval_df['generated_texts'].tolist() if eval_df is not None and 'generated_texts' in eval_df.columns else []
        }