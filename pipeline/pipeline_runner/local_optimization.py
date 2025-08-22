from typing import Dict, Any
import pandas as pd
from pipeline.utils import Utils


class LocalOptimizationHandler:
    def __init__(self, component_runners, evaluator, intermediate_handler=None):
        self.component_runners = component_runners
        self.evaluator = evaluator
        self.intermediate_handler = intermediate_handler
        self.component_results = {}
    
    def set_component_results(self, component_results: Dict):
        self.component_results = component_results
    
    def handle_local_optimization_component(self, current_component: str, config: Dict[str, Any], 
                                            trial_dir: str, working_df: pd.DataFrame, 
                                            qa_subset: pd.DataFrame, save_intermediate: bool) -> Dict[str, Any]:
        print(f"\n[Local Optimization] Current component: {current_component}")
        print(f"[Local Optimization] Will use saved outputs from previous components")
        
        last_retrieval_component = "retrieval"
        last_retrieval_score = 0.0
        
        if self.component_results:
            for comp in ['passage_compressor', 'passage_filter', 'passage_reranker', 'retrieval', 'query_expansion']:
                if comp in self.component_results:
                    last_retrieval_component = comp
                    last_retrieval_score = self.component_results[comp].get('best_score', 0.0)
                    print(f"[Local Optimization] Using score from {comp}: {last_retrieval_score}")
                    break
        
        component_handlers = {
            'query_expansion': self.handle_query_expansion_local,
            'retrieval': self.handle_retrieval_local,
            'passage_reranker': self.handle_reranker_local,
            'passage_filter': self.handle_filter_local,
            'passage_compressor': self.handle_compressor_local,
            'prompt_maker_generator': self.handle_prompt_generator_local
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
    
    def handle_query_expansion_local(self, config: Dict[str, Any], trial_dir: str, 
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
        
        if save_intermediate and self.intermediate_handler:
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
    
    def handle_retrieval_local(self, config: Dict[str, Any], trial_dir: str, 
                               working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                               last_retrieval_component: str, last_retrieval_score: float,
                               save_intermediate: bool) -> Dict[str, Any]:
        working_df, retrieval_results, last_retrieval_score, last_retrieval_results = self.component_runners.run_retrieval(
            config, trial_dir, working_df, qa_subset, is_local_optimization=True
        )
        
        if save_intermediate and self.intermediate_handler:
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
    
    def handle_reranker_local(self, config: Dict[str, Any], trial_dir: str, 
                              working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                              last_retrieval_component: str, last_retrieval_score: float,
                              save_intermediate: bool) -> Dict[str, Any]:
        reranked_df, reranker_results, reranker_applied, reranker_score, last_results = self.component_runners.run_reranker(
            config, trial_dir, working_df, qa_subset, is_local_optimization=True
        )
        
        if save_intermediate and self.intermediate_handler:
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
    
    def handle_filter_local(self, config: Dict[str, Any], trial_dir: str, 
                            working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                            last_retrieval_component: str, last_retrieval_score: float,
                            save_intermediate: bool) -> Dict[str, Any]:
        filtered_df, filter_results, filter_applied, filter_score, last_results = self.component_runners.run_filter(
            config, trial_dir, working_df, qa_subset, is_local_optimization=True, 
            component_results=self.component_results
        )
        
        if save_intermediate and self.intermediate_handler:
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
    
    def handle_compressor_local(self, config: Dict[str, Any], trial_dir: str, 
                                working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                                last_retrieval_component: str, last_retrieval_score: float,
                                save_intermediate: bool) -> Dict[str, Any]:
        compressed_df, token_eval_results, compressor_score, compression_results = self.component_runners.run_compressor(
            config, trial_dir, working_df, qa_subset, is_local_optimization=True
        )
        
        if save_intermediate and self.intermediate_handler:
            self.intermediate_handler.save_intermediate_result('passage_compressor', compressed_df, token_eval_results, trial_dir, config)
        
        return {
            'retrieval_score': 0.0,
            'last_retrieval_component': 'passage_compressor',
            'last_retrieval_score': compressor_score,
            'combined_score': compressor_score,
            'score': compressor_score,
            'compression_score': compressor_score,
            'compression_metrics': Utils.json_serializable(token_eval_results),
            'evaluation_method': 'local_optimization',
            'working_df': compressed_df
        }
    
    def handle_prompt_generator_local(self, config: Dict[str, Any], trial_dir: str, 
                                      working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                                      last_retrieval_component: str, last_retrieval_score: float,
                                      save_intermediate: bool) -> Dict[str, Any]:
        if self.component_results:
            for comp in ['passage_compressor', 'passage_filter', 'passage_reranker', 'retrieval', 'query_expansion']:
                if comp in self.component_results:
                    last_retrieval_component = comp
                    last_retrieval_score = self.component_results[comp].get('best_score', 0.0)
                    print(f"[Prompt/Generator] Using score from {comp}: {last_retrieval_score}")
                    break

        prompts_df, prompt_results = self.component_runners.run_prompt_maker(config, trial_dir, working_df, qa_subset)
        if save_intermediate and self.intermediate_handler:
            self.intermediate_handler.save_intermediate_result('prompt_maker', prompts_df, {}, trial_dir, config)

        eval_df = self.component_runners.run_generator(config, trial_dir, prompts_df, working_df, qa_subset)
        
        if eval_df is not None and 'generated_texts' in eval_df.columns:
            working_df['generated_texts'] = eval_df['generated_texts']
            working_df['prompts'] = eval_df['prompts'] if 'prompts' in eval_df.columns else prompts_df['prompts']
        
        generation_results = self.evaluate_generation_traditional(eval_df, qa_subset) if eval_df is not None else {}
        generation_score = generation_results.get('mean_score', 0.0)
        
        if save_intermediate and self.intermediate_handler:
            self.intermediate_handler.save_intermediate_result('generator', eval_df if eval_df is not None else working_df, 
                                        {'generation_score': generation_score, 'generation_metrics': generation_results}, 
                                        trial_dir, config)
        
        return {
            'retrieval_score': 0.0,
            'last_retrieval_component': last_retrieval_component,
            'last_retrieval_score': last_retrieval_score,
            'generation_score': generation_score,
            'generation_metrics': generation_results,
            'combined_score': self.evaluator.calculate_combined_score(
                last_retrieval_score, generation_score, True
            ),
            'score': generation_score if generation_score > 0 else last_retrieval_score,
            'evaluation_method': 'local_optimization',
            'working_df': working_df,
            'generated_texts': eval_df['generated_texts'].tolist() if eval_df is not None and 'generated_texts' in eval_df.columns else []
        }
    
    def evaluate_generation_traditional(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        if eval_df is None:
            return {}
        return self.evaluator.evaluate_generation(eval_df, qa_subset)