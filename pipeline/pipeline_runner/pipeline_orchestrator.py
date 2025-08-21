import os
import pandas as pd
from typing import Dict, Any, Tuple
from pipeline.utils import Utils
from .pipeline_utils import EarlyStoppingException


class PipelineOrchestrator:
    def __init__(self, config_generator, evaluator, intermediate_handler, component_runners,
                 retrieval_weight: float = 0.5, generation_weight: float = 0.5,
                 generation_metrics=None, use_ragas: bool = False):
        self.config_generator = config_generator
        self.evaluator = evaluator
        self.intermediate_handler = intermediate_handler
        self.component_runners = component_runners
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.generation_metrics = generation_metrics or []
        self.use_ragas = use_ragas
    
    def execute_full_pipeline(self, config: Dict[str, Any], trial_dir: str, 
                              qa_subset: pd.DataFrame, save_intermediate: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        working_df = qa_subset.copy()
        pipeline_state = {
            'last_retrieval_component': 'retrieval',
            'last_retrieval_score': 0.0,
            'last_retrieval_results': {},
            'all_results': {},
            'early_stopped': False,
            'stopped_at': None
        }
        
        try:
            working_df, query_expansion_results, retrieval_df_from_qe, retrieval_done_in_qe = self.component_runners.run_query_expansion_with_retrieval(
                config, trial_dir, qa_subset
            )
            
            if save_intermediate and query_expansion_results:
                self.intermediate_handler.save_intermediate_result('query_expansion', working_df, query_expansion_results, trial_dir, config)
            
            pipeline_state['all_results']['query_expansion'] = query_expansion_results
            
            if retrieval_done_in_qe and retrieval_df_from_qe is not None:
                print("[Trial] Skipping retrieval node - already done in query expansion")
                working_df = retrieval_df_from_qe
                retrieval_results = query_expansion_results.get('metrics', {})
                pipeline_state['last_retrieval_score'] = query_expansion_results.get('mean_score', 0.0)
                pipeline_state['last_retrieval_results'] = retrieval_results
                pipeline_state['last_retrieval_component'] = "query_expansion"
            else:
                working_df, retrieval_results, last_retrieval_score, last_retrieval_results = self.component_runners.run_retrieval(
                    config, trial_dir, working_df, qa_subset
                )
                pipeline_state['last_retrieval_component'] = "retrieval"
                pipeline_state['last_retrieval_score'] = last_retrieval_score
                pipeline_state['last_retrieval_results'] = last_retrieval_results
                
                if save_intermediate:
                    self.intermediate_handler.save_intermediate_result('retrieval', working_df, retrieval_results, trial_dir, config)
            
            pipeline_state['all_results']['retrieval'] = retrieval_results if 'retrieval_results' in locals() else {}

            working_df, reranker_results, reranker_applied, reranker_score, reranker_last_results = self.component_runners.run_reranker(
                config, trial_dir, working_df, qa_subset
            )
            if reranker_applied:
                pipeline_state['last_retrieval_component'] = "passage_reranker"
                pipeline_state['last_retrieval_score'] = reranker_score
                pipeline_state['last_retrieval_results'] = reranker_last_results
                
                if save_intermediate:
                    self.intermediate_handler.save_intermediate_result('passage_reranker', working_df, reranker_results, trial_dir, config)
            
            pipeline_state['all_results']['reranker'] = reranker_results

            working_df, filter_results, filter_applied, filter_score, filter_last_results = self.component_runners.run_filter(
                config, trial_dir, working_df, qa_subset
            )
            if filter_applied:
                pipeline_state['last_retrieval_component'] = "passage_filter"
                pipeline_state['last_retrieval_score'] = filter_score
                pipeline_state['last_retrieval_results'] = filter_last_results
                
                if save_intermediate:
                    self.intermediate_handler.save_intermediate_result('passage_filter', working_df, filter_results, trial_dir, config)
            
            pipeline_state['all_results']['filter'] = filter_results

            compressor_method = config.get('passage_compressor_method')
            if compressor_method not in ['pass_compressor', 'pass', None]:
                working_df, token_eval_results, compressor_score, compression_results = self.component_runners.run_compressor(
                    config, trial_dir, working_df, qa_subset
                )
                if compression_results:
                    pipeline_state['last_retrieval_component'] = "passage_compressor"
                    pipeline_state['last_retrieval_score'] = compressor_score
                    pipeline_state['last_retrieval_results'] = compression_results
                    
                    if save_intermediate:
                        self.intermediate_handler.save_intermediate_result('passage_compressor', working_df, token_eval_results, trial_dir, config)
                
                pipeline_state['all_results']['compressor'] = token_eval_results
            else:
                print("[Trial] Skipping pass compressor evaluation in global optimization")

            prompts_df, prompt_results = self.component_runners.run_prompt_maker(config, trial_dir, working_df, qa_subset)
            if save_intermediate:
                self.intermediate_handler.save_intermediate_result('prompt_maker', prompts_df, prompt_results, trial_dir, config)
            
            pipeline_state['all_results']['prompt_maker'] = prompt_results

            eval_df = self.component_runners.run_generator(config, trial_dir, prompts_df, working_df, qa_subset)
            
            generation_results = {}
            if eval_df is not None:
                generation_results = self._evaluate_generation_traditional(eval_df, qa_subset)
                if save_intermediate:
                    self.intermediate_handler.save_intermediate_result('generator', eval_df, 
                                                {'generation_score': generation_results.get('mean_score', 0.0), 
                                                 'generation_metrics': generation_results}, 
                                                trial_dir, config)
            
            pipeline_state['all_results']['generator'] = generation_results

            Utils.update_dataframe_columns(qa_subset, working_df, exclude_cols=['query', 'retrieval_gt', 'generation_gt', 'qid'])
            
            return eval_df if eval_df is not None else working_df, pipeline_state
            
        except EarlyStoppingException as e:
            print(f"\n[Pipeline] Early stopping at {e.component} with score {e.score:.4f}")
            pipeline_state['early_stopped'] = True
            pipeline_state['stopped_at'] = e.component
            pipeline_state['stopped_score'] = e.score
            
            if save_intermediate:
                self.intermediate_handler.save_pipeline_summary(trial_dir, 
                                           pipeline_state['last_retrieval_component'], 
                                           pipeline_state['last_retrieval_score'], 
                                           0.0,
                                           {'early_stopped': True, 'stopped_at': e.component})
            
            raise
    
    def evaluate_pipeline_results(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame, 
                                  pipeline_state: Dict[str, Any], config: Dict[str, Any],
                                  trial_dir: str, save_intermediate: bool) -> Dict[str, Any]:
        if self.use_ragas and eval_df is not None:
            print("\n[Trial] Using RAGAS evaluation for global optimization")
            
            corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
            ragas_results = self.evaluator.evaluate_with_ragas(eval_df, qa_subset, corpus_path)
            ragas_mean_score = ragas_results.get('ragas_mean_score', 0.0)
            
            print(f"[Trial] RAGAS evaluation results:")
            for metric, score in ragas_results.items():
                if metric != 'ragas_metrics' and isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.4f}")
            
            combined_score = ragas_mean_score
            generation_score = ragas_mean_score
            
            results = {
                'retrieval_score': 0.0,
                'last_retrieval_component': pipeline_state['last_retrieval_component'],
                'last_retrieval_score': 0.0,
                'combined_score': combined_score,
                'score': combined_score,
                'ragas_mean_score': ragas_mean_score,
                'ragas_metrics': Utils.json_serializable(ragas_results),
                'retrieval_skipped': pipeline_state.get('retrieval_skipped', False),
                'evaluation_method': 'ragas'
            }
            
        else:
            generation_results = pipeline_state['all_results'].get('generator', {})
            generation_score = generation_results.get('mean_score', 0.0) if generation_results else 0.0
            
            combined_score = self.evaluator.calculate_combined_score(
                pipeline_state['last_retrieval_score'], generation_score, 
                self.config_generator.node_exists("generator")
            )
            
            print(f"[Trial] Final composite score calculation:")
            print(f"  Last retrieval component ({pipeline_state['last_retrieval_component']}): {pipeline_state['last_retrieval_score']}")
            if self.config_generator.node_exists("generator") and self.generation_metrics and generation_score > 0:
                print(f"  Generation score: {generation_score}")
                print(f"  Combined score: {combined_score} (weights: {self.retrieval_weight}/{self.generation_weight})")
            else:
                print(f"  No generation component")
                print(f"  Combined score: {combined_score} (using only retrieval)")
            
            results = {
                'retrieval_score': pipeline_state['all_results'].get('retrieval', {}).get('mean_accuracy', 0.0),
                'last_retrieval_component': pipeline_state['last_retrieval_component'],
                'last_retrieval_score': pipeline_state['last_retrieval_score'],
                'combined_score': combined_score,
                'score': generation_score if generation_score > 0 else pipeline_state['last_retrieval_score'],
                'retrieval_metrics': Utils.json_serializable(pipeline_state['all_results'].get('retrieval', {})),
                'last_retrieval_metrics': Utils.json_serializable(pipeline_state['last_retrieval_results']),
                'retrieval_skipped': pipeline_state.get('retrieval_skipped', False),
                'evaluation_method': 'traditional'
            }
            
            if generation_results:
                results['generation_score'] = generation_score
                results['generation_metrics'] = Utils.json_serializable(generation_results)
        
        for component, component_results in pipeline_state['all_results'].items():
            if component_results:
                if component == 'query_expansion':
                    results['query_expansion_score'] = component_results.get('mean_score', 0.0)
                    results['query_expansion_metrics'] = Utils.json_serializable(component_results)
                elif component == 'reranker':
                    results['reranker_score'] = component_results.get('mean_accuracy', 0.0)
                    results['reranker_metrics'] = Utils.json_serializable(component_results)
                elif component == 'filter':
                    results['filter_score'] = component_results.get('mean_accuracy', 0.0)
                    results['filter_metrics'] = Utils.json_serializable(component_results)
                elif component == 'compressor':
                    results['compression_score'] = component_results.get('mean_score', 0.0)
                    results['compression_metrics'] = Utils.json_serializable(component_results)
                elif component == 'prompt_maker':
                    results['prompt_maker_score'] = component_results.get('mean_score', 0.0)
                    results['prompt_metrics'] = Utils.json_serializable(component_results)
        
        if save_intermediate:
            self.intermediate_handler.save_pipeline_summary(trial_dir, pipeline_state['last_retrieval_component'], 
                                       pipeline_state['last_retrieval_score'], 
                                       generation_score if 'generation_score' in locals() else 0.0,
                                       results)
        
        return results
    
    def _evaluate_generation_traditional(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        if eval_df is None:
            return {}
        return self.evaluator.evaluate_generation(eval_df, qa_subset)