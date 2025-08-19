import os
import json
import time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from pipeline.config_manager import ConfigGenerator
from pipeline.pipeline_executor import RAGPipelineExecutor
from pipeline.pipeline_evaluator import RAGPipelineEvaluator
from pipeline.utils import Utils


class EarlyStoppingException(Exception):
    def __init__(self, message, score, component):
        self.message = message
        self.score = score
        self.component = component
        super().__init__(self.message)


class RAGPipelineRunner:
    def __init__(
        self,
        config_generator: ConfigGenerator,
        retrieval_metrics: List, 
        filter_metrics: List,
        compressor_metrics: List,
        generation_metrics: List,
        prompt_maker_metrics: List,
        query_expansion_metrics: List = [],
        reranker_metrics: List = [], 
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        json_manager=None,
        use_ragas: bool = False, 
        ragas_config: Optional[Dict[str, Any]] = None,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o",
        early_stopping_thresholds: Optional[Dict[str, float]] = None
    ):
        self.config_generator = config_generator
        self.json_manager = json_manager
        self.retrieval_metrics = retrieval_metrics
        self.filter_metrics = filter_metrics
        self.compressor_metrics = compressor_metrics
        self.generation_metrics = generation_metrics
        self.prompt_maker_metrics = prompt_maker_metrics
        self.query_expansion_metrics = query_expansion_metrics
        self.reranker_metrics = reranker_metrics 
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        
        self.use_ragas = use_ragas
        self.ragas_config = ragas_config or {}
        
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model

        if early_stopping_thresholds is None:
            self.early_stopping_thresholds = {
                'retrieval': 0.1,
                'query_expansion': 0.1,
                'reranker': 0.2,
                'filter': 0.25,
                'compressor': 0.3
            }
        else:
            self.early_stopping_thresholds = early_stopping_thresholds
        
        self.executor = RAGPipelineExecutor(config_generator)
        self.evaluator = RAGPipelineEvaluator(
            config_generator=config_generator,
            retrieval_metrics=retrieval_metrics,
            filter_metrics=filter_metrics,
            compressor_metrics=compressor_metrics,
            generation_metrics=generation_metrics,
            prompt_maker_metrics=prompt_maker_metrics,
            query_expansion_metrics=query_expansion_metrics,
            reranker_metrics=reranker_metrics,
            retrieval_weight=retrieval_weight,
            generation_weight=generation_weight,
            use_ragas=use_ragas,
            ragas_config=ragas_config,
            use_llm_compressor_evaluator=use_llm_compressor_evaluator,
            llm_evaluator_model=llm_evaluator_model
        )
    
    def _check_early_stopping(self, component: str, score: float, is_local_optimization: bool = False) -> None:
        if is_local_optimization:
            return
        
        threshold = self.early_stopping_thresholds.get(component)
        if threshold is not None and score < threshold:
            print(f"\n[EARLY STOPPING] {component} score {score:.4f} < threshold {threshold}")
            raise EarlyStoppingException(
                f"Early stopping at {component}: score {score:.4f} below threshold {threshold}",
                score=score,
                component=component
            )
    
    def save_intermediate_result(self, component: str, working_df: pd.DataFrame, 
                                results: Dict[str, Any], trial_dir: str, config: Dict[str, Any] = None):
        try:
            debug_dir = os.path.join(trial_dir, "debug_intermediate_results")
            os.makedirs(debug_dir, exist_ok=True)

            df_path = os.path.join(debug_dir, f"{component}_dataframe.parquet")
            working_df.to_parquet(df_path, index=False)

            results_to_save = {}
            for key, value in results.items():
                if not isinstance(value, pd.DataFrame):
                    results_to_save[key] = Utils.json_serializable(value)
            
            results_path = os.path.join(debug_dir, f"{component}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)

            summary = {
                'component': component,
                'timestamp': time.time(),
                'num_rows': len(working_df),
                'columns': list(working_df.columns),
                'score': self._get_component_score(component, results),
                'execution_time': results.get('execution_time', 0.0)
            }
            
            if config:
                summary['config_explored'] = {
                    'full_config': config,
                    'component_specific_config': self._extract_component_config(component, config)
                }

            if component in ['retrieval', 'query_expansion']:
                summary['retrieval_score'] = results.get('retrieval_score', results.get('last_retrieval_score', 0.0))
                summary['retrieval_metrics'] = results.get('retrieval_metrics', {})
            elif component == 'passage_reranker':
                summary['reranker_score'] = results.get('reranker_score', 0.0)
                summary['reranker_metrics'] = results.get('reranker_metrics', {})
            elif component == 'passage_filter':
                summary['filter_score'] = results.get('filter_score', 0.0)
                summary['filter_metrics'] = results.get('filter_metrics', {})
            elif component == 'passage_compressor':
                summary['compression_score'] = results.get('compression_score', 0.0)
                summary['compressor_metrics'] = results.get('compression_metrics', {})
            elif component == 'prompt_maker':
                summary['prompt_maker_score'] = results.get('prompt_maker_score', 0.0)
                summary['prompt_maker_metrics'] = results.get('prompt_metrics', {})
            elif component == 'generator':
                summary['generation_score'] = results.get('generation_score', 0.0)
                summary['generation_metrics'] = results.get('generation_metrics', {})
            
            summary_path = os.path.join(debug_dir, f"{component}_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"[DEBUG] Saved intermediate results for {component} to {debug_dir}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save intermediate results for {component}: {e}")
            pass
    
    def _extract_component_config(self, component: str, config: Dict[str, Any]) -> Dict[str, Any]:
        component_config = {}
        
        if component == 'retrieval':
            relevant_keys = ['retrieval_method', 'retriever_top_k', 'bm25_tokenizer', 'vectordb_name']
        elif component == 'query_expansion':
            relevant_keys = ['query_expansion_method', 'query_expansion_model', 
                            'query_expansion_temperature', 'query_expansion_max_token']
        elif component == 'passage_reranker':
            relevant_keys = ['passage_reranker_method', 'reranker_top_k', 'reranker_model']
        elif component == 'passage_filter':
            relevant_keys = ['passage_filter_method', 'threshold', 'percentile']
        elif component == 'passage_compressor':
            relevant_keys = ['passage_compressor_method', 'compressor_model', 
                            'compressor_compression_ratio', 'compressor_spacy_model']
        elif component == 'prompt_maker':
            relevant_keys = ['prompt_maker_method', 'prompt_template_idx']
        elif component == 'generator':
            relevant_keys = ['generator_module_type', 'generator_model', 'generator_temperature']
        else:
            relevant_keys = []
        
        for key in relevant_keys:
            if key in config:
                component_config[key] = config[key]
        
        return component_config
    
    def _get_component_score(self, component: str, results: Dict[str, Any]) -> float:
        score_keys = {
            'query_expansion': ['query_expansion_score', 'mean_score', 'last_retrieval_score'],
            'retrieval': ['retrieval_score', 'mean_accuracy', 'last_retrieval_score'],
            'passage_reranker': ['reranker_score', 'mean_accuracy', 'last_retrieval_score'],
            'passage_filter': ['filter_score', 'mean_accuracy', 'last_retrieval_score'],
            'passage_compressor': ['compression_score', 'mean_score', 'last_retrieval_score'],
            'prompt_maker': ['prompt_maker_score', 'mean_score'],
            'generator': ['generation_score', 'mean_score', 'combined_score']
        }
        
        for key in score_keys.get(component, ['score']):
            if key in results and results[key] is not None:
                return results[key]
        
        return 0.0
    
    def _save_pipeline_summary(self, trial_dir: str, last_retrieval_component: str, 
                              last_retrieval_score: float, generation_score: float, 
                              results: Dict[str, Any]):
        try:
            debug_dir = os.path.join(trial_dir, "debug_intermediate_results")
            os.makedirs(debug_dir, exist_ok=True)
            
            pipeline_summary = {
                'timestamp': time.time(),
                'last_retrieval_component': last_retrieval_component,
                'last_retrieval_score': last_retrieval_score,
                'generation_score': generation_score,
                'combined_score': results.get('combined_score', 0.0),
                'evaluation_method': results.get('evaluation_method', 'unknown'),
                'component_scores': {},
                'components_executed': []
            }

            for file in os.listdir(debug_dir):
                if file.endswith('_summary.json'):
                    component = file.replace('_summary.json', '')
                    with open(os.path.join(debug_dir, file), 'r') as f:
                        summary = json.load(f)
                        pipeline_summary['component_scores'][component] = summary.get('score', 0.0)
                        pipeline_summary['components_executed'].append(component)
            
            summary_path = os.path.join(debug_dir, "pipeline_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2)
                
        except Exception as e:
            print(f"[WARNING] Failed to save pipeline summary: {e}")
    
    def _print_configuration(self, config: Dict[str, Any]):
        print("\n" + "="*80)
        print("SELECTED CONFIGURATION FROM OPTIMIZER:")
        print("="*80)
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
        print("="*80 + "\n")
    
    def _validate_global_optimization(self, config: Dict[str, Any]) -> bool:
        if not self.use_ragas:
            return True
            
        has_retrieval = ('retrieval_method' in config or 'query_expansion_method' in config)
        has_prompt_maker = any('prompt' in k for k in config.keys())
        has_generator = any('generator' in k for k in config.keys())
        
        if not (has_retrieval and has_prompt_maker and has_generator):
            print("WARNING: Global optimization with RAGAS requires retrieval, prompt_maker, and generator components")
            return False
        return True
    
    def _handle_local_optimization_component(self, current_component: str, config: Dict[str, Any], 
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
        working_df, query_expansion_results, retrieval_df_from_qe, retrieval_done_in_qe = self._run_query_expansion_with_retrieval(
            config, trial_dir, working_df, is_local_optimization=True
        )
        
        if retrieval_done_in_qe and retrieval_df_from_qe is not None:
            working_df = retrieval_df_from_qe
            last_retrieval_score = query_expansion_results.get('mean_score', 0.0)
            last_retrieval_component = "query_expansion"
        
        if save_intermediate:
            self.save_intermediate_result('query_expansion', working_df, query_expansion_results, trial_dir, config)
        
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
        working_df, retrieval_results, last_retrieval_score, last_retrieval_results = self._run_retrieval(
            config, trial_dir, working_df, qa_subset, is_local_optimization=True
        )
        
        if save_intermediate:
            self.save_intermediate_result('retrieval', working_df, retrieval_results, trial_dir, config)
        
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
            self.save_intermediate_result('passage_reranker', reranked_df, reranker_results, trial_dir, config)
        
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
            self.save_intermediate_result('passage_filter', filtered_df, filter_results, trial_dir, config)
        
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
            self.save_intermediate_result('passage_compressor', compressed_df, token_eval_results, trial_dir, config)
        
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
            self.save_intermediate_result('prompt_maker', prompts_df, {}, trial_dir, config)

        eval_df = self.executor.execute_generator(config, trial_dir, prompts_df, working_df, qa_subset)
        
        if eval_df is not None and 'generated_texts' in eval_df.columns:
            working_df['generated_texts'] = eval_df['generated_texts']
            working_df['prompts'] = eval_df['prompts'] if 'prompts' in eval_df.columns else prompts_df['prompts']
        
        generation_results = self._evaluate_generation_traditional(eval_df, qa_subset) if eval_df is not None else {}
        generation_score = generation_results.get('mean_score', 0.0)
        
        combined_score = (self.retrieval_weight * last_retrieval_score + 
                         self.generation_weight * generation_score)
        
        if save_intermediate:
            self.save_intermediate_result('generator', eval_df if eval_df is not None else working_df, 
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
    
    def _execute_full_pipeline(self, config: Dict[str, Any], trial_dir: str, 
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
            working_df, query_expansion_results, retrieval_df_from_qe, retrieval_done_in_qe = self._run_query_expansion_with_retrieval(
                config, trial_dir, qa_subset
            )
            
            if save_intermediate and query_expansion_results:
                self.save_intermediate_result('query_expansion', working_df, query_expansion_results, trial_dir, config)
            
            pipeline_state['all_results']['query_expansion'] = query_expansion_results
            
            if retrieval_done_in_qe and retrieval_df_from_qe is not None:
                print("[Trial] Skipping retrieval node - already done in query expansion")
                working_df = retrieval_df_from_qe
                retrieval_results = query_expansion_results.get('metrics', {})
                pipeline_state['last_retrieval_score'] = query_expansion_results.get('mean_score', 0.0)
                pipeline_state['last_retrieval_results'] = retrieval_results
                pipeline_state['last_retrieval_component'] = "query_expansion"
            else:
                working_df, retrieval_results, last_retrieval_score, last_retrieval_results = self._run_retrieval(
                    config, trial_dir, working_df, qa_subset
                )
                pipeline_state['last_retrieval_component'] = "retrieval"
                pipeline_state['last_retrieval_score'] = last_retrieval_score
                pipeline_state['last_retrieval_results'] = last_retrieval_results
                
                if save_intermediate:
                    self.save_intermediate_result('retrieval', working_df, retrieval_results, trial_dir, config)
            
            pipeline_state['all_results']['retrieval'] = retrieval_results if 'retrieval_results' in locals() else {}

            working_df, reranker_results, reranker_applied, reranker_score, reranker_last_results = self._run_reranker(
                config, trial_dir, working_df, qa_subset
            )
            if reranker_applied:
                pipeline_state['last_retrieval_component'] = "passage_reranker"
                pipeline_state['last_retrieval_score'] = reranker_score
                pipeline_state['last_retrieval_results'] = reranker_last_results
                
                if save_intermediate:
                    self.save_intermediate_result('passage_reranker', working_df, reranker_results, trial_dir, config)
            
            pipeline_state['all_results']['reranker'] = reranker_results

            working_df, filter_results, filter_applied, filter_score, filter_last_results = self._run_filter(
                config, trial_dir, working_df, qa_subset
            )
            if filter_applied:
                pipeline_state['last_retrieval_component'] = "passage_filter"
                pipeline_state['last_retrieval_score'] = filter_score
                pipeline_state['last_retrieval_results'] = filter_last_results
                
                if save_intermediate:
                    self.save_intermediate_result('passage_filter', working_df, filter_results, trial_dir, config)
            
            pipeline_state['all_results']['filter'] = filter_results

            compressor_method = config.get('passage_compressor_method')
            if compressor_method not in ['pass_compressor', 'pass', None]:
                working_df, token_eval_results, compressor_score, compression_results = self._run_compressor(
                    config, trial_dir, working_df, qa_subset
                )
                if compression_results:
                    pipeline_state['last_retrieval_component'] = "passage_compressor"
                    pipeline_state['last_retrieval_score'] = compressor_score
                    pipeline_state['last_retrieval_results'] = compression_results
                    
                    if save_intermediate:
                        self.save_intermediate_result('passage_compressor', working_df, token_eval_results, trial_dir, config)
                
                pipeline_state['all_results']['compressor'] = token_eval_results
            else:
                print("[Trial] Skipping pass compressor evaluation in global optimization")

            prompts_df, prompt_results = self._run_prompt_maker(config, trial_dir, working_df, qa_subset)
            if save_intermediate:
                self.save_intermediate_result('prompt_maker', prompts_df, prompt_results, trial_dir, config)
            
            pipeline_state['all_results']['prompt_maker'] = prompt_results

            eval_df = self._run_generator(config, trial_dir, prompts_df, working_df, qa_subset)
            
            generation_results = {}
            if eval_df is not None:
                generation_results = self._evaluate_generation_traditional(eval_df, qa_subset)
                if save_intermediate:
                    self.save_intermediate_result('generator', eval_df, 
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
                self._save_pipeline_summary(trial_dir, 
                                           pipeline_state['last_retrieval_component'], 
                                           pipeline_state['last_retrieval_score'], 
                                           0.0,
                                           {'early_stopped': True, 'stopped_at': e.component})
            
            raise
    
    def _evaluate_pipeline_results(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame, 
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
            self._save_pipeline_summary(trial_dir, pipeline_state['last_retrieval_component'], 
                                       pipeline_state['last_retrieval_score'], 
                                       generation_score if 'generation_score' in locals() else 0.0,
                                       results)
        
        return results
    
    def run_pipeline(self, config: Dict[str, Any], trial_dir: str, qa_subset: pd.DataFrame, 
                     is_local_optimization: bool = False, current_component: str = None) -> Dict[str, Any]:
        try:
            self._print_configuration(config)
            
            save_intermediate = config.get('save_intermediate_results', True)
            
            if not self._validate_global_optimization(config):
                return {
                    'score': 0.0,
                    'combined_score': 0.0,
                    'error': 'Missing required components for RAGAS evaluation'
                }
            
            if is_local_optimization and current_component:
                return self._handle_local_optimization_component(
                    current_component, config, trial_dir, qa_subset.copy(), 
                    qa_subset, save_intermediate
                )
            
            else:
                eval_df, pipeline_state = self._execute_full_pipeline(
                    config, trial_dir, qa_subset, save_intermediate
                )
                
                results = self._evaluate_pipeline_results(
                    eval_df, qa_subset, pipeline_state, config, trial_dir, save_intermediate
                )
                
                results['working_df'] = eval_df if eval_df is not None else pipeline_state.get('working_df', qa_subset)
                
                return results
        
        except EarlyStoppingException as e:
            print(f"Early stopping triggered: {e.message}")
            return {
                'score': e.score,
                'combined_score': e.score,
                'early_stopped_at': e.component,
                'error': e.message
            }
        
        except Exception as e:
            print(f"Error running pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {
                'score': 0.0,
                'combined_score': 0.0,
                'retrieval_score': 0.0,
                'last_retrieval_component': 'none',
                'last_retrieval_score': 0.0,
                'error': str(e)
            }
    
    def _evaluate_generation_traditional(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        if eval_df is None:
            return {}
        return self.evaluator.evaluate_generation(eval_df, qa_subset)
    
    def _run_query_expansion_with_retrieval(self, config: Dict[str, Any], trial_dir: str, 
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
                
                self._check_early_stopping('retrieval', retrieval_score, is_local_optimization)
                
                query_expansion_results = {
                    'mean_score': retrieval_score,
                    'metrics': retrieval_results
                }
                working_df = retrieval_df
                retrieval_done_in_qe = True
            return working_df, query_expansion_results, retrieval_df, retrieval_done_in_qe
        
        if self.query_expansion_metrics and 'queries' in working_df.columns:
            query_expansion_results, retrieval_df = self._evaluate_query_expansion_with_retrieval(
                working_df, qa_subset, trial_dir, config
            )
            retrieval_done_in_qe = True
            
            qe_score = query_expansion_results.get('mean_score', 0.0)
            self._check_early_stopping('query_expansion', qe_score, is_local_optimization)

            if retrieval_df is not None and isinstance(retrieval_df, pd.DataFrame):
                Utils.update_dataframe_columns(working_df, retrieval_df, 
                    include_cols=['retrieved_ids', 'retrieved_contents', 'retrieve_scores'])

                query_expansion_results['retrieval_df'] = retrieval_df
        
        return working_df, query_expansion_results, retrieval_df, retrieval_done_in_qe
    
    def _get_centralized_project_dir(self):
        return Utils.get_centralized_project_dir()
    
    def _evaluate_query_expansion_with_retrieval(self, working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                    trial_dir: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        retrieval_df = self.executor.execute_query_expansion_retrieval(config, trial_dir, working_df)
        query_expansion_results = self.evaluator.evaluate_query_expansion(retrieval_df, qa_subset)
        return query_expansion_results, retrieval_df
    
    def _run_retrieval(self, config: Dict[str, Any], trial_dir: str, 
          working_df: pd.DataFrame, qa_subset: pd.DataFrame, 
          is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
    
        retrieval_df = self.executor.execute_retrieval(config, trial_dir, working_df)
        retrieval_results = self.evaluator.evaluate_retrieval(retrieval_df, qa_subset)

        if retrieval_df is not None and isinstance(retrieval_df, pd.DataFrame):
            Utils.update_dataframe_columns(working_df, retrieval_df,
                include_cols=['retrieved_ids', 'retrieved_contents', 'retrieve_scores', 'queries'])
        
        retrieval_score = retrieval_results.get('mean_accuracy', 0.0)
        self._check_early_stopping('retrieval', retrieval_score, is_local_optimization)
        
        return working_df, retrieval_results, retrieval_score, retrieval_results
    
    def _run_reranker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
        reranker_results = {}
        reranker_applied = False
        last_score = 0.0
        last_results = {}
        
        reranked_df = self.executor.execute_reranker(config, trial_dir, working_df)
        
        if reranked_df is not working_df:
            reranker_results = self.evaluator.evaluate_reranker(reranked_df, qa_subset)
            last_score = reranker_results.get('mean_accuracy', 0.0)
            last_results = reranker_results
            reranker_applied = True
            print(f"Applied reranking, new mean accuracy: {last_score}")
            self._check_early_stopping('reranker', last_score, is_local_optimization)
        else:
            print(f"Pass-through reranking")
        
        return reranked_df, reranker_results, reranker_applied, last_score, last_results
    
    def _run_filter(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
           qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
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
            print(f"[Filter] Pass-through filter detected")
            return working_df, filter_results, filter_applied, last_score, last_results

        filter_applied = True
        
        try:
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
            self._check_early_stopping('filter', last_score, is_local_optimization)
            
        except EarlyStoppingException:
            raise
            
        except Exception as e:
            print(f"[Filter] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

            filter_results = {'mean_accuracy': 0.0, 'error': str(e)}
            last_score = 0.0
            last_results = filter_results
        
        return filtered_df, filter_results, filter_applied, last_score, last_results
    
    def _run_compressor(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                   qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
    
        compressed_df = self.executor.execute_compressor(config, trial_dir, working_df)

        if compressed_df is not working_df:
            token_eval_results = self.evaluator.evaluate_compressor(compressed_df, qa_subset)
            last_score = token_eval_results.get('mean_score', 0.0) if token_eval_results else 0.0
            print(f"Applied compression, mean score: {last_score}")
            self._check_early_stopping('compressor', last_score, is_local_optimization)
            return compressed_df, token_eval_results, last_score, token_eval_results
        else:
            return working_df, {}, 0.0, {}

    def _run_prompt_maker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                     qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        prompts_df = self.executor.execute_prompt_maker(config, trial_dir, working_df)
        prompt_results = {}
        print("[Trial] Skipping prompt maker evaluation for optimization runs")
        return prompts_df, prompt_results
    
    def _run_generator(self, config: Dict[str, Any], trial_dir: str, prompts_df: pd.DataFrame,
                  working_df: pd.DataFrame, qa_subset: pd.DataFrame) -> pd.DataFrame:
        
        eval_df = self.executor.execute_generator(config, trial_dir, prompts_df, working_df, qa_subset)
        return eval_df