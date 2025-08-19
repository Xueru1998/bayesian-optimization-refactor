import os
import json
import time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from pipeline.config_manager import ConfigGenerator
from pipeline.pipeline_executor import RAGPipelineExecutor
from pipeline.pipeline_evaluator import RAGPipelineEvaluator
from pipeline.utils import Utils


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
        llm_evaluator_model: str = "gpt-4o"  
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
                component_specific_config = {}
                
                if component == 'query_expansion':
                    relevant_params = [
                        'query_expansion_method', 'query_expansion_model', 
                        'query_expansion_temperature', 'query_expansion_max_token',
                        'retriever_top_k', 'retrieval_method', 'bm25_tokenizer', 'vectordb_name'
                    ]
                elif component == 'retrieval':
                    relevant_params = [
                        'retrieval_method', 'retriever_top_k', 
                        'bm25_tokenizer', 'vectordb_name'
                    ]
                elif component == 'passage_reranker':
                    relevant_params = [
                        'passage_reranker_method', 'reranker_top_k',
                        'reranker_model', 'reranker_model_name'
                    ]
                elif component == 'passage_filter':
                    relevant_params = [
                        'passage_filter_method', 'threshold', 'percentile'
                    ]
                elif component == 'passage_compressor':
                    relevant_params = [
                        'passage_compressor_method', 'compressor_generator_module_type',
                        'compressor_model', 'compressor_llm', 'compressor_api_url',
                        'compressor_batch', 'compressor_temperature', 'compressor_max_tokens'
                    ]
                elif component == 'prompt_maker':
                    relevant_params = [
                        'prompt_maker_method', 'prompt_template_idx'
                    ]
                elif component == 'generator':
                    relevant_params = [
                        'generator_model', 'generator_temperature', 'generator_module_type',
                        'generator_llm', 'generator_api_url'
                    ]
                else:
                    relevant_params = []
                
                for param in relevant_params:
                    if param in config:
                        component_specific_config[param] = config[param]
                
                summary['config'] = component_specific_config
                summary['full_trial_config'] = Utils.json_serializable(config)

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
    
    def run_pipeline(self, config: Dict[str, Any], trial_dir: str, qa_subset: pd.DataFrame, 
                     is_local_optimization: bool = False, current_component: str = None) -> Dict[str, Any]:
        try:
            print("\n" + "="*80)
            print("SELECTED CONFIGURATION FROM OPTIMIZER:")
            print("="*80)
            for key, value in sorted(config.items()):
                print(f"  {key}: {value}")
            print("="*80 + "\n")
            
            save_intermediate = config.get('save_intermediate_results', True)
            
            is_global_optimization = self.use_ragas

            if is_global_optimization:
                has_retrieval = ('retrieval_method' in config or 'query_expansion_method' in config)
                has_prompt_maker = any('prompt' in k for k in config.keys())
                has_generator = any('generator' in k for k in config.keys())
                
                if not (has_retrieval and has_prompt_maker and has_generator):
                    print("WARNING: Global optimization with RAGAS requires retrieval, prompt_maker, and generator components")
                    return {
                        'score': 0.0,
                        'combined_score': 0.0,
                        'error': 'Missing required components for RAGAS evaluation'
                    }
            
            last_retrieval_component = "retrieval"
            last_retrieval_score = 0.0
            last_retrieval_results = {}
            working_df = qa_subset.copy()

            if is_local_optimization and current_component:
                print(f"\n[Local Optimization] Current component: {current_component}")
                print(f"[Local Optimization] Using saved outputs from previous components")

                if hasattr(self, 'component_results'):
                    for comp in ['passage_compressor', 'passage_filter', 'passage_reranker', 'retrieval', 'query_expansion']:
                        if comp in self.component_results:
                            last_retrieval_component = comp
                            last_retrieval_score = self.component_results[comp].get('best_score', 0.0)
                            print(f"[Local Optimization] Using score from {comp}: {last_retrieval_score}")
                            break
                
                if current_component == 'query_expansion':
                    working_df, query_expansion_results, retrieval_df_from_qe, retrieval_done_in_qe = self._run_query_expansion_with_retrieval(
                        config, trial_dir, working_df
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
                
                elif current_component == 'retrieval':
                    working_df, retrieval_results, last_retrieval_score, last_retrieval_results = self._run_retrieval(
                        config, trial_dir, working_df, qa_subset
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
                
                elif current_component == 'passage_reranker':
                    
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
                
                elif current_component == 'passage_filter':
                    
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
                
                elif current_component == 'passage_compressor':
                    
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
                
                elif current_component == 'prompt_maker_generator':
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

                            
            else:
                working_df, query_expansion_results, retrieval_df_from_qe, retrieval_done_in_qe = self._run_query_expansion_with_retrieval(
                    config, trial_dir, qa_subset
                )
                
                if save_intermediate and query_expansion_results:
                    self.save_intermediate_result('query_expansion', working_df, query_expansion_results, trial_dir, config)
                
                if retrieval_done_in_qe and retrieval_df_from_qe is not None:
                    print("[Trial] Skipping retrieval node - already done in query expansion")
                    working_df = retrieval_df_from_qe
                    retrieval_results = query_expansion_results.get('metrics', {})
                    last_retrieval_score = query_expansion_results.get('mean_score', 0.0)
                    last_retrieval_results = retrieval_results
                    last_retrieval_component = "query_expansion"
                else:
                    working_df, retrieval_results, last_retrieval_score, last_retrieval_results = self._run_retrieval(
                        config, trial_dir, working_df, qa_subset
                    )
                    last_retrieval_component = "retrieval"
                    
                    if save_intermediate:
                        self.save_intermediate_result('retrieval', working_df, retrieval_results, trial_dir, config)

                working_df, reranker_results, reranker_applied, reranker_score, reranker_last_results = self._run_reranker(
                    config, trial_dir, working_df, qa_subset
                )
                if reranker_applied:
                    last_retrieval_component = "passage_reranker"
                    last_retrieval_score = reranker_score
                    last_retrieval_results = reranker_last_results
                    
                    if save_intermediate:
                        self.save_intermediate_result('passage_reranker', working_df, reranker_results, trial_dir, config)
                
                working_df, filter_results, filter_applied, filter_score, filter_last_results = self._run_filter(
                    config, trial_dir, working_df, qa_subset
                )
                if filter_applied:
                    last_retrieval_component = "passage_filter"
                    last_retrieval_score = filter_score
                    last_retrieval_results = filter_last_results
                    
                    if save_intermediate:
                        self.save_intermediate_result('passage_filter', working_df, filter_results, trial_dir, config)
                
                compressor_method = config.get('passage_compressor_method')
                if compressor_method not in ['pass_compressor', 'pass', None]:
                    working_df, token_eval_results, compressor_score, compression_results = self._run_compressor(
                        config, trial_dir, working_df, qa_subset
                    )
                    if compression_results:
                        last_retrieval_component = "passage_compressor"
                        last_retrieval_score = compressor_score
                        last_retrieval_results = compression_results
                        
                        if save_intermediate:
                            self.save_intermediate_result('passage_compressor', working_df, token_eval_results, trial_dir, config)
                else:
                    print("[Trial] Skipping pass compressor evaluation in global optimization")
                
                prompts_df, prompt_results = self._run_prompt_maker(config, trial_dir, working_df, qa_subset)
                if save_intermediate:
                    self.save_intermediate_result('prompt_maker', prompts_df, prompt_results, trial_dir, config)
                
                eval_df = self._run_generator(config, trial_dir, prompts_df, working_df, qa_subset)
                if save_intermediate and eval_df is not None:
                    generation_results = self._evaluate_generation_traditional(eval_df, qa_subset) if eval_df is not None else {}
                    self.save_intermediate_result('generator', eval_df, 
                                                {'generation_score': generation_results.get('mean_score', 0.0), 
                                                'generation_metrics': generation_results}, 
                                                trial_dir, config)

            if save_intermediate:
                self._save_pipeline_summary(trial_dir, last_retrieval_component, last_retrieval_score, 
                                          generation_results.get('mean_score', 0.0) if 'generation_results' in locals() else 0.0,
                                          results if 'results' in locals() else {})
            
            Utils.update_dataframe_columns(qa_subset, working_df, exclude_cols=['query', 'retrieval_gt', 'generation_gt', 'qid'])
            
            if is_local_optimization and current_component != 'prompt_maker_generator':
                if current_component == 'query_expansion':
                    results = {
                        'retrieval_score': 0.0,
                        'last_retrieval_component': last_retrieval_component,
                        'last_retrieval_score': last_retrieval_score,
                        'combined_score': last_retrieval_score,
                        'score': last_retrieval_score,
                        'query_expansion_score': query_expansion_results.get('mean_score', 0.0) if 'query_expansion_results' in locals() else 0.0,
                        'query_expansion_metrics': Utils.json_serializable(query_expansion_results) if 'query_expansion_results' in locals() else {},
                        'retrieval_skipped': retrieval_done_in_qe if 'retrieval_done_in_qe' in locals() else False,
                        'evaluation_method': 'local_optimization',
                        'working_df': working_df
                    }
                elif current_component == 'retrieval':
                    results = {
                        'retrieval_score': retrieval_results.get('mean_accuracy', 0.0) if 'retrieval_results' in locals() else 0.0,
                        'last_retrieval_component': last_retrieval_component,
                        'last_retrieval_score': last_retrieval_score,
                        'combined_score': last_retrieval_score,
                        'score': last_retrieval_score,
                        'retrieval_metrics': Utils.json_serializable(retrieval_results) if 'retrieval_results' in locals() else {},
                        'evaluation_method': 'local_optimization',
                        'working_df': working_df
                    }
                elif current_component == 'passage_reranker':
                    results = {
                        'retrieval_score': 0.0,
                        'last_retrieval_component': last_retrieval_component,
                        'last_retrieval_score': last_retrieval_score,
                        'combined_score': last_retrieval_score,
                        'score': last_retrieval_score,
                        'reranker_score': reranker_results.get('mean_accuracy', 0.0) if 'reranker_results' in locals() else 0.0,
                        'reranker_metrics': Utils.json_serializable(reranker_results) if 'reranker_results' in locals() else {},
                        'evaluation_method': 'local_optimization',
                        'working_df': working_df
                    }
                elif current_component == 'passage_filter':
                    results = {
                        'retrieval_score': 0.0,
                        'last_retrieval_component': last_retrieval_component,
                        'last_retrieval_score': last_retrieval_score,
                        'combined_score': last_retrieval_score,
                        'score': last_retrieval_score,
                        'filter_score': filter_results.get('mean_accuracy', 0.0) if 'filter_results' in locals() else 0.0,
                        'filter_metrics': Utils.json_serializable(filter_results) if 'filter_results' in locals() else {},
                        'evaluation_method': 'local_optimization',
                        'working_df': working_df
                    }
                elif current_component == 'passage_compressor':
                    results = {
                        'retrieval_score': 0.0,
                        'last_retrieval_component': last_retrieval_component,
                        'last_retrieval_score': last_retrieval_score,
                        'combined_score': last_retrieval_score,
                        'score': last_retrieval_score,
                        'compression_score': token_eval_results.get('mean_score', 0.0) if 'token_eval_results' in locals() else 0.0,
                        'compression_metrics': Utils.json_serializable(token_eval_results) if 'token_eval_results' in locals() else {},
                        'evaluation_method': 'local_optimization',
                        'working_df': working_df
                    }
                
                return results
            
            if is_global_optimization and 'eval_df' in locals() and eval_df is not None:
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
                    'last_retrieval_component': last_retrieval_component,
                    'last_retrieval_score': 0.0, 
                    'combined_score': combined_score,
                    'score': combined_score,
                    'ragas_mean_score': ragas_mean_score,
                    'ragas_metrics': Utils.json_serializable(ragas_results),
                    'retrieval_skipped': retrieval_done_in_qe if 'retrieval_done_in_qe' in locals() else False,
                    'evaluation_method': 'ragas'
                }
                
            else:
                generation_results = self._evaluate_generation_traditional(eval_df, qa_subset) if 'eval_df' in locals() and eval_df is not None else {}
                generation_score = generation_results.get('mean_score', 0.0) if generation_results else 0.0
                
                combined_score = self.evaluator.calculate_combined_score(
                    last_retrieval_score, generation_score, 
                    self.config_generator.node_exists("generator")
                )
                
                print(f"[Trial] Final composite score calculation:")
                print(f"  Last retrieval component ({last_retrieval_component}): {last_retrieval_score}")
                if self.config_generator.node_exists("generator") and self.generation_metrics and generation_score > 0:
                    print(f"  Generation score: {generation_score}")
                    print(f"  Combined score: {combined_score} (weights: {self.retrieval_weight}/{self.generation_weight})")
                else:
                    print(f"  No generation component")
                    print(f"  Combined score: {combined_score} (using only retrieval)")
                
                results = {
                    'retrieval_score': retrieval_results.get('mean_accuracy', 0.0) if 'retrieval_results' in locals() and not retrieval_done_in_qe else 0.0,
                    'last_retrieval_component': last_retrieval_component,
                    'last_retrieval_score': last_retrieval_score,
                    'combined_score': combined_score,
                    'score': generation_score if generation_score > 0 else last_retrieval_score,
                    'retrieval_metrics': Utils.json_serializable(retrieval_results) if 'retrieval_results' in locals() and not retrieval_done_in_qe else {},
                    'last_retrieval_metrics': Utils.json_serializable(last_retrieval_results),
                    'retrieval_skipped': retrieval_done_in_qe if 'retrieval_done_in_qe' in locals() else False,
                    'evaluation_method': 'traditional'
                }
                
                if generation_results:
                    results['generation_score'] = generation_score
                    results['generation_metrics'] = Utils.json_serializable(generation_results)

            if 'query_expansion_results' in locals() and query_expansion_results:
                results['query_expansion_score'] = query_expansion_results.get('mean_score', 0.0)
                results['query_expansion_metrics'] = Utils.json_serializable(query_expansion_results)
            
            if 'reranker_applied' in locals() and reranker_applied and 'reranker_results' in locals() and reranker_results:
                results['reranker_score'] = reranker_results.get('mean_accuracy', 0.0)
                results['reranker_metrics'] = Utils.json_serializable(reranker_results)
                
            if 'filter_applied' in locals() and filter_applied and 'filter_results' in locals() and filter_results:
                results['filter_score'] = filter_results.get('mean_accuracy', 0.0)
                results['filter_metrics'] = Utils.json_serializable(filter_results)
                
            if 'token_eval_results' in locals() and token_eval_results:
                results['compression_score'] = token_eval_results.get('mean_score', 0.0)
                results['compression_metrics'] = Utils.json_serializable(token_eval_results)
                
            if 'prompt_results' in locals() and prompt_results:
                results['prompt_maker_score'] = prompt_results.get('mean_score', 0.0)
                results['prompt_metrics'] = Utils.json_serializable(prompt_results)
                
            results['working_df'] = working_df
            
            return results
        
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
    
    def _run_query_expansion_with_retrieval(self, config: Dict[str, Any], trial_dir: str, 
               qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, bool]:
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
                query_expansion_results = {
                    'mean_score': retrieval_results.get('mean_accuracy', 0.0),
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
          working_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
    
        retrieval_df = self.executor.execute_retrieval(config, trial_dir, working_df)
        retrieval_results = self.evaluator.evaluate_retrieval(retrieval_df, qa_subset)

        if retrieval_df is not None and isinstance(retrieval_df, pd.DataFrame):
            Utils.update_dataframe_columns(working_df, retrieval_df,
                include_cols=['retrieved_ids', 'retrieved_contents', 'retrieve_scores', 'queries'])
        
        return working_df, retrieval_results, retrieval_results.get('mean_accuracy', 0.0), retrieval_results
    
    def _run_reranker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
            qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
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
        else:
            print(f"Pass-through reranking")
        
        return reranked_df, reranker_results, reranker_applied, last_score, last_results
    
    def _run_filter(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
       qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
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
        except Exception as e:
            print(f"[Filter] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

            filter_results = {'mean_accuracy': 0.0, 'error': str(e)}
            last_score = 0.0
            last_results = filter_results
        
        return filtered_df, filter_results, filter_applied, last_score, last_results
    
    
    def _run_compressor(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
            qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
    
        compressed_df = self.executor.execute_compressor(config, trial_dir, working_df)

        if compressed_df is not working_df:
            token_eval_results = self.evaluator.evaluate_compressor(compressed_df, qa_subset)
            last_score = token_eval_results.get('mean_score', 0.0) if token_eval_results else 0.0
            print(f"Applied compression, mean score: {last_score}")
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