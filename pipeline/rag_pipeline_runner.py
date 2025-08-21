import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from pipeline.config_manager import ConfigGenerator
from pipeline.pipeline_executor import RAGPipelineExecutor
from pipeline.pipeline_evaluator import RAGPipelineEvaluator

from pipeline.pipeline_runner.pipeline_utils import (
    EarlyStoppingException, 
    EarlyStoppingHandler,
    IntermediateResultsHandler,
    PipelineUtilities
)
from pipeline.pipeline_runner.local_optimization import LocalOptimizationHandler
from pipeline.pipeline_runner.component_runners import ComponentRunners
from pipeline.pipeline_runner.pipeline_orchestrator import PipelineOrchestrator

__all__ = ['RAGPipelineRunner', 'EarlyStoppingException']


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
        
        self.early_stopping_handler = EarlyStoppingHandler(self.early_stopping_thresholds)
        self.intermediate_handler = IntermediateResultsHandler()
        self.utilities = PipelineUtilities(use_ragas)
        
        self.component_runners = ComponentRunners(
            executor=self.executor,
            evaluator=self.evaluator,
            config_generator=self.config_generator,
            early_stopping_handler=self.early_stopping_handler,
            query_expansion_metrics=query_expansion_metrics
        )
        
        self.local_optimization_handler = LocalOptimizationHandler(
            executor=self.executor,
            evaluator=self.evaluator,
            intermediate_handler=self.intermediate_handler,
            component_runners=self.component_runners,
            retrieval_weight=retrieval_weight,
            generation_weight=generation_weight
        )
        
        self.pipeline_orchestrator = PipelineOrchestrator(
            config_generator=config_generator,
            evaluator=self.evaluator,
            intermediate_handler=self.intermediate_handler,
            component_runners=self.component_runners,
            retrieval_weight=retrieval_weight,
            generation_weight=generation_weight,
            generation_metrics=generation_metrics,
            use_ragas=use_ragas
        )
    
    def _check_early_stopping(self, component: str, score: float, is_local_optimization: bool = False) -> None:
        self.early_stopping_handler.check_early_stopping(component, score, is_local_optimization)
    
    def save_intermediate_result(self, component: str, working_df: pd.DataFrame, 
                                results: Dict[str, Any], trial_dir: str, config: Dict[str, Any] = None):
        self.intermediate_handler.save_intermediate_result(component, working_df, results, trial_dir, config)
    
    def _extract_component_config(self, component: str, config: Dict[str, Any]) -> Dict[str, Any]:
        return self.intermediate_handler._extract_component_config(component, config)
    
    def _get_component_score(self, component: str, results: Dict[str, Any]) -> float:
        return self.intermediate_handler._get_component_score(component, results)
    
    def _save_pipeline_summary(self, trial_dir: str, last_retrieval_component: str, 
                              last_retrieval_score: float, generation_score: float, 
                              results: Dict[str, Any]):
        self.intermediate_handler.save_pipeline_summary(trial_dir, last_retrieval_component, 
                                                       last_retrieval_score, generation_score, results)
    
    def _print_configuration(self, config: Dict[str, Any]):
        self.utilities.print_configuration(config)
    
    def _validate_global_optimization(self, config: Dict[str, Any]) -> bool:
        return self.utilities.validate_global_optimization(config)
    
    def _get_centralized_project_dir(self):
        return self.utilities.get_centralized_project_dir()
    
    def _handle_local_optimization_component(self, current_component: str, config: Dict[str, Any], 
                                            trial_dir: str, working_df: pd.DataFrame, 
                                            qa_subset: pd.DataFrame, save_intermediate: bool) -> Dict[str, Any]:
        if hasattr(self.local_optimization_handler, 'component_results'):
            self.local_optimization_handler.component_results = getattr(self, 'component_results', {})
        return self.local_optimization_handler.handle_local_optimization_component(
            current_component, config, trial_dir, working_df, qa_subset, save_intermediate
        )
    
    def _run_query_expansion_with_retrieval(self, config: Dict[str, Any], trial_dir: str, 
               qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, bool]:
        return self.component_runners.run_query_expansion_with_retrieval(config, trial_dir, qa_subset, is_local_optimization)
    
    def _evaluate_query_expansion_with_retrieval(self, working_df: pd.DataFrame, qa_subset: pd.DataFrame,
                    trial_dir: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        return self.component_runners._evaluate_query_expansion_with_retrieval(working_df, qa_subset, trial_dir, config)
    
    def _run_retrieval(self, config: Dict[str, Any], trial_dir: str, 
          working_df: pd.DataFrame, qa_subset: pd.DataFrame, 
          is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
        return self.component_runners.run_retrieval(config, trial_dir, working_df, qa_subset, is_local_optimization)
    
    def _run_reranker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
        return self.component_runners.run_reranker(config, trial_dir, working_df, qa_subset, is_local_optimization)
    
    def _run_filter(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
           qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], bool, float, Dict[str, Any]]:
        return self.component_runners.run_filter(config, trial_dir, working_df, qa_subset, is_local_optimization)
    
    def _run_compressor(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                   qa_subset: pd.DataFrame, is_local_optimization: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
        return self.component_runners.run_compressor(config, trial_dir, working_df, qa_subset, is_local_optimization)

    def _run_prompt_maker(self, config: Dict[str, Any], trial_dir: str, working_df: pd.DataFrame,
                     qa_subset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return self.component_runners.run_prompt_maker(config, trial_dir, working_df, qa_subset)
    
    def _run_generator(self, config: Dict[str, Any], trial_dir: str, prompts_df: pd.DataFrame,
                  working_df: pd.DataFrame, qa_subset: pd.DataFrame) -> pd.DataFrame:
        return self.component_runners.run_generator(config, trial_dir, prompts_df, working_df, qa_subset)
    
    def _execute_full_pipeline(self, config: Dict[str, Any], trial_dir: str, 
                              qa_subset: pd.DataFrame, save_intermediate: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return self.pipeline_orchestrator.execute_full_pipeline(config, trial_dir, qa_subset, save_intermediate)
    
    def _evaluate_pipeline_results(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame, 
                                  pipeline_state: Dict[str, Any], config: Dict[str, Any],
                                  trial_dir: str, save_intermediate: bool) -> Dict[str, Any]:
        return self.pipeline_orchestrator.evaluate_pipeline_results(eval_df, qa_subset, pipeline_state, config, trial_dir, save_intermediate)
    
    def _evaluate_generation_traditional(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        return self.pipeline_orchestrator._evaluate_generation_traditional(eval_df, qa_subset)
    
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