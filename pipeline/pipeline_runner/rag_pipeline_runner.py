from typing import Dict, Any, List, Optional
import pandas as pd
import traceback

from pipeline.config_manager import ConfigGenerator
from pipeline.pipeline_executor import RAGPipelineExecutor
from pipeline.pipeline_evaluator import RAGPipelineEvaluator
from pipeline.utils import Utils

from pipeline.pipeline_runner import (
    EarlyStoppingException,
    EarlyStoppingHandler,
    SAPEmbeddingsInitializer,
    IntermediateResultsHandler,
    PipelineUtilities,
    ComponentRunners,
    LocalOptimizationHandler,
    PipelineOrchestrator
)


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
        use_llm_evaluator: bool = False,
        llm_evaluator_config: Optional[Dict[str, Any]] = None,
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
        
        self.use_llm_evaluator = use_llm_evaluator
        self.llm_evaluator_config = llm_evaluator_config or {}
        
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
            use_llm_evaluator=self.use_llm_evaluator, 
            llm_evaluator_config=self.llm_evaluator_config
        )
        
        self.early_stopping_handler = EarlyStoppingHandler(early_stopping_thresholds)
        self.intermediate_handler = IntermediateResultsHandler()
        self.utilities = PipelineUtilities(use_ragas)
        self.sap_initializer = SAPEmbeddingsInitializer()
        
        self.component_runners = ComponentRunners(
            self.executor, 
            self.evaluator, 
            query_expansion_metrics,
            self.early_stopping_handler
        )
        
        self.local_optimization_handler = LocalOptimizationHandler(
            self.component_runners,
            self.evaluator,
            self.intermediate_handler
        )
        
        self.pipeline_orchestrator = PipelineOrchestrator(
            self.component_runners,
            self.evaluator,
            config_generator,
            self.intermediate_handler,
            use_ragas,
            retrieval_weight,
            generation_weight
        )
        
    def _get_centralized_project_dir(self):
        return Utils.get_centralized_project_dir()
        
    def run_pipeline(self, config: Dict[str, Any], trial_dir: str, qa_subset: pd.DataFrame, 
                     is_local_optimization: bool = False, current_component: str = None) -> Dict[str, Any]:
        try:
            self.sap_initializer.ensure_sap_embeddings_initialized(config, trial_dir)
            self.utilities.print_configuration(config)
            
            save_intermediate = config.get('save_intermediate_results', True)
            
            if not self.utilities.validate_global_optimization(config):
                return {
                    'score': 0.0,
                    'combined_score': 0.0,
                    'error': 'Missing required components for RAGAS evaluation'
                }
            
            if is_local_optimization and current_component:
                if hasattr(self, 'component_results'):
                    self.local_optimization_handler.set_component_results(self.component_results)
                return self.local_optimization_handler.handle_local_optimization_component(
                    current_component, config, trial_dir, qa_subset.copy(), 
                    qa_subset, save_intermediate
                )
            
            else:
                eval_df, pipeline_state = self.pipeline_orchestrator.execute_full_pipeline(
                    config, trial_dir, qa_subset, save_intermediate
                )
                
                results = self.pipeline_orchestrator.evaluate_pipeline_results(
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
            traceback.print_exc()
            return {
                'score': 0.0,
                'combined_score': 0.0,
                'retrieval_score': 0.0,
                'last_retrieval_component': 'none',
                'last_retrieval_score': 0.0,
                'error': str(e)
            }