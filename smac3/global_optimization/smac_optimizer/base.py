import os
import time
from typing import Dict, Any, Optional
import pandas as pd

from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_calculator import CombinationCalculator
from pipeline.pipeline_runner.rag_pipeline_runner import RAGPipelineRunner
from pipeline.utils import Utils
from smac3.global_optimization.config_space_builder import SMACConfigSpaceBuilder


class BaseOptimizer:
    
    def __init__(self):
        self.start_time = time.time()
        self.early_stopped_trials = []
        self.trial_counter = 0
        self.all_trials = []
    
    def _initialize_paths(self, project_dir: str, result_dir: Optional[str]):
        self.project_root = Utils.find_project_root()
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                f"SMAC_{self.optimizer}_results"
            )
        os.makedirs(self.result_dir, exist_ok=True)
    
    def _initialize_data(self, config_template: Dict[str, Any], 
                        qa_data: pd.DataFrame, corpus_data: pd.DataFrame):
        self.config_template = config_template
        self.qa_data = qa_data
        self.corpus_data = corpus_data
        self.total_samples = len(qa_data)
    
    def _initialize_optimization_params(self, n_trials, sample_percentage, cpu_per_trial,
                                retrieval_weight, generation_weight, use_cached_embeddings,
                                walltime_limit, n_workers, seed, early_stopping_threshold,
                                optimizer, use_multi_fidelity, min_budget_percentage,
                                max_budget_percentage, eta, use_ragas, ragas_config,
                                use_llm_compressor_evaluator, llm_evaluator_config,
                                disable_early_stopping, early_stopping_thresholds):
        self.n_trials = n_trials
        self.sample_percentage = sample_percentage
        self.cpu_per_trial = cpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.walltime_limit = walltime_limit
        self.n_workers = n_workers
        self.seed = seed
        self.early_stopping_threshold = early_stopping_threshold
        
        self.optimizer = optimizer.lower()
        assert self.optimizer in ["smac", "bohb"], f"optimizer must be 'smac' or 'bohb', got {optimizer}"
        
        self.use_multi_fidelity = use_multi_fidelity
        self.min_budget_percentage = min_budget_percentage
        self.max_budget_percentage = max_budget_percentage
        self.eta = eta
        
        self.min_budget = max(1, int(self.total_samples * self.min_budget_percentage))
        self.max_budget = max(self.min_budget, int(self.total_samples * self.max_budget_percentage))
        self.use_ragas = use_ragas  
        self.ragas_config = ragas_config  
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_config = llm_evaluator_config
        
        self.disable_early_stopping = disable_early_stopping
        if self.disable_early_stopping:
            self.early_stopping_thresholds = None
        else:
            self.early_stopping_thresholds = early_stopping_thresholds or {
                'retrieval': 0.1,
                'query_expansion': 0.1,
                'reranker': 0.2,
                'filter': 0.25,
                'compressor': 0.3
            }
        
    def _initialize_wandb_params(self, use_wandb, wandb_project, wandb_entity, wandb_run_name):
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
    
    def _setup_components(self):
        self.config_generator = ConfigGenerator(self.config_template)
        self.combination_calculator = CombinationCalculator(
            self.config_generator,
            search_type='bo'
        )
        self.config_space_builder = SMACConfigSpaceBuilder(self.config_generator, seed=self.seed)
        
        self._setup_runner()
    
    def _setup_runner(self):
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config('prompt_maker')
        self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()

        use_ragas = self.use_ragas
        
        ragas_config = self.ragas_config or {
            'retrieval_metrics': ["context_precision", "context_recall"],
            'generation_metrics': ["answer_relevancy", "faithfulness", "factual_correctness", "semantic_similarity"],
            'llm_model': "gpt-4o-mini",
            'embedding_model': "text-embedding-ada-002"
        }
        
        use_llm_evaluator = self.use_llm_compressor_evaluator
        llm_evaluator_config = self.llm_evaluator_config or {
            "llm_model": "gpt-4o",
            "temperature": 0.0
        }
        
        self.runner = RAGPipelineRunner(
            config_generator=self.config_generator,
            retrieval_metrics=self.retrieval_metrics,
            filter_metrics=self.filter_metrics,
            compressor_metrics=self.compressor_metrics,
            generation_metrics=self.generation_metrics,
            prompt_maker_metrics=self.prompt_maker_metrics,
            query_expansion_metrics=self.query_expansion_metrics,
            reranker_metrics=self.reranker_metrics,
            retrieval_weight=self.retrieval_weight,
            generation_weight=self.generation_weight,
            use_ragas=use_ragas,
            ragas_config=ragas_config,
            use_llm_evaluator=use_llm_evaluator,
            llm_evaluator_config=llm_evaluator_config,
            early_stopping_thresholds=self.early_stopping_thresholds
        )