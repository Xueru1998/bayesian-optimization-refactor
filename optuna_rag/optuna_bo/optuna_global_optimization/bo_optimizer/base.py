import os
import time
import yaml
from typing import Dict, Any, List, Optional
import pandas as pd

from pipeline.config_manager import ConfigGenerator
from pipeline.utils import Utils
from pipeline.search_space_calculator import SearchSpaceCalculator
from optuna_rag.config_extractor import OptunaConfigExtractor

from .initializer import PipelineInitializer
from .objective_handler import ObjectiveHandler
from .results_manager import ResultsManager
from .optimizer_core import OptimizerCore


class BOPipelineOptimizer:
    def __init__(
        self,
        config_path: str,
        qa_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        project_dir: str,
        n_trials: Optional[int] = None,
        sample_percentage: float = 0.1,
        cpu_per_trial: int = 4,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        continue_study: bool = False,
        use_wandb: bool = True,
        wandb_project: str = "BO & AutoRAG",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        optimizer: str = "tpe",
        early_stopping_threshold: float = 0.9,
        use_ragas: bool = False,
        ragas_llm_model: str = "gpt-4o-mini",
        ragas_embedding_model: str = "text-embedding-ada-002",
        ragas_metrics: Optional[Dict[str, List[str]]] = None,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o",
        component_early_stopping_enabled: bool = True,
        component_early_stopping_thresholds: Optional[Dict[str, float]] = None
    ):
        self.start_time = time.time()
        
        self.project_root = Utils.find_project_root()
        self.config_path = Utils.get_centralized_config_path(config_path)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        print(f"BO using config file: {self.config_path}")
        
        self.qa_df = qa_df
        self.corpus_df = corpus_df
        
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        print(f"BO using project directory: {self.project_dir}")
        
        with open(self.config_path, 'r') as f:
            self.config_template = yaml.safe_load(f)
        
        self.config_generator = ConfigGenerator(self.config_template)
        
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        
        if n_trials is None:
            suggestion = self.search_space_calculator.suggest_num_samples(
                sample_percentage=sample_percentage,
                min_samples=10,
                max_samples=50,
                max_combinations=500
            )
            self.n_trials = suggestion['num_samples']
            print(f"Use default trials: {self.n_trials}")
        else:
            self.n_trials = n_trials
            print(f"Using provided n_trials: {self.n_trials}")
        
        self.sample_percentage = sample_percentage
        self.cpu_per_trial = cpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.study_name = study_name if study_name else f"Optuna_rag_opt_{int(time.time())}"
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name or self.study_name
        self.optimizer = optimizer
        self.early_stopping_threshold = early_stopping_threshold

        self.component_early_stopping_enabled = component_early_stopping_enabled
        if component_early_stopping_thresholds is None:
            self.component_early_stopping_thresholds = {
                'retrieval': 0.1,
                'query_expansion': 0.1,
                'reranker': 0.2,
                'filter': 0.25,
                'compressor': 0.3
            }
        else:
            self.component_early_stopping_thresholds = component_early_stopping_thresholds
        
        self.use_ragas = use_ragas
        self.ragas_llm_model = ragas_llm_model
        self.ragas_embedding_model = ragas_embedding_model
        
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
        if ragas_metrics is None and use_ragas:
            self.ragas_metrics = {
                'retrieval': ['context_precision', 'context_recall'],
                'generation': ['answer_relevancy', 'faithfulness', 'factual_correctness', 'semantic_similarity']
            }
        else:
            self.ragas_metrics = ragas_metrics or {}
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
        
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.best_score = {"value": 0.0, "config": None, "latency": float('inf')}
        self.best_latency = {"value": float('inf'), "config": None, "score": 0.0}
        self.all_trials = []
        
        self._params_table_data = []
        
        self.initializer = PipelineInitializer(self)
        self.initializer.initialize_metrics()
        Utils.ensure_centralized_data(self.project_dir, self.corpus_df, self.qa_df)
        self.pipeline_runner = self.initializer.initialize_pipeline_runner()
        
        self.config_extractor = OptunaConfigExtractor(self.config_generator, search_type='bo')
        self.search_space = self.config_extractor.extract_search_space()
        
        self.initializer.print_initialization_summary()
        
        self.objective_handler = ObjectiveHandler(self)
        self.results_manager = ResultsManager(self)
        self.optimizer_core = OptimizerCore(self)
    
    def objective(self, trial):
        return self.objective_handler.objective(trial)
    
    def save_study_results(self, study):
        return self.results_manager.save_study_results(study)
    
    def optimize(self):
        return self.optimizer_core.optimize()