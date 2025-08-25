import time
from typing import Dict, Any, Optional
import pandas as pd
from pipeline.utils import Utils

from .base import BaseOptimizer
from .search_space_handler import SearchSpaceHandler
from .config_processor import ConfigProcessor
from .trial_runner import TrialRunner
from .optimizer_core import OptimizerCore
from .results_processor import ResultsProcessor
from .output_handler import OutputHandler


class SMACRAGOptimizer(
    BaseOptimizer,
    SearchSpaceHandler,
    ConfigProcessor,
    TrialRunner,
    OptimizerCore,
    ResultsProcessor,
    OutputHandler
):
    
    def __init__(
        self,
        config_template: Dict[str, Any],
        qa_data: pd.DataFrame,
        corpus_data: pd.DataFrame,
        project_dir: str,
        n_trials: Optional[int] = None,
        sample_percentage: float = 0.1,
        cpu_per_trial: int = 4,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        walltime_limit: int = 3600,
        n_workers: int = 1,
        seed: int = 42,
        early_stopping_threshold: float = 0.9,
        use_wandb: bool = True,
        wandb_project: str = "BO & AutoRAG",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        optimizer: str = "smac",
        use_multi_fidelity: bool = True,
        min_budget_percentage: float = 0.1,
        max_budget_percentage: float = 1.0,
        eta: int = 3,
        use_ragas: bool = False,
        ragas_config: Optional[Dict[str, Any]] = None,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_config: Optional[Dict[str, Any]] = None,
        disable_early_stopping: bool = False,
        early_stopping_thresholds: Optional[Dict[str, float]] = None
    ):
        print(f"[SMACRAGOptimizer] Initializing with:")
        print(f"  use_llm_compressor_evaluator: {use_llm_compressor_evaluator}")
        print(f"  llm_evaluator_config: {llm_evaluator_config}")
        
        super().__init__()
        
        self._initialize_paths(project_dir, result_dir)
        self._initialize_data(config_template, qa_data, corpus_data)
        self._initialize_optimization_params(
            n_trials, sample_percentage, cpu_per_trial, retrieval_weight,
            generation_weight, use_cached_embeddings, walltime_limit, n_workers,
            seed, early_stopping_threshold, optimizer, use_multi_fidelity,
            min_budget_percentage, max_budget_percentage, eta, use_ragas, ragas_config,
            use_llm_compressor_evaluator, llm_evaluator_config, disable_early_stopping,
            early_stopping_thresholds
        )
        self._initialize_wandb_params(use_wandb, wandb_project, wandb_entity, wandb_run_name)
        
        self.study_name = study_name if study_name else f"{optimizer}_opt_SAP_api_scifact_{int(time.time())}"
        
        self._setup_components()
        self._calculate_trials_if_needed()
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
        
        self.trial_counter = 0
        self.all_trials = []
        
        self._print_initialization_summary()