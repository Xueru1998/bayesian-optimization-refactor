import time
from typing import Dict, Any, Optional
import pandas as pd
from pipeline.utils import Utils

from .initialization_and_setup import InitializationAndSetup
from .config_processing import ConfigProcessing
from .trial_execution import TrialExecution
from .optimization_core import OptimizationCore


class SMACRAGOptimizer(
    InitializationAndSetup,
    ConfigProcessing,
    TrialExecution,
    OptimizationCore
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
        llm_evaluator_model: str = "gpt-4o",
        component_early_stopping_enabled: bool = True,
        component_early_stopping_thresholds: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
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
        
        self._initialize_paths(project_dir, result_dir)
        self._initialize_data(config_template, qa_data, corpus_data)
        self._initialize_optimization_params(
            n_trials, sample_percentage, cpu_per_trial, retrieval_weight,
            generation_weight, use_cached_embeddings, walltime_limit, n_workers,
            seed, early_stopping_threshold, optimizer, use_multi_fidelity,
            min_budget_percentage, max_budget_percentage, eta, use_ragas, ragas_config 
        )
        self._initialize_wandb_params(use_wandb, wandb_project, wandb_entity, wandb_run_name)
        
        self.study_name = study_name if study_name else f"{optimizer}_opt_{int(time.time())}"
        
        self._setup_components()
        self._calculate_trials_if_needed()
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
        
        self._print_initialization_summary()