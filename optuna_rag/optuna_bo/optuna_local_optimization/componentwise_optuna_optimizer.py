from typing import Dict, Any, Optional
import pandas as pd

from .optimizers.componentwise_grid_search import ComponentwiseGridSearch
from .optimizers.componentwise_bayesian_optimization import ComponentwiseBayesianOptimization


class ComponentwiseOptunaOptimizer:
    
    def __init__(
        self,
        config_template: Dict[str, Any],
        qa_data: pd.DataFrame,
        corpus_data: pd.DataFrame,
        project_dir: str,
        n_trials_per_component: Optional[int] = None,
        sample_percentage: float = 0.1,
        cpu_per_trial: int = 4,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        walltime_limit_per_component: Optional[int] = None,
        n_workers: int = 1,
        seed: int = 42,
        early_stopping_threshold: float = 0.9,
        use_wandb: bool = True,
        wandb_project: str = "Component-wise Optuna Optimization",
        wandb_entity: Optional[str] = None,
        optimizer: str = "tpe",
        use_multi_objective: bool = False,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o",
        resume_study: bool = False,
    ):
        self.optimizer_type = optimizer.lower()
        
        optimizer_class = (
            ComponentwiseGridSearch 
            if self.optimizer_type in ['grid', 'grid_search'] 
            else ComponentwiseBayesianOptimization
        )
        
        self.optimizer_instance = optimizer_class(
            config_template=config_template,
            qa_data=qa_data,
            corpus_data=corpus_data,
            project_dir=project_dir,
            n_trials_per_component=n_trials_per_component,
            sample_percentage=sample_percentage,
            cpu_per_trial=cpu_per_trial,
            retrieval_weight=retrieval_weight,
            generation_weight=generation_weight,
            use_cached_embeddings=use_cached_embeddings,
            result_dir=result_dir,
            study_name=study_name,
            walltime_limit_per_component=walltime_limit_per_component,
            n_workers=n_workers,
            seed=seed,
            early_stopping_threshold=early_stopping_threshold,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            optimizer=optimizer,
            use_multi_objective=use_multi_objective,
            use_llm_compressor_evaluator=use_llm_compressor_evaluator,
            llm_evaluator_model=llm_evaluator_model,
            resume_study=resume_study
        )
    
    def optimize(self) -> Dict[str, Any]:
        return self.optimizer_instance.optimize()
    
    @property
    def best_configs(self):
        return self.optimizer_instance.best_configs
    
    @property
    def component_results(self):
        return self.optimizer_instance.component_results
    
    @property
    def result_dir(self):
        return self.optimizer_instance.result_dir
    
    @property
    def study_name(self):
        return self.optimizer_instance.study_name