import optuna
import time
from typing import Dict, Any, Tuple
from pipeline_component.nodes.retrieval import RetrievalModule


class TrialExecutor:
    def __init__(self, pipeline_runner, corpus_df, qa_df, disable_early_stopping=False):
        self.pipeline_runner = pipeline_runner
        self.corpus_df = corpus_df
        self.qa_df = qa_df
        self.disable_early_stopping = disable_early_stopping
    
    def run_trial(self, trial: optuna.Trial, params: Dict[str, Any], trial_dir: str) -> Tuple[float, float]:
        trial_start_time = time.time()
        
        retrieval_module = RetrievalModule(
            base_project_dir=trial_dir,
            use_pregenerated_embeddings=True,
            centralized_project_dir=self.pipeline_runner._get_centralized_project_dir()
        )
        retrieval_module.prepare_project_dir(trial_dir, self.corpus_df, self.qa_df)
        
        if self.disable_early_stopping:
            original_thresholds = self.pipeline_runner.early_stopping_handler.early_stopping_thresholds
            self.pipeline_runner.early_stopping_handler.early_stopping_thresholds = None
            
        results = self.pipeline_runner.run_pipeline(params, trial_dir, self.qa_df)
        
        if self.disable_early_stopping:
            self.pipeline_runner.early_stopping_handler.early_stopping_thresholds = original_thresholds
        
        execution_time = time.time() - trial_start_time
        
        score = results.get("combined_score", 0.0)
        
        for key, value in results.items():
            if key not in ['config', 'trial_dir', 'timestamp', 'training_iteration', 
                        'iteration_scores', 'iteration_combined_scores', 'weighted_score', 
                        'weighted_combined_score', 'score', 'error']:
                trial.set_user_attr(key, value)
        
        return score, execution_time