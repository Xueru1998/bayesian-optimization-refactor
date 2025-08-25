import optuna
import tempfile
import os
import shutil
from typing import Dict, Any

from .params_suggester import ParameterSuggester
from .params_validator import ParameterValidator
from .trial_executor import TrialExecutor


class OptunaObjective:
    def __init__(self, search_space, config_generator, pipeline_runner, 
                 corpus_df, qa_df, disable_early_stopping=False):
        self.search_space = search_space
        self.config_generator = config_generator
        self.pipeline_runner = pipeline_runner
        self.corpus_df = corpus_df
        self.qa_df = qa_df
        self.disable_early_stopping = disable_early_stopping
        
        self.has_query_expansion = self.config_generator.node_exists("query_expansion")
        self.has_retrieval = self.config_generator.node_exists("retrieval")
        
        self.param_suggester = ParameterSuggester(
            search_space=search_space,
            config_generator=config_generator,
            has_query_expansion=self.has_query_expansion,
            has_retrieval=self.has_retrieval
        )
        
        self.param_validator = ParameterValidator(search_space)
        self.valid_param_combinations = self.param_validator.precompute_valid_combinations()
        
        self.trial_executor = TrialExecutor(
            pipeline_runner=pipeline_runner,
            corpus_df=corpus_df,
            qa_df=qa_df,
            disable_early_stopping=disable_early_stopping
        )
    
    def __call__(self, trial: optuna.Trial) -> float:
        params = self.param_suggester.suggest_params(trial)
        
        print(f"\nRunning trial {trial.number} with params: {params}")
        
        trial_dir = tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_")
        
        try:
            score, execution_time = self.trial_executor.run_trial(trial, params, trial_dir)
            
            trial.set_user_attr('execution_time', execution_time)
            
            return score
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            if os.path.exists(trial_dir):
                try:
                    shutil.rmtree(trial_dir)
                except:
                    pass