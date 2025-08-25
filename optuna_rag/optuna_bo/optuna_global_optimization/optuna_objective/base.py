import optuna
import tempfile
import os
import shutil
from typing import Dict, Any

from .params_suggester import ParameterSuggester
from .params_validator import ParameterValidator
from .trial_executor import TrialExecutor
from .config_processor import ConfigProcessor


class OptunaObjective:
    def __init__(self, search_space, config_generator, pipeline_runner, 
                 corpus_df, qa_df, result_dir=None, save_intermediate_results=True):
        self.search_space = search_space
        self.config_generator = config_generator
        self.pipeline_runner = pipeline_runner
        self.corpus_df = corpus_df
        self.qa_df = qa_df
        self.result_dir = result_dir or "optuna_results"
        self.save_intermediate_results = save_intermediate_results
        
        self.has_query_expansion = self.config_generator.node_exists("query_expansion")
        self.has_retrieval = self.config_generator.node_exists("retrieval")
        
        self.param_suggester = ParameterSuggester(
            search_space=search_space,
            config_generator=config_generator,
            has_query_expansion=self.has_query_expansion,
            has_retrieval=self.has_retrieval
        )
        
        self.param_validator = ParameterValidator(search_space)
        self.trial_executor = TrialExecutor(
            pipeline_runner=pipeline_runner,
            corpus_df=corpus_df,
            qa_df=qa_df,
            config_generator=config_generator,
            search_space=search_space
        )
        self.config_processor = ConfigProcessor(config_generator)
    
    def __call__(self, trial: optuna.Trial) -> float:
        params = self.param_suggester.suggest_params(trial)
        
        self._clean_params(params)
        
        params['save_intermediate_results'] = self.save_intermediate_results
        
        print(f"\nRunning trial {trial.number} with params: {params}")
        
        if self.result_dir:
            trial_dir = os.path.join(self.result_dir, f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        else:
            trial_dir = tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_")
        
        try:
            score = self.trial_executor.run_trial(trial, params, trial_dir)
            
            if self.save_intermediate_results and os.path.exists(trial_dir):
                debug_dir = os.path.join(trial_dir, "debug_intermediate_results")
                if os.path.exists(debug_dir):
                    print(f"[DEBUG] Intermediate results saved in: {debug_dir}")
            
            return score
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            if not self.result_dir and os.path.exists(trial_dir):
                try:
                    shutil.rmtree(trial_dir)
                except:
                    pass
    
    def _clean_params(self, params: Dict[str, Any]):
        params_to_remove = [
            'compressor_bearer_token',
            'generator_bearer_token',
            'query_expansion_bearer_token',
            'generator_config',
            'passage_compressor_config',
            'query_expansion_config'
        ]
        
        for param in params_to_remove:
            params.pop(param, None)
        
        if params.get('passage_compressor_method') in ['refine', 'tree_summarize']:
            params.pop('compressor_temperature', None)
            params.pop('compressor_max_tokens', None)