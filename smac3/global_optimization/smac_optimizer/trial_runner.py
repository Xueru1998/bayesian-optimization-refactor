import os
import json
import time
import yaml
import shutil
import numpy as np
import pandas as pd
from typing import Dict, Any
from pipeline.pipeline_runner.rag_pipeline_runner import EarlyStoppingException
from pipeline.logging.wandb import WandBLogger


class TrialRunner:
    
    def _run_trial(self, config: Dict[str, Any], budget: int, seed: int = 0) -> Dict[str, Any]:
        self.trial_counter += 1
        
        budget_percentage = budget / self.total_samples
        
        trial_dir = self._setup_trial_directory(budget)
        sampled_qa_data = self._sample_data(budget, seed)
        self._copy_corpus_data(trial_dir)
        
        trial_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Trial {self.trial_counter}/{self.n_trials} | Budget: {budget} samples ({budget_percentage:.1%})")
        print(f"{'='*60}")
        
        try:
            config_dict = self._prepare_config(config)
            trial_config = self.config_generator.generate_trial_config(config_dict)
            self._save_trial_config(trial_dir, trial_config)
            
            try:
                results = self.runner.run_pipeline(config_dict, trial_dir, sampled_qa_data)
                score = results.get('combined_score', 0.0)
                latency = time.time() - trial_start_time
                
                trial_result = self._create_trial_result(
                    config_dict, score, latency, budget, budget_percentage, results
                )
                trial_result['early_stopped'] = False
                self.all_trials.append(trial_result)
                
            except EarlyStoppingException as e:
                print(f"\n[TRIAL] Early stopped at {e.component} with score {e.score:.4f}")
                
                actual_score = e.score
                latency = time.time() - trial_start_time
                
                trial_result = self._create_trial_result(
                    config_dict, actual_score, latency, budget, budget_percentage, 
                    {
                        'early_stopped': True, 
                        'stopped_at': e.component, 
                        'stopped_score': e.score,
                        'combined_score': actual_score,
                        f'{e.component}_score': e.score,
                        'incomplete_pipeline': True
                    }
                )
                trial_result['early_stopped'] = True
                trial_result['stopped_at'] = e.component
                trial_result['stopped_score'] = e.score
                trial_result['actual_score'] = actual_score
                
                self.all_trials.append(trial_result)
                self.early_stopped_trials.append(trial_result)
            
            self._save_trial_results(trial_dir, trial_result)
            self._print_trial_summary(actual_score if 'actual_score' in locals() else score, 
                                    latency, budget, budget_percentage, trial_result)
            
            if self.use_wandb:
                self._log_trial_to_wandb(config_dict, trial_result)
            
            return {"score": -(actual_score if 'actual_score' in locals() else score), "latency": latency}
            
        except Exception as e:
            print(f"Error in trial {self.trial_counter}: {e}")
            import traceback
            traceback.print_exc()
            return {"score": 0.0, "latency": float('inf')}
    
    def _setup_trial_directory(self, budget: int) -> str:
        trial_dir = os.path.join(self.result_dir, f"trial_{self.trial_counter}_budget_{budget}")
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        return trial_dir
    
    def _sample_data(self, budget: int, seed: int) -> pd.DataFrame:
        actual_samples = min(budget, self.total_samples)
        if actual_samples < self.total_samples:
            return self.qa_data.sample(n=actual_samples, random_state=seed)
        return self.qa_data
    
    def _copy_corpus_data(self, trial_dir: str):
        centralized_corpus_path = os.path.join(self.project_dir, "data", "corpus.parquet")
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
    
    def _save_trial_config(self, trial_dir: str, trial_config: Dict[str, Any]):
        config_file = os.path.join(trial_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(trial_config, f, default_flow_style=False)
    
    def _create_trial_result(self, config_dict, score, latency, budget, budget_percentage, results):
        trial_result = {
            "trial_number": int(self.trial_counter), 
            "config": self._convert_numpy_types(config_dict),
            "score": float(score),
            "latency": float(latency),
            "budget": int(budget),
            "budget_percentage": budget_percentage,
            "retrieval_score": float(results.get('retrieval_score', 0.0)),
            "generation_score": float(results.get('generation_score', 0.0)),
            "combined_score": float(score),
            "timestamp": float(time.time()) 
        }
        
        for k, v in results.items():
            if k.endswith('_score'):
                trial_result[k] = float(v) if isinstance(v, (int, float, np.number)) else 0.0
            elif k.endswith('_metrics'):
                if isinstance(v, dict):
                    cleaned_metrics = {}
                    for metric_key, metric_value in v.items():
                        if not isinstance(metric_value, pd.DataFrame):
                            cleaned_metrics[metric_key] = self._convert_numpy_types(metric_value)
                    trial_result[k] = cleaned_metrics
                elif not isinstance(v, pd.DataFrame):
                    trial_result[k] = self._convert_numpy_types(v)
                    
        if 'config' in trial_result:
            for key in ['generator_temperature', 'query_expansion_temperature', 'temperature']:
                if key in trial_result['config']:
                    trial_result['config'][key] = float(trial_result['config'][key])
        
        return trial_result
    
    def _save_trial_results(self, trial_dir: str, trial_result: Dict[str, Any]):
        results_file = os.path.join(trial_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(trial_result, f, indent=2)
    
    def _print_trial_summary(self, score, latency, budget, budget_percentage, trial_result=None):
        print(f"Trial {self.trial_counter} completed:")
        print(f"  Score: {score:.4f}")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Budget: {budget} samples ({budget_percentage:.1%})")
        
        if trial_result and trial_result.get('early_stopped', False):
            print(f"  ⚠️  EARLY STOPPED at {trial_result.get('stopped_at', 'unknown')}")
            print(f"  ⚠️  Component score: {trial_result.get('stopped_score', 0.0):.4f}")
    
    def _log_trial_to_wandb(self, config_dict, trial_result):
        WandBLogger.log_trial_metrics(
            self.trial_counter, 
            trial_result['score'],
            config=config_dict,
            results=trial_result
        )
    
    def target_function_standard(self, config: Dict[str, Any], seed: int = 0) -> Dict[str, float]:
        config_dict = config.get_dictionary() if hasattr(config, 'get_dictionary') else dict(config)
        config_dict = self._validate_topk_constraints(config_dict)
        config_dict = self._ensure_conditional_parameters(config_dict, seed=seed)
        return self._run_trial(config_dict, self.max_budget, seed)
    
    def target_function_multifidelity(self, config: Dict[str, Any], seed: int = 0, budget: float = None) -> Dict[str, float]:
        config_dict = config.get_dictionary() if hasattr(config, 'get_dictionary') else dict(config)
        config_dict = self._validate_topk_constraints(config_dict)
        config_dict = self._ensure_conditional_parameters(config_dict, seed=seed)
        budget = int(budget) if budget is not None else self.max_budget
        return self._run_trial(config_dict, budget, seed)