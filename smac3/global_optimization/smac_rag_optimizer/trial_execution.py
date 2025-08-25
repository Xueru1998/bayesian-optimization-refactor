import os
import json
import time
import yaml
import shutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pipeline.pipeline_runner.rag_pipeline_runner import EarlyStoppingException
from pipeline.logging.wandb import WandBLogger


class TrialExecution:
    
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
        
        early_stopped = False
        early_stopped_component = None
        early_stopped_score = None
        
        try:
            config_dict = self._prepare_config(config)
            trial_config = self.config_generator.generate_trial_config(config_dict)
            self._save_trial_config(trial_dir, trial_config)
            
            try:
                results = self.runner.run_pipeline(config_dict, trial_dir, sampled_qa_data)
                score = results.get('combined_score', 0.0)
                
            except EarlyStoppingException as e:
                early_stopped = True
                early_stopped_component = e.component
                early_stopped_score = e.score
                score = e.score
                self.early_stopped_trials_count += 1
                
                print(f"\n[Trial {self.trial_counter}] Early stopped at {e.component} with score {e.score:.4f}")
                
                results = {
                    'combined_score': score,
                    'early_stopped_at': e.component,
                    'early_stopped_score': e.score,
                    'error': e.message
                }
            
            latency = time.time() - trial_start_time
            
            trial_result = self._create_trial_result(
                config_dict, score, latency, budget, budget_percentage, results
            )
            
            if early_stopped:
                trial_result['early_stopped'] = True
                trial_result['early_stopped_component'] = early_stopped_component
                trial_result['early_stopped_score'] = early_stopped_score
            
            self.all_trials.append(trial_result)
            
            self._save_trial_results(trial_dir, trial_result)
            self._print_trial_summary(score, latency, budget, budget_percentage, early_stopped, early_stopped_component)
            
            if self.use_wandb:
                self._log_trial_to_wandb(config_dict, trial_result)
            
            return {"score": -score, "latency": latency}
            
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
    
    def _print_trial_summary(self, score, latency, budget, budget_percentage, early_stopped=False, early_stopped_component=None):
        print(f"Trial {self.trial_counter} completed:")
        if early_stopped:
            print(f"  Status: EARLY STOPPED at {early_stopped_component}")
        print(f"  Score: {score:.4f}")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Budget: {budget} samples ({budget_percentage:.1%})")
    
    def _log_trial_to_wandb(self, config_dict, trial_result):
        log_data = {
            'trial': trial_result['trial_number'],
            'score': trial_result['score'],
            'latency': trial_result['latency'],
            'early_stopped': trial_result.get('early_stopped', False)
        }
        
        if trial_result.get('early_stopped', False):
            log_data['early_stopped_component'] = trial_result.get('early_stopped_component', 'unknown')
            log_data['early_stopped_score'] = trial_result.get('early_stopped_score', 0.0)
        
        WandBLogger.log_trial_metrics(
            self.trial_counter, 
            trial_result['score'],
            config=config_dict,
            results={**trial_result, **log_data}
        )
    
    def _convert_numpy_types(self, obj):
        if isinstance(obj, pd.DataFrame):
            return {"type": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(i) for i in obj)
        elif hasattr(obj, 'item'): 
            return obj.item()
        else:
            return obj
    
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