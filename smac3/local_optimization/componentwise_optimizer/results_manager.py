import os
import json
import yaml
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from ConfigSpace import ConfigurationSpace

from pipeline.utils import Utils


class ResultsManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def create_component_result(self, component: str, best_trial: Optional[Dict[str, Any]], 
                           best_config: Dict[str, Any], fixed_config: Dict[str, Any],
                           cs: ConfigurationSpace, start_time: float) -> Dict[str, Any]:
        if not best_trial and not self.optimizer.component_trials:
            print(f"[WARNING] No successful trials for component {component}")
            return {
                'component': component,
                'best_config': {},
                'best_score': 0.0,
                'best_trial': None,
                'all_trials': [],
                'n_trials': 0,
                'fixed_config': fixed_config,
                'search_space_size': len(cs.get_hyperparameters()),
                'optimization_time': time.time() - start_time,
                'detailed_metrics': [],
                'optimization_method': 'smac'  
            }

        return {
            'component': component,
            'best_config': best_trial['config'] if best_trial else best_config,
            'best_score': best_trial['score'] if best_trial else 0.0,
            'best_latency': best_trial.get('latency', 0.0) if best_trial else 0.0,
            'best_trial': best_trial,
            'all_trials': self.optimizer.component_trials,
            'n_trials': len(self.optimizer.component_trials),
            'fixed_config': fixed_config,
            'search_space_size': len(cs.get_hyperparameters()),
            'optimization_time': time.time() - start_time,
            'detailed_metrics': self.optimizer.component_detailed_metrics.get(component, []),
            'best_output_path': best_trial.get('output_parquet') if best_trial else None,
            'optimization_method': 'smac' 
        }
    
    def save_component_result(self, component_dir: str, result: Dict[str, Any]):
        result_for_json = result.copy()

        if 'best_trial' in result_for_json and result_for_json['best_trial']:
            if 'results' in result_for_json['best_trial']:
                trial_results = result_for_json['best_trial']['results'].copy()
                keys_to_remove = [k for k, v in trial_results.items() if isinstance(v, pd.DataFrame)]
                for key in keys_to_remove:
                    trial_results.pop(key, None)
                result_for_json['best_trial']['results'] = trial_results
                
        if 'all_trials' in result_for_json:
            cleaned_trials = []
            for trial in result_for_json['all_trials']:
                trial_copy = trial.copy()
                if 'results' in trial_copy:
                    trial_results = trial_copy['results'].copy()
                    keys_to_remove = [k for k, v in trial_results.items() if isinstance(v, pd.DataFrame)]
                    for key in keys_to_remove:
                        trial_results.pop(key, None)
                    trial_copy['results'] = trial_results
                cleaned_trials.append(trial_copy)
            result_for_json['all_trials'] = cleaned_trials

        result_serializable = self.convert_numpy_types(result_for_json)

        with open(os.path.join(component_dir, "optimization_result.json"), 'w') as f:
            json.dump(result_serializable, f, indent=2)
    
    def save_trial_results(self, trial_dir: str, results: Dict[str, Any], 
                          cleaned_config: Dict[str, Any], component: str, latency: float):
        results['trial_number'] = self.optimizer.current_trial
        results['time_taken'] = latency
        results['config'] = cleaned_config
        results['component'] = component
        
        results_for_json = results.copy()

        keys_to_remove = []
        for key, value in results_for_json.items():
            if isinstance(value, pd.DataFrame):
                keys_to_remove.append(key)
            elif isinstance(value, dict):
                nested_keys_to_remove = []
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, pd.DataFrame):
                        nested_keys_to_remove.append(nested_key)
                for nested_key in nested_keys_to_remove:
                    value.pop(nested_key, None)

        for key in keys_to_remove:
            results_for_json.pop(key, None)
        
        results_serializable = self.convert_numpy_types(results_for_json)
        
        with open(os.path.join(trial_dir, "results.json"), 'w') as f:
            json.dump(results_serializable, f, indent=2)
    
    def save_error_results(self, trial_dir: str, cleaned_config: Dict[str, Any], 
                          component: str, error: Exception):
        import traceback
        
        error_results = {
            'trial_number': self.optimizer.current_trial,
            'config': cleaned_config,
            'component': component,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        
        error_results_serializable = self.convert_numpy_types(error_results)
        
        with open(os.path.join(trial_dir, "error.json"), 'w') as f:
            json.dump(error_results_serializable, f, indent=2)
    
    def save_final_results(self, results: Dict[str, Any]):
        summary_file = os.path.join(self.optimizer.result_dir, "component_optimization_summary.json")

        results_serializable = self.convert_numpy_types(results)
        
        with open(summary_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        if not results.get('validation_failed', False):
            final_config = {}
            for component in results['component_order']:
                if component in results['best_configs'] and results['best_configs'][component]:
                    final_config.update(results['best_configs'][component])
            
            if final_config:
                final_config_file = os.path.join(self.optimizer.result_dir, "final_best_config.yaml")
                final_config_serializable = self.convert_numpy_types(final_config)
                with open(final_config_file, 'w') as f:
                    yaml.dump(self.optimizer.config_generator.generate_trial_config(final_config_serializable), f)
            else:
                print("Warning: No successful configurations found, skipping final config generation")
    
    def print_final_summary(self, results: Dict[str, Any]):
        print(f"\nTotal optimization time: {Utils.format_time_duration(results['optimization_time'])}")
        
        if results.get('validation_failed', False):
            print("\nOptimization failed due to insufficient search space.")
        else:
            for component in results['component_order']:
                comp_result = results['component_results'][component]
                print(f"\n{component.upper()}:")
                print(f"  Best score: {comp_result['best_score']:.4f}")
                print(f"  Best config: {comp_result['best_config']}")
                print(f"  Trials run: {comp_result['n_trials']}")
    
    def convert_numpy_types(self, obj):    
        if isinstance(obj, pd.DataFrame):
            return None  
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                converted = self.convert_numpy_types(v)
                if converted is not None: 
                    result[k] = converted
            return result
        elif isinstance(obj, list):
            result = []
            for item in obj:
                converted = self.convert_numpy_types(item)
                if converted is not None:  
                    result.append(converted)
            return result
        else:
            return obj