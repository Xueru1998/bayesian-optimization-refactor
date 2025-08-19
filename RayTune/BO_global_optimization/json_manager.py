import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd

class JsonManager:
    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        self.explored_configs = []
        self.iteration_summary = {}
        os.makedirs(self.result_dir, exist_ok=True)
    
    def add_explored_config(self, config: Dict[str, Any], results: Dict[str, Any]) -> None:
        serializable_results = self.convert_to_json_serializable(results)

        if 'retrieval_metrics' in serializable_results:
            serializable_results['retrieval_metrics'] = {
                'summary': serializable_results['retrieval_metrics'].get('mean_accuracy', 0.0)
            }
        if 'query_expansion_metrics' in serializable_results:
            serializable_results['query_expansion_metrics'] = {
                'summary': serializable_results['query_expansion_metrics'].get('mean_accuracy', 0.0)
            }
        if 'reranker_metrics' in serializable_results:
            serializable_results['reranker_metrics'] = {
                'summary': serializable_results['reranker_metrics'].get('mean_accuracy', 0.0)
            }
        if 'filter_metrics' in serializable_results:
            serializable_results['filter_metrics'] = {
                'summary': serializable_results['filter_metrics'].get('mean_accuracy', 0.0)
            }
        if 'compression_metrics' in serializable_results:
            serializable_results['compression_metrics'] = {
                'summary': serializable_results['compression_metrics'].get('mean_score', 0.0)
            }
        if 'generation_metrics' in serializable_results:
            serializable_results['generation_metrics'] = {
                'summary': serializable_results['generation_metrics'].get('mean_score', 0.0)
            }
        if 'prompt_metrics' in serializable_results:
            serializable_results['prompt_metrics'] = {
                'summary': serializable_results['prompt_metrics'].get('mean_score', 0.0)
            }
        
        if 'trial_dir' in serializable_results:
            del serializable_results['trial_dir']

        self.explored_configs.append({
            "config": config,
            "results": serializable_results,
            "timestamp": time.time()
        })
        
        self.save_explored_configs()
        self.save_iteration_summary()
        
        print(f"Saved config #{len(self.explored_configs)} to explored_configs.json and iteration files")
    
    def save_explored_configs(self) -> None:
        if not self.explored_configs:
            return
            
        configs_file = os.path.join(self.result_dir, "explored_configs.json")
        self.save_to_json_file(self.explored_configs, configs_file)
        print(f"Saved {len(self.explored_configs)} explored configurations to {configs_file}")
    
    def save_iteration_summary(self, force_create: bool = False) -> None:
        print(f"\n===== Saving iteration summary =====")
        print(f"Result directory: {self.result_dir}")
        print(f"Explored configs count: {len(self.explored_configs)}")
        
        iterations_file = os.path.join(self.result_dir, "iteration_summary.json")
        stats_file = os.path.join(self.result_dir, "iteration_stats.json")

        if not self.explored_configs and not force_create:
            print("No explored configs to save - returning early")
            return
        
        try:
            if not os.path.exists(iterations_file) or force_create:
                self.save_to_json_file({}, iterations_file)
            
            if not os.path.exists(stats_file) or force_create:
                self.save_to_json_file({
                    "total_configs_explored": 0,
                    "unique_configs_tested": 0,
                    "top_configs": []
                }, stats_file)
                    
            if not self.explored_configs:
                print(f"✓ Ensured iteration files exist:")
                print(f"  - {iterations_file}")
                print(f"  - {stats_file}")
                return
            
            existing_summary = self.load_from_json_file(iterations_file, {})
            existing_stats = self.load_from_json_file(stats_file, {
                "total_configs_explored": 0,
                "unique_configs_tested": 0,
                "top_configs": []
            })
                    
            summary = existing_summary.copy()
            
            for trial in self.explored_configs:
                config_key = "_".join([f"{k}:{v}" for k, v in sorted(trial['config'].items())])
                
                if config_key not in summary:
                    summary[config_key] = {
                        "config": trial['config'],
                        "trials": [],
                        "best_score": 0.0
                    }
                
                trial_timestamps = [t.get('timestamp') for t in summary[config_key]["trials"]]
                if trial['timestamp'] in trial_timestamps:
                    continue
                    
                trial_data = {
                    "timestamp": trial['timestamp'],
                    "combined_score": trial['results'].get('combined_score', 0.0),
                    "last_retrieval_component": trial['results'].get('last_retrieval_component', 'unknown'),
                    "last_retrieval_score": trial['results'].get('last_retrieval_score', 0.0),
                    "training_iteration": trial['results'].get('training_iteration', 1)
                }
                
                if trial['results'].get('retrieval_score') is not None:
                    trial_data['retrieval_score'] = trial['results']['retrieval_score']
                if trial['results'].get('query_expansion_score') is not None:
                    trial_data['query_expansion_score'] = trial['results']['query_expansion_score']
                if trial['results'].get('filter_score') is not None:
                    trial_data['filter_score'] = trial['results']['filter_score']
                if trial['results'].get('compression_score') is not None:
                    trial_data['compression_score'] = trial['results']['compression_score']
                if trial['results'].get('reranker_score') is not None:
                    trial_data['reranker_score'] = trial['results']['reranker_score']
                if trial['results'].get('generation_score') is not None:
                    trial_data['generation_score'] = trial['results']['generation_score']
                if trial['results'].get('prompt_maker_score') is not None:
                    trial_data['prompt_maker_score'] = trial['results']['prompt_maker_score']
                    
                summary[config_key]["trials"].append(trial_data)
                
                if trial_data['combined_score'] > summary[config_key]['best_score']:
                    summary[config_key]['best_score'] = trial_data['combined_score']
            
            sorted_summary = dict(sorted(
                summary.items(), 
                key=lambda item: item[1]['best_score'], 
                reverse=True
            ))
            
            self.save_to_json_file(sorted_summary, iterations_file)
            self.iteration_summary = sorted_summary

            stats = {
                "total_configs_explored": existing_stats["total_configs_explored"] + 
                                        len([t for t in self.explored_configs if t['timestamp'] not in 
                                            [trial.get('timestamp') for config in existing_summary.values() 
                                            for trial in config.get('trials', [])]]),
                "unique_configs_tested": len(summary),
                "top_configs": []
            }

            for i, (config_key, config_data) in enumerate(sorted_summary.items()):
                if i >= 10:
                    break
                    
                stats["top_configs"].append({
                    "config": config_data['config'],
                    "best_score": config_data['best_score'],
                    "trials": len(config_data['trials'])
                })
            
            self.save_to_json_file(stats, stats_file)
                
            print(f"Successfully appended iteration data to:")
            print(f"  - {iterations_file}")
            print(f"  - {stats_file}")
        
        except Exception as e:
            print(f"ERROR saving iteration summary: {e}")
            import traceback
            traceback.print_exc()
    
    def save_metrics_as_csv(self, csv_filename: str = "trial_metrics.csv") -> None:
        if not self.explored_configs:
            return

        records = []
        for idx, trial in enumerate(self.explored_configs):
            record = {"trial_index": idx}

            for param_name, param_value in trial["config"].items():
                record[f"config_{param_name}"] = param_value

            for metric_name, metric_value in trial["results"].items():
                if isinstance(metric_value, (int, float, str, bool)) or metric_value is None:
                    record[metric_name] = metric_value
                
            records.append(record)

        metrics_df = pd.DataFrame(records)
        csv_path = os.path.join(self.result_dir, csv_filename)
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved metrics to CSV: {csv_path}")
    
    @staticmethod
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: JsonManager.convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [JsonManager.convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [JsonManager.convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'toJSON'):
            return obj.toJSON()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            try:
                return JsonManager.convert_to_json_serializable(obj.__dict__)
            except:
                return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)
    
    @staticmethod
    def save_to_json_file(data, file_path: str, indent: int = 2) -> bool:
        try:
            serializable_data = JsonManager.convert_to_json_serializable(data)
            with open(file_path, 'w') as f:
                json.dump(serializable_data, f, indent=indent)
            return True
        except Exception as e:
            print(f"Warning: Could not save to JSON file {file_path} - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def load_from_json_file(file_path: str, default_value: Any = None) -> Any:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load from JSON file {file_path} - {e}")
            return default_value