import os
import shutil
import pandas as pd
import time
import json
import numpy as np
from typing import Dict, Any, List, Union, Optional

import yaml

from pipeline_component.nodes.generator import create_generator


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


class Utils:
    PROJECT_MARKERS = [
        'config.yaml',
        'autorag_project',
        '.git',
        'requirements.txt',
        'pipeline'
    ]
    
    @staticmethod
    def ensure_list(value):
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]
    
    @staticmethod
    def add_to_result_list(values, result_list):
        values_list = Utils.ensure_list(values)
        for value in values_list:
            if value not in result_list:
                result_list.append(value)
    
    @staticmethod
    def save_dataframe(output_dir: str, file_name: str, data: Union[List[Dict[str, Any]], pd.DataFrame], 
                      additional_columns: Dict[str, Any] = None, format: str = 'csv'):
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        if additional_columns:
            for key, value in additional_columns.items():
                df[key] = value

        file_path = os.path.join(output_dir, file_name)
        
        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
            
        print(f"Results saved to: {file_path}")

    @staticmethod
    def save_results_to_csv(output_dir: str, file_name: str, data: Union[List[Dict[str, Any]], pd.DataFrame], 
                           additional_columns: Dict[str, Any] = None):
        Utils.save_dataframe(output_dir, file_name, data, additional_columns, format='csv')

    @staticmethod
    def save_results_to_parquet(output_dir: str, file_name: str, data: pd.DataFrame):
        Utils.save_dataframe(output_dir, file_name, data, format='parquet')

    @staticmethod
    def ensure_evaluation_results_format(evaluation_results: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(evaluation_results, dict):
            return [evaluation_results]
        return evaluation_results

    @staticmethod
    def create_output_directory(base_dir: str, node_name: str) -> str:
        output_dir = os.path.join(base_dir, node_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def record_execution_time(start_time: float) -> float:
        return time.time() - start_time

    @staticmethod
    def save_json(filepath: str, data: dict):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
            
    @staticmethod
    def save_results_to_json(output_dir: str, file_name: str, data: Dict[str, Any]):
        file_path = os.path.join(output_dir, file_name)
        Utils.save_json(file_path, data)
        print(f"JSON results saved to: {file_path}")

    @staticmethod
    def find_project_root(start_path: str = None) -> str:
        project_root_env = os.environ.get('PROJECT_ROOT')
        if project_root_env and os.path.exists(project_root_env):
            print(f"Using PROJECT_ROOT from environment: {project_root_env}")
            return project_root_env
        
        if start_path is None:
            import inspect
            frame = inspect.currentframe().f_back
            start_path = os.path.dirname(os.path.abspath(frame.f_globals['__file__']))
        
        project_root = Utils._search_for_markers(start_path)
        if project_root:
            return project_root
            
        cwd = os.getcwd()
        if Utils._has_project_markers(cwd):
            print(f"Using current working directory as project root: {cwd}")
            return cwd
        
        workspace_root = '/workspace'
        if os.path.exists(workspace_root) and Utils._has_project_markers(workspace_root):
            print(f"Using workspace directory as project root: {workspace_root}")
            return workspace_root
        
        print(f"Could not find project root with markers, using current directory: {cwd}")
        return cwd
    
    @staticmethod
    def _search_for_markers(start_path: str, max_levels: int = 10) -> Optional[str]:
        current_dir = start_path
        
        for _ in range(max_levels):
            if Utils._has_project_markers(current_dir):
                return current_dir
                
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
                
            current_dir = parent
        
        return None
    
    @staticmethod
    def _has_project_markers(directory: str) -> bool:
        return any(os.path.exists(os.path.join(directory, marker)) for marker in Utils.PROJECT_MARKERS)
    
    @staticmethod
    def get_centralized_config_path(config_path: str = "config.yaml") -> str:
        project_root = Utils.find_project_root()
        
        if config_path == "config.yaml" or not os.path.isabs(config_path):
            return os.path.join(project_root, config_path)
        return config_path
    
    @staticmethod
    def get_centralized_project_dir(project_dir: str = "autorag_project") -> str:
        if project_dir == "autorag_project" or not os.path.isabs(project_dir):
            project_root = Utils.find_project_root()
            return os.path.join(project_root, "autorag_project")
        return project_dir
    
    @staticmethod
    def get_centralized_data_paths(project_dir: str = None):
        if project_dir is None:
            project_dir = Utils.get_centralized_project_dir()

        data_files = ["qa_validation.parquet", "corpus.parquet"]
        paths = []
        
        for filename in data_files:
            primary_path = os.path.join(project_dir, "data", filename)
            fallback_path = os.path.join(project_dir, filename)
            paths.append(primary_path if os.path.exists(primary_path) else fallback_path)
            
        return tuple(paths)
    
    @staticmethod
    def json_serializable(obj):
        try:
            if isinstance(obj, (np.integer, np.floating, np.ndarray)):
                return NumpyEncoder().default(obj)
            elif isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            elif hasattr(obj, 'to_json'):
                return obj.to_json()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: Utils.json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [Utils.json_serializable(item) for item in obj]
            else:
                return str(obj)
        except:
            return str(obj)
    
    @staticmethod
    def ensure_centralized_data(project_dir: str, corpus_df: pd.DataFrame = None, 
                               qa_df: pd.DataFrame = None) -> Dict[str, str]:
        os.makedirs(os.path.join(project_dir, "resources"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
        
        paths = {}
        data_config = [
            ('corpus', corpus_df, os.path.join(project_dir, "data", "corpus.parquet")),
            ('qa', qa_df, os.path.join(project_dir, "data", "qa_validation.parquet"))
        ]
        
        for key, df, file_path in data_config:
            if df is not None and not os.path.exists(file_path):
                df.to_parquet(file_path)
                print(f"Saved {key} to: {file_path}")
            else:
                print(f"Using existing {key} at: {file_path}")
            paths[key] = file_path
            
        return paths
    
    @staticmethod
    def find_pareto_front(trials: List[Dict[str, Any]], 
                         score_key: str = "score", 
                         latency_key: str = "latency") -> List[Dict[str, Any]]:
        if not trials:
            return []
            
        valid_trials = [t for t in trials if t[score_key] > 0 and t[latency_key] < float('inf')]
        if not valid_trials:
            return []
            
        pareto_indices = []
        for i, trial_i in enumerate(valid_trials):
            is_dominated = any(
                Utils._dominates(valid_trials[j], trial_i, score_key, latency_key)
                for j in range(len(valid_trials)) if j != i
            )
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return [valid_trials[i] for i in pareto_indices]
    
    @staticmethod
    def _dominates(trial_a: Dict[str, Any], trial_b: Dict[str, Any], 
                   score_key: str, latency_key: str) -> bool:
        score_a, latency_a = -trial_a[score_key], trial_a[latency_key]
        score_b, latency_b = -trial_b[score_key], trial_b[latency_key]
        
        return ((score_a <= score_b and latency_a <= latency_b) and 
                (score_a < score_b or latency_a < latency_b))
    
    @staticmethod
    def format_time_duration(total_seconds: float) -> str:
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    @staticmethod
    def save_optimization_results(result_dir: str, all_trials: List[Dict[str, Any]], 
                                best_score: Dict[str, Any], best_latency: Dict[str, Any],
                                additional_data: Optional[Dict[str, Any]] = None):
        os.makedirs(result_dir, exist_ok=True)
        
        Utils.save_json(os.path.join(result_dir, "all_trials.json"), all_trials)
        
        pareto_front = Utils.find_pareto_front(all_trials)
        
        best_trials = {
            "best_score": best_score,
            "best_latency": best_latency,
            "pareto_front": pareto_front
        }
        
        if additional_data:
            best_trials.update(additional_data)
            
        Utils.save_json(os.path.join(result_dir, "best_trials.json"), best_trials)
        
        csv_result = Utils._save_trials_as_csv(result_dir, all_trials)
        
        return {"pareto_front": pareto_front, **csv_result}
    
    @staticmethod
    def _save_trials_as_csv(result_dir: str, all_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            df_records = []
            metric_names = ["retrieval_score", "reranker_score", "filter_score", 
                           "compression_score", "prompt_maker_score", "generation_score", 
                           "combined_score"]
            
            for trial in all_trials:
                record = {
                    "trial_number": trial.get("trial_number", 0), 
                    "score": trial.get("score", 0), 
                    "latency": trial.get("latency", float('inf'))
                }
                
                for param_name, param_value in trial.get("config", {}).items():
                    record[f"config_{param_name}"] = param_value
                
                for metric_name in metric_names:
                    if metric_name in trial:
                        record[metric_name] = trial[metric_name]
                        
                df_records.append(record)
                
            df = pd.DataFrame(df_records)
            csv_path = os.path.join(result_dir, "all_trials.csv")
            df.to_csv(csv_path, index=False)
            
            return {"csv_path": csv_path}
            
        except Exception as e:
            print(f"Warning: Error saving CSV: {e}")
            return {}        
  

    
    @staticmethod
    def update_dataframe_columns(target_df: pd.DataFrame, source_df: pd.DataFrame, 
                               include_cols: List[str] = None, exclude_cols: List[str] = None):
        if include_cols:
            cols_to_update = [col for col in include_cols if col in source_df.columns]
        else:
            cols_to_update = source_df.columns.tolist()
            
        if exclude_cols:
            cols_to_update = [col for col in cols_to_update if col not in exclude_cols]
        
        for col in cols_to_update:
            target_df[col] = source_df[col]
    
    @staticmethod
    def find_generator_config(config_generator, node_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        node_config = config_generator.extract_node_config(node_name)
        if not node_config:
            return None
            
        modules_key = 'modules' if node_name == 'generator' else 'strategy'
        if modules_key not in node_config:
            return None
            
        modules_list = node_config[modules_key]
        if node_name == 'prompt_maker' and 'generator_modules' in modules_list:
            modules_list = modules_list['generator_modules']
            
        for module in modules_list:
            module_llm = module.get('llm', '')
            module_model = module.get('model', '')
            
            if (model_name == module_llm or 
                model_name == module_model or
                model_name in str(module_llm) or
                model_name in str(module_model) or
                (isinstance(module_llm, str) and module_llm in model_name)):
                return module
                
        return None
    
    @staticmethod
    def create_generator_from_config(model_name: str, generator_config: Optional[Dict[str, Any]], 
                                module_type: str = None) -> Any:
        
        if not generator_config:
            return create_generator(model=model_name)
        
        generator_kwargs = {
            'model': model_name,
            'batch_size': generator_config.get('batch', 8)
        }
        
        provider_mapping = {
            'vllm': 'vllm',
            'openai': 'openai',
            'openai_llm': 'openai',
            'llama_index': 'llama_index',
            'llama_index_llm': 'llama_index',
            'sap_api': 'sap_api',
        }
        
        provider = provider_mapping.get(module_type)
        if provider:
            generator_kwargs['provider'] = provider
    
        if module_type == 'sap_api':
            generator_kwargs['api_url'] = generator_config.get('api_url')
            generator_kwargs['bearer_token'] = generator_config.get('bearer_token')
            generator_kwargs['llm'] = generator_config.get('llm', 'mistralai')
            generator_kwargs['max_tokens'] = generator_config.get('max_tokens', 500)
            
        elif module_type == 'vllm':
            for key in ['tensor_parallel_size', 'gpu_memory_utilization']:
                if key in generator_config:
                    generator_kwargs[key] = generator_config[key]
                    
        elif module_type in ['llama_index', 'llama_index_llm']:
            generator_kwargs['llm'] = generator_config.get('llm', 'openai')
            
        
        try:
            return create_generator(**generator_kwargs)
        except Exception as e:
            print(f"Failed to create generator with config, falling back to default: {e}")
            return create_generator(model=model_name)

    
    @staticmethod
    def get_temperature_from_config(config: Dict[str, Any], generator_config: Optional[Dict[str, Any]], 
                                  temp_key: str = 'temperature') -> float:
        temperature = config.get(temp_key, 0.7)
        
        if generator_config and 'temperature' in generator_config:
            temp_values = generator_config['temperature']
            if isinstance(temp_values, list):
                temperature = temp_values[0] 
            else:
                temperature = temp_values
                
        return round(float(temperature), 4)
    
    @staticmethod
    def detect_module_type(config: Dict[str, Any], generator_config: Optional[Dict[str, Any]], 
                        model_name: str) -> str:
        module_type_mapping = {
            'openai_llm': 'openai',
            'llama_index_llm': 'llama_index',
            'vllm': 'vllm',
            'sap_api': 'sap_api',
        }

        if config.get('generator_module_type'):
            raw_module_type = config['generator_module_type']
            return module_type_mapping.get(raw_module_type, raw_module_type)

        elif generator_config and generator_config.get('module_type'):
            detected_module_type = generator_config.get('module_type')
            return module_type_mapping.get(detected_module_type, detected_module_type)

        else:
            if any(x in model_name.lower() for x in ['gpt', 'o1-mini', 'o1-preview']):
                return 'openai'

            elif 'mistralai-large-instruct' in model_name:
                return 'sap_api'
 
            elif any(x in model_name.lower() for x in ['mistral', 'llama', 'qwen', 'mixtral', 'yi', 'gemma']):
                return 'vllm'

            else:
                return 'llama_index'
            
    @staticmethod
    def convert_numpy_types(obj):
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
                converted = Utils.convert_numpy_types(v)
                if converted is not None:
                    result[k] = converted
            return result
        elif isinstance(obj, list):
            result = []
            for item in obj:
                converted = Utils.convert_numpy_types(item)
                if converted is not None:
                    result.append(converted)
            return result
        else:
            return obj
        
    @staticmethod
    def save_component_optimization_results(result_dir: str, results: Dict[str, Any], 
                                           config_generator=None):
        summary_file = os.path.join(result_dir, "component_optimization_summary.json")
        results_serializable = Utils.convert_numpy_types(results)
        
        with open(summary_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        if config_generator and not results.get('validation_failed', False):
            final_config = {}
            for component in results.get('component_order', []):
                if component in results.get('best_configs', {}) and results['best_configs'][component]:
                    final_config.update(results['best_configs'][component])
            
            if final_config:
                final_config_file = os.path.join(result_dir, "final_best_config.yaml")
                final_config_serializable = Utils.convert_numpy_types(final_config)
                with open(final_config_file, 'w') as f:
                    yaml.dump(config_generator.generate_trial_config(final_config_serializable), f)
                    
    @staticmethod
    def find_best_trial_from_component(component_trials: List[Dict], 
                                    component: str = None) -> Optional[Dict]:
        if not component_trials:
            return None
        
        valid_trials = []
        for trial in component_trials:
            if 'status' in trial:
                if trial.get('status') != 'FAILED':
                    valid_trials.append(trial)
            else:
                valid_trials.append(trial)
        
        if not valid_trials:
            return None
        
        score_groups = {}
        for trial in valid_trials:
            score = trial.get('score', 0.0)
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(trial)
        
        if not score_groups:
            return None
        
        max_score = max(score_groups.keys())
        trials_with_max_score = score_groups[max_score]
        
        if len(trials_with_max_score) > 1:
            if component:
                print(f"[{component}] Found {len(trials_with_max_score)} trials with score {max_score:.4f}, selecting by latency")
            best_trial = min(trials_with_max_score, key=lambda t: t.get('latency', float('inf')))
            if component:
                print(f"[{component}] Selected trial {best_trial.get('trial_number', 'unknown')} with latency {best_trial.get('latency', 0.0):.2f}s")
        else:
            best_trial = trials_with_max_score[0]
            if component:
                print(f"[{component}] Selected trial {best_trial.get('trial_number', 'unknown')} with score {max_score:.4f} and latency {best_trial.get('latency', 0.0):.2f}s")

        if component:
            print(f"[{component}] Best config: {best_trial.get('config', {})}")

        return best_trial