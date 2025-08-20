import os
import shutil
import pandas as pd
import time
import json
import numpy as np
from typing import Dict, Any, List, Union, Optional

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
                print(f"Saved {key} to location: {file_path}")
            else:
                print(f"Using existing {key} at location: {file_path}")
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
    def get_storage_type(storage_name: str) -> str:
        db_keywords = ['postgresql', 'mysql']
        return "cloud_database" if any(kw in storage_name.lower() for kw in db_keywords) else "local_sqlite"
        
    @staticmethod
    def copy_data_to_trial_dir(centralized_project_dir: str, trial_dir: str):
        src_data_dir = os.path.join(centralized_project_dir, "data")
        dst_data_dir = os.path.join(trial_dir, "data")
        os.makedirs(dst_data_dir, exist_ok=True)

        for fname in ["corpus.parquet", "qa_validation.parquet"]:
            src_file = os.path.join(src_data_dir, fname)
            dst_file = os.path.join(dst_data_dir, fname)
            if os.path.exists(src_file):
                shutil.copyfile(src_file, dst_file)
    
    @staticmethod
    def populate_retrieved_contents(df: pd.DataFrame, trial_dir: str) -> pd.DataFrame:
        try:
            corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
            if not os.path.exists(corpus_path):
                corpus_path = os.path.join(Utils.get_centralized_project_dir(), "data", "corpus.parquet")
            
            if not os.path.exists(corpus_path):
                print(f"[WARNING] Corpus file not found at {corpus_path}")
                return df
            
            corpus_df = pd.read_parquet(corpus_path)

            if 'doc_id' in corpus_df.columns and 'contents' in corpus_df.columns:
                id_to_content = dict(zip(corpus_df['doc_id'], corpus_df['contents']))
            else:
                print(f"[WARNING] Corpus missing required columns. Found: {corpus_df.columns.tolist()}")
                return df

            if 'retrieved_ids' in df.columns:
                def get_contents_from_ids(ids):
                    if isinstance(ids, list):
                        return [id_to_content.get(doc_id, "") for doc_id in ids]
                    return []
                
                df['retrieved_contents'] = df['retrieved_ids'].apply(get_contents_from_ids)
                print(f"[Local Optimization] Successfully populated retrieved_contents for {len(df)} rows")
            
            return df
            
        except Exception as e:
            print(f"[WARNING] Failed to populate retrieved_contents: {e}")
            return df
    
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
            
        if module_type == 'vllm_api':
            if 'uri' not in generator_config:
                raise ValueError("URI is required for vllm_api generator")
            
            return create_generator(
                model=model_name,
                module_type='vllm_api',
                uri=generator_config.get('uri'),
                max_tokens=generator_config.get('max_tokens', 400)
            )
        elif module_type == 'vllm':
            try:
                return create_generator(
                    model=model_name,
                    module_type='vllm'
                )
            except Exception as e:
                print(f"Failed to initialize vLLM, falling back to llama_index: {e}")
                return create_generator(
                    model=model_name,
                    module_type='llama_index'
                )
        elif module_type == 'openai':
            return create_generator(
                model=model_name,
                module_type='openai'
            )
        elif module_type == 'llama_index':
            return create_generator(
                model=model_name,
                module_type='llama_index',
                llm=generator_config.get('llm', 'openai')
            )
        else:
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
            'vllm_api': 'vllm_api',
        }
        
        if config.get('generator_module_type'):
            raw_module_type = config['generator_module_type']
            return module_type_mapping.get(raw_module_type, raw_module_type)
        elif generator_config and generator_config.get('module_type'):
            detected_module_type = generator_config.get('module_type')
            return module_type_mapping.get(detected_module_type, detected_module_type)
        else:
            if 'gpt' in model_name.lower():
                return 'openai'
            elif any(x in model_name.lower() for x in ['mistral', 'llama', 'qwen', 'mixtral']):
                return 'vllm'
            else:
                return 'llama_index'