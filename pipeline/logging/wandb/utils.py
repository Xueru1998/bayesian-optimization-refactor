import hashlib
import json
import numpy as np
from typing import Dict, Any, List


class WandBUtils:
    @staticmethod
    def normalize_value(value: Any) -> Any:
        if value is None:
            return None
        elif value == '':
            return None
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (np.integer, int)):
            return int(value)
        elif isinstance(value, (np.floating, float)):  
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return str(value)
        elif isinstance(value, dict):
            return str(value)
        elif isinstance(value, str):
            try:
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val)
                return float_val
            except (ValueError, TypeError):
                return str(value)
        elif hasattr(value, '__str__'):
            return str(value)
        else:
            return str(value)
    
    @staticmethod
    def get_config_id(config: Dict[str, Any]) -> str:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        config_clean = convert_numpy(config)
        config_str = json.dumps(config_clean, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    @staticmethod
    def format_config_for_table(config: Dict[str, Any], component: str = None) -> str:
        exclude_keys = []
        
        if component == 'query_expansion':
            exclude_keys = ['query_expansion_generator_module_type', 'query_expansion_llm']
        elif component == 'generator':
            exclude_keys = ['generator_llm', 'generator_module_type']
        
        config_items = []
        for key in sorted(config.keys()):
            if 'api_url' in key or key in exclude_keys:
                continue
            
            value = config[key]
            if isinstance(value, list):
                value_str = str(value)
            elif isinstance(value, (int, float)):
                value_str = str(value)
            else:
                value_str = str(value)
            
            config_items.append(f"{key}: {value_str}")
        
        return " | ".join(config_items)
    
    @staticmethod
    def normalize_table_value(value: Any, expected_type: str = None) -> Any:
        if value is None:
            return None
        elif value == '':
            return None
        
        if expected_type == 'score' or '_score' in str(expected_type):
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0
            elif isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            else:
                return 0.0
        
        if isinstance(value, bool):
            return value
        elif isinstance(value, (np.integer, int)):
            return int(value)
        elif isinstance(value, (np.floating, float)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, list):
            return str(value)
        elif isinstance(value, dict):
            return str(value)
        elif isinstance(value, str):
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            try:
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val)
                return float_val
            except (ValueError, TypeError):
                return str(value)
        elif hasattr(value, '__str__'):
            return str(value)
        else:
            return str(value)
    
    @staticmethod
    def group_trials_by_config(trials_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        config_groups = {}
        for trial in trials_data:
            config = trial.get('config', {})
            config_id = WandBUtils.get_config_id(config)
            
            if config_id not in config_groups:
                config_groups[config_id] = []
            config_groups[config_id].append(trial)
        
        return config_groups
    
    @staticmethod
    def get_best_trials_by_config(trials_data: List[Dict[str, Any]], full_budget_only: bool = False) -> List[Dict[str, Any]]:
        config_groups = WandBUtils.group_trials_by_config(trials_data)
        
        best_trials = []
        for config_id, trials in config_groups.items():
            if full_budget_only:
                full_budget_trials = [t for t in trials if t.get('budget_percentage', 1.0) >= 0.99]
                if not full_budget_trials:
                    continue
                best_trial = max(full_budget_trials, key=lambda t: t.get('score', 0))
            else:
                best_trial = max(trials, key=lambda t: (t.get('budget', 0), t.get('score', 0)))
            
            best_trial['config_id'] = config_id
            best_trial['num_evaluations'] = len(trials)
            
            budgets = sorted([t.get('budget', 0) for t in trials])
            best_trial['budget_progression'] = budgets
            
            best_trials.append(best_trial)
        
        return best_trials
    
    @staticmethod
    def _safe_get_score(attrs: Dict[str, Any], key: str) -> float:
        value = attrs.get(key)
        if value is None:
            return 0.0
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0
    
    @staticmethod
    def detect_active_components(study_or_trials: Any) -> Dict[str, bool]:
        components = {
            "query_expansion": False,
            "retrieval": False,
            "reranker": False,
            "filter": False,
            "compressor": False,
            "prompt_maker": False,
            "generator": False
        }
        
        if hasattr(study_or_trials, 'trials'):
            import optuna
            trials = study_or_trials.trials
            for trial in trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    attrs = trial.user_attrs
                    
                    if WandBUtils._safe_get_score(attrs, "query_expansion_score") > 0:
                        components["query_expansion"] = True
                    if WandBUtils._safe_get_score(attrs, "retrieval_score") > 0:
                        components["retrieval"] = True
                    if WandBUtils._safe_get_score(attrs, "reranker_score") > 0:
                        components["reranker"] = True
                    if WandBUtils._safe_get_score(attrs, "filter_score") > 0:
                        components["filter"] = True
                    if (WandBUtils._safe_get_score(attrs, "compression_score") > 0 or 
                        WandBUtils._safe_get_score(attrs, "compressor_score") > 0):
                        components["compressor"] = True
                    if WandBUtils._safe_get_score(attrs, "prompt_maker_score") > 0:
                        components["prompt_maker"] = True
                    if WandBUtils._safe_get_score(attrs, "generation_score") > 0:
                        components["generator"] = True
                    
                    if all(components.values()):
                        break
        else:
            for trial in study_or_trials:
                for comp in components.keys():
                    if comp == "compressor":
                        compression_score = WandBUtils._safe_get_score(trial, "compression_score")
                        compressor_score = WandBUtils._safe_get_score(trial, "compressor_score")
                        if compression_score > 0 or compressor_score > 0:
                            components[comp] = True
                    elif comp == "generator":
                        if WandBUtils._safe_get_score(trial, "generation_score") > 0:
                            components[comp] = True
                    else:
                        score_key = f"{comp}_score"
                        if WandBUtils._safe_get_score(trial, score_key) > 0:
                            components[comp] = True
        
        return components