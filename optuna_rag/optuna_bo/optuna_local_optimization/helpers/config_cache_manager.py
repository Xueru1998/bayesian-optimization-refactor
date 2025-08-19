import os
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class ConfigCacheManager:
    
    def __init__(self, cache_dir: str, verbose: bool = True):
        self.cache_dir = os.path.join(cache_dir, "config_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_index_file = os.path.join(self.cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
        self.hits = 0
        self.misses = 0
        self.verbose = verbose
        
    def _load_cache_index(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_index_file):
            with open(self.cache_index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _generate_config_hash(self, component: str, config: Dict[str, Any]) -> str:
        config_sorted = json.dumps(config, sort_keys=True)
        config_string = f"{component}:{config_sorted}"
        return hashlib.md5(config_string.encode()).hexdigest()
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in config.items():
            if isinstance(value, float):
                normalized[key] = round(value, 6)
            elif isinstance(value, (list, tuple)):
                normalized[key] = tuple(value) if isinstance(value, list) else value
            else:
                normalized[key] = value
        return normalized
    
    def get_cached_result(self, component: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        normalized_config = self._normalize_config(config)
        config_hash = self._generate_config_hash(component, normalized_config)
        
        if config_hash in self.cache_index:
            cache_entry = self.cache_index[config_hash]
            
            result_file = cache_entry.get('result_file')
            parquet_file = cache_entry.get('parquet_file')
            
            if result_file and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    cached_result = json.load(f)
                
                cached_result['from_cache'] = True
                cached_result['cache_hit'] = True
                cached_result['original_trial_id'] = cache_entry.get('trial_id')
                
                if parquet_file and os.path.exists(parquet_file):
                    cached_result['output_parquet'] = parquet_file
                    try:
                        working_df = pd.read_parquet(parquet_file)
                        cached_result['working_df'] = working_df
                    except Exception as e:
                        print(f"[Cache] Warning: Could not load parquet file: {e}")
                
                self.hits += 1
                print(f"\n[CACHE HIT] Component: {component}")
                print(f"  Config hash: {config_hash}")
                print(f"  Score: {cached_result.get('score', 0.0):.4f}")
                print(f"  Original trial: {cache_entry.get('trial_id')}")
                print(f"  Cache stats: {self.hits} hits, {self.misses} misses (hit rate: {self.get_hit_rate():.1%})")
                
                if self.verbose:
                    print(f"\n[CACHE DEBUG] Current config being checked:")
                    for key in sorted(normalized_config.keys()):
                        print(f"    {key}: {normalized_config[key]}")
                    
                    cached_config = cache_entry.get('config', {})
                    print(f"\n[CACHE DEBUG] Cached config that matched:")
                    for key in sorted(cached_config.keys()):
                        print(f"    {key}: {cached_config[key]}")
                    
                    config_diff = set(normalized_config.keys()) ^ set(cached_config.keys())
                    if config_diff:
                        print(f"\n[CACHE DEBUG] Key differences: {config_diff}")
                    
                    for key in set(normalized_config.keys()) & set(cached_config.keys()):
                        if normalized_config[key] != cached_config[key]:
                            print(f"\n[CACHE DEBUG] Value mismatch for '{key}':")
                            print(f"    Current: {normalized_config[key]}")
                            print(f"    Cached:  {cached_config[key]}")
                
                return cached_result
        
        self.misses += 1
        return None
    
    def save_to_cache(self, component: str, config: Dict[str, Any], 
                      results: Dict[str, Any], trial_id: str, 
                      output_parquet_path: Optional[str] = None):
        normalized_config = self._normalize_config(config)
        config_hash = self._generate_config_hash(component, normalized_config)
        
        result_file = os.path.join(self.cache_dir, f"{config_hash}_result.json")
        
        results_to_cache = {
            k: v for k, v in results.items() 
            if k != 'working_df' and not isinstance(v, pd.DataFrame)
        }
        results_to_cache['cached_at_trial'] = trial_id
        results_to_cache['component'] = component
        results_to_cache['config'] = normalized_config
        
        with open(result_file, 'w') as f:
            json.dump(results_to_cache, f, indent=2)
        
        cache_entry = {
            'component': component,
            'config': normalized_config,
            'trial_id': trial_id,
            'result_file': result_file,
            'score': results.get('score', 0.0),
            'latency': results.get('latency', 0.0)
        }
        
        if output_parquet_path and os.path.exists(output_parquet_path):
            cache_parquet = os.path.join(self.cache_dir, f"{config_hash}_output.parquet")
            try:
                df = pd.read_parquet(output_parquet_path)
                df.to_parquet(cache_parquet)
                cache_entry['parquet_file'] = cache_parquet
            except Exception as e:
                print(f"[Cache] Warning: Could not cache parquet file: {e}")
        
        self.cache_index[config_hash] = cache_entry
        self._save_cache_index()
        
        print(f"\n[CACHE SAVE] Saved results for {component} (hash: {config_hash})")
        print(f"  Score: {results.get('score', 0.0):.4f}")
        if self.verbose:
            print(f"  Config being cached:")
            for key in sorted(normalized_config.keys()):
                print(f"    {key}: {normalized_config[key]}")
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'total_cached_configs': len(self.cache_index),
            'cache_size_mb': self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)
    
    def clear_cache(self):
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"[Cache] Error deleting {file_path}: {e}")
        
        self.cache_index = {}
        self._save_cache_index()
        self.hits = 0
        self.misses = 0
        print("[Cache] Cache cleared")
    
    def get_cached_configs_for_component(self, component: str) -> list:
        cached_configs = []
        for config_hash, entry in self.cache_index.items():
            if entry.get('component') == component:
                cached_configs.append({
                    'config': entry.get('config'),
                    'score': entry.get('score'),
                    'trial_id': entry.get('trial_id')
                })
        return cached_configs