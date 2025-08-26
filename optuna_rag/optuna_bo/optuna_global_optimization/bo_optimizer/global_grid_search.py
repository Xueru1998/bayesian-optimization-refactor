import itertools
import os
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from pipeline.utils import Utils
from pipeline.logging.wandb import WandBLogger
from pipeline.search_space_calculator import CombinationCalculator
import wandb


class GlobalGridSearchOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.all_combinations = []
        self.current_index = 0
        self.explored_configs = set()
        self.combination_calculator = None
        
    def initialize(self):
        self.combination_calculator = CombinationCalculator(
            self.optimizer.config_generator,
            search_type='grid'
        )
        
        search_space = self._get_search_space()
        self.all_combinations = self._build_all_combinations(search_space)
        self.current_index = 0
        self.explored_configs.clear()
        
        print(f"\n[Grid Search] Generated {len(self.all_combinations)} total combinations")
        
        if self.optimizer.result_dir:
            self._save_grid_search_sequence()
    
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.optimizer.use_wandb:
            self._initialize_wandb()
        
        print("\n===== Starting Global Grid Search =====")
        print(f"Total configurations to test: {len(self.all_combinations)}")
        print("Note: Grid search will test ALL combinations without early stopping")
        
        try:
            for idx, config in enumerate(self.all_combinations):
                if self._is_duplicate_config(config):
                    print(f"\n[Trial {idx}/{len(self.all_combinations)}] Skipping duplicate config")
                    continue
                
                self._execute_trial(idx, config)
                
                if idx > 0 and (idx + 1) % 10 == 0:
                    self._print_progress(idx + 1)
            
            return self._finalize_results(start_time)
            
        except Exception as e:
            print(f"Grid search failed: {str(e)}")
            if self.optimizer.use_wandb:
                wandb.finish(exit_code=1)
            raise
    
    def _get_search_space(self) -> Dict[str, Any]:
        if hasattr(self.optimizer.config_extractor, 'extract_grid_search_space'):
            return self.optimizer.config_extractor.extract_grid_search_space()
        return self.optimizer.search_space
    
    def _build_all_combinations(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        grid_space = self._normalize_search_space(search_space)
        
        if self._has_composite_configs(grid_space):
            return self._build_composite_combinations(grid_space)
        return self._build_simple_combinations(grid_space)
    
    def _normalize_search_space(self, search_space: Dict[str, Any]) -> Dict[str, List[Any]]:
        grid_space = {}
        for param_name, param_spec in search_space.items():
            if isinstance(param_spec, list):
                grid_space[param_name] = param_spec
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                grid_space[param_name] = list(param_spec)
            else:
                grid_space[param_name] = [param_spec] if param_spec is not None else []
        return grid_space
    
    def _has_composite_configs(self, grid_space: Dict[str, Any]) -> bool:
        composite_keys = ['query_expansion_config', 'retrieval_config', 
                         'passage_compressor_config', 'generator_config']
        return any(key in grid_space for key in composite_keys)
    
    def _build_composite_combinations(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        combinations = []
        base_configs = self._build_base_configs(grid_space)
        
        for base_config in base_configs:
            downstream_configs = self._build_downstream_configs(grid_space, base_config)
            combinations.extend(downstream_configs)
        
        return combinations
    
    def _build_base_configs(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_configs = []
        
        if 'query_expansion_config' in grid_space:
            qe_configs = self._expand_query_expansion(grid_space)
            base_configs.extend(qe_configs)
        elif 'retrieval_config' in grid_space:
            retrieval_configs = self._expand_retrieval(grid_space)
            base_configs.extend(retrieval_configs)
        else:
            retriever_top_k = grid_space.get('retriever_top_k', [4])
            for top_k in retriever_top_k:
                base_configs.append({'retriever_top_k': top_k})
        
        return base_configs
    
    def _expand_query_expansion(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        qe_configs = grid_space.get('query_expansion_config', ['pass_query_expansion'])
        retriever_top_k = grid_space.get('retriever_top_k', [4])
        
        for qe_config in qe_configs:
            if qe_config == 'pass_query_expansion':
                configs.extend(self._build_pass_qe_configs(grid_space, retriever_top_k))
            else:
                configs.extend(self._build_active_qe_configs(grid_space, qe_config, retriever_top_k))
        
        return configs
    
    def _build_pass_qe_configs(self, grid_space: Dict[str, Any], top_k_values: List[int]) -> List[Dict[str, Any]]:
        configs = []
        retrieval_methods = grid_space.get('retrieval_method', ['vectordb'])
        
        for top_k in top_k_values:
            for method in retrieval_methods:
                base = {
                    'query_expansion_config': 'pass_query_expansion',
                    'retriever_top_k': top_k,
                    'retrieval_method': method
                }
                
                if method == 'bm25':
                    for tokenizer in grid_space.get('bm25_tokenizer', ['space']):
                        config = base.copy()
                        config['bm25_tokenizer'] = tokenizer
                        configs.append(config)
                elif method == 'vectordb':
                    for vdb_name in grid_space.get('vectordb_name', ['gemini']):
                        config = base.copy()
                        config['vectordb_name'] = vdb_name
                        configs.append(config)
                else:
                    configs.append(base)
        
        return configs
    
    def _build_active_qe_configs(self, grid_space: Dict[str, Any], qe_config: str, 
                                top_k_values: List[int]) -> List[Dict[str, Any]]:
        configs = []
        parts = qe_config.split('::')
        if len(parts) < 3:
            return configs
        
        method = parts[0]
        qe_retrieval_methods = grid_space.get('query_expansion_retrieval_method', ['vectordb'])
        
        for top_k in top_k_values:
            for qe_method in qe_retrieval_methods:
                base = {
                    'query_expansion_config': qe_config,
                    'retriever_top_k': top_k,
                    'query_expansion_retrieval_method': qe_method
                }
                
                if qe_method == 'bm25':
                    tokenizers = grid_space.get('query_expansion_bm25_tokenizer', ['space'])
                    for tokenizer in tokenizers:
                        config = base.copy()
                        config['query_expansion_bm25_tokenizer'] = tokenizer
                        configs.extend(self._add_qe_method_params(config, method, grid_space))
                elif qe_method == 'vectordb':
                    vdb_names = grid_space.get('query_expansion_vectordb_name', ['gemini'])
                    for vdb_name in vdb_names:
                        config = base.copy()
                        config['query_expansion_vectordb_name'] = vdb_name
                        configs.extend(self._add_qe_method_params(config, method, grid_space))
                else:
                    configs.extend(self._add_qe_method_params(base, method, grid_space))
        
        return configs
    
    def _add_qe_method_params(self, base_config: Dict[str, Any], method: str, 
                             grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        
        if method == 'hyde' and 'query_expansion_max_token' in grid_space:
            for max_token in grid_space['query_expansion_max_token']:
                config = base_config.copy()
                config['query_expansion_max_token'] = max_token
                configs.append(config)
        elif method == 'multi_query_expansion' and 'query_expansion_temperature' in grid_space:
            for temp in grid_space['query_expansion_temperature']:
                config = base_config.copy()
                config['query_expansion_temperature'] = temp
                configs.append(config)
        else:
            configs.append(base_config)
        
        return configs
    
    def _expand_retrieval(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        retrieval_configs = grid_space.get('retrieval_config', [])
        retriever_top_k = grid_space.get('retriever_top_k', [4])
        
        for retrieval_config in retrieval_configs:
            for top_k in retriever_top_k:
                configs.append({
                    'retrieval_config': retrieval_config,
                    'retriever_top_k': top_k
                })
        
        return configs
    
    def _build_downstream_configs(self, grid_space: Dict[str, Any], 
                                 base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        retriever_top_k = base_config.get('retriever_top_k', 4)
        
        reranker_configs = self._build_reranker_configs(grid_space, retriever_top_k)
        filter_configs = self._build_filter_configs(grid_space)
        compressor_configs = self._build_compressor_configs(grid_space)
        prompt_configs = self._build_prompt_configs(grid_space)
        generator_configs = self._build_generator_configs(grid_space)
        
        for reranker in reranker_configs:
            for filter_cfg in filter_configs:
                for compressor in compressor_configs:
                    for prompt in prompt_configs:
                        for generator in generator_configs:
                            config = base_config.copy()
                            config.update(reranker)
                            config.update(filter_cfg)
                            config.update(compressor)
                            config.update(prompt)
                            config.update(generator)
                            configs.append(config)
        
        return configs
    
    def _build_reranker_configs(self, grid_space: Dict[str, Any], 
                           retriever_top_k: int) -> List[Dict[str, Any]]:
        configs = []
        
        methods = grid_space.get('passage_reranker_method', ['sentence_transformer_reranker'])
        top_k_values = grid_space.get('reranker_top_k', [3])
        
        if not methods:
            return [{}]
        
        for method in methods:
            if method == 'pass_reranker':
                configs.append({'passage_reranker_method': method})
            elif method == 'sentence_transformer_reranker':
                models = grid_space.get('sentence_transformer_reranker_model_name', 
                                    ['cross-encoder/ms-marco-MiniLM-L12-v2'])
                for model in models:
                    if top_k_values:
                        for top_k in top_k_values:
                            if top_k <= retriever_top_k:
                                configs.append({
                                    'passage_reranker_method': method,
                                    'sentence_transformer_reranker_model_name': model,
                                    'reranker_top_k': top_k
                                })
                    else:
                        configs.append({
                            'passage_reranker_method': method,
                            'sentence_transformer_reranker_model_name': model
                        })
            else:
                if top_k_values:
                    for top_k in top_k_values:
                        if top_k <= retriever_top_k:
                            configs.append({
                                'passage_reranker_method': method,
                                'reranker_top_k': top_k
                            })
                else:
                    configs.append({'passage_reranker_method': method})
        
        return configs if configs else [{}]
    
    def _build_filter_configs(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        methods = grid_space.get('passage_filter_method', [])
        
        if not methods:
            return [{}]
        
        for method in methods:
            if method == 'pass_passage_filter':
                configs.append({'passage_filter_method': method})
            elif method == 'percentile_cutoff':
                percentiles = grid_space.get('percentile_cutoff_percentile', [0.4])
                for percentile in percentiles:
                    configs.append({
                        'passage_filter_method': method,
                        'percentile_cutoff_percentile': percentile
                    })
            else:
                configs.append({'passage_filter_method': method})
        
        return configs
    
    def _build_compressor_configs(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = grid_space.get('passage_compressor_config', ['pass_compressor'])
        return [{'passage_compressor_config': cfg} for cfg in configs]
    
    def _build_prompt_configs(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        methods = grid_space.get('prompt_maker_method', ['fstring'])
        indices = grid_space.get('prompt_template_idx', [0])
        
        for method in methods:
            for idx in indices:
                configs.append({
                    'prompt_maker_method': method,
                    'prompt_template_idx': idx
                })
        
        return configs if configs else [{}]
    
    def _build_generator_configs(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        configs = []
        gen_configs = grid_space.get('generator_config', [])
        temperatures = grid_space.get('generator_temperature', [0.0])
        
        for gen_cfg in gen_configs:
            for temp in temperatures:
                configs.append({
                    'generator_config': gen_cfg,
                    'generator_temperature': temp
                })
        
        return configs if configs else [{}]
    
    def _build_simple_combinations(self, grid_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        keys = list(grid_space.keys())
        values = [grid_space[key] for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            if self._is_valid_combination(config):
                combinations.append(config)
        
        return combinations
    
    def _is_valid_combination(self, config: Dict[str, Any]) -> bool:
        if 'retrieval_method' in config:
            method = config['retrieval_method']
            if method == 'bm25' and 'vectordb_name' in config:
                return False
            if method == 'vectordb' and 'bm25_tokenizer' in config:
                return False
        
        if 'reranker_top_k' in config and 'retriever_top_k' in config:
            if config['reranker_top_k'] > config['retriever_top_k']:
                return False
        
        return True
    
    def _is_duplicate_config(self, config: Dict[str, Any]) -> bool:
        config_hash = self._config_to_hash(config)
        if config_hash in self.explored_configs:
            return True
        self.explored_configs.add(config_hash)
        return False
    
    def _config_to_hash(self, config: Dict[str, Any]) -> str:
        sorted_config = dict(sorted(config.items()))
        config_str = json.dumps(sorted_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _execute_trial(self, idx: int, config: Dict[str, Any]):
        print(f"\n[Trial {idx + 1}/{len(self.all_combinations)}]")
        print(f"Testing configuration: {config}")
        
        trial_dir = self._setup_trial_directory(idx)
        trial_start_time = time.time()
        
        processed_config = self._prepare_config(config)
        results = self._run_pipeline(processed_config, trial_dir)
        
        execution_time = time.time() - trial_start_time
        self._record_trial_results(idx, config, results, execution_time)
    
    def _setup_trial_directory(self, idx: int) -> str:
        trial_dir = os.path.join(self.optimizer.result_dir, f"trial_{idx}")
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        
        self._copy_corpus_data(trial_dir)
        self._save_qa_data(trial_dir)
        
        return trial_dir
    
    def _copy_corpus_data(self, trial_dir: str):
        centralized_corpus_path = os.path.join(
            self.optimizer.pipeline_runner._get_centralized_project_dir(),
            "data", "corpus.parquet"
        )
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            import shutil
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
    
    def _save_qa_data(self, trial_dir: str):
        qa_subset_path = os.path.join(trial_dir, "data", "qa.parquet")
        self.optimizer.qa_df.to_parquet(qa_subset_path, index=False)
    
    def _prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        processed = config.copy()
        self._process_composite_configs(processed)
        
        if 'generator_temperature' in processed:
            processed['generator_temperature'] = round(processed['generator_temperature'], 4)
        if 'query_expansion_temperature' in processed:
            processed['query_expansion_temperature'] = round(processed['query_expansion_temperature'], 4)
        
        return processed
    
    def _process_composite_configs(self, params: Dict[str, Any]):
        if 'generator_config' in params:
            self._expand_generator_config(params)
        
        if 'query_expansion_config' in params:
            self._expand_qe_config(params)
        
        if 'passage_compressor_config' in params:
            self._expand_compressor_config(params)
    
    def _expand_generator_config(self, params: Dict[str, Any]):
        gen_config_str = params['generator_config']
        module_type, model = gen_config_str.split('::', 1)
        
        params['generator_module_type'] = module_type
        params['generator_model'] = model
        
        metadata = self._get_config_metadata('generator', gen_config_str)
        if metadata:
            if 'api_url' in metadata:
                params['generator_api_url'] = metadata['api_url']
            if 'llm' in metadata:
                params['generator_llm'] = metadata['llm']
    
    def _expand_qe_config(self, params: Dict[str, Any]):
        qe_config_str = params['query_expansion_config']
        if qe_config_str == 'pass_query_expansion' or '::' not in qe_config_str:
            return
        
        parts = qe_config_str.split('::', 2)
        if len(parts) >= 3:
            method, gen_type, model = parts
            params['query_expansion_method'] = method
            params['query_expansion_generator_module_type'] = gen_type
            params['query_expansion_model'] = model
            
            metadata = self._get_config_metadata('query_expansion', qe_config_str)
            if metadata:
                if 'api_url' in metadata:
                    params['query_expansion_api_url'] = metadata['api_url']
                if 'llm' in metadata:
                    params['query_expansion_llm'] = metadata['llm']
    
    def _expand_compressor_config(self, params: Dict[str, Any]):
        comp_config_str = params['passage_compressor_config']
        if comp_config_str == 'pass_compressor' or '::' not in comp_config_str:
            return
        
        parts = comp_config_str.split('::', 2)
        if len(parts) >= 3:
            method, gen_type, model = parts
            params['passage_compressor_method'] = method
            params['compressor_generator_module_type'] = gen_type
            params['compressor_model'] = model
            
            metadata = self._get_config_metadata('passage_compressor', comp_config_str)
            if metadata:
                params['compressor_llm'] = metadata.get('llm', 'openai')
                if 'api_url' in metadata:
                    params['compressor_api_url'] = metadata['api_url']
    
    def _get_config_metadata(self, component: str, config_str: str) -> Optional[Dict[str, Any]]:
        if not hasattr(self.optimizer, 'search_space'):
            return None
        
        component_key = f'{component}_config' if component != 'passage_compressor' else 'passage_compressor_config'
        configs = self.optimizer.search_space.get(component_key, {})
        
        if isinstance(configs, dict) and 'metadata' in configs:
            return configs['metadata'].get(config_str)
        
        return None
    
    def _run_pipeline(self, config: Dict[str, Any], trial_dir: str) -> Dict[str, Any]:
        return self.optimizer.pipeline_runner.run_pipeline(
            config=config,
            trial_dir=trial_dir,
            qa_subset=self.optimizer.qa_df,
            is_local_optimization=False,
            current_component=None
        )
    
    def _record_trial_results(self, idx: int, config: Dict[str, Any], 
                             results: Dict[str, Any], execution_time: float):
        score = self._extract_score(results)
        
        trial_result = {
            "trial_number": idx,
            "config": config.copy(),
            "score": score,
            "latency": execution_time,
            "execution_time": execution_time,
            "retrieval_f1": results.get("retrieval_f1", 0.0),
            "generation_f1": results.get("generation_f1", 0.0)
        }
        
        if self.optimizer.use_ragas and 'ragas_metrics' in results:
            trial_result['ragas_metrics'] = results['ragas_metrics']
        
        self.optimizer.all_trials.append(trial_result)
        self._update_best_results(score, execution_time, config, idx)
        
        print(f"  Score: {score:.4f}, Latency: {execution_time:.2f}s")
        
        if self.optimizer.use_wandb:
            self._log_to_wandb(idx, score, execution_time, results)
    
    def _extract_score(self, results: Dict[str, Any]) -> float:
        if self.optimizer.use_ragas and 'ragas_mean_score' in results:
            return results['ragas_mean_score']
        return results.get("combined_score", 0.0)
    
    def _update_best_results(self, score: float, execution_time: float, 
                            config: Dict[str, Any], idx: int):
        if score > self.optimizer.best_score["value"]:
            self.optimizer.best_score = {
                "value": score,
                "config": config.copy(),
                "latency": execution_time,
                "trial_number": idx
            }
        
        if execution_time < self.optimizer.best_latency["value"]:
            self.optimizer.best_latency = {
                "value": execution_time,
                "config": config.copy(),
                "score": score,
                "trial_number": idx
            }
    
    def _log_to_wandb(self, idx: int, score: float, execution_time: float, 
                     results: Dict[str, Any]):
        wandb.log({
            "trial": idx,
            "score": score,
            "latency": execution_time,
            "retrieval_f1": results.get("retrieval_f1", 0.0),
            "generation_f1": results.get("generation_f1", 0.0)
        })
    
    def _print_progress(self, completed: int):
        print(f"\nProgress: {completed}/{len(self.all_combinations)} trials completed")
        print(f"Current best score: {self.optimizer.best_score['value']:.4f}")
        print(f"Current best latency: {self.optimizer.best_latency['value']:.2f}s")
    
    def _finalize_results(self, start_time: float) -> Dict[str, Any]:
        total_time = time.time() - start_time
        
        self._save_final_results()
        
        if self.optimizer.use_wandb:
            self._log_wandb_results()
        
        results = self._prepare_results(total_time)
        self._print_results_summary(total_time, results)
        
        Utils.save_results_to_json(
            self.optimizer.result_dir,
            "grid_search_summary.json",
            results
        )
        
        if self.optimizer.use_wandb:
            self._update_wandb_summary(results)
            wandb.finish()
        
        return results
    
    def _save_grid_search_sequence(self):
        sequence_file = os.path.join(self.optimizer.result_dir, "grid_search_sequence.txt")
        
        with open(sequence_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GLOBAL GRID SEARCH SEQUENCE\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total configurations to test: {len(self.all_combinations)}\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("-"*80 + "\n\n")
            
            for idx, config in enumerate(self.all_combinations, 1):
                f.write(f"Configuration #{idx:04d}:\n")
                
                sorted_params = sorted(config.items())
                
                for param, value in sorted_params:
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    elif isinstance(value, (list, dict)):
                        value_str = str(value)
                    else:
                        value_str = str(value)
                    
                    f.write(f"  {param:40s} = {value_str}\n")
                
                f.write("\n")
                
                if idx % 10 == 0 and idx < len(self.all_combinations):
                    f.write("-"*40 + f" [{idx}/{len(self.all_combinations)}] " + "-"*40 + "\n\n")
            
            f.write("-"*80 + "\n")
            f.write(f"END OF SEQUENCE - {len(self.all_combinations)} configurations\n")
            f.write("="*80 + "\n")
        
        print(f"[Grid Search] Saved sequence to: {sequence_file}")
    
    def _save_final_results(self):
        results_df = pd.DataFrame(self.optimizer.all_trials)
        results_df.to_csv(
            os.path.join(self.optimizer.result_dir, "grid_search_results.csv"),
            index=False
        )
        
        Utils.save_results_to_json(
            self.optimizer.result_dir,
            "grid_search_all_trials.json",
            self.optimizer.all_trials
        )
    
    def _initialize_wandb(self):
        search_space_size = len(self.all_combinations)
        
        wandb_config = {
            "search_type": "grid_search",
            "optimizer": "GRID",
            "total_combinations": search_space_size,
            "retrieval_weight": self.optimizer.retrieval_weight,
            "generation_weight": self.optimizer.generation_weight,
            "search_space": self.optimizer.search_space,
            "study_name": self.optimizer.study_name,
            "evaluation_method": "RAGAS" if self.optimizer.use_ragas else "Traditional",
            "ragas_enabled": self.optimizer.use_ragas
        }
        
        wandb.init(
            project=self.optimizer.wandb_project,
            entity=self.optimizer.wandb_entity,
            name=self.optimizer.wandb_run_name or f"grid_search_{int(time.time())}",
            config=wandb_config,
            reinit=True
        )
    
    def _log_wandb_results(self):
        print("\nLogging results to W&B...")
        
        if self.optimizer.use_ragas:
            WandBLogger.log_ragas_comparison_plot(self.optimizer.all_trials, prefix="grid")
            WandBLogger.log_ragas_summary_table(self.optimizer.all_trials, prefix="grid")
        
        WandBLogger.log_final_tables(self.optimizer.all_trials, None, prefix="grid_final")
    
    def _prepare_results(self, total_time: float) -> Dict[str, Any]:
        pareto_front = Utils.find_pareto_front(self.optimizer.all_trials)
        
        best_config = None
        high_score_configs = [t for t in self.optimizer.all_trials if t['score'] > 0.9]
        if high_score_configs:
            best_config = min(high_score_configs, key=lambda x: x['latency'])
        elif self.optimizer.all_trials:
            best_config = max(self.optimizer.all_trials, key=lambda x: x['score'])
        
        results = {
            "best_config": best_config,
            "best_score_config": self.optimizer.best_score["config"],
            "best_score": self.optimizer.best_score["value"],
            "best_score_latency": self.optimizer.best_score["latency"],
            "best_latency_config": self.optimizer.best_latency["config"],
            "best_latency": self.optimizer.best_latency["value"],
            "best_latency_score": self.optimizer.best_latency["score"],
            "pareto_front": pareto_front,
            "optimization_time": total_time,
            "n_trials": len(self.optimizer.all_trials),
            "total_combinations_tested": len(self.explored_configs),
            "total_combinations_available": len(self.all_combinations),
            "optimizer": "grid",
            "all_trials": self.optimizer.all_trials
        }
        
        return results
    
    def _print_results_summary(self, total_time: float, results: Dict[str, Any]):
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n===== Grid Search Results =====")
        print(f"Total optimization time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Total configurations tested: {len(self.explored_configs)}/{len(self.all_combinations)}")
        
        if results.get('best_config'):
            best_config = results['best_config']
            print("\nBest configuration (score > 0.9 with minimum latency):")
            print(f"  Trial: {best_config.get('trial_number', 'N/A')}")
            print(f"  Score: {best_config.get('score', 0.0):.4f}")
            print(f"  Latency: {best_config.get('latency', float('inf')):.2f}s")
            print(f"  Config: {best_config.get('config', {})}")
        
        print("\nBest trial by score:")
        print(f"  Score: {self.optimizer.best_score['value']:.4f}")
        print(f"  Latency: {self.optimizer.best_score['latency']:.2f}s")
        
        print("\nBest trial by latency:")
        print(f"  Score: {self.optimizer.best_latency['score']:.4f}")
        print(f"  Latency: {self.optimizer.best_latency['value']:.2f}s")
        
        pareto_front = results['pareto_front']
        print(f"\nPareto optimal solutions: {len(pareto_front)}")
        for i, trial in enumerate(pareto_front[:5]):
            print(f"  Solution {i+1}: Score={trial['score']:.4f}, Latency={trial['latency']:.2f}s")
    
    def _update_wandb_summary(self, results: Dict[str, Any]):
        wandb.summary["best_score"] = self.optimizer.best_score["value"]
        wandb.summary["best_latency"] = self.optimizer.best_latency["value"]
        wandb.summary["total_trials"] = len(self.optimizer.all_trials)
        wandb.summary["optimization_time"] = results['optimization_time']
        wandb.summary["total_combinations_tested"] = len(self.explored_configs)
        wandb.summary["optimizer"] = "grid"