import json
import optuna
import pandas as pd
import tempfile
import os
import shutil
from typing import Dict, Any, Union, Tuple
import time

class OptunaObjective:
    def __init__(self, search_space, config_generator, pipeline_runner, 
                 component_cache, corpus_df, qa_df, use_cache=True, search_type='grid', result_dir=None, save_intermediate_results=True):
        self.search_space = search_space
        self.config_generator = config_generator
        self.pipeline_runner = pipeline_runner
        self.component_cache = component_cache
        self.corpus_df = corpus_df
        self.qa_df = qa_df
        self.use_cache = use_cache
        self.search_type = search_type
        self.cached_pipeline_runner = None
        self.result_dir = result_dir or "optuna_results"
        self.save_intermediate_results = save_intermediate_results
        
        self.has_query_expansion = self.config_generator.node_exists("query_expansion")
        self.has_retrieval = self.config_generator.node_exists("retrieval")
        
        if self.search_type == 'grid' and self.has_query_expansion:
            self._precompute_valid_param_combinations()
    
    def _precompute_valid_param_combinations(self):
        self.valid_param_combinations = []
        
        qe_methods = self.search_space.get('query_expansion_method', ['pass_query_expansion'])
        qe_models = self.search_space.get('query_expansion_model', [])
        qe_max_tokens = self.search_space.get('query_expansion_max_token', [])
        qe_temperatures = self.search_space.get('query_expansion_temperature', [])
        
        qe_retrieval_methods = self.search_space.get('query_expansion_retrieval_method', [])
        qe_bm25_tokenizers = self.search_space.get('query_expansion_bm25_tokenizer', [])
        qe_vectordb_names = self.search_space.get('query_expansion_vectordb_name', [])
        
        retrieval_methods = self.search_space.get('retrieval_method', [])
        bm25_tokenizers = self.search_space.get('bm25_tokenizer', [])
        vectordb_names = self.search_space.get('vectordb_name', [])
        
        retriever_top_k_values = self.search_space.get('retriever_top_k', [10])
        
        print(f"[DEBUG] Starting combination generation:")
        print(f"  QE methods: {qe_methods}")
        print(f"  QE retrieval methods: {qe_retrieval_methods}")
        print(f"  Retrieval methods: {retrieval_methods}")
        print(f"  Top-K values: {retriever_top_k_values}")
        
        for top_k in retriever_top_k_values:
            for qe_method in qe_methods:
                if qe_method == 'pass_query_expansion':
                    for retrieval_method in retrieval_methods:
                        base_params = {
                            'query_expansion_method': qe_method,
                            'retrieval_method': retrieval_method,
                            'retriever_top_k': top_k
                        }
                        
                        if retrieval_method == 'bm25':
                            for tokenizer in bm25_tokenizers:
                                params = base_params.copy()
                                params['bm25_tokenizer'] = tokenizer
                                self._add_other_component_combinations(params)
                        elif retrieval_method == 'vectordb':
                            for vdb_name in vectordb_names:
                                params = base_params.copy()
                                params['vectordb_name'] = vdb_name
                                self._add_other_component_combinations(params)
                        else:
                            self._add_other_component_combinations(base_params)
                else:
                    for qe_retrieval_method in qe_retrieval_methods:
                        base_qe_params = {
                            'query_expansion_method': qe_method,
                            'query_expansion_retrieval_method': qe_retrieval_method,
                            'retriever_top_k': top_k
                        }
                        
                        if qe_method in ['query_decompose', 'hyde'] and qe_models:
                            for model in qe_models:
                                model_params = base_qe_params.copy()
                                model_params['query_expansion_model'] = model
                                
                                if qe_method == 'hyde' and qe_max_tokens:
                                    for max_token in qe_max_tokens:
                                        hyde_params = model_params.copy()
                                        hyde_params['query_expansion_max_token'] = max_token
                                        self._add_qe_retrieval_params(hyde_params, qe_retrieval_method, 
                                                                    qe_bm25_tokenizers, qe_vectordb_names)
                                else:
                                    self._add_qe_retrieval_params(model_params, qe_retrieval_method,
                                                                qe_bm25_tokenizers, qe_vectordb_names)
                                    
                        elif qe_method == 'multi_query_expansion':
                            if qe_models and qe_temperatures:
                                for model in qe_models:
                                    for temp in qe_temperatures:
                                        temp_params = base_qe_params.copy()
                                        temp_params['query_expansion_model'] = model
                                        temp_params['query_expansion_temperature'] = temp
                                        self._add_qe_retrieval_params(temp_params, qe_retrieval_method,
                                                                    qe_bm25_tokenizers, qe_vectordb_names)
                            else:
                                self._add_qe_retrieval_params(base_qe_params, qe_retrieval_method,
                                                            qe_bm25_tokenizers, qe_vectordb_names)
                        else:
                            self._add_qe_retrieval_params(base_qe_params, qe_retrieval_method,
                                                        qe_bm25_tokenizers, qe_vectordb_names)
        
        print(f"[DEBUG] Total valid combinations generated: {len(self.valid_param_combinations)}")

    def _add_other_component_combinations(self, base_params):
        filter_configs = self.search_space.get('passage_filter_config', ['pass_passage_filter'])
        reranker_top_k_values = self.search_space.get('reranker_top_k', [])
        prompt_configs = self.search_space.get('prompt_config', self.search_space.get('prompt_maker_config', []))
        generator_models = self.search_space.get('generator_model', [])
        generator_temps = self.search_space.get('generator_temperature', [])
        
        for filter_config in filter_configs:
            filter_params = base_params.copy()
            filter_params['passage_filter_config'] = filter_config

            reranker_configs = self.search_space.get('reranker_config', [])
            
            if reranker_configs:
                for reranker_config in reranker_configs:
                    if reranker_top_k_values:
                        for reranker_top_k in reranker_top_k_values:
                            if reranker_top_k <= filter_params['retriever_top_k']:
                                reranker_params = filter_params.copy()
                                reranker_params['reranker_config'] = reranker_config
                                reranker_params['reranker_top_k'] = reranker_top_k
                                self._add_prompt_and_generator_combinations(reranker_params, prompt_configs, generator_models, generator_temps)
                    else:
                        reranker_params = filter_params.copy()
                        reranker_params['reranker_config'] = reranker_config
                        self._add_prompt_and_generator_combinations(reranker_params, prompt_configs, generator_models, generator_temps)
            else:
                self._add_prompt_and_generator_combinations(filter_params, prompt_configs, generator_models, generator_temps)
                
    def _add_qe_retrieval_params(self, base_params, qe_retrieval_method, 
                            qe_bm25_tokenizers, qe_vectordb_names):
        if qe_retrieval_method == 'bm25' and qe_bm25_tokenizers:
            for tokenizer in qe_bm25_tokenizers:
                params = base_params.copy()
                params['query_expansion_bm25_tokenizer'] = tokenizer
                self._add_other_component_combinations(params)
        elif qe_retrieval_method == 'vectordb' and qe_vectordb_names:
            for vdb_name in qe_vectordb_names:
                params = base_params.copy()
                params['query_expansion_vectordb_name'] = vdb_name
                self._add_other_component_combinations(params)
        else:
            self._add_other_component_combinations(base_params)
                
    def _add_prompt_and_generator_combinations(self, base_params, prompt_configs, generator_models, generator_temps):
        if prompt_configs and generator_models:
            for prompt_config in prompt_configs:
                for generator_model in generator_models:
                    for generator_temp in generator_temps:
                        final_params = base_params.copy()
                        final_params['prompt_config'] = prompt_config
                        final_params['generator_model'] = generator_model
                        final_params['generator_temperature'] = generator_temp
                        self.valid_param_combinations.append(final_params)
        elif generator_models:
            for generator_model in generator_models:
                for generator_temp in generator_temps:
                    final_params = base_params.copy()
                    final_params['generator_model'] = generator_model
                    final_params['generator_temperature'] = generator_temp
                    self.valid_param_combinations.append(final_params)
        elif prompt_configs:
            for prompt_config in prompt_configs:
                final_params = base_params.copy()
                final_params['prompt_config'] = prompt_config
                self.valid_param_combinations.append(final_params)
        else:
            self.valid_param_combinations.append(base_params)
        
    def _add_qe_retrieval_combinations(self, base_params):
        qe_retrieval_methods = self.search_space.get('query_expansion_retrieval_method', ['bm25'])
        
        for method in qe_retrieval_methods:
            method_params = base_params.copy()
            method_params['query_expansion_retrieval_method'] = method
            
            if method == 'bm25':
                tokenizers = self.search_space.get('query_expansion_bm25_tokenizer', [])
                for tokenizer in tokenizers:
                    tokenizer_params = method_params.copy()
                    tokenizer_params['query_expansion_bm25_tokenizer'] = tokenizer
                    self._add_other_component_combinations(tokenizer_params)
            elif method == 'vectordb':
                vdb_names = self.search_space.get('query_expansion_vectordb_name', [])
                for vdb_name in vdb_names:
                    vdb_params = method_params.copy()
                    vdb_params['query_expansion_vectordb_name'] = vdb_name
                    self._add_other_component_combinations(vdb_params)
    
    
    def __call__(self, trial: optuna.Trial) -> float:
        if self.search_type == 'grid' and self.has_query_expansion:
            if trial.number < len(self.valid_param_combinations):
                params = self.valid_param_combinations[trial.number].copy()
                for key, value in params.items():
                    trial._cached_frozen_trial.distributions[key] = optuna.distributions.CategoricalDistribution([value])
                    trial._cached_frozen_trial.params[key] = value
            else:
                return 0.0
        else:
            params = self._suggest_params(trial)
        
        self._clean_params(params)
        
        params['save_intermediate_results'] = self.save_intermediate_results
        
        print(f"\nRunning trial {trial.number} with params: {params}")
        
        if self.use_cache:
            cached_score = self._check_cache(trial, params)
            if cached_score is not None:
                return cached_score
        
        if self.result_dir:
            trial_dir = os.path.join(self.result_dir, f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        else:
            trial_dir = tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_")
        
        try:
            score = self._run_trial(trial, params, trial_dir)
            
            if self.use_cache:
                self._save_to_cache(params, score, trial.user_attrs)
            
            if self.save_intermediate_results and os.path.exists(trial_dir):
                debug_dir = os.path.join(trial_dir, "debug_intermediate_results")
                if os.path.exists(debug_dir):
                    print(f"[DEBUG] Intermediate results saved in: {debug_dir}")
                    self._list_saved_files(debug_dir)
            
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
                
    def _list_saved_files(self, debug_dir: str):
        parquet_files = [f for f in os.listdir(debug_dir) if f.endswith('.parquet')]
        json_files = [f for f in os.listdir(debug_dir) if f.endswith('.json')]
        
        print(f"[DEBUG] Found {len(parquet_files)} parquet files and {len(json_files)} JSON files")
        
        if parquet_files:
            print("[DEBUG] Parquet files:")
            for pf in sorted(parquet_files):
                file_path = os.path.join(debug_dir, pf)
                file_size = os.path.getsize(file_path) / 1024
                try:
                    df = pd.read_parquet(file_path)
                    print(f"  - {pf} ({file_size:.1f} KB, {len(df)} rows, columns: {list(df.columns)[:5]}...)")
                except:
                    print(f"  - {pf} ({file_size:.1f} KB)")
        
        if json_files:
            print("[DEBUG] JSON files:")
            for jf in sorted(json_files):
                file_size = os.path.getsize(os.path.join(debug_dir, jf)) / 1024
                print(f"  - {jf} ({file_size:.1f} KB)")
    
                
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
            
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        
        self._suggest_query_expansion_params(trial, params)
        self._suggest_retrieval_params(trial, params)
        self._suggest_reranker_params(trial, params)
        self._suggest_filter_params(trial, params)
        self._suggest_compressor_params(trial, params)
        self._suggest_prompt_maker_params(trial, params)
        self._suggest_generator_params(trial, params)
        
        return params
    
    def _suggest_value(self, trial: optuna.Trial, param_name: str, 
                  param_spec: Union[list, Tuple[float, float]], 
                  param_type: str = 'categorical') -> Any:
        if isinstance(param_spec, list):
            return trial.suggest_categorical(param_name, param_spec)
        elif isinstance(param_spec, tuple) and len(param_spec) == 2:
            if self.search_type == 'grid':
                return trial.suggest_categorical(param_name, list(param_spec))
            else:
                if param_type == 'int':
                    return trial.suggest_int(param_name, param_spec[0], param_spec[1])
                else:
                    return trial.suggest_float(param_name, param_spec[0], param_spec[1])
        else:
            raise ValueError(f"Invalid parameter specification for {param_name}: {param_spec}")
    
    def _suggest_query_expansion_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'query_expansion_config' in self.search_space:
            qe_config_str = trial.suggest_categorical('query_expansion_config',
                                                    self.search_space['query_expansion_config'])
            
            if qe_config_str == 'pass_query_expansion':
                params['query_expansion_method'] = 'pass_query_expansion'
                return

            parts = qe_config_str.split('::')
            if len(parts) >= 3:
                method, gen_type, model = parts[0], parts[1], parts[2]
                params['query_expansion_method'] = method
                params['query_expansion_generator_module_type'] = gen_type
                params['query_expansion_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                for gen_config in unified_params.get('generator_configs', []):
                    if (gen_config['method'] == method and 
                        gen_config['generator_module_type'] == gen_type and 
                        model in gen_config['models']):
                        if gen_type == 'sap_api':
                            params['query_expansion_api_url'] = gen_config.get('api_url')
                            params['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                        else:
                            params['query_expansion_llm'] = gen_config.get('llm', 'openai')
                        break
                
                if 'query_expansion_temperature' in self.search_space:
                    params['query_expansion_temperature'] = self._suggest_value(
                        trial, 'query_expansion_temperature', 
                        self.search_space['query_expansion_temperature'], 'float'
                    )
                
                if 'query_expansion_max_token' in self.search_space:
                    params['query_expansion_max_token'] = self._suggest_value(
                        trial, 'query_expansion_max_token', 
                        self.search_space['query_expansion_max_token'], 'int'
                    )
                
                self._add_query_expansion_retrieval_params(trial, params)
            
            return

        if 'query_expansion_method' not in self.search_space:
            return
            
        params['query_expansion_method'] = trial.suggest_categorical('query_expansion_method', 
                                                                self.search_space['query_expansion_method'])
        
        if params['query_expansion_method'] == 'pass_query_expansion':
            return

        if 'query_expansion_generator_module_type' in self.search_space:
            params['query_expansion_generator_module_type'] = trial.suggest_categorical(
                'query_expansion_generator_module_type', 
                self.search_space['query_expansion_generator_module_type']
            )
        
        if 'query_expansion_llm' in self.search_space:
            params['query_expansion_llm'] = trial.suggest_categorical(
                'query_expansion_llm', 
                self.search_space['query_expansion_llm']
            )
        
        if 'query_expansion_model' in self.search_space:
            params['query_expansion_model'] = trial.suggest_categorical(
                'query_expansion_model', 
                self.search_space['query_expansion_model']
            )

        if 'query_expansion_temperature' in self.search_space:
            params['query_expansion_temperature'] = self._suggest_value(
                trial, 'query_expansion_temperature', 
                self.search_space['query_expansion_temperature'], 'float'
            )
        
        if 'query_expansion_max_token' in self.search_space:
            params['query_expansion_max_token'] = self._suggest_value(
                trial, 'query_expansion_max_token', 
                self.search_space['query_expansion_max_token'], 'int'
            )
        
        self._add_query_expansion_retrieval_params(trial, params)


    def _add_query_expansion_retrieval_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'query_expansion_retrieval_method' in self.search_space:
            params['query_expansion_retrieval_method'] = trial.suggest_categorical(
                'query_expansion_retrieval_method', 
                self.search_space['query_expansion_retrieval_method']
            )
            
            if params['query_expansion_retrieval_method'] == 'bm25' and 'query_expansion_bm25_tokenizer' in self.search_space:
                params['query_expansion_bm25_tokenizer'] = trial.suggest_categorical(
                    'query_expansion_bm25_tokenizer', 
                    self.search_space['query_expansion_bm25_tokenizer']
                )
            elif params['query_expansion_retrieval_method'] == 'vectordb' and 'query_expansion_vectordb_name' in self.search_space:
                params['query_expansion_vectordb_name'] = trial.suggest_categorical(
                    'query_expansion_vectordb_name', 
                    self.search_space['query_expansion_vectordb_name']
                )

                    
    def _suggest_retrieval_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        # Always suggest retriever_top_k if available
        if 'retriever_top_k' in self.search_space:
            params['retriever_top_k'] = self._suggest_value(
                trial, 'retriever_top_k', 
                self.search_space['retriever_top_k'], 'int'
            )
        
        # Check if query expansion is active
        is_qe_active = False
        if 'query_expansion_config' in params:
            is_qe_active = params['query_expansion_config'] != 'pass_query_expansion'
        elif 'query_expansion_method' in params:
            is_qe_active = params['query_expansion_method'] != 'pass_query_expansion'
        
        # If query expansion is active, skip retrieval method params
        if is_qe_active:
            print(f"[DEBUG] Active query expansion detected, skipping retrieval method params")
            return
        
        # Only suggest retrieval params if no active query expansion
        if 'retrieval_config' in self.search_space:
            params['retrieval_config'] = trial.suggest_categorical('retrieval_config', 
                                                                self.search_space['retrieval_config'])
            return
        
        if 'retrieval_method' in self.search_space:
            params['retrieval_method'] = trial.suggest_categorical('retrieval_method', 
                                                                self.search_space['retrieval_method'])
            
            if params.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in self.search_space:
                params['bm25_tokenizer'] = trial.suggest_categorical('bm25_tokenizer', 
                                                                self.search_space['bm25_tokenizer'])
            elif params.get('retrieval_method') == 'vectordb' and 'vectordb_name' in self.search_space:
                params['vectordb_name'] = trial.suggest_categorical('vectordb_name', 
                                                                self.search_space['vectordb_name'])
            
    def _suggest_filter_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        reranker_top_k = params.get('reranker_top_k', None)
        if reranker_top_k == 1:
            params['passage_filter_config'] = 'pass_passage_filter'
            params['passage_filter_method'] = 'pass_passage_filter'
            print(f"  Auto-setting filter to 'pass' because reranker_top_k=1")
            return
        
        if self.search_type == 'bo' and 'passage_filter_method' in self.search_space:
            params['passage_filter_method'] = trial.suggest_categorical(
                'passage_filter_method', 
                self.search_space['passage_filter_method']
            )
            
            if params['passage_filter_method'] == 'pass_passage_filter':
                return
            
            filter_method = params['passage_filter_method']

            if filter_method == 'threshold_cutoff' and 'threshold_cutoff_threshold' in self.search_space:
                threshold_range = self.search_space['threshold_cutoff_threshold']
                params['threshold'] = trial.suggest_float(
                    'threshold_cutoff_threshold',
                    threshold_range[0], threshold_range[1]
                )
            elif filter_method == 'percentile_cutoff' and 'percentile_cutoff_percentile' in self.search_space:
                percentile_range = self.search_space['percentile_cutoff_percentile']
                params['percentile'] = trial.suggest_float(
                    'percentile_cutoff_percentile',
                    percentile_range[0], percentile_range[1]
                )
            elif filter_method == 'similarity_threshold_cutoff' and 'similarity_threshold_cutoff_threshold' in self.search_space:
                threshold_range = self.search_space['similarity_threshold_cutoff_threshold']
                params['threshold'] = trial.suggest_float(
                    'similarity_threshold_cutoff_threshold',
                    threshold_range[0], threshold_range[1]
                )
            elif filter_method == 'similarity_percentile_cutoff' and 'similarity_percentile_cutoff_percentile' in self.search_space:
                percentile_range = self.search_space['similarity_percentile_cutoff_percentile']
                params['percentile'] = trial.suggest_float(
                    'similarity_percentile_cutoff_percentile',
                    percentile_range[0], percentile_range[1]
                )
            
            return
        
        if 'passage_filter_config' in self.search_space:
            filter_config = trial.suggest_categorical('passage_filter_config', 
                                                    self.search_space['passage_filter_config'])
            params['passage_filter_config'] = filter_config
            return
        
        if 'passage_filter_method' not in self.search_space:
            return
            
        params['passage_filter_method'] = trial.suggest_categorical('passage_filter_method', 
                                                                self.search_space['passage_filter_method'])
        
        if params['passage_filter_method'] == 'pass_passage_filter':
            return
            
        filter_method = params['passage_filter_method']

        if filter_method == 'threshold_cutoff' and 'threshold_cutoff_threshold' in self.search_space:
            params['threshold'] = self._suggest_value(
                trial, 'threshold_cutoff_threshold',
                self.search_space['threshold_cutoff_threshold'], 'float'
            )
        elif filter_method == 'similarity_threshold_cutoff' and 'similarity_threshold_cutoff_threshold' in self.search_space:
            params['threshold'] = self._suggest_value(
                trial, 'similarity_threshold_cutoff_threshold',
                self.search_space['similarity_threshold_cutoff_threshold'], 'float'
            )
        elif filter_method == 'percentile_cutoff' and 'percentile_cutoff_percentile' in self.search_space:
            params['percentile'] = self._suggest_value(
                trial, 'percentile_cutoff_percentile',
                self.search_space['percentile_cutoff_percentile'], 'float'
            )
        elif filter_method == 'similarity_percentile_cutoff' and 'similarity_percentile_cutoff_percentile' in self.search_space:
            params['percentile'] = self._suggest_value(
                trial, 'similarity_percentile_cutoff_percentile',
                self.search_space['similarity_percentile_cutoff_percentile'], 'float'
            )
        
    def _suggest_reranker_params(self, trial: optuna.Trial, params: Dict[str, Any]):
    
        if 'reranker_config' in self.search_space:
            params['reranker_config'] = trial.suggest_categorical('reranker_config',
                                                                self.search_space['reranker_config'])
            
            if 'reranker_top_k' in self.search_space:
                if self.search_type == 'bo' and 'retriever_top_k' in params:
                    reranker_range = self.search_space['reranker_top_k']
                    if isinstance(reranker_range, tuple):
                        max_reranker_k = min(reranker_range[1], params['retriever_top_k'])
                        params['reranker_top_k'] = trial.suggest_int('reranker_top_k', 
                                                                    reranker_range[0], 
                                                                    max_reranker_k)
                    else:
                        params['reranker_top_k'] = self._suggest_value(
                            trial, 'reranker_top_k', 
                            self.search_space['reranker_top_k'], 'int'
                        )
                else:
                    params['reranker_top_k'] = self._suggest_value(
                        trial, 'reranker_top_k', 
                        self.search_space['reranker_top_k'], 'int'
                    )
            return

        if 'passage_reranker_method' not in self.search_space:
            return
            
        params['passage_reranker_method'] = trial.suggest_categorical('passage_reranker_method', 
                                                                    self.search_space['passage_reranker_method'])
        
        if params['passage_reranker_method'] == 'pass_reranker':
            return
        
        reranker_method = params['passage_reranker_method']

        if reranker_method == 'sap_api':
            unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
            api_endpoints = unified_params.get('api_endpoints', {})
            
            if 'sap_api' in api_endpoints:
                params['reranker_api_url'] = api_endpoints['sap_api']

            params['reranker_model_name'] = 'cohere-rerank-v3.5'
        else:
            model_key = f"{reranker_method}_model_name"
            if model_key in self.search_space:
                params['reranker_model_name'] = trial.suggest_categorical(model_key,
                                                                        self.search_space[model_key])
            else:
                model_key = f"{reranker_method}_model"
                if model_key in self.search_space:
                    params['reranker_model'] = trial.suggest_categorical(model_key,
                                                                    self.search_space[model_key])
            
        if 'reranker_top_k' in self.search_space:
            if self.search_type == 'bo' and 'retriever_top_k' in params:
                reranker_range = self.search_space['reranker_top_k']
                if isinstance(reranker_range, tuple):
                    max_reranker_k = min(reranker_range[1], params['retriever_top_k'])
                    params['reranker_top_k'] = trial.suggest_int('reranker_top_k', 
                                                                reranker_range[0], 
                                                                max_reranker_k)
                else:
                    params['reranker_top_k'] = self._suggest_value(
                        trial, 'reranker_top_k', 
                        self.search_space['reranker_top_k'], 'int'
                    )
            else:
                params['reranker_top_k'] = self._suggest_value(
                    trial, 'reranker_top_k', 
                    self.search_space['reranker_top_k'], 'int'
                )

    def _suggest_compressor_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'passage_compressor_config' in self.search_space:
            comp_config_str = trial.suggest_categorical('passage_compressor_config',
                                                    self.search_space['passage_compressor_config'])
            
            if comp_config_str != 'pass_compressor':
                # Handle lexrank
                if comp_config_str == 'lexrank':
                    params['passage_compressor_method'] = 'lexrank'

                    if 'lexrank_compression_ratio' in self.search_space:
                        params['compression_ratio'] = self._suggest_value(
                            trial, 'lexrank_compression_ratio',
                            self.search_space['lexrank_compression_ratio'], 'float'
                        )
                    
                    if 'lexrank_threshold' in self.search_space:
                        params['threshold'] = self._suggest_value(
                            trial, 'lexrank_threshold',
                            self.search_space['lexrank_threshold'], 'float'
                        )
                    
                    if 'lexrank_damping' in self.search_space:
                        params['damping'] = self._suggest_value(
                            trial, 'lexrank_damping',
                            self.search_space['lexrank_damping'], 'float'
                        )
                    
                    if 'lexrank_max_iterations' in self.search_space:
                        params['max_iterations'] = self._suggest_value(
                            trial, 'lexrank_max_iterations',
                            self.search_space['lexrank_max_iterations'], 'int'
                        )
                
                # Handle spacy
                elif comp_config_str.startswith('spacy::'):
                    parts = comp_config_str.split('::', 1)
                    params['passage_compressor_method'] = 'spacy'
                    if len(parts) > 1:
                        params['spacy_model'] = parts[1]

                    if 'spacy_compression_ratio' in self.search_space:
                        params['compression_ratio'] = self._suggest_value(
                            trial, 'spacy_compression_ratio',
                            self.search_space['spacy_compression_ratio'], 'float'
                        )
                
                # Handle LLM-based compressors 
                elif '::' in comp_config_str:
                    parts = comp_config_str.split('::', 3)
                    if len(parts) >= 3:
                        method, gen_type, model = parts[0], parts[1], parts[2]
                        params['passage_compressor_method'] = method
                        params['compressor_generator_module_type'] = gen_type
                        params['compressor_model'] = model
                        
                        unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                        for comp_config in unified_params.get('compressor_configs', []):
                            if (comp_config['method'] == method and 
                                comp_config['generator_module_type'] == gen_type and 
                                model in comp_config['models']):
                                params['compressor_llm'] = comp_config.get('llm', 'openai')
                                if gen_type == 'sap_api':
                                    params['compressor_api_url'] = comp_config.get('api_url')
                                elif gen_type == 'vllm':
                                    params['compressor_llm'] = model
                                break
                else:
                    params['passage_compressor_method'] = comp_config_str
            else:
                params['passage_compressor_method'] = comp_config_str
            
            if 'compressor_batch' in self.search_space:
                params['compressor_batch'] = trial.suggest_categorical('compressor_batch',
                                                                    self.search_space['compressor_batch'])
            return

        if 'passage_compressor_method' not in self.search_space:
            return
            
        params['passage_compressor_method'] = trial.suggest_categorical('passage_compressor_method', 
                                                                    self.search_space['passage_compressor_method'])
        
        if params['passage_compressor_method'] == 'pass_compressor':
            return

        method = params['passage_compressor_method']
        
        if method == 'lexrank':
            if 'compression_ratio' in self.search_space:
                params['compression_ratio'] = self._suggest_value(
                    trial, 'compression_ratio',
                    self.search_space['compression_ratio'], 'float'
                )
            if 'threshold' in self.search_space:
                params['threshold'] = self._suggest_value(
                    trial, 'threshold',
                    self.search_space['threshold'], 'float'
                )
            if 'damping' in self.search_space:
                params['damping'] = self._suggest_value(
                    trial, 'damping',
                    self.search_space['damping'], 'float'
                )
            if 'max_iterations' in self.search_space:
                params['max_iterations'] = self._suggest_value(
                    trial, 'max_iterations',
                    self.search_space['max_iterations'], 'int'
                )
        
        elif method == 'spacy':
            if 'compression_ratio' in self.search_space:
                params['compression_ratio'] = self._suggest_value(
                    trial, 'compression_ratio',
                    self.search_space['compression_ratio'], 'float'
                )
            if 'spacy_model' in self.search_space:
                params['spacy_model'] = trial.suggest_categorical(
                    'spacy_model',
                    self.search_space['spacy_model']
                )
        
        elif method in ['tree_summarize', 'refine']:
            if 'compressor_generator_module_type' in self.search_space:
                params['compressor_generator_module_type'] = trial.suggest_categorical(
                    'compressor_generator_module_type', 
                    self.search_space['compressor_generator_module_type']
                )
            
            if 'compressor_llm' in self.search_space:
                params['compressor_llm'] = trial.suggest_categorical('compressor_llm', 
                                                                self.search_space['compressor_llm'])
            
            if 'compressor_model' in self.search_space:
                params['compressor_model'] = trial.suggest_categorical('compressor_model', 
                                                                    self.search_space['compressor_model'])
            
            if 'compressor_batch' in self.search_space:
                params['compressor_batch'] = self._suggest_value(
                    trial, 'compressor_batch',
                    self.search_space['compressor_batch'], 'int'
                )

    def _parse_composite_config(self, config_str: str, component: str, 
                            params: Dict[str, Any], config_generator) -> None:
        
        if component == 'passage_compressor' and config_str != 'pass_compressor':
            parts = config_str.split('::', 3)
            if len(parts) == 3:
                method, gen_type, model = parts
                params['passage_compressor_method'] = method
                params['compressor_generator_module_type'] = gen_type
                params['compressor_model'] = model

                unified_params = config_generator.extract_unified_parameters('passage_compressor')
                for comp_config in unified_params.get('compressor_configs', []):
                    if (comp_config['method'] == method and 
                        comp_config['generator_module_type'] == gen_type and 
                        model in comp_config['models']):
                        
                        params['compressor_llm'] = comp_config.get('llm', 'openai')
                        
                        if gen_type == 'sap_api':
                            params['compressor_api_url'] = comp_config.get('api_url')
                            params['compressor_bearer_token'] = comp_config.get('bearer_token')
                        elif gen_type == 'vllm':
                            params['compressor_llm'] = model
                            if 'tensor_parallel_size' in comp_config:
                                params['compressor_tensor_parallel_size'] = comp_config['tensor_parallel_size']
                        
                        if 'batch' in comp_config:
                            params['compressor_batch'] = comp_config['batch']
                        break
    
    def _suggest_prompt_maker_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'prompt_config' in self.search_space:
            prompt_configs = self.search_space['prompt_config']
            if prompt_configs:  
                params['prompt_config'] = trial.suggest_categorical('prompt_config', prompt_configs)
            return
        
        if 'prompt_maker_method' not in self.search_space:
            return
        
        prompt_methods = self.search_space.get('prompt_maker_method', [])
        if not prompt_methods:
            return
            
        params['prompt_maker_method'] = trial.suggest_categorical('prompt_maker_method', prompt_methods)
        
        if params['prompt_maker_method'] == 'pass_prompt_maker':
            return
            
        if 'prompt_template_idx' in self.search_space:
            template_indices = self.search_space.get('prompt_template_idx', [])
            if template_indices: 
                params['prompt_template_idx'] = self._suggest_value(
                    trial, 'prompt_template_idx',
                    template_indices, 'int'
                )

    
    def _suggest_generator_params(self, trial: optuna.Trial, params: Dict[str, Any]):
        if 'generator_config' in self.search_space:
            generator_configs = self.search_space.get('generator_config', {})
            
            if isinstance(generator_configs, dict) and 'values' in generator_configs:
                gen_config_str = trial.suggest_categorical('generator_config', 
                                                        generator_configs['values'])
                
                metadata = generator_configs.get('metadata', {})
                if metadata and gen_config_str in metadata:
                    config_metadata = metadata[gen_config_str]
                    
                    module_type, model = gen_config_str.split('::', 1)
                    params['generator_module_type'] = module_type
                    params['generator_model'] = model
                    
                    if 'api_url' in config_metadata:
                        params['generator_api_url'] = config_metadata['api_url']
                    if 'llm' in config_metadata:
                        params['generator_llm'] = config_metadata['llm']
            else:
                gen_config_str = trial.suggest_categorical('generator_config', generator_configs)
                module_type, model = gen_config_str.split('::', 1)
                params['generator_module_type'] = module_type
                params['generator_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('generator')
                for module_config in unified_params.get('module_configs', []):
                    if module_config['module_type'] == module_type and model in module_config['models']:
                        if module_type == 'sap_api':
                            params['generator_api_url'] = module_config.get('api_url')
                            params['generator_llm'] = module_config.get('llm', 'mistralai')
                        elif module_type == 'vllm':
                            params['generator_llm'] = model
                        else:
                            params['generator_llm'] = module_config.get('llm', 'openai')
                        break
            
            if 'generator_temperature' in self.search_space:
                temp_spec = self.search_space.get('generator_temperature')
                if temp_spec:
                    temp_value = self._suggest_value(
                        trial, 'generator_temperature',
                        temp_spec, 'float'
                    )
                    params['generator_temperature'] = round(float(temp_value), 4)
            
            if 'generator_max_tokens' in self.search_space:
                if params.get('generator_module_type') == 'sap_api':
                    params['generator_max_tokens'] = trial.suggest_int(
                        'generator_max_tokens',
                        self.search_space['generator_max_tokens'][0],
                        self.search_space['generator_max_tokens'][1]
                    )
            
            return
        
        if 'generator_module_type' in self.search_space:
            module_types = self.search_space.get('generator_module_type', [])
            if module_types:
                params['generator_module_type'] = trial.suggest_categorical('generator_module_type', module_types)

        if 'generator_model' in self.search_space:
            models = self.search_space.get('generator_model', [])
            if not models:
                raise ValueError("No generator models available in search space")
            params['generator_model'] = trial.suggest_categorical('generator_model', models)
            
        if 'generator_temperature' in self.search_space:
            temp_spec = self.search_space.get('generator_temperature')
            if temp_spec:
                temp_value = self._suggest_value(
                    trial, 'generator_temperature',
                    temp_spec, 'float'
                )
                params['generator_temperature'] = round(float(temp_value), 4)
            
    def _check_cache(self, trial: optuna.Trial, params: Dict[str, Any]):
        cached_df, cached_metrics = self.component_cache.check_cache('generator', params)
        if cached_metrics is None:
            return None
            
        print(f"Using cached results for trial {trial.number}")
        
        cached_score = cached_metrics.get('final_combined_score', 0.0)
        if cached_score == 0.0:
            cached_score = cached_metrics.get('combined_score', 0.0)
            if cached_score == 0.0:
                cached_score = cached_metrics.get('mean_score', 0.0)
        
        for key, value in cached_metrics.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                trial.set_user_attr(key, value)
        
        return cached_score
    
    def _run_trial(self, trial: optuna.Trial, params: Dict[str, Any], trial_dir: str) -> float:
        trial_start_time = time.time()

        self._process_composite_configs(params)
        
        centralized_corpus_path = os.path.join(self.pipeline_runner._get_centralized_project_dir(), "data", "corpus.parquet")
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
        
        qa_subset_path = os.path.join(trial_dir, "data", "qa.parquet")
        self.qa_df.to_parquet(qa_subset_path, index=False)

        results = self.pipeline_runner.run_pipeline(
            config=params,
            trial_dir=trial_dir,
            qa_subset=self.qa_df,
            is_local_optimization=False,
            current_component=None
        )
    
        execution_time = time.time() - trial_start_time
        trial.set_user_attr('execution_time', execution_time)
        
        score = results.get("combined_score", 0.0)
        
        if self.pipeline_runner.use_ragas and 'ragas_mean_score' in results:
            score = results['ragas_mean_score']
            trial.set_user_attr('ragas_mean_score', score)
            if 'ragas_metrics' in results:
                for metric_name, metric_value in results['ragas_metrics'].items():
                    if isinstance(metric_value, (int, float)):
                        trial.set_user_attr(f'ragas_{metric_name}', metric_value)
        
        if 'compression_score' in results and 'compressor_score' not in results:
            trial.set_user_attr('compressor_score', results['compression_score'])
        
        for key, value in results.items():
            if key not in ['config', 'trial_dir', 'timestamp', 'training_iteration', 
                        'iteration_scores', 'iteration_combined_scores', 'weighted_score', 
                        'weighted_combined_score', 'score', 'error', 'working_df']:
                if not isinstance(value, pd.DataFrame):
                    trial.set_user_attr(key, value)
        
        return score
    
    def _process_composite_configs(self, params: Dict[str, Any]):
 
        if 'generator_config' in params:
            gen_config_str = params['generator_config']
            module_type, model = gen_config_str.split('::', 1)
            
            params['generator_module_type'] = module_type
            params['generator_model'] = model

            if hasattr(self, 'search_space') and 'generator_config' in self.search_space:
                generator_configs = self.search_space.get('generator_config', {})

                if isinstance(generator_configs, dict) and 'metadata' in generator_configs:
                    metadata = generator_configs.get('metadata', {})
                    
                    if metadata and gen_config_str in metadata:
                        config_metadata = metadata[gen_config_str]

                        if 'api_url' in config_metadata:
                            params['generator_api_url'] = config_metadata['api_url']
                        if 'llm' in config_metadata:
                            params['generator_llm'] = config_metadata['llm']
                        if 'max_tokens' in config_metadata and not isinstance(config_metadata['max_tokens'], list):
                            params['generator_max_tokens'] = config_metadata['max_tokens']
                else:
                    unified_params = self.config_generator.extract_unified_parameters('generator')
                    for module_config in unified_params.get('module_configs', []):
                        if module_config['module_type'] == module_type and model in module_config['models']:
                            if module_type == 'sap_api':
                                params['generator_api_url'] = module_config.get('api_url')
                                params['generator_llm'] = module_config.get('llm', 'mistralai')
                            elif module_type == 'vllm':
                                params['generator_llm'] = model  
                            else:
                                params['generator_llm'] = module_config.get('llm', 'openai')
                            break

        if 'query_expansion_config' in params:
            qe_config_str = params['query_expansion_config']
            if qe_config_str != 'pass_query_expansion' and '::' in qe_config_str:
                parts = qe_config_str.split('::', 2)
                if len(parts) >= 3:
                    method, gen_type, model = parts
                    params['query_expansion_method'] = method
                    params['query_expansion_generator_module_type'] = gen_type
                    params['query_expansion_model'] = model

                    if hasattr(self, 'search_space') and 'query_expansion_config' in self.search_space:
                        qe_configs = self.search_space.get('query_expansion_config', {})

                        if isinstance(qe_configs, dict) and 'metadata' in qe_configs:
                            metadata = qe_configs.get('metadata', {})
                            
                            if metadata and qe_config_str in metadata:
                                config_metadata = metadata[qe_config_str]
                                if 'api_url' in config_metadata:
                                    params['query_expansion_api_url'] = config_metadata['api_url']
                                if 'llm' in config_metadata:
                                    params['query_expansion_llm'] = config_metadata['llm']
                        else:
                            unified_params = self.config_generator.extract_unified_parameters('query_expansion')
                            for gen_config in unified_params.get('generator_configs', []):
                                if (gen_config['method'] == method and 
                                    gen_config['generator_module_type'] == gen_type and 
                                    model in gen_config['models']):
                                    if gen_type == 'sap_api':
                                        params['query_expansion_api_url'] = gen_config.get('api_url')
                                        params['query_expansion_llm'] = gen_config.get('llm', 'mistralai')
                                    else:
                                        params['query_expansion_llm'] = gen_config.get('llm', 'openai')
                                    break

        if 'passage_compressor_config' in params:
            comp_config_str = params['passage_compressor_config']
            if comp_config_str != 'pass_compressor' and '::' in comp_config_str:
                parts = comp_config_str.split('::', 3)
                if len(parts) >= 3:
                    method, gen_type, model = parts
                    params['passage_compressor_method'] = method
                    params['compressor_generator_module_type'] = gen_type
                    params['compressor_model'] = model

                    if hasattr(self, 'search_space') and 'passage_compressor_config' in self.search_space:
                        comp_configs = self.search_space.get('passage_compressor_config', {})

                        if isinstance(comp_configs, dict) and 'metadata' in comp_configs:
                            metadata = comp_configs.get('metadata', {})
                            
                            if metadata and comp_config_str in metadata:
                                config_metadata = metadata[comp_config_str]
                                if 'api_url' in config_metadata:
                                    params['compressor_api_url'] = config_metadata['api_url']
                                if 'llm' in config_metadata:
                                    params['compressor_llm'] = config_metadata['llm']
                        else:
                            unified_params = self.config_generator.extract_unified_parameters('passage_compressor')
                            for comp_config in unified_params.get('compressor_configs', []):
                                if (comp_config['method'] == method and 
                                    comp_config['generator_module_type'] == gen_type and 
                                    model in comp_config['models']):
                                    params['compressor_llm'] = comp_config.get('llm', 'openai')
                                    if gen_type == 'sap_api':
                                        params['compressor_api_url'] = comp_config.get('api_url')
                                    elif gen_type == 'vllm':
                                        params['compressor_llm'] = model
                                    break
    
    def _save_to_cache(self, params: Dict[str, Any], score: float, user_attrs: Dict[str, Any]):
        dummy_df = pd.DataFrame({'completed': [True]})
        cache_metrics = {
            'final_combined_score': score,
            **user_attrs
        }
        self.component_cache.save_to_cache('generator', params, dummy_df, cache_metrics)