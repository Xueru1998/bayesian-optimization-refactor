import optuna
import pandas as pd
import tempfile
import os
import shutil
from typing import Dict, Any, Union, Tuple

class OptunaObjective:
    def __init__(self, search_space, config_generator, pipeline_runner, 
                 component_cache, corpus_df, qa_df, use_cache=True):
        self.search_space = search_space
        self.config_generator = config_generator
        self.pipeline_runner = pipeline_runner
        self.component_cache = component_cache
        self.corpus_df = corpus_df
        self.qa_df = qa_df
        self.use_cache = use_cache
        self.cached_pipeline_runner = None
        
        self.has_query_expansion = self.config_generator.node_exists("query_expansion")
        self.has_retrieval = self.config_generator.node_exists("retrieval")
    
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

        params = self._suggest_params(trial)
        
        print(f"\nRunning trial {trial.number} with params: {params}")
        
        if self.use_cache:
            cached_score = self._check_cache(trial, params)
            if cached_score is not None:
                return cached_score
        
        trial_dir = tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_")
        
        try:
            score = self._run_trial(trial, params, trial_dir)
            
            if self.use_cache:
                self._save_to_cache(params, score, trial.user_attrs)
            
            return score
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            if os.path.exists(trial_dir):
                try:
                    shutil.rmtree(trial_dir)
                except:
                    pass
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}

        self._suggest_query_expansion_params(trial, params)

        if not self.has_query_expansion or params.get('query_expansion_method') == 'pass_query_expansion':
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
            if param_type == 'int':
                return trial.suggest_int(param_name, param_spec[0], param_spec[1])
            else:
                return trial.suggest_float(param_name, param_spec[0], param_spec[1])
        else:
            raise ValueError(f"Invalid parameter specification for {param_name}: {param_spec}")
    
    def _suggest_query_expansion_params(self, trial: optuna.Trial, params: Dict[str, Any]):
    
        if 'query_expansion_config' in self.search_space:
            params['query_expansion_config'] = trial.suggest_categorical('query_expansion_config',
                                                                    self.search_space['query_expansion_config'])
            return
        
        if 'query_expansion_method' not in self.search_space:
            return
            
        params['query_expansion_method'] = trial.suggest_categorical('query_expansion_method', 
                                                                self.search_space['query_expansion_method'])
        
        if params['query_expansion_method'] == 'pass_query_expansion':
            return

        if 'retriever_top_k' in self.search_space:
            params['retriever_top_k'] = self._suggest_value(
                trial, 'retriever_top_k', 
                self.search_space['retriever_top_k'], 'int'
            )

        if 'retrieval_method' in params:
            del params['retrieval_method']
        if 'bm25_tokenizer' in params:
            del params['bm25_tokenizer']
        if 'vectordb_name' in params:
            del params['vectordb_name']
        
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
        
        if params['query_expansion_method'] == 'hyde' and 'query_expansion_max_token' in self.search_space:
            params['query_expansion_max_token'] = self._suggest_value(
                trial, 'query_expansion_max_token', 
                self.search_space['query_expansion_max_token'], 'int'
            )
        elif params['query_expansion_method'] == 'multi_query_expansion' and 'query_expansion_temperature' in self.search_space:
            params['query_expansion_temperature'] = self._suggest_value(
                trial, 'query_expansion_temperature', 
                self.search_space['query_expansion_temperature'], 'float'
            )
        
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
        if self.has_query_expansion and params.get('query_expansion_method') and params.get('query_expansion_method') != 'pass_query_expansion':
            print(f"[DEBUG] Skipping ALL retrieval params - query expansion is active: {params.get('query_expansion_method')}")
            return

        if 'retriever_top_k' in self.search_space:
            params['retriever_top_k'] = self._suggest_value(
                trial, 'retriever_top_k', 
                self.search_space['retriever_top_k'], 'int'
            )

        if 'retrieval_config' in self.search_space:
            params['retrieval_config'] = trial.suggest_categorical('retrieval_config', 
                                                                self.search_space['retrieval_config'])
            return

        if 'retrieval_method' not in self.search_space:
            if self.has_retrieval and 'retrieval_method' not in params:
                retrieval_params = self.config_generator.extract_unified_parameters('retrieval')
                if retrieval_params.get('methods'):
                    params['retrieval_method'] = retrieval_params['methods'][0]
                    
                    if params['retrieval_method'] == 'bm25' and retrieval_params.get('bm25_tokenizers'):
                        params['bm25_tokenizer'] = retrieval_params['bm25_tokenizers'][0]
                    elif params['retrieval_method'] == 'vectordb' and retrieval_params.get('vectordb_names'):
                        params['vectordb_name'] = retrieval_params['vectordb_names'][0]
            return
        
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
        
        if 'passage_filter_method' in self.search_space:
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
                if 'retriever_top_k' in params:
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
            if 'retriever_top_k' in params:
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
            params['passage_compressor_config'] = trial.suggest_categorical('passage_compressor_config',
                                                                self.search_space['passage_compressor_config'])
            return
        
        if 'compressor_config' in self.search_space:
            params['compressor_config'] = trial.suggest_categorical('compressor_config',
                                                                self.search_space['compressor_config'])
            
            if 'compressor_batch' in self.search_space:
                params['compressor_batch'] = self._suggest_value(
                    trial, 'compressor_batch',
                    self.search_space['compressor_batch'], 'int'
                )
            return
        
        if 'passage_compressor_method' not in self.search_space:
            return
            
        params['passage_compressor_method'] = trial.suggest_categorical('passage_compressor_method', 
                                                                    self.search_space['passage_compressor_method'])
        
        if params['passage_compressor_method'] == 'pass_compressor':
            return
        
        if params['passage_compressor_method'] in ['tree_summarize', 'refine']:
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
        
        elif params['passage_compressor_method'] == 'lexrank':
            if 'compressor_compression_ratio' in self.search_space:
                params['compressor_compression_ratio'] = self._suggest_value(
                    trial, 'compressor_compression_ratio',
                    self.search_space['compressor_compression_ratio'], 'float'
                )
            if 'compressor_threshold' in self.search_space:
                params['compressor_threshold'] = self._suggest_value(
                    trial, 'compressor_threshold', 
                    self.search_space['compressor_threshold'], 'float'
                )
            if 'compressor_damping' in self.search_space:
                params['compressor_damping'] = self._suggest_value(
                    trial, 'compressor_damping',
                    self.search_space['compressor_damping'], 'float'
                )
            if 'compressor_max_iterations' in self.search_space:
                params['compressor_max_iterations'] = self._suggest_value(
                    trial, 'compressor_max_iterations',
                    self.search_space['compressor_max_iterations'], 'int'
                )
        
        elif params['passage_compressor_method'] == 'spacy':
            if 'compressor_compression_ratio' in self.search_space:
                params['compressor_compression_ratio'] = self._suggest_value(
                    trial, 'compressor_compression_ratio',
                    self.search_space['compressor_compression_ratio'], 'float'
                )
            if 'compressor_spacy_model' in self.search_space:
                params['compressor_spacy_model'] = trial.suggest_categorical('compressor_spacy_model',
                                                                        self.search_space['compressor_spacy_model'])
    
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
        import time
        from pipeline_component.nodes.retrieval import RetrievalModule
        
        trial_start_time = time.time()
        
        retrieval_module = RetrievalModule(
            base_project_dir=trial_dir,
            use_pregenerated_embeddings=True,
            centralized_project_dir=self.pipeline_runner._get_centralized_project_dir()
        )
        retrieval_module.prepare_project_dir(trial_dir, self.corpus_df, self.qa_df)
        

        results = self.pipeline_runner.run_pipeline(params, trial_dir, self.qa_df)
        
        execution_time = time.time() - trial_start_time
        trial.set_user_attr('execution_time', execution_time)
        
        score = results.get("combined_score", 0.0)
        
        if 'compression_score' in results and 'compressor_score' not in results:
            trial.set_user_attr('compressor_score', results['compression_score'])
        
        for key, value in results.items():
            if key not in ['config', 'trial_dir', 'timestamp', 'training_iteration', 
                        'iteration_scores', 'iteration_combined_scores', 'weighted_score', 
                        'weighted_combined_score', 'score', 'error']:
                trial.set_user_attr(key, value)
        
        return score
    
    def _save_to_cache(self, params: Dict[str, Any], score: float, user_attrs: Dict[str, Any]):
        dummy_df = pd.DataFrame({'completed': [True]})
        cache_metrics = {
            'final_combined_score': score,
            **user_attrs
        }
        self.component_cache.save_to_cache('generator', params, dummy_df, cache_metrics)