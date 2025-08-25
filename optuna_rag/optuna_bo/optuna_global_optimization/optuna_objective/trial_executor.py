import optuna
import time
import os
import shutil
import pandas as pd
from typing import Dict, Any


class TrialExecutor:
    def __init__(self, pipeline_runner, corpus_df, qa_df, config_generator, search_space):
        self.pipeline_runner = pipeline_runner
        self.corpus_df = corpus_df
        self.qa_df = qa_df
        self.config_generator = config_generator
        self.search_space = search_space
    
    def run_trial(self, trial: optuna.Trial, params: Dict[str, Any], trial_dir: str) -> float:
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