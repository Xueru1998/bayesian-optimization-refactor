import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any
from pipeline.utils import Utils, NumpyEncoder


class EarlyStoppingException(Exception):
    def __init__(self, message, score, component):
        self.message = message
        self.score = score
        self.component = component
        super().__init__(self.message)


class EarlyStoppingHandler:
    def __init__(self, early_stopping_thresholds=None):
        if early_stopping_thresholds is None:
            self.early_stopping_thresholds = {
                'retrieval': 0.1,
                'query_expansion': 0.1,
                'reranker': 0.2,
                'filter': 0.25,
                'compressor': 0.3
            }
        else:
            self.early_stopping_thresholds = early_stopping_thresholds
    
    def check_early_stopping(self, component: str, score: float, is_local_optimization: bool = False) -> None:
        if is_local_optimization:
            return
        
        if self.early_stopping_thresholds is None:
            return
        
        threshold = self.early_stopping_thresholds.get(component)
        if threshold is not None and score < threshold:
            print(f"\n[EARLY STOPPING] {component} score {score:.4f} < threshold {threshold}")
            raise EarlyStoppingException(
                f"Early stopping at {component}: score {score:.4f} below threshold {threshold}",
                score=score,
                component=component
            )


class PipelineUtilities:
    def __init__(self, use_ragas: bool = False):
        self.use_ragas = use_ragas
    
    def print_configuration(self, config: Dict[str, Any]):
        print("\n" + "="*80)
        print("SELECTED CONFIGURATION FROM OPTIMIZER:")
        print("="*80)
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
        print("="*80 + "\n")
    
    def validate_global_optimization(self, config: Dict[str, Any]) -> bool:
        if not self.use_ragas:
            return True
            
        has_retrieval = ('retrieval_method' in config or 'query_expansion_method' in config)
        has_prompt_maker = any('prompt' in k for k in config.keys())
        has_generator = any('generator' in k for k in config.keys())
        
        if not (has_retrieval and has_prompt_maker and has_generator):
            print("WARNING: Global optimization with RAGAS requires retrieval, prompt_maker, and generator components")
            return False
        return True
    
    def get_centralized_project_dir(self):
        return Utils.get_centralized_project_dir()


class IntermediateResultsHandler:
    def save_intermediate_result(self, component: str, working_df: pd.DataFrame, 
                                results: Dict[str, Any], trial_dir: str, config: Dict[str, Any] = None):
        try:
            debug_dir = os.path.join(trial_dir, "debug_intermediate_results")
            os.makedirs(debug_dir, exist_ok=True)

            df_path = os.path.join(debug_dir, f"{component}_dataframe.parquet")
            working_df.to_parquet(df_path, index=False)

            results_to_save = {}
            for key, value in results.items():
                if not isinstance(value, pd.DataFrame):
                    results_to_save[key] = Utils.json_serializable(value)
            
            results_path = os.path.join(debug_dir, f"{component}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results_to_save, f, indent=2, cls=NumpyEncoder)

            summary = {
                'component': component,
                'timestamp': time.time(),
                'num_rows': len(working_df),
                'columns': list(working_df.columns),
                'score': self._get_component_score(component, results),
                'execution_time': results.get('execution_time', 0.0)
            }
            
            if config:
                summary['config_explored'] = {
                    'full_config': config,
                    'component_specific_config': self._extract_component_config(component, config)
                }

            if component in ['retrieval', 'query_expansion']:
                summary['retrieval_score'] = results.get('retrieval_score', results.get('last_retrieval_score', 0.0))
                summary['retrieval_metrics'] = results.get('retrieval_metrics', {})
            elif component == 'passage_reranker':
                summary['reranker_score'] = results.get('reranker_score', 0.0)
                summary['reranker_metrics'] = results.get('reranker_metrics', {})
            elif component == 'passage_filter':
                summary['filter_score'] = results.get('filter_score', 0.0)
                summary['filter_metrics'] = results.get('filter_metrics', {})
            elif component == 'passage_compressor':
                summary['compression_score'] = results.get('compression_score', 0.0)
                summary['compressor_metrics'] = results.get('compression_metrics', {})
            elif component == 'prompt_maker':
                summary['prompt_maker_score'] = results.get('prompt_maker_score', 0.0)
                summary['prompt_maker_metrics'] = results.get('prompt_metrics', {})
            elif component == 'generator':
                summary['generation_score'] = results.get('generation_score', 0.0)
                summary['generation_metrics'] = results.get('generation_metrics', {})
            
            summary_path = os.path.join(debug_dir, f"{component}_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(Utils.json_serializable(summary), f, indent=2, cls=NumpyEncoder)
            
            print(f"[DEBUG] Saved intermediate results for {component} to {debug_dir}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save intermediate results for {component}: {e}")
            pass
    
    def _extract_component_config(self, component: str, config: Dict[str, Any]) -> Dict[str, Any]:
        component_config = {}
        
        if component == 'retrieval':
            relevant_keys = ['retrieval_method', 'retriever_top_k', 'bm25_tokenizer', 'vectordb_name']
        elif component == 'query_expansion':
            relevant_keys = ['query_expansion_method', 'query_expansion_model', 
                            'query_expansion_temperature', 'query_expansion_max_token']
        elif component == 'passage_reranker':
            relevant_keys = ['passage_reranker_method', 'reranker_top_k', 'reranker_model']
        elif component == 'passage_filter':
            relevant_keys = ['passage_filter_method', 'threshold', 'percentile']
        elif component == 'passage_compressor':
            relevant_keys = ['passage_compressor_method', 'compressor_model', 
                            'compressor_compression_ratio', 'compressor_spacy_model']
        elif component == 'prompt_maker':
            relevant_keys = ['prompt_maker_method', 'prompt_template_idx']
        elif component == 'generator':
            relevant_keys = ['generator_module_type', 'generator_model', 'generator_temperature']
        else:
            relevant_keys = []
        
        for key in relevant_keys:
            if key in config:
                component_config[key] = config[key]
        
        return component_config
    
    def _get_component_score(self, component: str, results: Dict[str, Any]) -> float:
        score_keys = {
            'query_expansion': ['query_expansion_score', 'mean_score', 'last_retrieval_score'],
            'retrieval': ['retrieval_score', 'mean_accuracy', 'last_retrieval_score'],
            'passage_reranker': ['reranker_score', 'mean_accuracy', 'last_retrieval_score'],
            'passage_filter': ['filter_score', 'mean_accuracy', 'last_retrieval_score'],
            'passage_compressor': ['compression_score', 'mean_score', 'last_retrieval_score'],
            'prompt_maker': ['prompt_maker_score', 'mean_score'],
            'generator': ['generation_score', 'mean_score', 'combined_score']
        }
        
        for key in score_keys.get(component, ['score']):
            if key in results and results[key] is not None:
                return results[key]
        
        return 0.0
    
    def save_pipeline_summary(self, trial_dir: str, last_retrieval_component: str, 
                              last_retrieval_score: float, generation_score: float, 
                              results: Dict[str, Any]):
        try:
            debug_dir = os.path.join(trial_dir, "debug_intermediate_results")
            os.makedirs(debug_dir, exist_ok=True)
            
            pipeline_summary = {
                'timestamp': time.time(),
                'last_retrieval_component': last_retrieval_component,
                'last_retrieval_score': last_retrieval_score,
                'generation_score': generation_score,
                'combined_score': results.get('combined_score', 0.0),
                'evaluation_method': results.get('evaluation_method', 'unknown'),
                'component_scores': {},
                'components_executed': []
            }

            for file in os.listdir(debug_dir):
                if file.endswith('_summary.json'):
                    component = file.replace('_summary.json', '')
                    with open(os.path.join(debug_dir, file), 'r') as f:
                        summary = json.load(f)
                        pipeline_summary['component_scores'][component] = summary.get('score', 0.0)
                        pipeline_summary['components_executed'].append(component)
            
            summary_path = os.path.join(debug_dir, "pipeline_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(Utils.json_serializable(pipeline_summary), f, indent=2, cls=NumpyEncoder)
                
        except Exception as e:
            print(f"[WARNING] Failed to save pipeline summary: {e}")