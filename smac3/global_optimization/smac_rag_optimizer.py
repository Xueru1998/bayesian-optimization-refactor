import os
import json
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import wandb
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier import Hyperband, SuccessiveHalving
from smac.initial_design import SobolInitialDesign
from smac.facade import AbstractFacade
from smac.callback import Callback

from pipeline.config_manager import ConfigGenerator
from pipeline.pipeline_runner.rag_pipeline_runner import RAGPipelineRunner, EarlyStoppingException
from pipeline.search_space_calculator import SearchSpaceCalculator
from pipeline.utils import Utils
from smac3.global_optimization.config_space_builder import SMACConfigSpaceBuilder
from pipeline.logging.wandb import WandBLogger


class SMACRAGOptimizer:
    
    def __init__(
        self,
        config_template: Dict[str, Any],
        qa_data: pd.DataFrame,
        corpus_data: pd.DataFrame,
        project_dir: str,
        n_trials: Optional[int] = None,
        sample_percentage: float = 0.1,
        cpu_per_trial: int = 4,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        walltime_limit: int = 3600,
        n_workers: int = 1,
        seed: int = 42,
        early_stopping_threshold: float = 0.9,
        use_wandb: bool = True,
        wandb_project: str = "BO & AutoRAG",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        optimizer: str = "smac",
        use_multi_fidelity: bool = True,
        min_budget_percentage: float = 0.1,
        max_budget_percentage: float = 1.0,
        eta: int = 3,
        use_ragas: bool = False,  
        ragas_config: Optional[Dict[str, Any]] = None,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o",
        component_early_stopping_enabled: bool = True,
        component_early_stopping_thresholds: Optional[Dict[str, float]] = None
    ):
        self.start_time = time.time()
        
        self._initialize_paths(project_dir, result_dir)
        self._initialize_data(config_template, qa_data, corpus_data)
        self._initialize_optimization_params(
            n_trials, sample_percentage, cpu_per_trial, retrieval_weight,
            generation_weight, use_cached_embeddings, walltime_limit, n_workers,
            seed, early_stopping_threshold, optimizer, use_multi_fidelity,
            min_budget_percentage, max_budget_percentage, eta, use_ragas, ragas_config 
        )
        self._initialize_wandb_params(use_wandb, wandb_project, wandb_entity, wandb_run_name)
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
        self.component_early_stopping_enabled = component_early_stopping_enabled
        if component_early_stopping_thresholds is None:
            self.component_early_stopping_thresholds = {
                'retrieval': 0.1,
                'query_expansion': 0.1,
                'reranker': 0.2,
                'filter': 0.25,
                'compressor': 0.3
            }
        else:
            self.component_early_stopping_thresholds = component_early_stopping_thresholds
        
        self.study_name = study_name if study_name else f"{optimizer}_opt_{int(time.time())}"
        
        self._setup_components()
        self._calculate_trials_if_needed()
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
        
        self.trial_counter = 0
        self.all_trials = []
        self.early_stopped_trials_count = 0
        
        self._print_initialization_summary()
    
    def _initialize_paths(self, project_dir: str, result_dir: Optional[str]):
        self.project_root = Utils.find_project_root()
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                f"SMAC_{self.optimizer}_results"
            )
        os.makedirs(self.result_dir, exist_ok=True)
    
    def _initialize_data(self, config_template: Dict[str, Any], 
                        qa_data: pd.DataFrame, corpus_data: pd.DataFrame):
        self.config_template = config_template
        self.qa_data = qa_data
        self.corpus_data = corpus_data
        self.total_samples = len(qa_data)
    
    def _initialize_optimization_params(self, n_trials, sample_percentage, cpu_per_trial,
                                      retrieval_weight, generation_weight, use_cached_embeddings,
                                      walltime_limit, n_workers, seed, early_stopping_threshold,
                                      optimizer, use_multi_fidelity, min_budget_percentage,
                                      max_budget_percentage, eta, use_ragas, ragas_config):
        self.n_trials = n_trials
        self.sample_percentage = sample_percentage
        self.cpu_per_trial = cpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.walltime_limit = walltime_limit
        self.n_workers = n_workers
        self.seed = seed
        self.early_stopping_threshold = early_stopping_threshold
        
        self.optimizer = optimizer.lower()
        assert self.optimizer in ["smac", "bohb"], f"optimizer must be 'smac' or 'bohb', got {optimizer}"
        
        self.use_multi_fidelity = use_multi_fidelity
        self.min_budget_percentage = min_budget_percentage
        self.max_budget_percentage = max_budget_percentage
        self.eta = eta
        
        self.min_budget = max(1, int(self.total_samples * self.min_budget_percentage))
        self.max_budget = max(self.min_budget, int(self.total_samples * self.max_budget_percentage))
        self.use_ragas = use_ragas  
        self.ragas_config = ragas_config  
        
    def _initialize_wandb_params(self, use_wandb, wandb_project, wandb_entity, wandb_run_name):
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
    
    def _setup_components(self):
        self.config_generator = ConfigGenerator(self.config_template)
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        self.config_space_builder = SMACConfigSpaceBuilder(self.config_generator, seed=self.seed)
        
        self._setup_runner()
    
    def _setup_runner(self):
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config('prompt_maker')
        self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()

        use_ragas = self.use_ragas  
        
        ragas_config = self.ragas_config or {
            'retrieval_metrics': ["context_precision", "context_recall"],
            'generation_metrics': ["answer_relevancy", "faithfulness", "factual_correctness", "semantic_similarity"],
            'llm_model': "gpt-4o-mini",
            'embedding_model': "text-embedding-ada-002" 
        }
        
        early_stopping_thresholds = self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None
        
        self.runner = RAGPipelineRunner(
            config_generator=self.config_generator,
            retrieval_metrics=self.retrieval_metrics,
            filter_metrics=self.filter_metrics,
            compressor_metrics=self.compressor_metrics,
            generation_metrics=self.generation_metrics,
            prompt_maker_metrics=self.prompt_maker_metrics,
            query_expansion_metrics=self.query_expansion_metrics,
            reranker_metrics=self.reranker_metrics,
            retrieval_weight=self.retrieval_weight,
            generation_weight=self.generation_weight,
            use_ragas=use_ragas, 
            ragas_config=ragas_config,
            use_llm_compressor_evaluator=self.use_llm_compressor_evaluator,
            llm_evaluator_model=self.llm_evaluator_model,
            early_stopping_thresholds=early_stopping_thresholds
        )

    
    def _calculate_trials_if_needed(self):
        if self.n_trials is None:
            suggestion = self.search_space_calculator.suggest_num_samples(
                sample_percentage=self.sample_percentage,
                min_samples=20,
                max_samples=50,
                max_combinations=500
            )
            self.n_trials = suggestion['num_samples']
            print(f"Use Default num_trials: {self.n_trials}")
        else:
            self.n_trials = max(20, self.n_trials)
            if self.n_trials < 20:
                print(f"Minimum 20 trials recommended for SMAC. Increased to {self.n_trials}")
            else:
                print(f"Using provided num_trials: {self.n_trials}")
    
    def _ensure_conditional_parameters(self, config_dict: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        rng = np.random.RandomState(seed)

        reranker_key = None
        for key in ['reranker_topk', 'reranker_top_k']:
            if key in config_dict:
                reranker_key = key
                break
        
        if reranker_key and config_dict[reranker_key] == 1:
            config_dict['passage_filter_method'] = 'pass_passage_filter'
            config_dict['passage_filter_config'] = 'pass_passage_filter'

            filter_params = ['threshold', 'percentile', 'threshold_cutoff_threshold', 
                        'percentile_cutoff_percentile', 'similarity_threshold_cutoff_threshold',
                        'similarity_percentile_cutoff_percentile']
            for param in filter_params:
                if param in config_dict:
                    del config_dict[param]
            
            print(f"[CONSTRAINT] Set passage_filter to 'pass' because {reranker_key}=1")

        if 'query_expansion_method' in config_dict and config_dict['query_expansion_method'] != 'pass_query_expansion':
            for param in ['retrieval_method', 'bm25_tokenizer', 'vectordb_name']:
                if param in config_dict:
                    del config_dict[param]
                    print(f"[DEBUG] Removed {param} since query expansion is active")
            
            qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()

            if 'query_expansion_retrieval_method' not in config_dict:
                if qe_retrieval_options and qe_retrieval_options.get('methods'):
                    config_dict['query_expansion_retrieval_method'] = rng.choice(qe_retrieval_options['methods'])
                    print(f"[DEBUG] Added missing query_expansion_retrieval_method: {config_dict['query_expansion_retrieval_method']}")
                else:
                    config_dict['query_expansion_retrieval_method'] = 'bm25'
                    print(f"[DEBUG] No QE retrieval options available, defaulting to bm25")

            qe_retrieval_method = config_dict.get('query_expansion_retrieval_method')
            
            if qe_retrieval_method == 'bm25':
                if 'query_expansion_vectordb_name' in config_dict:
                    del config_dict['query_expansion_vectordb_name']
                    print(f"[DEBUG] Removed query_expansion_vectordb_name since using BM25")

                if 'query_expansion_bm25_tokenizer' not in config_dict:
                    if qe_retrieval_options and qe_retrieval_options.get('bm25_tokenizers'):
                        config_dict['query_expansion_bm25_tokenizer'] = rng.choice(qe_retrieval_options['bm25_tokenizers'])
                        print(f"[DEBUG] Added missing query_expansion_bm25_tokenizer: {config_dict['query_expansion_bm25_tokenizer']}")
                    else:
                        config_dict['query_expansion_bm25_tokenizer'] = 'space'
                        print(f"[DEBUG] No BM25 tokenizers available, using default: space")
            
            elif qe_retrieval_method == 'vectordb':
                if 'query_expansion_bm25_tokenizer' in config_dict:
                    del config_dict['query_expansion_bm25_tokenizer']
                    print(f"[DEBUG] Removed query_expansion_bm25_tokenizer since using vectordb")

                if 'query_expansion_vectordb_name' not in config_dict:
                    if qe_retrieval_options and qe_retrieval_options.get('vectordb_names'):
                        config_dict['query_expansion_vectordb_name'] = rng.choice(qe_retrieval_options['vectordb_names'])
                        print(f"[DEBUG] Added missing query_expansion_vectordb_name: {config_dict['query_expansion_vectordb_name']}")
                    else:
                        config_dict['query_expansion_vectordb_name'] = 'default'
                        print(f"[DEBUG] No vectordb names available, using default: default")
        
        else:
            for param in ['query_expansion_retrieval_method', 'query_expansion_bm25_tokenizer', 'query_expansion_vectordb_name']:
                if param in config_dict:
                    del config_dict[param]
                    print(f"[DEBUG] Removed {param} since query expansion is not active")

        if 'passage_filter_method' in config_dict and config_dict['passage_filter_method'] != 'pass_passage_filter':
            filter_method = config_dict['passage_filter_method']
            
            unified_space = self.config_space_builder.unified_extractor.extract_search_space('smac')
            
            if filter_method in ['threshold_cutoff', 'similarity_threshold_cutoff']:
                self._add_threshold_if_missing(config_dict, unified_space, filter_method, rng)
            elif filter_method in ['percentile_cutoff', 'similarity_percentile_cutoff']:
                self._add_percentile_if_missing(config_dict, unified_space, filter_method, rng)
        
        config_dict = self._validate_topk_constraints(config_dict)
        
        return config_dict
    
    def _add_threshold_if_missing(self, config_dict, unified_space, filter_method, rng):
        if 'threshold' not in config_dict:
            if 'threshold' in unified_space and 'method_values' in unified_space['threshold']:
                method_values = unified_space['threshold']['method_values']
                if filter_method in method_values:
                    values = method_values[filter_method]
                    config_dict['threshold'] = self._sample_from_values(values, rng, 0.0, 1.0)
                    print(f"[DEBUG] Added threshold: {config_dict['threshold']:.4f}")
                else:
                    config_dict['threshold'] = rng.uniform(0.65, 0.85)
            else:
                config_dict['threshold'] = rng.uniform(0.65, 0.85)
    
    def _add_percentile_if_missing(self, config_dict, unified_space, filter_method, rng):
        if 'percentile' not in config_dict:
            if 'percentile' in unified_space and 'method_values' in unified_space['percentile']:
                method_values = unified_space['percentile']['method_values']
                if filter_method in method_values:
                    values = method_values[filter_method]
                    config_dict['percentile'] = self._sample_from_values(values, rng, 0.0, 1.0)
                    print(f"[DEBUG] Added percentile: {config_dict['percentile']:.4f}")
                else:
                    config_dict['percentile'] = rng.uniform(0.6, 0.8)
            else:
                config_dict['percentile'] = rng.uniform(0.6, 0.8)
    
    def _sample_from_values(self, values: List[float], rng, min_limit: float, max_limit: float) -> float:
        if isinstance(values, list) and len(values) >= 2:
            return rng.uniform(min(values), max(values))
        elif isinstance(values, list) and len(values) == 1:
            val = values[0]
            min_val = max(min_limit, val - 0.2)
            max_val = min(max_limit, val + 0.2)
            return rng.uniform(min_val, max_val)
        else:
            return rng.uniform(0.65, 0.85)
    
    def _run_trial(self, config: Dict[str, Any], budget: int, seed: int = 0) -> Dict[str, Any]:
        
        self.trial_counter += 1
        
        budget_percentage = budget / self.total_samples
        
        trial_dir = self._setup_trial_directory(budget)
        sampled_qa_data = self._sample_data(budget, seed)
        self._copy_corpus_data(trial_dir)
        
        trial_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Trial {self.trial_counter}/{self.n_trials} | Budget: {budget} samples ({budget_percentage:.1%})")
        print(f"{'='*60}")
        
        early_stopped = False
        early_stopped_component = None
        early_stopped_score = None
        
        try:
            config_dict = self._prepare_config(config)
            trial_config = self.config_generator.generate_trial_config(config_dict)
            self._save_trial_config(trial_dir, trial_config)
            
            try:
                results = self.runner.run_pipeline(config_dict, trial_dir, sampled_qa_data)
                score = results.get('combined_score', 0.0)
                
            except EarlyStoppingException as e:
                early_stopped = True
                early_stopped_component = e.component
                early_stopped_score = e.score
                score = e.score
                self.early_stopped_trials_count += 1
                
                print(f"\n[Trial {self.trial_counter}] Early stopped at {e.component} with score {e.score:.4f}")
                
                results = {
                    'combined_score': score,
                    'early_stopped_at': e.component,
                    'early_stopped_score': e.score,
                    'error': e.message
                }
            
            latency = time.time() - trial_start_time
            
            trial_result = self._create_trial_result(
                config_dict, score, latency, budget, budget_percentage, results
            )
            
            if early_stopped:
                trial_result['early_stopped'] = True
                trial_result['early_stopped_component'] = early_stopped_component
                trial_result['early_stopped_score'] = early_stopped_score
            
            self.all_trials.append(trial_result)
            
            self._save_trial_results(trial_dir, trial_result)
            self._print_trial_summary(score, latency, budget, budget_percentage, early_stopped, early_stopped_component)
            
            if self.use_wandb:
                self._log_trial_to_wandb(config_dict, trial_result)
            
            return {"score": -score, "latency": latency}
            
        except Exception as e:
            print(f"Error in trial {self.trial_counter}: {e}")
            import traceback
            traceback.print_exc()
            return {"score": 0.0, "latency": float('inf')}
    
    def _setup_trial_directory(self, budget: int) -> str:
        trial_dir = os.path.join(self.result_dir, f"trial_{self.trial_counter}_budget_{budget}")
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        return trial_dir
    
    def _sample_data(self, budget: int, seed: int) -> pd.DataFrame:
        actual_samples = min(budget, self.total_samples)
        if actual_samples < self.total_samples:
            return self.qa_data.sample(n=actual_samples, random_state=seed)
        return self.qa_data
    
    def _copy_corpus_data(self, trial_dir: str):
        centralized_corpus_path = os.path.join(self.project_dir, "data", "corpus.parquet")
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
            
    def _validate_topk_constraints(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        retriever_key = None
        reranker_key = None
        
        for key in ['retriever_topk', 'retriever_top_k']:
            if key in config_dict:
                retriever_key = key
                break
        
        for key in ['reranker_topk', 'reranker_top_k']:
            if key in config_dict:
                reranker_key = key
                break
        
        if retriever_key and reranker_key:
            retriever_value = config_dict[retriever_key]
            reranker_value = config_dict[reranker_key]
            
            if isinstance(retriever_value, (int, float, np.integer, np.floating)):
                retriever_value = int(retriever_value)
            if isinstance(reranker_value, (int, float, np.integer, np.floating)):
                reranker_value = int(reranker_value)
            
            if reranker_value > retriever_value:
                original_value = reranker_value
                config_dict[reranker_key] = retriever_value
                print(f"[CONSTRAINT] Adjusted {reranker_key} from {original_value} to {retriever_value} to not exceed {retriever_key}")
        
        return config_dict
    
    def _prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config_dict = config if isinstance(config, dict) else dict(config)

        if 'query_expansion_method' not in config_dict:
            if 'retrieval_method' not in config_dict:
                retrieval_options = self.config_generator.extract_retrieval_options()
                if retrieval_options and retrieval_options.get('methods'):
                    config_dict['retrieval_method'] = retrieval_options['methods'][0]

                    if config_dict['retrieval_method'] == 'bm25' and 'bm25_tokenizer' not in config_dict:
                        if retrieval_options.get('bm25_tokenizers'):
                            config_dict['bm25_tokenizer'] = retrieval_options['bm25_tokenizers'][0]
                    elif config_dict['retrieval_method'] == 'vectordb' and 'vectordb_name' not in config_dict:
                        if retrieval_options.get('vectordb_names'):
                            config_dict['vectordb_name'] = retrieval_options['vectordb_names'][0]

        config_dict = self.config_space_builder.clean_trial_config(config_dict)
        config_dict = self._convert_numpy_types(config_dict)

        for temp_param in ['generator_temperature', 'query_expansion_temperature', 'temperature']:
            if temp_param in config_dict:
                try:
                    config_dict[temp_param] = float(config_dict[temp_param])
                except:
                    config_dict[temp_param] = 0.7
        
        config_dict = self._validate_topk_constraints(config_dict)
        
        return config_dict
    
    def _convert_numpy_types(self, obj):
        if isinstance(obj, pd.DataFrame):
            return {"type": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(i) for i in obj)
        elif hasattr(obj, 'item'): 
            return obj.item()
        else:
            return obj
    
    def _save_trial_config(self, trial_dir: str, trial_config: Dict[str, Any]):
        config_file = os.path.join(trial_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(trial_config, f, default_flow_style=False)
    
    def _create_trial_result(self, config_dict, score, latency, budget, budget_percentage, results):
        trial_result = {
            "trial_number": int(self.trial_counter), 
            "config": self._convert_numpy_types(config_dict),
            "score": float(score),
            "latency": float(latency),
            "budget": int(budget),
            "budget_percentage": budget_percentage,
            "retrieval_score": float(results.get('retrieval_score', 0.0)),
            "generation_score": float(results.get('generation_score', 0.0)),
            "combined_score": float(score),
            "timestamp": float(time.time()) 
        }
        
        for k, v in results.items():
            if k.endswith('_score'):
                trial_result[k] = float(v) if isinstance(v, (int, float, np.number)) else 0.0
            elif k.endswith('_metrics'):
                if isinstance(v, dict):
                    cleaned_metrics = {}
                    for metric_key, metric_value in v.items():
                        if not isinstance(metric_value, pd.DataFrame):
                            cleaned_metrics[metric_key] = self._convert_numpy_types(metric_value)
                    trial_result[k] = cleaned_metrics
                elif not isinstance(v, pd.DataFrame):
                    trial_result[k] = self._convert_numpy_types(v)
                    
        if 'config' in trial_result:
            for key in ['generator_temperature', 'query_expansion_temperature', 'temperature']:
                if key in trial_result['config']:
                    trial_result['config'][key] = float(trial_result['config'][key])
        
        return trial_result
    
    def _save_trial_results(self, trial_dir: str, trial_result: Dict[str, Any]):
        results_file = os.path.join(trial_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(trial_result, f, indent=2)
    
    def _print_trial_summary(self, score, latency, budget, budget_percentage, early_stopped=False, early_stopped_component=None):
        print(f"Trial {self.trial_counter} completed:")
        if early_stopped:
            print(f"  Status: EARLY STOPPED at {early_stopped_component}")
        print(f"  Score: {score:.4f}")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Budget: {budget} samples ({budget_percentage:.1%})")

    
    def _log_trial_to_wandb(self, config_dict, trial_result):
        log_data = {
            'trial': trial_result['trial_number'],
            'score': trial_result['score'],
            'latency': trial_result['latency'],
            'early_stopped': trial_result.get('early_stopped', False)
        }
        
        if trial_result.get('early_stopped', False):
            log_data['early_stopped_component'] = trial_result.get('early_stopped_component', 'unknown')
            log_data['early_stopped_score'] = trial_result.get('early_stopped_score', 0.0)
        
        WandBLogger.log_trial_metrics(
            self.trial_counter, 
            trial_result['score'],
            config=config_dict,
            results={**trial_result, **log_data}
        )
    
    def target_function_standard(self, config: Dict[str, Any], seed: int = 0) -> Dict[str, float]:
        config_dict = config.get_dictionary() if hasattr(config, 'get_dictionary') else dict(config)
        config_dict = self._validate_topk_constraints(config_dict)
        config_dict = self._ensure_conditional_parameters(config_dict, seed=seed)
        return self._run_trial(config_dict, self.max_budget, seed)
    
    def target_function_multifidelity(self, config: Dict[str, Any], seed: int = 0, budget: float = None) -> Dict[str, float]:
        config_dict = config.get_dictionary() if hasattr(config, 'get_dictionary') else dict(config)
        config_dict = self._validate_topk_constraints(config_dict)
        config_dict = self._ensure_conditional_parameters(config_dict, seed=seed)
        budget = int(budget) if budget is not None else self.max_budget
        return self._run_trial(config_dict, budget, seed)
        
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.use_wandb:
            self._initialize_wandb()
        
        cs = self.config_space_builder.build_configuration_space()
        early_stopping_callback = self._create_early_stopping_callback()
        
        scenario = self._create_scenario(cs)
        initial_design = self._create_initial_design(scenario)
        
        target_function = (self.target_function_multifidelity if self.use_multi_fidelity 
                          else self.target_function_standard)
        
        smac = self._create_optimizer(scenario, target_function, initial_design, [early_stopping_callback])
        
        self._print_optimization_start_info()
        
        incumbents = self._run_optimization(smac, early_stopping_callback)
        pareto_front = self._extract_pareto_front(smac, incumbents)
        
        optimization_results = self._create_optimization_results(
            pareto_front, early_stopping_callback.should_stop, incumbents, time.time() - start_time
        )
        
        self._save_and_log_results(optimization_results, pareto_front)
        self._print_optimization_summary(optimization_results, pareto_front)
        
        return optimization_results
    
    def _initialize_wandb(self):
        WandBLogger.reset_step_counter()
        
        cs = self.config_space_builder.build_configuration_space()
        search_space_info = self.config_space_builder.get_search_space_info()
        
        wandb_config = {
            "optimizer": f"{self.optimizer.upper()}{' Multi-Fidelity' if self.use_multi_fidelity else ''}",
            "n_trials": self.n_trials,
            "retrieval_weight": self.retrieval_weight,
            "generation_weight": self.generation_weight,
            "search_space_size": search_space_info['n_hyperparameters'],
            "study_name": self.study_name,
            "early_stopping_threshold": self.early_stopping_threshold,
            "component_early_stopping_enabled": self.component_early_stopping_enabled,
            "component_early_stopping_thresholds": self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None,
            "n_workers": self.n_workers,
            "walltime_limit": self.walltime_limit if self.walltime_limit is not None else "No limit",
            "use_multi_fidelity": self.use_multi_fidelity,
            "min_budget": self.min_budget,
            "max_budget": self.max_budget,
            "eta": self.eta if self.use_multi_fidelity else None
        }
        
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.wandb_run_name or self.study_name,
            config=wandb_config,
            reinit=True
        )
    
    def _create_early_stopping_callback(self):
        class EarlyStoppingCallback(Callback):
            def __init__(self, threshold: float):
                super().__init__()
                self.threshold = threshold
                self.should_stop = False
                
            def on_tell(self, smbo, info, value):
                if info and value:
                    if hasattr(value, 'cost') and isinstance(value.cost, list):
                        score = -value.cost[0]
                    else:
                        score = 0
                    
                    if score >= self.threshold:
                        print(f"\n*** Early stopping triggered! Score {score:.4f} >= {self.threshold} ***")
                        self.should_stop = True
                        smbo._stop = True
        
        return EarlyStoppingCallback(self.early_stopping_threshold)
    
    def _create_scenario(self, cs) -> Scenario:
        base_params = {
            'configspace': cs,
            'deterministic': True,
            'n_trials': self.n_trials,
            'n_workers': self.n_workers,
            'seed': self.seed,
            'objectives': ["score", "latency"],
            'output_directory': self.result_dir,
            'name': self.study_name
        }
        
        if self.walltime_limit is not None:
            base_params['walltime_limit'] = self.walltime_limit
        
        if self.use_multi_fidelity:
            base_params['min_budget'] = self.min_budget
            base_params['max_budget'] = self.max_budget
        
        return Scenario(**base_params)
    
    def _create_initial_design(self, scenario: Scenario) -> SobolInitialDesign:
        n_init = min(self.n_trials // 4, 10)
        n_init = max(n_init, 2)
        
        print(f"Using {n_init} initial random configurations")
        
        return SobolInitialDesign(
            scenario=scenario,
            n_configs=n_init,
            max_ratio=1.0,
            additional_configs=[]
        )
    
    def _create_optimizer(self, scenario, target_function, initial_design, callbacks):
        if self.use_multi_fidelity:
            return self._create_multi_fidelity_optimizer(scenario, target_function, initial_design, callbacks)
        else:
            return self._create_standard_optimizer(scenario, target_function, initial_design, callbacks)
    
    def _create_multi_fidelity_optimizer(self, scenario, target_function, initial_design, callbacks):
        if self.optimizer == "bohb":
            intensifier = Hyperband(
                scenario=scenario,
                incumbent_selection="highest_budget",
                eta=self.eta
            )
            print(f"Using BOHB (Bayesian Optimization Hyperband) with eta={self.eta}")
        else:
            intensifier = SuccessiveHalving(
                scenario=scenario,
                incumbent_selection="highest_budget",
                eta=self.eta
            )
            print(f"Using SMAC3 with Successive Halving, eta={self.eta}")
        
        return MultiFidelityFacade(
            scenario=scenario,
            target_function=target_function,
            intensifier=intensifier,
            callbacks=callbacks,
            initial_design=initial_design,
            overwrite=True
        )
    
    def _create_standard_optimizer(self, scenario, target_function, initial_design, callbacks):
        return HPOFacade(
            scenario=scenario,
            target_function=target_function,
            multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
                scenario,
                objective_weights=[self.generation_weight, self.retrieval_weight]
            ),
            callbacks=callbacks,
            initial_design=initial_design,
            overwrite=True
        )
    
    def _print_optimization_start_info(self):
        if self.use_multi_fidelity:
            print(f"\nStarting {self.optimizer.upper()} multi-fidelity optimization")
            print(f"Budget range: {self.min_budget} to {self.max_budget} samples")
            print(f"Budget percentage: {self.min_budget_percentage:.1%} to {self.max_budget_percentage:.1%}")
        else:
            print(f"\nStarting standard SMAC3 optimization (no multi-fidelity)")
        
        print(f"Total trials: {self.n_trials}")
        print(f"Objectives: score (weight={self.generation_weight}), latency (weight={self.retrieval_weight})")
        print(f"Early stopping threshold: {self.early_stopping_threshold}")
    
    def _run_optimization(self, smac, early_stopping_callback):
        try:
            incumbents = smac.optimize()
        except Exception as e:
            print(f"Optimization stopped: {e}")
            incumbents = self._extract_incumbents_from_smac(smac)
        
        return incumbents if isinstance(incumbents, list) else ([incumbents] if incumbents else [])
    
    def _extract_incumbents_from_smac(self, smac):
        incumbents = []
        try:
            if hasattr(smac, 'intensifier') and hasattr(smac.intensifier, 'get_incumbents'):
                incumbents = smac.intensifier.get_incumbents()
            elif hasattr(smac, 'get_incumbents'):
                incumbents = smac.get_incumbents()
            elif hasattr(smac, 'runhistory'):
                incumbents = self._extract_from_runhistory(smac)
        except Exception as e:
            print(f"Could not retrieve incumbents: {e}")
        return incumbents
    
    def _extract_from_runhistory(self, smac):
        incumbents = []
        if hasattr(smac.runhistory, 'get_incumbents'):
            incumbents = smac.runhistory.get_incumbents()
        else:
            configs = smac.runhistory.get_configs()
            if configs:
                for config in configs[:20]:
                    try:
                        cost = smac.runhistory.get_cost(config)
                        if cost is not None:
                            incumbents.append(config)
                    except:
                        continue
        return incumbents
    
    def _extract_pareto_front(self, smac, incumbents):
        pareto_front = []
        for incumbent in incumbents:
            try:
                cost = self._get_incumbent_cost(smac, incumbent)
                if cost is None:
                    continue
                
                score_val, latency_val = self._extract_cost_values(cost)
                pareto_solution = self._create_pareto_solution(incumbent, score_val, latency_val)
                pareto_front.append(pareto_solution)
                
            except Exception as e:
                print(f"Error processing incumbent: {e}")
                continue
        
        self._update_pareto_front_trial_numbers(pareto_front)
        return pareto_front
    
    def _get_incumbent_cost(self, smac, incumbent):
        if hasattr(smac, 'validate'):
            return smac.validate(incumbent)
        elif hasattr(smac, 'runhistory'):
            try:
                incumbent_cost = smac.runhistory.get_cost(incumbent)
                if isinstance(incumbent_cost, list):
                    return {"score": incumbent_cost[0], "latency": incumbent_cost[1]}
                else:
                    return {"score": incumbent_cost, "latency": 0.0}
            except:
                return None
        return None
    
    def _extract_cost_values(self, cost):
        if isinstance(cost, dict):
            score_val = cost.get("score", 0.0)
            latency_val = cost.get("latency", 0.0)
        elif isinstance(cost, (list, np.ndarray)):
            if len(cost) >= 2:
                score_val = float(cost[0])
                latency_val = float(cost[1])
            else:
                score_val = float(cost[0]) if len(cost) > 0 else 0.0
                latency_val = 0.0
        else:
            score_val = float(cost)
            latency_val = 0.0
        
        if hasattr(score_val, '__len__') and not isinstance(score_val, str):
            score_val = float(score_val[0] if len(score_val) > 0 else 0.0)
        if hasattr(latency_val, '__len__') and not isinstance(latency_val, str):
            latency_val = float(latency_val[0] if len(latency_val) > 0 else 0.0)
        
        return score_val, latency_val
    
    def _create_pareto_solution(self, incumbent, score_val, latency_val):
        config_dict = dict(incumbent)
        pareto_solution = {
            'config': config_dict,
            'score': -float(score_val) if score_val else 0.0,
            'latency': float(latency_val) if latency_val else 0.0,
            'trial_number': None
        }
        
        for trial in self.all_trials:
            if trial['config'] == config_dict:
                pareto_solution.update({
                    'score': trial['score'],
                    'latency': trial['latency'],
                    'trial_number': trial['trial_number'],
                    'budget': trial.get('budget', self.max_budget),
                    'budget_percentage': trial.get('budget_percentage', 1.0)
                })
                break
        
        return pareto_solution
    
    def _update_pareto_front_trial_numbers(self, pareto_front):
        for pf_solution in pareto_front:
            for trial in self.all_trials:
                if trial['config'] == pf_solution['config']:
                    pf_solution['trial_number'] = trial['trial_number']
                    break
    
    def _create_optimization_results(self, pareto_front, early_stopped, incumbents, total_time):
        best_configs = self._find_best_configurations(pareto_front)
        
        valid_trials = [t for t in self.all_trials if not t.get('early_stopped', False)]
        
        results = {
            'optimizer': self.optimizer,
            'use_multi_fidelity': bool(self.use_multi_fidelity),
            'min_budget': int(self.min_budget),
            'max_budget': int(self.max_budget),
            'best_config': self._convert_numpy_types(best_configs['best_balanced']),
            'best_score_config': self._convert_numpy_types(best_configs['best_score']['config']),
            'best_score': float(best_configs['best_score']['score']),
            'best_score_latency': float(best_configs['best_score']['latency']),
            'best_latency_config': self._convert_numpy_types(best_configs['best_latency']['config']),
            'best_latency': float(best_configs['best_latency']['latency']),
            'best_latency_score': float(best_configs['best_latency']['score']),
            'pareto_front': [self._convert_numpy_types(pf) for pf in pareto_front],
            'optimization_time': float(total_time),
            'n_trials': int(self.trial_counter),
            'total_trials': int(self.trial_counter),
            'early_stopped_trials': int(self.early_stopped_trials_count),
            'completed_trials': int(self.trial_counter - self.early_stopped_trials_count),
            'early_stopped': bool(early_stopped),
            'incumbents': [self._convert_numpy_types(dict(inc)) for inc in incumbents],
            'all_trials': [self._convert_numpy_types(trial) for trial in self.all_trials],
            'component_early_stopping_enabled': self.component_early_stopping_enabled,
            'component_early_stopping_thresholds': self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None
        }
        
        return results
    
    def _find_best_configurations(self, pareto_front):
        valid_trials = [t for t in self.all_trials if not t.get('early_stopped', False)]
        
        if not valid_trials:
            default_trial = {'config': {}, 'score': 0.0, 'latency': float('inf')}
            return {
                'best_score': default_trial,
                'best_latency': default_trial,
                'best_balanced': default_trial
            }
        
        if self.use_multi_fidelity:
            return self._find_best_multifidelity_configs_with_early_stopping(pareto_front, valid_trials)
        else:
            return self._find_best_standard_configs_with_early_stopping(pareto_front, valid_trials)
    
    def _find_best_multifidelity_configs_with_early_stopping(self, pareto_front, valid_trials):
        full_budget_trials = [
            t for t in valid_trials 
            if t.get('budget_percentage', 1.0) >= 0.99
        ]
        
        if full_budget_trials:
            best_score_trial = max(full_budget_trials, key=lambda x: x['score'])
            best_latency_trial = min(full_budget_trials, key=lambda x: x['latency'])
            
            high_score_trials = [t for t in full_budget_trials if t['score'] > 0.9]
            if high_score_trials:
                best_balanced = min(high_score_trials, key=lambda x: x['latency'])
            else:
                best_balanced = max(full_budget_trials, key=lambda x: x['score'])
        else:
            best_score_trial = max(valid_trials, key=lambda x: x['score'])
            best_latency_trial = min(valid_trials, key=lambda x: x['latency'])
            best_balanced = best_score_trial
        
        return {
            'best_score': best_score_trial,
            'best_latency': best_latency_trial,
            'best_balanced': best_balanced
        }

    def _find_best_standard_configs_with_early_stopping(self, pareto_front, valid_trials):
        best_score_trial = max(valid_trials, key=lambda x: x['score'])
        best_latency_trial = min(valid_trials, key=lambda x: x['latency'])
        
        high_score_trials = [t for t in valid_trials if t['score'] > 0.9]
        if high_score_trials:
            best_balanced = min(high_score_trials, key=lambda x: x['latency'])
        else:
            valid_pareto = [p for p in pareto_front if not any(t.get('early_stopped', False) for t in self.all_trials if t['config'] == p['config'])]
            best_balanced = max(valid_pareto, key=lambda x: x['score']) if valid_pareto else best_score_trial
        
        return {
            'best_score': best_score_trial,
            'best_latency': best_latency_trial,
            'best_balanced': best_balanced
        }
    
    def _save_and_log_results(self, optimization_results, pareto_front):
        Utils.save_results_to_json(self.result_dir, "optimization_summary.json", optimization_results)
        
        if self.all_trials:
            converted_trials = self._convert_trials_for_csv(self.all_trials)
            Utils.save_results_to_csv(self.result_dir, "all_trials.csv", converted_trials)
        
        if self.use_wandb:
            WandBLogger.log_optimization_plots(None, self.all_trials, pareto_front, prefix=self.optimizer)
            WandBLogger.log_final_tables(self.all_trials, pareto_front, prefix="final")
            WandBLogger.log_summary(optimization_results)
            
            wandb.run.summary["tables_logged"] = True
            
            wandb.finish()
    
    def _convert_trials_for_csv(self, trials):
        converted_trials = []
        for trial in trials:
            converted_trial = {}
            for k, v in trial.items():
                if isinstance(v, np.integer):
                    converted_trial[k] = int(v)
                elif isinstance(v, np.floating):
                    converted_trial[k] = float(v)
                elif isinstance(v, np.ndarray):
                    converted_trial[k] = v.tolist()
                elif isinstance(v, dict):
                    converted_trial[k] = str(v)
                else:
                    converted_trial[k] = v
            converted_trials.append(converted_trial)
        return converted_trials
    
    def _print_optimization_summary(self, optimization_results, pareto_front):
        time_str = Utils.format_time_duration(optimization_results['optimization_time'])
        
        print(f"\n{'='*60}")
        print(f"{self.optimizer.upper()} Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total optimization time: {time_str}")
        print(f"Total trials: {self.trial_counter}")
        print(f"Early stopped trials: {self.early_stopped_trials_count}")
        print(f"Completed trials: {self.trial_counter - self.early_stopped_trials_count}")
        
        if self.use_multi_fidelity and self.all_trials:
            self._print_multifidelity_summary()
        
        if optimization_results['early_stopped']:
            print("⚡ Optimization stopped early due to achieving target score!")
        
        self._print_best_config_summary(optimization_results['best_config'])
        self._print_pareto_front_summary(pareto_front)
        
        print(f"\nResults saved to: {self.result_dir}")

    
    def _print_multifidelity_summary(self):
        full_budget_count = len([
            t for t in self.all_trials 
            if t.get('budget_percentage', 1.0) >= 0.99
        ])
        unique_configs = len(set(
            self._get_config_hash(t.get('config', {})) 
            for t in self.all_trials
        ))
        print(f"Unique configurations tested: {unique_configs}")
        print(f"Fully evaluated configurations: {full_budget_count}")
    
    def _print_best_config_summary(self, best_config):
        if best_config and isinstance(best_config, dict):
            print("\nBest balanced configuration (high score with low latency):")
            print(f"  Score: {best_config.get('score', 'N/A')}")
            print(f"  Latency: {best_config.get('latency', 'N/A')}")
            if 'budget' in best_config:
                print(f"  Budget: {best_config['budget']} samples")
            if 'config' in best_config:
                print(f"  Config: {best_config['config']}")
    
    def _print_pareto_front_summary(self, pareto_front):
        print(f"\nPareto front contains {len(pareto_front)} solutions")
        if pareto_front:
            print("Top 5 Pareto optimal solutions:")
            for i, solution in enumerate(sorted(pareto_front, key=lambda x: -x['score'])[:5]):
                budget_str = f", Budget: {solution.get('budget', 'N/A')}" if 'budget' in solution else ""
                print(f"  {i+1}. Score: {solution['score']:.4f}, Latency: {solution['latency']:.2f}s{budget_str}")
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        config_str = json.dumps(dict(sorted(config.items())), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _print_initialization_summary(self):    
        print(f"\n===== {self.optimizer.upper()} {'Multi-Fidelity ' if self.use_multi_fidelity else ''}RAG Pipeline Optimizer =====")
        print(f"Using {self.n_trials} trials")
        print(f"Objectives: maximize score (weight={self.generation_weight}), minimize latency (weight={self.retrieval_weight})")
        
        if self.component_early_stopping_enabled:
            print(f"\nComponent-level early stopping ENABLED with thresholds:")
            for component, threshold in self.component_early_stopping_thresholds.items():
                print(f"  {component}: {threshold}")
        else:
            print(f"\nComponent-level early stopping DISABLED")
        
        print(f"\nHigh-score early stopping threshold: {self.early_stopping_threshold}")
        
        if self.use_multi_fidelity:
            print(f"\nMulti-fidelity settings:")
            print(f"  Min budget: {self.min_budget} samples ({self.min_budget_percentage:.1%})")
            print(f"  Max budget: {self.max_budget} samples ({self.max_budget_percentage:.1%})")
            print(f"  Eta: {self.eta}")      

        print(f"Using cached embeddings: {self.use_cached_embeddings}")
        if self.walltime_limit is not None:
            print(f"Wall time limit: {self.walltime_limit}s")
        else:
            print(f"Wall time limit: No limit")
        print(f"Number of workers: {self.n_workers}")