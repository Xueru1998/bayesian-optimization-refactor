import os
import shutil
import sys
import tempfile
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import math
import wandb
import hashlib
import threading

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search import ConcurrencyLimiter

from pipeline.config_manager import ConfigGenerator
from pipeline.utils import Utils
from pipeline_component.retrieval import RetrievalModule
from RayTune.BO_global_optimization.json_manager import JsonManager
from RayTune.search_space_manager import SearchSpaceManager
from pipeline.rag_pipeline_runner import RAGPipelineRunner
from pipeline.search_space_calculator import SearchSpaceCalculator
from pipeline.wandb_logger import WandBLogger

@ray.remote
class TrialCounter:
    def __init__(self):
        self.counter = 0
    
    def increment_and_get(self):
        self.counter += 1
        return self.counter
    
    def reset(self):
        self.counter = 0
        

def run_trial_fn(config: Dict[str, Any], checkpoint_dir=None):
    optimizer_config = config.pop('_optimizer_config')
    
    qa_df = optimizer_config['qa_df']
    project_dir = optimizer_config['project_dir']
    config_generator = optimizer_config['config_generator']
    pipeline_runner = optimizer_config['pipeline_runner']
    json_manager = optimizer_config['json_manager']
    use_cached_embeddings = optimizer_config['use_cached_embeddings']
    min_budget = optimizer_config['min_budget']
    max_budget = optimizer_config['max_budget']
    total_samples = optimizer_config['total_samples']
    num_samples = optimizer_config['num_samples']
    trial_counter_actor = optimizer_config['trial_counter']
    use_wandb = optimizer_config['use_wandb']
    
    trial_number = ray.get(trial_counter_actor.increment_and_get.remote())
    
    trial_start_time = time.time()
    trial_dir = tempfile.mkdtemp(prefix="rag_tune_")
    
    print(f"\n{'='*60}")
    print(f"Trial {trial_number}/{num_samples}")
    print(f"{'='*60}")
    
    def _make_json_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_json_serializable(i) for i in obj]
        else:
            return obj
    
    def _setup_trial_directory(trial_dir: str, config: Dict[str, Any]) -> bool:
        try:
            os.makedirs(trial_dir, exist_ok=True)
            os.makedirs(os.path.join(trial_dir, "configs"), exist_ok=True)
            os.makedirs(os.path.join(trial_dir, "resources"), exist_ok=True)
            os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
            
            centralized_corpus_path = os.path.join(project_dir, "data", "corpus.parquet")
            trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
            if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
                shutil.copy2(centralized_corpus_path, trial_corpus_path)
            
            retrieval_module = RetrievalModule(
                base_project_dir=trial_dir,
                use_pregenerated_embeddings=use_cached_embeddings,
                centralized_project_dir=project_dir
            )
            retrieval_module.setup_vectordb_config(trial_dir)
            
            trial_config = config_generator.generate_trial_config(config)
            config_path = os.path.join(trial_dir, "configs", "config.yaml")
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(trial_config, f)
                
            if config.get('retrieval_method') == 'bm25' and 'bm25_tokenizer' in config:
                from autorag.nodes.retrieval.bm25 import get_bm25_pkl_name
                
                bm25_tokenizer = config['bm25_tokenizer']
                centralized_bm25_file = os.path.join(project_dir, "resources", get_bm25_pkl_name(bm25_tokenizer))
                trial_bm25_file = os.path.join(trial_dir, "resources", get_bm25_pkl_name(bm25_tokenizer))
                
                if os.path.exists(centralized_bm25_file) and use_cached_embeddings:
                    shutil.copy(centralized_bm25_file, trial_bm25_file)
            
            elif config.get('retrieval_method') == 'vectordb' and use_cached_embeddings:
                vectordb_name = config.get('vectordb_name', 'default')
                centralized_vectordb_yaml = os.path.join(project_dir, "resources", "vectordb.yaml")
                trial_vectordb_yaml = os.path.join(trial_dir, "resources", "vectordb.yaml")
                if os.path.exists(centralized_vectordb_yaml):
                    shutil.copy(centralized_vectordb_yaml, trial_vectordb_yaml)
                
            return True
        except Exception as e:
            print(f"Error setting up trial directory: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    try:
        setup_success = _setup_trial_directory(trial_dir, config)
        if not setup_success:
            print("Failed to set up trial directory")
            trial_result = {
                "combined_score": 0.0,
                "score": 0.0,
                "last_retrieval_score": 0.0,
                "last_retrieval_component": "none",
                "error": "setup_failed", 
                "training_iteration": 1,
                "config": config,
                "timestamp": time.time(),
                "trial_duration_seconds": time.time() - trial_start_time,
                "trial_number": trial_number
            }
            session.report(trial_result)
            return
        
        for iteration in range(1, 4):
            iteration_start_time = time.time()
            
            fraction = iteration / 3.0
            budget_int = int(min_budget + (max_budget - min_budget) * fraction)
            budget_percentage = budget_int / total_samples
            
            actual_samples = min(budget_int, total_samples)
            if actual_samples < total_samples:
                qa_subset = qa_df.sample(n=actual_samples, random_state=42 + iteration)
            else:
                qa_subset = qa_df
            
            print(f"\n[Trial {trial_number}] Iteration {iteration}/3 | Budget: {budget_int} samples ({budget_percentage:.1%})")
            
            results = pipeline_runner.run_pipeline(config, trial_dir, qa_subset)
            
            score = results.get('combined_score', 0.0)
            latency = time.time() - iteration_start_time
            
            trial_result = {
                "trial_number": trial_number,
                "config": config,
                "score": float(results.get('score', 0.0)),
                "combined_score": float(score),
                "latency": float(latency),
                "budget": budget_int,
                "budget_percentage": budget_percentage,
                "trial_size": len(qa_subset),
                "trial_dir": trial_dir,
                "timestamp": time.time(),
                "training_iteration": iteration,
                "iteration_duration_seconds": latency,
                "cumulative_duration_seconds": time.time() - trial_start_time,
                "retrieval_score": float(results.get('retrieval_score', 0.0)),
                "generation_score": float(results.get('generation_score', 0.0)),
                "last_retrieval_component": results.get('last_retrieval_component', 'none'),
                "last_retrieval_score": float(results.get('last_retrieval_score', 0.0)),
                "compressor_score": float(results.get('compressor_score', 0.0) or results.get('compression_score', 0.0)),
                "filter_score": float(results.get('filter_score', 0.0)),
                "reranker_score": float(results.get('reranker_score', 0.0)),
                "prompt_maker_score": float(results.get('prompt_maker_score', 0.0)),
                "query_expansion_score": float(results.get('query_expansion_score', 0.0)),
            }
            
            for key, value in results.items():
                if key.endswith('_score') or key.endswith('_metrics'):
                    trial_result[key] = _make_json_serializable(value)
            
            print(f"  Score: {score:.4f}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Budget: {budget_int} samples ({budget_percentage:.1%})")
            
            json_manager.add_explored_config(config, trial_result)
            
            if use_wandb:
                global_step = (trial_number - 1) * 3 + iteration - 1
                WandBLogger.log_trial_metrics(
                    trial=trial_number,
                    score=trial_result['combined_score'],
                    config=config,
                    results=trial_result,
                    step=global_step
                )
            
            session.report(trial_result)
            
    except Exception as e:
        print(f"Error in trial {trial_number}: {e}")
        import traceback
        traceback.print_exc()
        error_result = {
            "combined_score": 0.0,
            "score": 0.0,
            "last_retrieval_score": 0.0,
            "last_retrieval_component": "none",
            "error": str(e), 
            "training_iteration": 1,
            "config": config,
            "timestamp": time.time(),
            "trial_duration_seconds": time.time() - trial_start_time,
            "trial_number": trial_number
        }
        session.report(error_result)
    finally:
        if os.path.exists(trial_dir):
            try:
                shutil.rmtree(trial_dir)
            except Exception as e:
                print(f"Warning: Could not clean up trial directory: {e}")

class RAGPipelineOptimizer:    
    def __init__(
        self,
        config_path: str,
        qa_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        project_dir: str,
        num_samples: Optional[int] = None,
        sample_percentage: float = 0.1,
        max_concurrent: int = 1,
        cpu_per_trial: Optional[int] = None,
        gpu_per_trial: Optional[float] = None,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        early_stop_threshold: float = 0.9,
        use_wandb: bool = True,
        wandb_project: str = "BO & AutoRAG",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        study_name: Optional[str] = None,
        min_budget_percentage: float = 0.33,
        max_budget_percentage: float = 1.0,
        eta: int = 3
    ):
        self.start_time = time.time() 
        self.project_root = Utils.find_project_root()
        self.config_path = Utils.get_centralized_config_path(config_path)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        print(f"BOHB using centralized config file: {self.config_path}")
        
        self.qa_df = qa_df
        self.corpus_df = corpus_df
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        print(f"BOHB using centralized project directory: {self.project_dir}")
        
        with open(self.config_path, 'r') as f:
            import yaml
            self.config_template = yaml.safe_load(f)
            
        self.config_generator = ConfigGenerator(self.config_template)
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator) 
        self.search_space_manager = SearchSpaceManager(self.config_generator)
        
        self.study_name = study_name if study_name else f"RayTune_bohb_opt_{int(time.time())}"
        
        if num_samples is None:
            suggestion = self.search_space_calculator.suggest_num_samples(
                sample_percentage=sample_percentage,
                min_samples=10,
                max_samples=50,
                max_combinations=500
            )
            self.num_samples = suggestion['num_samples']
            print(f"Auto-calculated num_samples: {self.num_samples}")
            print(f"Reasoning: {suggestion['reasoning']}")
        else:
            self.num_samples = num_samples
            print(f"Using provided num_samples: {self.num_samples}")
            
        self.sample_percentage = sample_percentage
        self.max_concurrent = max_concurrent
        self.cpu_per_trial = cpu_per_trial
        self.gpu_per_trial = gpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.early_stop_threshold = early_stop_threshold
        
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        
        self.min_budget_percentage = min_budget_percentage
        self.max_budget_percentage = max_budget_percentage
        self.eta = eta
        
        self.total_samples = len(qa_df)
        self.min_budget = max(1, int(self.total_samples * self.min_budget_percentage))
        self.max_budget = max(self.min_budget, int(self.total_samples * self.max_budget_percentage))
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.ray_storage_path = os.path.join(self.result_dir, "ray_results")
        os.makedirs(self.ray_storage_path, exist_ok=True)
        
        self.json_manager = JsonManager(self.result_dir)
        
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.query_expansion_metrics = []
        if self.config_generator.node_exists("query_expansion"):
            self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()

        self.filter_metrics = []
        if self.config_generator.node_exists("passage_filter"):
            self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        
        self.compressor_metrics = []
        if self.config_generator.node_exists("passage_compressor"):
            self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        
        self.reranker_metrics = []
        if self.config_generator.node_exists("passage_reranker"):
            self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        
        self.generation_metrics = []
        if self.config_generator.node_exists("generator"):
            self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        
        self.prompt_maker_metrics = []
        if self.config_generator.node_exists("prompt_maker"):
            self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config(node_type='prompt_maker')

        Utils.ensure_centralized_data(self.project_dir, self.corpus_df, self.qa_df)

        self.pipeline_runner = RAGPipelineRunner(
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
            json_manager=self.json_manager
        )

        if not ray.is_initialized():
            ray.init()
        
        self.all_trials = []
        
        self._print_initialization_summary()

    def _print_initialization_summary(self):
        summary = self.search_space_calculator.get_search_space_summary()
        
        print(f"\n===== BOHB Multi-Fidelity RAG Pipeline Optimizer =====")
        print(f"Using {self.num_samples} trials")
        print(f"Total search space combinations: {summary['search_space_size']}")
        print(f"Objectives: maximize score (weight={self.generation_weight}), minimize latency (weight={self.retrieval_weight})")
        
        print(f"\nMulti-fidelity settings:")
        print(f"  Min budget: {self.min_budget} samples ({self.min_budget_percentage:.1%})")
        print(f"  Max budget: {self.max_budget} samples ({self.max_budget_percentage:.1%})")
        print(f"  Eta: {self.eta}")
        
        for component, info in summary.items():
            if component != "search_space_size" and info['combinations'] > 1:
                print(f"\n{component.title()}:")
                print(f"  Combinations: {info['combinations']}")
        
        if self.cpu_per_trial is None:
            print(f"\nCPUs per trial: No limit (using all available)")
        else:
            print(f"\nCPUs per trial: {self.cpu_per_trial}")
        print(f"Using cached embeddings: {self.use_cached_embeddings}")
        print(f"Early stop threshold: {self.early_stop_threshold}")
        print(f"Number of workers: {self.max_concurrent}")

    def define_search_space(self) -> Dict[str, Any]:
        return self.search_space_manager.define_search_space()

    def _make_json_serializable(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        else:
            return obj

    def _format_trials_for_wandb(self, all_trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_trials = []
        
        for trial in all_trials:
            formatted_trial = {
                'trial_number': trial.get('trial_number', len(formatted_trials) + 1),
                'config': trial.get('config', {}),
                'score': float(trial.get('combined_score', trial.get('score', 0.0))),
                'latency': float(trial.get('latency', 0.0)),
                'budget': trial.get('budget', self.max_budget),
                'budget_percentage': trial.get('budget_percentage', 1.0),
                'status': trial.get('status', 'COMPLETE'),
                'trial': trial.get('trial_number', len(formatted_trials) + 1),
                'execution_time_s': float(trial.get('latency', 0.0))
            }
            
            component_scores = [
                'retrieval_score', 'generation_score', 'reranker_score', 
                'filter_score', 'compressor_score', 'compression_score',  
                'prompt_maker_score', 'query_expansion_score', 'last_retrieval_score'
            ]
            
            for score_key in component_scores:
                if score_key in trial:
                    if score_key == 'compression_score':
                        formatted_trial['compressor_score'] = float(trial.get(score_key, 0.0))
                    else:
                        formatted_trial[score_key] = float(trial.get(score_key, 0.0))

            if 'generation_score' not in formatted_trial and 'generation_score' in trial:
                formatted_trial['generation_score'] = float(trial.get('generation_score', 0.0))
            
            for key, value in trial.items():
                if key.endswith('_metrics') and key not in formatted_trial:
                    formatted_trial[key] = value
            
            formatted_trials.append(formatted_trial)
        
        formatted_trials = sorted(formatted_trials, key=lambda x: x['trial_number'])
        
        return formatted_trials
    
    
    def optimize(self):
        os.makedirs(self.result_dir, exist_ok=True)
        self.json_manager.save_iteration_summary(force_create=True)
        
        start_time = time.time()
        
        trial_counter = TrialCounter.remote()
        
        if self.use_wandb:
            search_space_info = self.search_space_calculator.get_search_space_summary()
            
            wandb_config = {
                "optimizer": "BOHB Multi-Fidelity",
                "n_trials": self.num_samples,
                "retrieval_weight": self.retrieval_weight,
                "generation_weight": self.generation_weight,
                "search_space_size": search_space_info['search_space_size'],
                "study_name": self.study_name,
                "early_stopping_threshold": self.early_stop_threshold,
                "n_workers": self.max_concurrent,
                "use_multi_fidelity": True,
                "min_budget": self.min_budget,
                "max_budget": self.max_budget,
                "eta": self.eta
            }
            
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.wandb_run_name or self.study_name,
                config=wandb_config,
                reinit=True
            )
        
        search_space = self.define_search_space()
        
        search_space['_optimizer_config'] = {
            'qa_df': self.qa_df,
            'project_dir': self.project_dir,
            'config_generator': self.config_generator,
            'pipeline_runner': self.pipeline_runner,
            'json_manager': self.json_manager,
            'use_cached_embeddings': self.use_cached_embeddings,
            'min_budget': self.min_budget,
            'max_budget': self.max_budget,
            'total_samples': self.total_samples,
            'num_samples': self.num_samples,
            'trial_counter': trial_counter,
            'use_wandb': self.use_wandb  
        }
        
        self.search_space_manager.print_search_space_summary(
            search_space, 
            self.num_samples, 
            self.sample_percentage, 
            self.max_concurrent, 
            self.cpu_per_trial, 
            self.retrieval_weight, 
            self.generation_weight
        )
        
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="combined_score",
            mode="max",
            max_t=3,
            grace_period=1,
            reduction_factor=self.eta,
            brackets=1
        )
        
        bohb_search = TuneBOHB(
            metric="combined_score", 
            mode="max",
            bohb_config={
                "min_points_in_model": 10,
                "top_n_percent": 15,
                "num_samples": 32,
                "random_fraction": 0.3,
                "bandwidth_factor": 3,
                "min_bandwidth": 0.001
            }
        )
        
        search_alg = ConcurrencyLimiter(bohb_search, max_concurrent=self.max_concurrent)
                
        run_name = f"rag_bohb_opt_{int(time.time())}"
        
        from ray.tune.stopper import Stopper
        
        class ScoreThresholdStopper(Stopper):
            def __init__(self, threshold, parent):
                self.threshold = threshold
                self._should_stop = False
                self.parent = parent
                
            def __call__(self, trial_id, result):
                if (result.get("training_iteration", 0) == 3 and 
                    result.get("combined_score", 0.0) > self.threshold):
                    print(f"\n*** Trial {trial_id} achieved score > {self.threshold}, stopping experiment! ***")
                    self._should_stop = True
                return False
                
            def stop_all(self):
                return self._should_stop
        
        experiment_stopper = ScoreThresholdStopper(self.early_stop_threshold, self)
        
        if self.cpu_per_trial is None:
            ray_resources = ray.available_resources()
            available_cpus = int(ray_resources.get("CPU", 1))
            available_gpus = ray_resources.get("GPU", 0)
            
            cpus_per_trial = max(1, available_cpus // self.max_concurrent)
            gpus_per_trial = available_gpus / self.max_concurrent if available_gpus > 0 else 0
            
            resources_per_trial = {
                "cpu": cpus_per_trial,
                "gpu": gpus_per_trial
            }
            
            print(f"\nResource allocation:")
            print(f"  Total: {available_cpus} CPUs, {available_gpus} GPUs")
            print(f"  Per trial: {cpus_per_trial} CPUs, {gpus_per_trial:.2f} GPUs")
        else:
            resources_per_trial = {"cpu": self.cpu_per_trial}
            available_gpus = ray.available_resources().get("GPU", 0)
            if available_gpus > 0:
                resources_per_trial["gpu"] = available_gpus / self.max_concurrent
        
        analysis = tune.run(
            run_trial_fn,
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial=resources_per_trial,
            stop=experiment_stopper,
            verbose=2,
            storage_path=self.ray_storage_path,
            name=run_name
        )

        self.all_trials = []

        trial_results = []
        for trial in analysis.trials:
            if hasattr(trial, 'last_result') and trial.last_result:
                result = trial.last_result
                trial_number = result.get('trial_number', 0)
                if trial_number > 0:  
                    trial_results.append((trial_number, trial, result))

        trial_results.sort(key=lambda x: x[0])

        for trial_number, trial, result in trial_results:
            clean_config = dict(trial.config)
            clean_config.pop('_optimizer_config', None)

            if 'query_expansion_retrieval_method' in clean_config:
                trial_data = {
                    'trial_number': trial_number,
                    'config': self._make_json_serializable(clean_config),
                    'score': float(result.get('score', 0.0)),
                    'combined_score': float(result.get('combined_score', 0.0)),
                    'latency': float(result.get('latency', 0.0)),
                    'budget': result.get('budget', self.max_budget),
                    'budget_percentage': result.get('budget_percentage', 1.0),
                    'retrieval_score': float(result.get('retrieval_score', 0.0)),
                    'generation_score': float(result.get('generation_score', 0.0)),
                    'last_retrieval_score': float(result.get('last_retrieval_score', 0.0)),
                    'last_retrieval_component': result.get('last_retrieval_component', 'none'),
                    'training_iteration': result.get('training_iteration', 0),
                    'status': 'COMPLETE' if result.get('training_iteration') == 3 else 'EARLY_STOPPED',
                    'query_expansion_top_k': 10  
                }
            else:
                trial_data = {
                    'trial_number': trial_number,
                    'config': self._make_json_serializable(clean_config),
                    'score': float(result.get('score', 0.0)),
                    'combined_score': float(result.get('combined_score', 0.0)),
                    'latency': float(result.get('latency', 0.0)),
                    'budget': result.get('budget', self.max_budget),
                    'budget_percentage': result.get('budget_percentage', 1.0),
                    'retrieval_score': float(result.get('retrieval_score', 0.0)),
                    'generation_score': float(result.get('generation_score', 0.0)),
                    'last_retrieval_score': float(result.get('last_retrieval_score', 0.0)),
                    'last_retrieval_component': result.get('last_retrieval_component', 'none'),
                    'training_iteration': result.get('training_iteration', 0),
                    'status': 'COMPLETE' if result.get('training_iteration') == 3 else 'EARLY_STOPPED'
                }
                
                if 'retriever_top_k' in clean_config:
                    trial_data['retriever_top_k'] = clean_config['retriever_top_k']

            for key, value in result.items():
                if key.endswith('_score') or key.endswith('_metrics'):
                    trial_data[key] = value
            
            self.all_trials.append(trial_data)

        best_trial = analysis.get_best_trial(metric="combined_score", mode="max")
        
        end_time = time.time()
        total_time = end_time - start_time
        early_stopped = experiment_stopper._should_stop

        best_config_result = None
        best_score = -float('inf')
        best_latency = float('inf')
        
        if best_trial and best_trial.last_result:
            best_config = dict(best_trial.config)
            best_config.pop('_optimizer_config', None)
            best_config_result = {
                'config': best_config,
                'score': best_trial.last_result.get('combined_score', 0.0),
                'latency': best_trial.last_result.get('latency', 0.0),
                'trial_number': best_trial.last_result.get('trial_number'),
                'budget': best_trial.last_result.get('budget', self.max_budget),
                'budget_percentage': best_trial.last_result.get('budget_percentage', 1.0)
            }
            best_score = best_config_result['score']
            best_latency = best_config_result['latency']

        full_budget_best = None
        if self.all_trials:
            full_budget_trials = [t for t in self.all_trials if t.get('status') == 'COMPLETE' and t.get('budget_percentage', 1.0) >= 0.99]
            if full_budget_trials:
                full_budget_best = max(full_budget_trials, key=lambda x: x['combined_score'])
                
        best_trial_config = dict(best_trial.config) if best_trial else {}
        best_trial_config.pop('_optimizer_config', None)
        
        optimization_results = {
            'optimizer': 'bohb',
            'use_multi_fidelity': True,
            'min_budget': self.min_budget,
            'max_budget': self.max_budget,
            'best_config': best_config_result,
            'best_trial_config': best_trial_config,
            'best_trial_score': best_score,
            'best_full_budget_config': full_budget_best['config'] if full_budget_best else {},
            'best_full_budget_score': full_budget_best['combined_score'] if full_budget_best else 0.0,
            'optimization_time': total_time,
            'n_trials': len(analysis.trials),
            'total_trials': len(self.all_trials),
            'early_stopped': early_stopped,
            'all_trials': self.all_trials
        }
        
        Utils.save_results_to_json(self.result_dir, "optimization_summary.json", optimization_results)
        
        if self.all_trials:
            Utils.save_results_to_csv(self.result_dir, "all_trials.csv", self._convert_for_csv(self.all_trials))
        
        if self.use_wandb and self.all_trials:
            formatted_trials = self._format_trials_for_wandb(self.all_trials)
            
            WandBLogger.log_optimization_plots(
                study_or_facade=None,
                all_trials=formatted_trials,
                pareto_front=[],
                prefix="bohb"
            )
            
            WandBLogger.log_component_comparison_plot(
                study_or_trials=formatted_trials,
                prefix="bohb"
            )
            
            WandBLogger.log_final_tables(
                all_trials=formatted_trials,
                pareto_front=[],
                prefix="bohb"
            )
            
            completed_trials = [t for t in formatted_trials if t.get('status') == 'COMPLETE']
            if completed_trials:
                WandBLogger.log_ranked_trials_table(
                    trials_data=completed_trials,
                    table_name="bohb/best_completed_trials",
                    top_n=20
                )
            
            WandBLogger.log_ranked_trials_table(
                trials_data=formatted_trials,
                table_name="bohb/all_trials_ranked",
                top_n=None
            )
            
            WandBLogger.log_component_metrics_table(
                study_or_trials=formatted_trials,
                table_name="bohb/component_metrics",
                step=len(formatted_trials)
            )
            
            summary_data = {
                'optimizer': 'bohb',
                'best_score': best_score,
                'best_latency': best_config_result['latency'] if best_config_result else float('inf'),
                'total_trials': len(self.all_trials),
                'completed_trials': len([t for t in self.all_trials if t.get('status') == 'COMPLETE']),
                'early_stopped_trials': len([t for t in self.all_trials if t.get('status') == 'EARLY_STOPPED']),
                'optimization_time': total_time,
                'early_stopped': early_stopped,
                'use_multi_fidelity': True,
                'min_budget': self.min_budget,
                'max_budget': self.max_budget
            }
            
            WandBLogger.log_summary(summary_data)
            
            wandb.finish()
        
        time_str = Utils.format_time_duration(total_time)
        
        print(f"\n{'='*60}")
        print(f"BOHB Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total optimization time: {time_str}")
        print(f"Total trials completed: {len(self.all_trials)}")
        print(f"Total Ray Tune trials: {len(analysis.trials)}")
        
        if self.all_trials:
            completed_trials = [t for t in self.all_trials if t.get('status') == 'COMPLETE']
            full_budget_count = len([t for t in completed_trials if t.get('budget_percentage', 1.0) >= 0.99])
            unique_configs = len(set(self._get_config_hash(t.get('config', {})) for t in self.all_trials))
            print(f"Unique configurations tested: {unique_configs}")
            print(f"Fully evaluated configurations (100% budget): {full_budget_count}")
            print(f"Early stopped trials: {len([t for t in self.all_trials if t.get('status') == 'EARLY_STOPPED'])}")
        
        if early_stopped:
            print("⚡ Optimization stopped early due to achieving target score!")
        
        if best_config_result:
            print(f"\nBest configuration found:")
            print(f"  Score: {best_config_result['score']:.4f}")
            print(f"  Latency: {best_config_result['latency']:.2f}s")
            print(f"  Budget: {best_config_result['budget']} samples ({best_config_result['budget_percentage']:.1%})")
            print(f"  Trial number: {best_config_result['trial_number']}")
        
        if full_budget_best and full_budget_best != best_config_result:
            print(f"\nBest full-budget configuration:")
            print(f"  Score: {full_budget_best['combined_score']:.4f}")
            print(f"  Latency: {full_budget_best['latency']:.2f}s")
            print(f"  Trial number: {full_budget_best['trial_number']}")
        
        print(f"\nResults saved to: {self.result_dir}")
        
        self.json_manager.save_metrics_as_csv("trial_metrics.csv")
        
        ray.kill(trial_counter)
        
        email_results = {
            'best_config': best_config_result,
            'best_score': best_score,
            'best_latency': best_latency,
            'total_trials': len(self.all_trials),
            'early_stopped': early_stopped,
            'all_trials': [
                {
                    'trial_number': t['trial_number'],
                    'score': t['combined_score'],
                    'latency': t['latency'],
                    'config': t['config']
                }
                for t in self.all_trials
            ],
            'pareto_front': []
        }
        
        return email_results


    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        serializable_config = self._make_json_serializable(config)
        config_str = json.dumps(dict(sorted(serializable_config.items())), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _convert_for_csv(self, trials):
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