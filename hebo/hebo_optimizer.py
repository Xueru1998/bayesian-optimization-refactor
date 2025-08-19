import os
import json
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import hashlib
import wandb
from typing import Dict, Any, List, Optional, Tuple, Union

from hebo.optimizers.hebo import HEBO

from pipeline.config_manager import ConfigGenerator
from pipeline.rag_pipeline_runner import RAGPipelineRunner
from pipeline.search_space_calculator import SearchSpaceCalculator
from pipeline.utils import Utils
from pipeline.wandb_logger import WandBLogger
from hebo_config_space_builder import HEBOConfigSpaceBuilder

class HEBORAGOptimizer:
    
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
        early_stopping_threshold: float = 0.9,
        use_wandb: bool = True,
        wandb_project: str = "BO & AutoRAG",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        seed: int = 42,
        batch_size: int = 1,
        n_suggestions: int = 1
    ):
        self.start_time = time.time()
        
        self._initialize_paths(project_dir, result_dir)
        self._initialize_data(config_template, qa_data, corpus_data)
        self._initialize_optimization_params(
            n_trials, sample_percentage, cpu_per_trial, retrieval_weight,
            generation_weight, use_cached_embeddings, walltime_limit,
            early_stopping_threshold, seed, batch_size, n_suggestions
        )
        self._initialize_wandb_params(use_wandb, wandb_project, wandb_entity, wandb_run_name)
        
        self.study_name = study_name if study_name else f"hebo_opt_{int(time.time())}"
        
        self._setup_components()
        self._calculate_trials_if_needed()
        
        Utils.ensure_centralized_data(self.project_dir, self.corpus_data, self.qa_data)
        
        self.trial_counter = 0
        self.all_trials = []
        self.early_stopped = False
        
        self._print_initialization_summary()
    
    def _initialize_paths(self, project_dir: str, result_dir: Optional[str]):
        self.project_root = Utils.find_project_root()
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "HEBO_results"
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
                                      walltime_limit, early_stopping_threshold, seed, 
                                      batch_size, n_suggestions):
        self.n_trials = n_trials
        self.sample_percentage = sample_percentage
        self.cpu_per_trial = cpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.walltime_limit = walltime_limit
        self.early_stopping_threshold = early_stopping_threshold
        self.seed = seed
        self.batch_size = batch_size
        self.n_suggestions = n_suggestions
    
    def _initialize_wandb_params(self, use_wandb, wandb_project, wandb_entity, wandb_run_name):
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
    
    def _setup_components(self):
        self.config_generator = ConfigGenerator(self.config_template)
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        self.config_space_builder = HEBOConfigSpaceBuilder(self.config_generator)
        
        self._setup_runner()
    
    def _setup_runner(self):
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config('prompt_maker')
        self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()
        
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
            generation_weight=self.generation_weight
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
            print(f"Auto-calculated num_trials: {self.n_trials}")
            print(f"Reasoning: {suggestion['reasoning']}")
        else:
            self.n_trials = max(20, self.n_trials)
            print(f"Using provided num_trials: {self.n_trials}")
    
    def _run_trial(self, config: Dict[str, Any]) -> Dict[str, Any]:
        self.trial_counter += 1
        
        trial_dir = self._setup_trial_directory()
        sampled_qa_data = self._sample_data()
        self._copy_corpus_data(trial_dir)
        
        trial_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Trial {self.trial_counter}/{self.n_trials}")
        print(f"{'='*60}")
        
        try:
            config_dict = self._prepare_config(config)
            trial_config = self.config_generator.generate_trial_config(config_dict)
            self._save_trial_config(trial_dir, trial_config)
            
            results = self.runner.run_pipeline(config_dict, trial_dir, sampled_qa_data)
            score = results.get('combined_score', 0.0)
            latency = time.time() - trial_start_time
            
            trial_result = self._create_trial_result(
                config_dict, score, latency, results
            )
            self.all_trials.append(trial_result)
            
            self._save_trial_results(trial_dir, trial_result)
            self._print_trial_summary(score, latency)
            
            if self.use_wandb:
                self._log_trial_to_wandb(config_dict, trial_result)
            
            if score >= self.early_stopping_threshold:
                print(f"\n*** Early stopping triggered! Score {score:.4f} >= {self.early_stopping_threshold} ***")
                self.early_stopped = True
            
            return {"score": score, "latency": latency}
            
        except Exception as e:
            print(f"Error in trial {self.trial_counter}: {e}")
            import traceback
            traceback.print_exc()
            return {"score": 0.0, "latency": float('inf')}
    
    def _setup_trial_directory(self) -> str:
        trial_dir = os.path.join(self.result_dir, f"trial_{self.trial_counter}")
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        return trial_dir
    
    def _sample_data(self) -> pd.DataFrame:
        return self.qa_data
    
    def _copy_corpus_data(self, trial_dir: str):
        centralized_corpus_path = os.path.join(self.project_dir, "data", "corpus.parquet")
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
    
    def _prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config_dict = self._convert_numpy_types(config)
        
        for temp_param in ['generator_temperature', 'query_expansion_temperature', 'temperature']:
            if temp_param in config_dict:
                try:
                    config_dict[temp_param] = float(config_dict[temp_param])
                except:
                    config_dict[temp_param] = 0.7
        
        return config_dict
    
    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    def _save_trial_config(self, trial_dir: str, trial_config: Dict[str, Any]):
        config_file = os.path.join(trial_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(trial_config, f, default_flow_style=False)
    
    def _create_trial_result(self, config_dict, score, latency, results):
        trial_result = {
            "trial_number": self.trial_counter,
            "config": self._convert_numpy_types(config_dict),
            "score": float(score),
            "latency": float(latency),
            "retrieval_score": float(results.get('retrieval_score', 0.0)),
            "generation_score": float(results.get('generation_score', 0.0)),
            "combined_score": float(score),
            "timestamp": time.time()
        }
        
        for k, v in results.items():
            if k.endswith('_score') or k.endswith('_metrics'):
                trial_result[k] = self._convert_numpy_types(v)
        
        return trial_result
    
    def _save_trial_results(self, trial_dir: str, trial_result: Dict[str, Any]):
        results_file = os.path.join(trial_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(trial_result, f, indent=2)
    
    def _print_trial_summary(self, score, latency):
        print(f"Trial {self.trial_counter} completed:")
        print(f"  Score: {score:.4f}")
        print(f"  Latency: {latency:.2f}s")
    
    def _log_trial_to_wandb(self, config_dict, trial_result):
        WandBLogger.log_trial_metrics(
            self.trial_counter, 
            trial_result['score'],
            config=config_dict,
            results=trial_result,
            step=self.trial_counter
        )
    
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.use_wandb:
            self._initialize_wandb()
        
        design_space, param_info = self.config_space_builder.build_design_space()
        
        print(f"\nInitializing HEBO optimizer with design space of {param_info.get('_n_params', 0)} parameters")
        
        hebo = HEBO(design_space, scramble_seed=self.seed)
        
        self._print_optimization_start_info()
        
        n_iterations = self.n_trials // self.n_suggestions
        if self.n_trials % self.n_suggestions != 0:
            n_iterations += 1
        
        for iteration in range(n_iterations):
            if self.early_stopped:
                break
            
            if self.walltime_limit is not None and time.time() - start_time > self.walltime_limit:
                print(f"\nWalltime limit reached ({self.walltime_limit}s)")
                break
            
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            n_suggest = min(self.n_suggestions, self.n_trials - self.trial_counter)
            if n_suggest <= 0:
                break
            
            print(f"Requesting {n_suggest} suggestions from HEBO...")
            recommendations = hebo.suggest(n_suggestions=n_suggest)
            print(f"HEBO suggested {len(recommendations)} configurations")
            
            batch_scores = []
            batch_latencies = []
            
            for idx in range(len(recommendations)):
                config = recommendations.iloc[idx].to_dict()
                
                cleaned_config = self.config_space_builder.clean_config(config, param_info)
                
                print(f"\n[DEBUG] Trial {self.trial_counter + 1} config after cleaning:")
                for k, v in sorted(cleaned_config.items()):
                    print(f"  {k}: {v}")
                
                result = self._run_trial(cleaned_config)
                
                batch_scores.append(-result['score'])
                batch_latencies.append(result['latency'])
                
                if self.early_stopped:
                    break
            
            if len(batch_scores) > 0:
                success = self._safe_observe(hebo, recommendations, batch_scores, batch_latencies)
                
                if success:
                    all_scores = [trial['score'] for trial in self.all_trials]
                    overall_best = max(all_scores) if all_scores else 0.0
                    print(f"Iteration {iteration + 1} completed. Best score so far: {overall_best:.4f}")
                else:
                    print(f"Warning: Could not update HEBO with observations from iteration {iteration + 1}")
            else:
                print(f"Iteration {iteration + 1} completed with no trials")
        
        pareto_front = self._extract_pareto_front()
        
        optimization_results = self._create_optimization_results(
            pareto_front, self.early_stopped, time.time() - start_time
        )
        
        self._save_and_log_results(optimization_results, pareto_front)
        self._print_optimization_summary(optimization_results, pareto_front)
        
        return optimization_results
    
    def _safe_observe(self, hebo, recommendations, scores, latencies):

        valid_indices = []
        valid_scores = []
        valid_latencies = []
        
        for i, (score, latency) in enumerate(zip(scores, latencies)):
            if np.isfinite(score) and np.isfinite(latency):
                valid_indices.append(i)
                valid_scores.append(score)
                valid_latencies.append(latency)
            else:
                print(f"Warning: Skipping observation {i} with score={score}, latency={latency}")
        
        if not valid_indices:
            print("No valid observations to report to HEBO")
            return False

        obs_array = np.column_stack([valid_scores, valid_latencies])
        valid_recommendations = recommendations.iloc[valid_indices].reset_index(drop=True)
        
        try:
            hebo.observe(valid_recommendations, obs_array)
            return True
        except Exception as e:
            print(f"Failed to observe in HEBO: {e}")
            # Try 1: As DataFrame
            try:
                obs_df = pd.DataFrame({
                    'objective_0': valid_scores,
                    'objective_1': valid_latencies
                })
                hebo.observe(valid_recommendations, obs_df)
                print("Successfully observed using DataFrame format")
                return True
            except:
                pass

            try:
                single_obj = np.array(valid_scores).reshape(-1, 1)
                hebo.observe(valid_recommendations, single_obj)
                print("Successfully observed using single objective")
                return True
            except:
                pass
            
            return False
    
    def _initialize_wandb(self):
        design_space, param_info = self.config_space_builder.build_design_space()
        
        n_params = param_info.get('_n_params', 0)
        
        wandb_config = {
            "optimizer": "HEBO",
            "n_trials": self.n_trials,
            "retrieval_weight": self.retrieval_weight,
            "generation_weight": self.generation_weight,
            "search_space_size": n_params,
            "study_name": self.study_name,
            "early_stopping_threshold": self.early_stopping_threshold,
            "walltime_limit": self.walltime_limit if self.walltime_limit is not None else "No limit",
            "batch_size": self.batch_size,
            "n_suggestions": self.n_suggestions
        }
        
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.wandb_run_name or self.study_name,
            config=wandb_config,
            reinit=True
        )
    
    def _extract_pareto_front(self) -> List[Dict[str, Any]]:
        if not self.all_trials:
            return []
        
        points = [(t['score'], t['latency']) for t in self.all_trials]
        
        pareto_indices = []
        for i, (score_i, latency_i) in enumerate(points):
            is_dominated = False
            for j, (score_j, latency_j) in enumerate(points):
                if i != j:
                    if score_j >= score_i and latency_j <= latency_i:
                        if score_j > score_i or latency_j < latency_i:
                            is_dominated = True
                            break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        pareto_front = [self.all_trials[i] for i in pareto_indices]
        return sorted(pareto_front, key=lambda x: -x['score'])
    
    def _create_optimization_results(self, pareto_front, early_stopped, total_time):
        best_configs = self._find_best_configurations(pareto_front)
        
        return {
            'optimizer': 'HEBO',
            'best_config': best_configs['best_balanced'],
            'best_score_config': best_configs['best_score']['config'],
            'best_score': best_configs['best_score']['score'],
            'best_score_latency': best_configs['best_score']['latency'],
            'best_latency_config': best_configs['best_latency']['config'],
            'best_latency': best_configs['best_latency']['latency'],
            'best_latency_score': best_configs['best_latency']['score'],
            'pareto_front': pareto_front,
            'optimization_time': total_time,
            'n_trials': self.trial_counter,
            'total_trials': self.trial_counter,
            'early_stopped': early_stopped,
            'all_trials': self.all_trials
        }
    
    def _find_best_configurations(self, pareto_front):
        if not self.all_trials:
            default_trial = {'config': {}, 'score': 0.0, 'latency': float('inf')}
            return {
                'best_score': default_trial,
                'best_latency': default_trial,
                'best_balanced': default_trial
            }
        
        best_score_trial = max(self.all_trials, key=lambda x: x['score'])
        best_latency_trial = min(self.all_trials, key=lambda x: x['latency'])
        
        high_score_trials = [t for t in self.all_trials if t['score'] > 0.9]
        if high_score_trials:
            best_balanced = min(high_score_trials, key=lambda x: x['latency'])
        else:
            best_balanced = max(pareto_front, key=lambda x: x['score']) if pareto_front else best_score_trial
        
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
            import wandb
            WandBLogger.log_optimization_plots(None, self.all_trials, pareto_front, prefix="hebo")
            WandBLogger.log_final_tables(self.all_trials, pareto_front, prefix="final")
            WandBLogger.log_summary(optimization_results)
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
        print(f"HEBO Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total optimization time: {time_str}")
        print(f"Total trials: {self.trial_counter}")
        
        if optimization_results['early_stopped']:
            print("⚡ Optimization stopped early due to achieving target score!")
        
        self._print_best_config_summary(optimization_results['best_config'])
        self._print_pareto_front_summary(pareto_front)
        
        print(f"\nResults saved to: {self.result_dir}")
    
    def _print_best_config_summary(self, best_config):
        if best_config and isinstance(best_config, dict):
            print("\nBest balanced configuration (high score with low latency):")
            print(f"  Score: {best_config.get('score', 'N/A')}")
            print(f"  Latency: {best_config.get('latency', 'N/A')}")
            if 'config' in best_config:
                print(f"  Config: {best_config['config']}")
    
    def _print_pareto_front_summary(self, pareto_front):
        print(f"\nPareto front contains {len(pareto_front)} solutions")
        if pareto_front:
            print("Top 5 Pareto optimal solutions:")
            for i, solution in enumerate(sorted(pareto_front, key=lambda x: -x['score'])[:5]):
                print(f"  {i+1}. Score: {solution['score']:.4f}, Latency: {solution['latency']:.2f}s")
    
    def _print_initialization_summary(self):
        summary = self.search_space_calculator.get_search_space_summary()
        
        print(f"\n===== HEBO RAG Pipeline Optimizer =====")
        print(f"Using {self.n_trials} trials")
        print(f"Total search space combinations: {summary['search_space_size']}")
        print(f"Objectives: maximize score (weight={self.generation_weight}), minimize latency (weight={self.retrieval_weight})")
        
        for component, info in summary.items():
            if component != "search_space_size" and info['combinations'] > 1:
                print(f"\n{component.title()}:")
                print(f"  Combinations: {info['combinations']}")
        
        print(f"\nCPUs per trial: {self.cpu_per_trial}")
        print(f"Using cached embeddings: {self.use_cached_embeddings}")
        if self.walltime_limit is not None:
            print(f"Wall time limit: {self.walltime_limit}s")
        else:
            print(f"Wall time limit: No limit")
        print(f"Batch size: {self.batch_size}")
        print(f"Suggestions per iteration: {self.n_suggestions}")
    
    def _print_optimization_start_info(self):
        print(f"\nStarting HEBO optimization")
        print(f"Total trials: {self.n_trials}")
        print(f"Objectives: score (maximize), latency (minimize)")
        print(f"Early stopping threshold: {self.early_stopping_threshold}")