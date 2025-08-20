import os
import sys
import time
import json
import yaml
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dotenv import load_dotenv

import optuna
import wandb
from optuna_rag.optuna_bo.optuna_global_optimization.plot_generator import PlotGenerator

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config_manager import ConfigGenerator
from pipeline.rag_pipeline_runner import RAGPipelineRunner, EarlyStoppingException
from pipeline.utils import Utils
from optuna_rag.config_extractor import OptunaConfigExtractor
from optuna_rag.optuna_bo.optuna_global_optimization.objective import OptunaObjective
from pipeline.logging.wandb import WandBLogger
from pipeline.search_space_calculator import SearchSpaceCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BOPipelineOptimizer:
    
    def __init__(
        self,
        config_path: str,
        qa_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        project_dir: str,
        n_trials: Optional[int] = None,
        sample_percentage: float = 0.1,
        cpu_per_trial: int = 4,
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_cached_embeddings: bool = True,
        result_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        continue_study: bool = False,
        use_wandb: bool = True,
        wandb_project: str = "BO & AutoRAG",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        optimizer: str = "tpe",
        early_stopping_threshold: float = 0.9,
        use_ragas: bool = False,
        ragas_llm_model: str = "gpt-4o-mini",
        ragas_embedding_model: str = "text-embedding-ada-002",
        ragas_metrics: Optional[Dict[str, List[str]]] = None,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o",
        component_early_stopping_enabled: bool = True,
        component_early_stopping_thresholds: Optional[Dict[str, float]] = None
    ):
        self.start_time = time.time()
        
        self.project_root = Utils.find_project_root()
        self.config_path = Utils.get_centralized_config_path(config_path)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        print(f"BO using config file: {self.config_path}")
        
        self.qa_df = qa_df
        self.corpus_df = corpus_df
        
        self.project_dir = Utils.get_centralized_project_dir(project_dir)
        print(f"BO using project directory: {self.project_dir}")
        
        with open(self.config_path, 'r') as f:
            self.config_template = yaml.safe_load(f)
        
        self.config_generator = ConfigGenerator(self.config_template)
        
        self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        
        if n_trials is None:
            suggestion = self.search_space_calculator.suggest_num_samples(
                sample_percentage=sample_percentage,
                min_samples=10,
                max_samples=50,
                max_combinations=500
            )
            self.n_trials = suggestion['num_samples']
            print(f"Use default trials: {self.n_trials}")
        else:
            self.n_trials = n_trials
            print(f"Using provided n_trials: {self.n_trials}")
        
        self.sample_percentage = sample_percentage
        self.cpu_per_trial = cpu_per_trial
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_cached_embeddings = use_cached_embeddings
        self.study_name = study_name if study_name else f"Optuna_rag_opt_{int(time.time())}"
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name or self.study_name
        self.optimizer = optimizer
        self.early_stopping_threshold = early_stopping_threshold

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
        
        self.use_ragas = use_ragas
        self.ragas_llm_model = ragas_llm_model
        self.ragas_embedding_model = ragas_embedding_model
        
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
        if ragas_metrics is None and use_ragas:
            self.ragas_metrics = {
                'retrieval': ['context_precision', 'context_recall'],
                'generation': ['answer_relevancy', 'faithfulness', 'factual_correctness', 'semantic_similarity']
            }
        else:
            self.ragas_metrics = ragas_metrics or {}
        
        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.best_score = {"value": 0.0, "config": None, "latency": float('inf')}
        self.best_latency = {"value": float('inf'), "config": None, "score": 0.0}
        self.all_trials = []
        
        self._params_table_data = []
        
        self._initialize_metrics()
        Utils.ensure_centralized_data(self.project_dir, self.corpus_df, self.qa_df)
        self._initialize_pipeline_runner()
        
        self.config_extractor = OptunaConfigExtractor(self.config_generator, search_type='bo')
        self.search_space = self.config_extractor.extract_search_space()
        
        self._print_initialization_summary()

    def _initialize_metrics(self):
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
            self.reranker_metrics = self.config_generator.extract_metrics_from_config(node_type='passage_reranker')
        
        self.generation_metrics = []
        if self.config_generator.node_exists("generator"):
            self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        
        self.prompt_maker_metrics = []
        if self.config_generator.node_exists("prompt_maker"):
            self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config(node_type='prompt_maker')

    def _initialize_pipeline_runner(self):
        early_stopping_thresholds = self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None
        
        if self.use_ragas:
            ragas_config = {
                'llm_model': self.ragas_llm_model,
                'embedding_model': self.ragas_embedding_model,
                'retrieval_metrics': self.ragas_metrics.get('retrieval', []),
                'generation_metrics': self.ragas_metrics.get('generation', [])
            }
            
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
                use_ragas=True,
                ragas_config=ragas_config,
                use_llm_compressor_evaluator=self.use_llm_compressor_evaluator,
                llm_evaluator_model=self.llm_evaluator_model,
                early_stopping_thresholds=early_stopping_thresholds  
            )
        else:
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
                use_llm_compressor_evaluator=self.use_llm_compressor_evaluator,
                llm_evaluator_model=self.llm_evaluator_model,
                early_stopping_thresholds=early_stopping_thresholds 
            )

    def _print_initialization_summary(self):
        summary = {}
        components = ['query_expansion', 'retrieval', 'passage_filter', 'passage_reranker', 
                    'passage_compressor', 'prompt_maker_generator']
        
        for component in components:
            combinations, note = self.search_space_calculator.calculate_component_combinations(component)
            summary[component] = {
                'combinations': combinations,
                'note': note
            }
        
        total_combinations = self.search_space_calculator.calculate_total_combinations()
        
        print("\n===== RAG Pipeline Optimizer Initialized =====")
        print(f"Using {self.n_trials} trials with {self.optimizer.upper()} sampler")

        if self.component_early_stopping_enabled:
            print(f"\nComponent-level early stopping ENABLED with thresholds:")
            for component, threshold in self.component_early_stopping_thresholds.items():
                print(f"  {component}: {threshold}")
        else:
            print(f"\nComponent-level early stopping DISABLED")
        
        print(f"\nHigh-score early stopping threshold: {self.early_stopping_threshold}")
        
        if self.use_ragas:
            print(f"\nEvaluation Method: RAGAS")
            print(f"  LLM Model: {self.ragas_llm_model}")
            print(f"  Embedding Model: {self.ragas_embedding_model}")
            print(f"  Metrics: {list(self.ragas_metrics.get('retrieval', [])) + list(self.ragas_metrics.get('generation', []))}")
        else:
            print(f"\nEvaluation Method: Traditional (component-wise)")
        
        for component in components:
            if summary[component]['combinations'] > 0:
                print(f"\n{component.replace('_', ' ').title()}:")
                
                if component == "retrieval":
                    print(f"  Metrics: {self.retrieval_metrics}")
                elif component == "query_expansion" and self.query_expansion_metrics:
                    print(f"  Metrics: {self.query_expansion_metrics}")
                elif component == "passage_filter" and self.filter_metrics:
                    print(f"  Metrics: {self.filter_metrics}")
                elif component == "passage_reranker" and self.reranker_metrics:
                    print(f"  Metrics: {self.reranker_metrics}")
                elif component == "passage_compressor" and self.compressor_metrics:
                    print(f"  Metrics: {self.compressor_metrics}")
                elif component == "prompt_maker_generator":
                    if self.generation_metrics:
                        print(f"  Generator Metrics: {self.generation_metrics}")
                        
        print(f"\nScore weights - Retrieval: {self.retrieval_weight}, Generation: {self.generation_weight}")
        
        print("\nSearch space summary:")
        for param, values in self.search_space.items():
            if isinstance(values, list):
                print(f"  {param}: {len(values)} options (categorical)")
            elif isinstance(values, tuple):
                print(f"  {param}: ({values[0]}, {values[1]}) (continuous)")

    def objective(self, trial: optuna.Trial) -> Tuple[float, float]:
        start_time = time.time()
        
        try:
            early_stopped = False
            early_stopped_component = None
            early_stopped_score = None
            
            try:
                objective_instance = OptunaObjective(
                    search_space=self.search_space,
                    config_generator=self.config_generator,
                    pipeline_runner=self.pipeline_runner,
                    component_cache=None,
                    corpus_df=self.corpus_df,
                    qa_df=self.qa_df,
                    use_cache=False,
                )

                score = objective_instance(trial)
                
            except EarlyStoppingException as e:
                early_stopped = True
                early_stopped_component = e.component
                early_stopped_score = e.score
                score = e.score 
                
                print(f"\n[Trial {trial.number}] Early stopped at {e.component} with score {e.score:.4f}")

                trial.set_user_attr('early_stopped', True)
                trial.set_user_attr('early_stopped_component', e.component)
                trial.set_user_attr('early_stopped_score', e.score)
            
            if score is None or score < 0:
                score = 0.0

            execution_time = time.time() - start_time
            trial.set_user_attr('execution_time', execution_time)
            latency = execution_time

            display_config = {k: v for k, v in trial.params.items()}

            if self.use_ragas and 'ragas_mean_score' in trial.user_attrs:
                ragas_score = trial.user_attrs.get('ragas_mean_score', 0.0)
                if ragas_score > 0:
                    score = ragas_score
                
                ragas_metrics = trial.user_attrs.get('ragas_metrics', {})
                for metric_name, metric_value in ragas_metrics.items():
                    trial.set_user_attr(f'ragas_{metric_name}', metric_value)

            if 'generation_score' in trial.user_attrs and 'generator_score' not in trial.user_attrs:
                trial.set_user_attr('generator_score', trial.user_attrs.get('generation_score', 0.0))
            elif 'generation_score' not in trial.user_attrs and 'generator_score' not in trial.user_attrs:
                for attr_key, attr_value in trial.user_attrs.items():
                    if 'generation' in attr_key.lower() and isinstance(attr_value, (int, float)):
                        trial.set_user_attr('generator_score', attr_value)
                        break

            trial_result = self._create_trial_result(
                trial.number, display_config, score, latency, trial.user_attrs
            )

            if early_stopped:
                trial_result['early_stopped'] = True
                trial_result['early_stopped_component'] = early_stopped_component
                trial_result['early_stopped_score'] = early_stopped_score

            if self.use_wandb:
                WandBLogger.log_trial_metrics(trial, score, results=trial_result)
                
                if self.use_ragas:
                    step = WandBLogger.get_next_step()
                    WandBLogger._log_ragas_detailed_metrics(trial_result, step=step)

                row_data = {
                    "trial": trial.number, 
                    "score": score,
                    "execution_time_s": round(execution_time, 2),
                    "status": "EARLY_STOPPED" if early_stopped else "COMPLETE"
                }
                
                if early_stopped:
                    row_data["early_stopped_component"] = early_stopped_component
                    row_data["early_stopped_score"] = early_stopped_score
                
                for param_name, param_value in trial.params.items():
                    row_data[param_name] = param_value

                if 'retriever_top_k' in trial_result and 'retriever_top_k' not in row_data:
                    row_data['retriever_top_k'] = trial_result['retriever_top_k']
                elif 'retriever_top_k' in trial.params:
                    row_data['retriever_top_k'] = trial.params['retriever_top_k']

                if 'generator_score' in trial_result and trial_result['generator_score'] > 0:
                    row_data['generator_score'] = trial_result['generator_score']
                elif 'generation_score' in trial_result and trial_result['generation_score'] > 0:
                    row_data['generator_score'] = trial_result['generation_score']

                if self.use_ragas and 'ragas_mean_score' in trial_result:
                    row_data['ragas_mean_score'] = trial_result['ragas_mean_score']
                
                self._params_table_data.append(row_data)
                params_table = WandBLogger.create_parameters_table(self._params_table_data)
                if params_table:
                    wandb.log({"parameters_table": params_table}, step=trial.number)

                wandb.log({
                    "multi_objective/score": score,
                    "multi_objective/latency": latency,
                    "early_stopped": early_stopped
                }, step=trial.number)

            self._update_best_results(score, latency, display_config)
            self.all_trials.append(trial_result)

            Utils.save_optimization_results(self.result_dir, self.all_trials, self.best_score, self.best_latency)

            print(f"\nTrial {trial.number} completed:")
            if early_stopped:
                print(f"  Status: EARLY STOPPED at {early_stopped_component}")
                print(f"  Component score: {early_stopped_score:.4f} < threshold {self.component_early_stopping_thresholds.get(early_stopped_component, 'N/A')}")
            print(f"  Score: {score:.4f}" + (" (RAGAS)" if self.use_ragas else ""))
            print(f"  Latency: {latency:.2f}s")
            
            if self.use_ragas and 'ragas_metrics' in trial_result:
                print("  RAGAS Metrics:")
                for metric, value in trial_result['ragas_metrics'].items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")
            
            print(f"  Best score so far: {self.best_score['value']:.4f}")
            print(f"  Best latency so far: {self.best_latency['value']:.2f}s")

            gen_score = trial_result.get('generator_score', trial_result.get('generation_score', 0.0))
            if gen_score > 0:
                print(f"  Generator Score: {gen_score:.4f}")

            return -score, latency
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, float('inf')
        
    def _create_trial_result(self, trial_number: int, config: Dict[str, Any], 
                   score: float, latency: float, user_attrs: Dict[str, Any]) -> Dict[str, Any]:
        all_component_metrics = {}
        
        metrics_mapping = {
            "retrieval": "retrieval",
            "reranker": "reranker",
            "filter": "filter",
            "compressor": "compressor",
            "prompt_maker": "prompt_maker",
            "generator": "generator"
        }
        
        for prefix in metrics_mapping.values():
            for key, value in user_attrs.items():
                if key.startswith(f"{prefix}_") and not key.endswith("_score"):
                    all_component_metrics[key] = value
        
        generator_score_value = user_attrs.get("generator_score", user_attrs.get("generation_score", 0.0))
        compressor_score_value = user_attrs.get("compressor_score", user_attrs.get("compression_score", 0.0))
        
        trial_result = {
            "trial_number": trial_number,
            "config": config,
            "score": score,
            "latency": latency,
            "retrieval_score": user_attrs.get("retrieval_score", 0.0),
            "reranker_score": user_attrs.get("reranker_score", 0.0),
            "filter_score": user_attrs.get("filter_score", 0.0),
            "compressor_score": compressor_score_value,
            "compression_score": compressor_score_value,
            "prompt_maker_score": user_attrs.get("prompt_maker_score", 0.0),
            "generator_score": generator_score_value,
            "generation_score": generator_score_value,
            "combined_score": user_attrs.get("combined_score", 0.0),
            "timestamp": time.time(),
            **all_component_metrics
        }

        if user_attrs.get("early_stopped", False):
            trial_result["early_stopped"] = True
            trial_result["early_stopped_component"] = user_attrs.get("early_stopped_component")
            trial_result["early_stopped_score"] = user_attrs.get("early_stopped_score")
        
        if self.use_ragas:
            trial_result["ragas_mean_score"] = user_attrs.get("ragas_mean_score", 0.0)
            trial_result["ragas_metrics"] = user_attrs.get("ragas_metrics", {})
            trial_result["evaluation_method"] = "ragas"
        else:
            trial_result["evaluation_method"] = "traditional"

        if 'retriever_top_k' in config:
            trial_result["retriever_top_k"] = config.get('retriever_top_k')
        elif config.get('query_expansion_method') and config.get('query_expansion_method') != 'pass_query_expansion':
            trial_result["retriever_top_k"] = 10
        
        return trial_result
    
    def _update_best_results(self, score: float, latency: float, config: Dict[str, Any]):
        if score > self.best_score["value"]:
            self.best_score = {"value": score, "config": config, "latency": latency}
            
        if latency < self.best_latency["value"]:
            self.best_latency = {"value": latency, "config": config, "score": score}

    def _generate_plots(self, study):
        try:
            plot_generator = PlotGenerator(self.result_dir)
            plot_generator.generate_all_plots(study, self.all_trials)
        except ImportError:
            print("Matplotlib or Optuna visualization not available. Skipping plots.")
        except Exception as e:
            print(f"Error generating plots: {e}")

    def save_study_results(self, study: optuna.Study):
        df = study.trials_dataframe()
        Utils.save_results_to_csv(self.result_dir, "optuna_trials.csv", df)
        
        all_metrics = []
        early_stopped_count = 0
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                gen_score = trial.user_attrs.get('generator_score', trial.user_attrs.get('generation_score', 0.0))
                
                metrics = {
                    'run_id': f"trial_{trial.number}",
                    'score': -trial.values[0] if trial.values else 0.0,
                    'latency': trial.values[1] if len(trial.values) > 1 else float('inf'),
                    **trial.params,
                    **trial.user_attrs
                }

                metrics['generator_score'] = gen_score
                
                if trial.user_attrs.get('early_stopped', False):
                    early_stopped_count += 1
                    metrics['early_stopped'] = True
                    metrics['early_stopped_component'] = trial.user_attrs.get('early_stopped_component')
                    metrics['early_stopped_score'] = trial.user_attrs.get('early_stopped_score')
                
                all_metrics.append(metrics)
        
        Utils.save_results_to_csv(self.result_dir, "all_metrics.csv", all_metrics)
        
        print(f"\nTotal early stopped trials: {early_stopped_count} out of {len(study.trials)}")
        
        best_trials = study.best_trials
        if best_trials:
            best_metrics = []
            for trial in best_trials:
                best_metric = {
                    'trial_number': trial.number,
                    'score': -trial.values[0] if trial.values else 0.0,
                    'latency': trial.values[1] if len(trial.values) > 1 else float('inf'),
                    'params': trial.params,
                    'metrics': trial.user_attrs
                }

                if 'metrics' in best_metric and isinstance(best_metric['metrics'], dict):
                    gen_score = best_metric['metrics'].get('generator_score', 
                                                         best_metric['metrics'].get('generation_score', 0.0))
                    best_metric['metrics']['generator_score'] = gen_score
                
                best_metrics.append(best_metric)
            
            Utils.save_results_to_json(self.result_dir, "best_params.json", best_metrics)
            
            print(f"\nBest trials found: {len(best_trials)}")
            for i, trial in enumerate(best_trials[:5]):
                print(f"\nBest trial #{i+1}: Trial {trial.number}")
                print(f"Score: {-trial.values[0]:.4f}, Latency: {trial.values[1]:.2f}s")
                print(f"Params: {trial.params}")

                gen_score = trial.user_attrs.get('generator_score', 
                                               trial.user_attrs.get('generation_score', 0.0))
                if gen_score > 0:
                    print(f"Generator Score: {gen_score:.4f}")
    
    def _find_best_config(self, study: optuna.Study) -> Optional[Dict[str, Any]]:
        # Filter out early stopped trials when finding best config
        valid_trials = [t for t in self.all_trials if not t.get('early_stopped', False)]
        
        high_score_trials = []
        for trial in valid_trials:
            if trial['score'] > 0.9:
                high_score_trials.append(trial)
        
        if high_score_trials:
            best_trial = min(high_score_trials, key=lambda x: x['latency'])
            return best_trial

        pareto_front = Utils.find_pareto_front(valid_trials)
        if pareto_front:
            for trial in sorted(pareto_front, key=lambda x: -x['score']):
                if trial['score'] > 0.8:
                    return trial
            return max(pareto_front, key=lambda x: x['score'])
        
        return None

    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.use_wandb:
            search_space_filtered = {}
            for param_name, param_value in self.search_space.items():
                if isinstance(param_value, list):
                    search_space_filtered[param_name] = param_value
                elif isinstance(param_value, tuple) and len(param_value) == 2:
                    search_space_filtered[param_name] = f"[{param_value[0]}, {param_value[1]}]"
                else:
                    search_space_filtered[param_name] = str(param_value)
            
            wandb_config = {
                "search_type": f"bayesian_optimization_{self.optimizer}",
                "optimizer": f"BO-{self.optimizer.upper()}",
                "n_trials": self.n_trials,
                "retrieval_weight": self.retrieval_weight,
                "generation_weight": self.generation_weight,
                "search_space": search_space_filtered,
                "search_space_size": self.search_space_calculator.calculate_total_combinations(),
                "study_name": self.study_name,
                "evaluation_method": "RAGAS" if self.use_ragas else "Traditional",
                "ragas_enabled": self.use_ragas,
                "ragas_llm_model": self.ragas_llm_model if self.use_ragas else None,
                "ragas_embedding_model": self.ragas_embedding_model if self.use_ragas else None,
                "ragas_metrics": self.ragas_metrics if self.use_ragas else None,
                "component_early_stopping_enabled": self.component_early_stopping_enabled,
                "component_early_stopping_thresholds": self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None
            }
            
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.wandb_run_name,
                config=wandb_config,
                reinit=True
            )
        
        class EarlyStoppingCallback:
            def __init__(self, score_threshold=0.9):
                self.score_threshold = score_threshold
                self.should_stop = False
            
            def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    score = -trial.values[0] if trial.values else 0.0
                    
                    if score > self.score_threshold:
                        print(f"\n*** Early stopping triggered! ***")
                        print(f"Trial {trial.number} achieved score {score:.4f} > {self.score_threshold}")
                        self.should_stop = True
                        study.stop()

        threshold = getattr(self, 'early_stopping_threshold', getattr(self, '_temp_early_stopping', 0.9))
        early_stopping = EarlyStoppingCallback(score_threshold=threshold)
        
        try:
            if self.optimizer == "tpe":
                from optuna.samplers import TPESampler
                sampler = TPESampler(
                    n_startup_trials=10,
                    n_ei_candidates=24,
                    seed=42,
                    multivariate=True,
                    constant_liar=True,
                    warn_independent_sampling=False
                )
                print("Using TPE (Tree-structured Parzen Estimator) sampler")
                
            elif self.optimizer == "botorch":
                from optuna.integration import BoTorchSampler
                sampler = BoTorchSampler(
                    n_startup_trials=10,
                    seed=42
                )
                print("Using BoTorch (Gaussian Process-based) sampler")
                                
            elif self.optimizer == "random":
                from optuna.samplers import RandomSampler
                sampler = RandomSampler(seed=42)
                print("Using Random sampler")
                
            else:
                from optuna.samplers import TPESampler
                sampler = TPESampler(
                    n_startup_trials=10,
                    n_ei_candidates=24,
                    seed=42,
                    multivariate=True,
                    constant_liar=True,
                    warn_independent_sampling=False
                )
                print(f"Unknown sampler type '{self.optimizer}', using TPE as default")
            
            study = optuna.create_study(
                directions=["minimize", "minimize"],
                sampler=sampler,
                study_name=self.study_name
            )
            
            callbacks = [early_stopping]
            
            try:
                study.optimize(
                    self.objective,
                    n_trials=self.n_trials,
                    callbacks=callbacks,
                    show_progress_bar=True
                )
            except:
                if early_stopping.should_stop:
                    print("Optimization stopped early due to achieving target score.")
                else:
                    raise
            
            end_time = time.time()
            total_time = end_time - start_time
            time_str = Utils.format_time_duration(total_time)
            
            self.save_study_results(study)
            self._generate_plots(study)
            
            if self.use_wandb and wandb.run is not None:
                print("\nGenerating Optuna visualization plots for W&B...")
                
                if self.use_ragas:
                    WandBLogger.log_ragas_comparison_plot(self.all_trials, prefix="bo")
                    WandBLogger.log_ragas_summary_table(self.all_trials, prefix="bo")

                WandBLogger.log_optimization_plots(study, self.all_trials, None, prefix="bo_plots")
                WandBLogger.log_final_tables(self.all_trials, None, prefix="final")
                WandBLogger.log_summary(study)
            
            best_config = self._find_best_config(study)
            
            # Count early stopped trials
            early_stopped_count = sum(1 for t in self.all_trials if t.get('early_stopped', False))
            
            print("\n===== Optimization Results =====")
            print(f"Total optimization time: {time_str}")
            print(f"Total trials: {len(self.all_trials)}")
            print(f"Early stopped trials: {early_stopped_count}")
            print(f"Completed trials: {len(self.all_trials) - early_stopped_count}")
            print(f"Sampler used: {self.optimizer.upper()}")
            
            if best_config:
                print("\nBest configuration (considering score > 0.9 with minimum latency):")
                print(f"  Trial: {best_config['trial_number']}")
                print(f"  Score: {best_config['score']:.4f}")
                print(f"  Latency: {best_config['latency']:.2f}s")
                print(f"  Config: {best_config['config']}")
            
            print("\nBest trial by score only:")
            print(f"  Score: {self.best_score['value']:.4f}")
            print(f"  Latency: {self.best_score['latency']:.2f}s")
            print(f"  Config: {self.best_score['config']}")
            
            print("\nBest trial by latency only:")
            print(f"  Score: {self.best_latency['score']:.4f}")
            print(f"  Latency: {self.best_latency['value']:.2f}s")
            print(f"  Config: {self.best_latency['config']}")
            
            # Only consider non-early-stopped trials for Pareto front
            valid_trials = [t for t in self.all_trials if not t.get('early_stopped', False)]
            pareto_front = Utils.find_pareto_front(valid_trials)
            
            print(f"\nPareto optimal solutions: {len(pareto_front)}")
            for i, trial in enumerate(pareto_front[:5]):
                print(f"  Solution {i+1}: Score={trial['score']:.4f}, Latency={trial['latency']:.2f}s (Trial #{trial['trial_number']})")
                print(f"    Config: {trial['config']}")
            
            results = {
                "best_config": best_config,
                "best_score_config": self.best_score["config"],
                "best_score": self.best_score["value"],
                "best_score_latency": self.best_score["latency"],
                "best_latency_config": self.best_latency["config"],
                "best_latency": self.best_latency["value"],
                "best_latency_score": self.best_latency["score"],
                "pareto_front": pareto_front,
                "optimization_time": total_time,
                "n_trials": len(self.all_trials),
                "early_stopped_trials": early_stopped_count,
                "completed_trials": len(self.all_trials) - early_stopped_count,
                "early_stopped": early_stopping.should_stop,
                "optimizer": self.optimizer,
                "total_trials": len(self.all_trials),
                "all_trials": self.all_trials,
                "component_early_stopping_enabled": self.component_early_stopping_enabled,
                "component_early_stopping_thresholds": self.component_early_stopping_thresholds if self.component_early_stopping_enabled else None
            }

            if best_config:
                results["best_config"] = {
                    "config": best_config.get("config", {}),
                    "score": best_config.get("score", 0.0),
                    "latency": best_config.get("latency", float('inf')),
                    "trial_number": best_config.get("trial_number", -1)
                }
            
            Utils.save_results_to_json(self.result_dir, "optimization_summary.json", results)
            
            if self.use_wandb:
                wandb.summary["best_score"] = self.best_score["value"]
                wandb.summary["best_latency"] = self.best_latency["value"]
                wandb.summary["total_trials"] = len(self.all_trials)
                wandb.summary["early_stopped_trials"] = early_stopped_count
                wandb.summary["completed_trials"] = len(self.all_trials) - early_stopped_count
                wandb.summary["optimization_time"] = total_time
                wandb.summary["early_stopped"] = early_stopping.should_stop
                wandb.summary["optimizer"] = self.optimizer
                wandb.summary["evaluation_method"] = "RAGAS" if self.use_ragas else "Traditional"
                
                if best_config:
                    for key, value in best_config['config'].items():
                        wandb.summary[f"best_config_{key}"] = value
                
                wandb.finish()
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            if self.use_wandb:
                wandb.finish(exit_code=1)
            raise