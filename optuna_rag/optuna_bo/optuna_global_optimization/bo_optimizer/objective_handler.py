import os
import time
import optuna
import wandb
from typing import Tuple, Dict, Any

from pipeline.pipeline_runner.rag_pipeline_runner import EarlyStoppingException
from pipeline.utils import Utils
from pipeline.logging.wandb import WandBLogger
from optuna_rag.optuna_bo.optuna_global_optimization.optuna_objective import OptunaObjective


class ObjectiveHandler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def objective(self, trial: optuna.Trial) -> Tuple[float, float]:
        start_time = time.time()
        
        try:
            objective_instance = OptunaObjective(
                search_space=self.optimizer.search_space,
                config_generator=self.optimizer.config_generator,
                pipeline_runner=self.optimizer.pipeline_runner,
                corpus_df=self.optimizer.corpus_df,
                qa_df=self.optimizer.qa_df
            )

            score = objective_instance(trial)
            
            if score is None or score < 0:
                score = 0.0

            execution_time = time.time() - start_time
            trial.set_user_attr('execution_time', execution_time)
            latency = execution_time

            display_config = {k: v for k, v in trial.params.items()}
            
            display_config['save_intermediate_results'] = True

            if self.optimizer.use_ragas and 'ragas_mean_score' in trial.user_attrs:
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

            trial_dir = os.path.join(self.optimizer.result_dir, f"trial_{trial.number}")
            if os.path.exists(trial_dir):
                debug_dir = os.path.join(trial_dir, "debug_intermediate_results")
                if os.path.exists(debug_dir):
                    print(f"[DEBUG] Intermediate results saved in: {debug_dir}")
                    parquet_files = [f for f in os.listdir(debug_dir) if f.endswith('.parquet')]
                    json_files = [f for f in os.listdir(debug_dir) if f.endswith('.json')]
                    print(f"[DEBUG] Found {len(parquet_files)} parquet files and {len(json_files)} JSON files")
                    for pf in parquet_files:
                        file_size = os.path.getsize(os.path.join(debug_dir, pf)) / 1024
                        print(f"[DEBUG]   - {pf} ({file_size:.1f} KB)")

            if self.optimizer.use_wandb:
                self._log_to_wandb(trial, score, latency, execution_time, trial_result)

            self.optimizer._update_best_results(score, latency, display_config)
            self.optimizer.all_trials.append(trial_result)

            Utils.save_optimization_results(self.optimizer.result_dir, self.optimizer.all_trials, 
                                           self.optimizer.best_score, self.optimizer.best_latency)

            self._print_trial_summary(trial.number, score, latency, trial_result)

            return -score, latency
            
        except EarlyStoppingException as e:
            print(f"\n[TRIAL {trial.number}] Early stopped at {e.component} with score {e.score:.4f}")
            
            execution_time = time.time() - start_time
            trial.set_user_attr('execution_time', execution_time)
            trial.set_user_attr('early_stopped', True)
            trial.set_user_attr('stopped_at', e.component)
            trial.set_user_attr('stopped_score', e.score)
            
            actual_score = e.score
            latency = execution_time
            
            display_config = {k: v for k, v in trial.params.items()}
            display_config['save_intermediate_results'] = True
            
            trial_result = self._create_trial_result(
                trial.number, display_config, actual_score, latency, trial.user_attrs
            )
            trial_result['early_stopped'] = True
            trial_result['stopped_at'] = e.component
            trial_result['stopped_score'] = e.score
            
            self.optimizer.all_trials.append(trial_result)
            self.optimizer.early_stopped_trials.append(trial_result)
            
            if self.optimizer.use_wandb:
                WandBLogger.log_trial_metrics(trial, actual_score, results=trial_result)
                wandb.log({
                    "early_stopped": True,
                    "stopped_at": e.component,
                    "stopped_score": e.score
                }, step=trial.number)
            
            print(f"  ⚠️  EARLY STOPPED - Component: {e.component}, Score: {e.score:.4f}")
            
            return -actual_score, latency
            
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
        
        if self.optimizer.use_ragas:
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
    
    def _log_to_wandb(self, trial, score, latency, execution_time, trial_result):
        WandBLogger.log_trial_metrics(trial, score, results=trial_result)
        
        if self.optimizer.use_ragas:
            step = WandBLogger.get_next_step()
            WandBLogger._log_ragas_detailed_metrics(trial_result, step=step)

        row_data = {
            "trial": trial.number, 
            "score": score,
            "execution_time_s": round(execution_time, 2),
            "status": "COMPLETE"
        }
        
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

        if self.optimizer.use_ragas and 'ragas_mean_score' in trial_result:
            row_data['ragas_mean_score'] = trial_result['ragas_mean_score']
        
        self.optimizer._params_table_data.append(row_data)
        params_table = WandBLogger.create_parameters_table(self.optimizer._params_table_data)
        if params_table:
            wandb.log({"parameters_table": params_table}, step=trial.number)

        wandb.log({
            "multi_objective/score": score,
            "multi_objective/latency": latency
        }, step=trial.number)
    
    def _print_trial_summary(self, trial_number, score, latency, trial_result):
        print(f"\nTrial {trial_number} completed:")
        print(f"  Score: {score:.4f}" + (" (RAGAS)" if self.optimizer.use_ragas else ""))
        print(f"  Latency: {latency:.2f}s")
        
        if self.optimizer.use_ragas and 'ragas_metrics' in trial_result:
            print("  RAGAS Metrics:")
            for metric, value in trial_result['ragas_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
        
        print(f"  Best score so far: {self.optimizer.best_score['value']:.4f}")
        print(f"  Best latency so far: {self.optimizer.best_latency['value']:.2f}s")

        gen_score = trial_result.get('generator_score', trial_result.get('generation_score', 0.0))
        if gen_score > 0:
            print(f"  Generator Score: {gen_score:.4f}")