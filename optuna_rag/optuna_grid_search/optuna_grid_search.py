import os
import json
import yaml
import optuna
from optuna.samplers import GridSampler
import wandb
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import time 
from dotenv import load_dotenv

load_dotenv()

from pipeline.config_manager import ConfigGenerator
from pipeline.utils import Utils
from optuna_rag.cache_manager import ComponentCacheManager
from pipeline.rag_pipeline_runner import RAGPipelineRunner
from optuna_rag.config_extractor import OptunaConfigExtractor
from optuna_rag.objective import OptunaObjective
from pipeline.wandb_logger import WandBLogger
from pipeline.email_notifier import ExperimentEmailNotifier


class OptunaGridSearchRunner:
    def __init__(self, config_path: str, corpus_path: str, qa_path: str, 
                 output_base_dir: str, use_cache: bool = True,
                 retrieval_weight: float = 0.5, generation_weight: float = 0.5,
                 use_wandb: bool = True, wandb_project: str = "BO & AutoRAG",
                 wandb_entity: Optional[str] = None, wandb_run_name: Optional[str] = None,
                 send_email: bool = False):
    
        self.project_root = Utils.find_project_root()
        self.config_path = Utils.get_centralized_config_path(config_path)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        print(f"Optuna GridSearch using centralized config file: {self.config_path}")
        
        self.corpus_path = corpus_path
        self.qa_path = qa_path
        self.output_base_dir = output_base_dir
        self.use_cache = use_cache
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.send_email = send_email

        self.corpus_df = pd.read_parquet(corpus_path)
        self.qa_df = pd.read_parquet(qa_path)

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_generator = ConfigGenerator(self.config)

        self.cache_dir = os.path.join(output_base_dir, "optuna_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.component_cache = ComponentCacheManager(self.cache_dir)

        self.study_name = f"Optuna_rag_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.wandb_run_name = wandb_run_name or self.study_name

        self._initialize_metrics()
        self._initialize_pipeline_runner()
        
        self.config_extractor = OptunaConfigExtractor(self.config_generator, search_type='grid')
        
        if self.send_email:
            try:
                self.email_notifier = ExperimentEmailNotifier()
                print("Email notifications enabled")
            except Exception as e:
                print(f"Warning: Failed to initialize email notifier: {e}")
                print("Continuing without email notifications")
                self.send_email = False
    
    def _initialize_metrics(self):
        self.retrieval_metrics = self.config_generator.extract_metrics_from_config()
        
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
            self.reranker_metrics = self.config_generator.extract_metrics_from_config(
                node_type='passage_reranker'
            )
        
        self.generation_metrics = []
        if self.config_generator.node_exists("generator"):
            self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        
        self.prompt_maker_metrics = []

    
    def _initialize_pipeline_runner(self):
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
            json_manager=None
        )
    
    def _check_wandb_login(self) -> bool:
        try:
            import wandb
            if wandb.api.api_key is None:
                print("You are not logged into Weights & Biases!")
                return False
            return True
        except ImportError:
            print("wandb package not installed!")
            return False
        except Exception:
            return False
        
    def _is_component_active(self, component: str, params: Dict[str, Any]) -> bool:
        if component == 'query_expansion':
            method = params.get('query_expansion_method')
            return method and method != 'pass_query_expansion'
        elif component == 'retrieval':
            qe_method = params.get('query_expansion_method')
            return not (qe_method and qe_method != 'pass_query_expansion')
        elif component == 'reranker':
            method = params.get('passage_reranker_method')
            return method and method != 'pass_reranker'
        elif component == 'filter':
            method = params.get('passage_filter_method')
            return method and method != 'pass_passage_filter'
        elif component == 'compressor':
            method = params.get('passage_compressor_method')
            return method and method != 'pass_compressor'
        elif component == 'prompt_maker':
            method = params.get('prompt_maker_method')
            return method and method != 'pass_prompt_maker'
        elif component == 'generator':
            return 'generator_model' in params or 'generator_llm' in params
        return False

    def run_grid_search(self, n_jobs: int = 1) -> optuna.Study:
        start_time = time.time()
        experiment_status = "completed"
        error_message = None
        results_dict = None

        try:
            self.search_space = self.config_extractor.extract_search_space()
            
            original_objective = OptunaObjective(
                search_space=self.search_space,
                config_generator=self.config_generator,
                pipeline_runner=self.pipeline_runner,
                component_cache=self.component_cache,
                corpus_df=self.corpus_df,
                qa_df=self.qa_df,
                use_cache=self.use_cache,
                search_type='grid'
            )
            
            if hasattr(original_objective, 'valid_param_combinations'):
                total_combinations = len(original_objective.valid_param_combinations)
                print(f"Total valid grid search combinations: {total_combinations}")
            else:
                total_combinations = 1
                for param, values in self.search_space.items():
                    if isinstance(values, list):
                        total_combinations *= len(values)
                print(f"Total grid search combinations: {total_combinations}")
            
            print("\nParameter space breakdown:")
            for param, values in self.search_space.items():
                if isinstance(values, list):
                    print(f"  {param}: {len(values)} options")
            
            if self.use_wandb:
                print(f"\n===== Weights & Biases Configuration =====")
                
                if not self._check_wandb_login():
                    print("\nContinuing without W&B tracking...")
                    self.use_wandb = False
                else:
                    print(" W&B authentication successful")
                    
                    wandb_config = {
                        "search_type": "grid_search",
                        "total_combinations": total_combinations,
                        "retrieval_weight": self.retrieval_weight,
                        "generation_weight": self.generation_weight,
                        "search_space": self.search_space,
                        "study_name": self.study_name,
                        "n_jobs": n_jobs,
                        "cache_enabled": self.use_cache,
                    }
                    
                    wandb.init(
                        project=self.wandb_project,
                        entity=self.wandb_entity,
                        name=self.wandb_run_name,
                        config=wandb_config,
                        reinit=True
                    )
                    print(f"\nWeights & Biases tracking enabled")
            
            study = optuna.create_study(
                direction="maximize",
                sampler=GridSampler(self.search_space),
                study_name=self.study_name,
            )
            
            self._params_table_data = []
            self._cache_stats = {
                'total_cache_hits': 0,
                'total_cache_misses': 0,
                'component_cache_hits': {},
                'component_cache_misses': {}
            }
            
            def objective_with_wandb_logging(trial):
                start_time = time.time()
                trial_cache_stats = {
                    'query_expansion': {'hit': False, 'miss': False},
                    'retrieval': {'hit': False, 'miss': False},
                    'reranker': {'hit': False, 'miss': False},
                    'filter': {'hit': False, 'miss': False},
                    'compressor': {'hit': False, 'miss': False},
                    'prompt_maker': {'hit': False, 'miss': False},
                    'generator': {'hit': False, 'miss': False}
                }

                original_check_cache = self.component_cache.check_cache
                
                def tracked_check_cache(component, params):
                    df, metrics = original_check_cache(component, params)
                    if df is not None and metrics is not None:
                        trial_cache_stats[component]['hit'] = True
                        self._cache_stats['total_cache_hits'] += 1
                        self._cache_stats['component_cache_hits'][component] = self._cache_stats['component_cache_hits'].get(component, 0) + 1
                    else:
                        if self._is_component_active(component, params): 
                            trial_cache_stats[component]['miss'] = True
                            self._cache_stats['total_cache_misses'] += 1
                            self._cache_stats['component_cache_misses'][component] = self._cache_stats['component_cache_misses'].get(component, 0) + 1
                    return df, metrics
                
                self.component_cache.check_cache = tracked_check_cache
                
                try:
                    score = original_objective(trial)
                finally:
                    self.component_cache.check_cache = original_check_cache
                
                execution_time = time.time() - start_time

                components_cached = sum(1 for stats in trial_cache_stats.values() if stats['hit'])
                components_computed = sum(1 for stats in trial_cache_stats.values() if stats['miss'])
                cache_hit_rate = components_cached / (components_cached + components_computed) if (components_cached + components_computed) > 0 else 0
                
                if self.use_wandb and wandb.run is not None:
                    WandBLogger.log_trial_metrics(trial, score)
                    cache_metrics = {
                        "cache/trial_components_cached": components_cached,
                        "cache/trial_components_computed": components_computed,
                        "cache/trial_cache_hit_rate": cache_hit_rate,
                        "cache/total_cache_hits": self._cache_stats['total_cache_hits'],
                        "cache/total_cache_misses": self._cache_stats['total_cache_misses'],
                        "cache/overall_hit_rate": self._cache_stats['total_cache_hits'] / (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) if (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) > 0 else 0,
                    }
                    
                    for component in trial_cache_stats:
                        if trial_cache_stats[component]['hit']:
                            cache_metrics[f"cache/{component}_hit"] = 1
                        elif trial_cache_stats[component]['miss']:
                            cache_metrics[f"cache/{component}_hit"] = 0
                    
                    wandb.log(cache_metrics, step=trial.number)

                    row_data = {
                        "trial": trial.number,
                        "score": score,
                        "execution_time_s": round(execution_time, 2),
                        "cached_components": components_cached,
                        "computed_components": components_computed,
                        "cache_hit_rate": round(cache_hit_rate, 2),
                        **trial.params,
                        "status": "COMPLETE"  
                    }
                    self._params_table_data.append(row_data)
                    params_table = WandBLogger.create_parameters_table(self._params_table_data)
                    if params_table:
                        wandb.log({"parameters_table": params_table}, step=trial.number)
                
                return score
            
            callbacks = []
            
            try:
                study.optimize(
                    objective_with_wandb_logging,
                    n_trials=total_combinations,
                    n_jobs=n_jobs,
                    show_progress_bar=True,
                    callbacks=callbacks 
                )
            except Exception as e:
                if "All trials in the search space have been evaluated" in str(e):
                    print("Grid search completed - all combinations evaluated")
                else:
                    raise e
            
            if self.use_wandb and wandb.run is not None:
                final_cache_stats = {
                    "cache/final_total_hits": self._cache_stats['total_cache_hits'],
                    "cache/final_total_misses": self._cache_stats['total_cache_misses'],
                    "cache/final_overall_hit_rate": self._cache_stats['total_cache_hits'] / (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) if (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) > 0 else 0,
                }

                for component in ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor', 'prompt_maker', 'generator']:
                    hits = self._cache_stats['component_cache_hits'].get(component, 0)
                    misses = self._cache_stats['component_cache_misses'].get(component, 0)
                    total = hits + misses
                    if total > 0:
                        final_cache_stats[f"cache/final_{component}_hits"] = hits
                        final_cache_stats[f"cache/final_{component}_misses"] = misses
                        final_cache_stats[f"cache/final_{component}_hit_rate"] = hits / total
                
                wandb.log(final_cache_stats)

                cache_summary_data = []
                for component in ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor', 'prompt_maker', 'generator']:
                    hits = self._cache_stats['component_cache_hits'].get(component, 0)
                    misses = self._cache_stats['component_cache_misses'].get(component, 0)
                    total = hits + misses
                    if total > 0:
                        cache_summary_data.append({
                            'component': component,
                            'cache_hits': hits,
                            'cache_misses': misses,
                            'total_calls': total,
                            'hit_rate': round(hits / total, 3)
                        })
                
                if cache_summary_data:
                    cache_table = wandb.Table(
                        columns=['component', 'cache_hits', 'cache_misses', 'total_calls', 'hit_rate'],
                        data=[[row[col] for col in ['component', 'cache_hits', 'cache_misses', 'total_calls', 'hit_rate']] 
                            for row in cache_summary_data]
                    )
                    wandb.log({"cache_summary_table": cache_table})
                
                print("\nGenerating Optuna visualization plots for W&B...")

                WandBLogger.log_optimization_plots(study, prefix="grid_search_plots")
                max_step = max([t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) if study.trials else 0
                WandBLogger.log_study_tables(study, self._params_table_data, prefix="final", step=max_step)
                WandBLogger.log_summary(study, additional_metrics={
                    "cache_enabled": self.use_cache,
                    "total_cache_hits": self._cache_stats['total_cache_hits'],
                    "total_cache_misses": self._cache_stats['total_cache_misses'],
                    "overall_cache_hit_rate": self._cache_stats['total_cache_hits'] / (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) if (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) > 0 else 0
                })
                
                wandb.finish()

            print("\n===== Cache Usage Summary =====")
            print(f"Total cache hits: {self._cache_stats['total_cache_hits']}")
            print(f"Total cache misses: {self._cache_stats['total_cache_misses']}")
            overall_hit_rate = self._cache_stats['total_cache_hits'] / (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) if (self._cache_stats['total_cache_hits'] + self._cache_stats['total_cache_misses']) > 0 else 0
            print(f"Overall hit rate: {overall_hit_rate:.2%}")
            
            print("\nPer-component cache statistics:")
            for component in ['query_expansion', 'retrieval', 'reranker', 'filter', 'compressor', 'prompt_maker', 'generator']:
                hits = self._cache_stats['component_cache_hits'].get(component, 0)
                misses = self._cache_stats['component_cache_misses'].get(component, 0)
                total = hits + misses
                if total > 0:
                    print(f"  {component}: {hits} hits, {misses} misses ({hits/total:.2%} hit rate)")
            
            self._save_study_results(study)
            
            if self.send_email:
                results_dict = self._prepare_email_results(study)
            
            return study
            
        except Exception as e:
            experiment_status = "failed"
            error_message = str(e)
            raise
            
        finally:
            if self.send_email and hasattr(self, 'email_notifier'):
                duration = time.time() - start_time
                try:
                    self.email_notifier.send_experiment_notification(
                        experiment_name=self.study_name,
                        results=results_dict,
                        duration=duration,
                        status=experiment_status,
                        error_message=error_message,
                        attach_plots=True
                    )
                except Exception as email_error:
                    print(f"Warning: Failed to send email notification: {email_error}")
    
    def _prepare_email_results(self, study: optuna.Study) -> Dict[str, Any]:
        best_trial = study.best_trial if study.best_trial else None
        
        all_trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_trials_data.append({
                    'trial_number': trial.number,
                    'score': trial.value,
                    'latency': trial.duration.total_seconds() if trial.duration else 0,
                    'params': trial.params
                })
        
        pareto_front = []
        if all_trials_data:
            sorted_trials = sorted(all_trials_data, key=lambda x: (-x['score'], x['latency']))
            for trial in sorted_trials[:10]:
                pareto_front.append({
                    'score': trial['score'],
                    'latency': trial['latency'],
                    'trial_number': trial['trial_number']
                })
        
        results = {
            'best_score': best_trial.value if best_trial else None,
            'best_latency': best_trial.duration.total_seconds() if best_trial and best_trial.duration else None,
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'best_config': {'config': best_trial.params} if best_trial else None,
            'all_trials': all_trials_data,
            'pareto_front': pareto_front,
            'cache_stats': self._cache_stats if hasattr(self, '_cache_stats') else {}
        }
        
        return results
    
    def _save_study_results(self, study: optuna.Study):
        df = study.trials_dataframe()
        df.to_csv(os.path.join(self.output_base_dir, "optuna_results.csv"), index=False)
        
        all_metrics = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                metrics = {
                    'run_id': f"trial_{trial.number}",
                    'score': trial.value,
                    **trial.params,
                    **trial.user_attrs
                }
                all_metrics.append(metrics)
        
        best_trial = None
        if study.trials:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if completed_trials:
                sorted_trials = sorted(
                    completed_trials,
                    key=lambda t: (-t.value, t.duration.total_seconds() if t.duration else float('inf'))
                )
                best_trial = sorted_trials[0]
                
                best_score = best_trial.value
                trials_with_best_score = [t for t in completed_trials if t.value == best_score]
                
                if len(trials_with_best_score) > 1:
                    print(f"\nFound {len(trials_with_best_score)} trials with the same best score ({best_score}).")
                    print("Selecting the one with shortest execution time:")
                    
                    for t in trials_with_best_score[:5]:
                        duration = t.duration.total_seconds() if t.duration else 0
                        print(f"  Trial {t.number}: score={t.value:.4f}, duration={duration:.2f}s")
        
        if best_trial:
            best_params_file = os.path.join(self.output_base_dir, "best_params.json")
            
            best_trial_data = {
                'params': best_trial.params,
                'score': best_trial.value,
                'trial_number': best_trial.number,
                'duration_seconds': best_trial.duration.total_seconds() if best_trial.duration else None,
                'metrics': best_trial.user_attrs
            }
            
            with open(best_params_file, 'w') as f:
                json.dump(best_trial_data, f, indent=2)
            
            print(f"\nBest trial: {best_trial.number}")
            print(f"Best score: {best_trial.value}")
            if best_trial.duration:
                print(f"Execution time: {best_trial.duration.total_seconds():.2f} seconds")
            print(f"Best params: {best_trial.params}")
        
        trials_summary = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                summary = {
                    'trial_number': trial.number,
                    'score': trial.value,
                    'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
                    'params': trial.params
                }
                trials_summary.append(summary)
        
        trials_summary.sort(key=lambda x: (-x['score'], x['duration_seconds'] or float('inf')))
        
        summary_file = os.path.join(self.output_base_dir, "trials_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(trials_summary, f, indent=2)
        
        print(f"\nTrials summary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run grid search using Optuna')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file (default: centralized config.yaml)')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to corpus.parquet file (default: centralized location)')
    parser.add_argument('--qa', type=str, default=None,
                        help='Path to qa_validation.parquet file (default: centralized location)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: ./optuna_grid_search_results)')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations after search')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases tracking')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Custom name for the W&B run (default: same as study_name)')
    parser.add_argument('--send-email', action='store_true', help='Send email notification when optimization completes')
    
    args = parser.parse_args()

    project_root = Utils.find_project_root()
    config_path = Utils.get_centralized_config_path(args.config)

    print(f"Using config: {config_path}")

    if args.corpus:
        corpus_path = args.corpus
    else:
        qa_path, corpus_path = Utils.get_centralized_data_paths()
    
    if args.qa:
        qa_path = args.qa
    else:
        qa_path, _ = Utils.get_centralized_data_paths()
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "optuna_grid_search_results")
    
    print(f"Using corpus: {corpus_path}")
    print(f"Using QA data: {qa_path}")
    print(f"Results will be saved to: {output_dir}")
    
    if args.send_email:
        print("\nEmail notifications enabled")

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        print("Please ensure config.yaml exists in the project root or specify --config")
        exit(1)
    
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus file not found at {corpus_path}")
        print("Please ensure corpus data exists in centralized location or specify --corpus")
        exit(1)
        
    if not os.path.exists(qa_path):
        print(f"ERROR: QA validation file not found at {qa_path}")
        print("Please ensure QA data exists in centralized location or specify --qa")
        exit(1)
    
    print("\n===== Starting Optuna Grid Search =====")
    print("All required files found. Starting optimization...")
    
    wandb_project = "BO & AutoRAG"
    wandb_entity = None
    
    runner = OptunaGridSearchRunner(
        config_path=config_path,
        corpus_path=corpus_path,
        qa_path=qa_path,
        output_base_dir=output_dir,
        use_cache=not args.no_cache,
        retrieval_weight=0.5,
        generation_weight=0.5,
        use_wandb=not args.no_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=args.wandb_run_name,
        send_email=args.send_email
    )
    
    study = runner.run_grid_search(n_jobs=args.n_jobs)
    
    if args.visualize:
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice, plot_contour
        import plotly
        
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            plots = {
                'optimization_history': plot_optimization_history(study),
                'param_importances': plot_param_importances(study),
                'parallel_coordinate': plot_parallel_coordinate(study),
                'slice': plot_slice(study),
            }
            
            if len([p for p in study.trials[0].params if isinstance(p, (int, float))] if study.trials else []) >= 2:
                plots['contour'] = plot_contour(study)
            
            for name, fig in plots.items():
                plotly.offline.plot(
                    fig,
                    filename=os.path.join(viz_dir, f"{name}.html"),
                    auto_open=False
                )
            
            print(f"Visualizations saved to {viz_dir}")
        except Exception as e:
            print(f"Warning: Some visualizations could not be generated: {e}")
        
    print("\n===== Optuna Grid Search Complete =====")
    print(f"Results saved to: {output_dir}")