import os
import time
import json
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.integration import BoTorchSampler
import wandb

from ..base.base_componentwise_optimizer import BaseComponentwiseOptimizer
from .componentwise_grid_search import ComponentwiseGridSearch
from ..helpers.component_grid_search_helper import ComponentGridSearchHelper
from ..helpers.config_cache_manager import ConfigCacheManager
from pipeline.utils import Utils
from pipeline.wandb_logger import WandBLogger


class ComponentwiseOptunaOptimizer(BaseComponentwiseOptimizer):
    """Main optimizer class that handles both Grid Search and Bayesian Optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        cache_verbose = True
        self.cache_manager = ConfigCacheManager(self.result_dir, verbose=cache_verbose)
        self.use_cache = True

        cache_stats = self.cache_manager.get_cache_stats()
        if cache_stats['total_cached_configs'] > 0:
            print(f"[Cache] Found {cache_stats['total_cached_configs']} cached configurations")
            print(f"[Cache] Cache size: {cache_stats['cache_size_mb']:.2f} MB")
        
        if self.optimizer in ['grid', 'grid_search']:
            self.grid_optimizer = ComponentwiseGridSearch(*args, **kwargs)
            self.grid_helper = ComponentGridSearchHelper()
    
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        is_valid, invalid_components, component_combinations, combination_note = self._validate_all_components()
        
        print(f"\n{'='*70}")
        print(f"COMPONENT VALIDATION FOR {'GRID SEARCH' if self.optimizer == 'grid' else 'BAYESIAN OPTIMIZATION'}")
        print(f"All components search space:")
        for comp, combos in component_combinations.items():
            print(f"  - {comp}: {combos} combinations")
        
        if self.optimizer != 'grid':
            print(f"\nNote: {combination_note}")
        
        print(f"{'='*70}\n")
        
        effective_early_stopping_threshold = self.early_stopping_threshold if self.optimizer != "grid" else 1.1
        all_results = {
            'study_name': self.study_name,
            'component_results': {},
            'best_configs': {},
            'optimization_time': 0,
            'component_order': [],
            'early_stopped': False,
            'retrieval_weight': self.retrieval_weight,
            'generation_weight': self.generation_weight
        }
        
        active_components = []
        has_active_query_expansion = False
        
        for comp in self.COMPONENT_ORDER:
            if comp == 'query_expansion' and self.config_generator.node_exists(comp):
                qe_config = self.config_generator.extract_node_config("query_expansion")
                qe_methods = []
                for module in qe_config.get("modules", []):
                    method = module.get("module_type")
                    if method:
                        qe_methods.append(method)
                if qe_methods:
                    has_active_query_expansion = True
                    active_components.append(comp)
            elif comp == 'retrieval' and has_active_query_expansion:
                print("[Component-wise] Skipping retrieval component since query expansion includes retrieval")
                continue
            elif comp == 'prompt_maker_generator':
                if self.config_generator.node_exists('prompt_maker') or self.config_generator.node_exists('generator'):
                    active_components.append(comp)
            elif self.config_generator.node_exists(comp):
                active_components.append(comp)
                
        if self.optimizer == 'grid':
            self.grid_helper.save_all_components_grid_sequence(
                active_components, 
                self.result_dir,
                self
            )
        
        if self.resume_study:
            completed_components = list(self.best_configs.keys())
            remaining_components = [c for c in active_components if c not in completed_components]
            
            if not remaining_components:
                print("\n[RESUME] All components already completed!")
                summary_file = os.path.join(self.result_dir, "component_optimization_summary.json")
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        return json.load(f)
                else:
                    print("[RESUME] Warning: Summary file not found, starting fresh")
                    self.resume_study = False
            else:
                print(f"\n[RESUME] Remaining components to optimize: {remaining_components}")
                active_components = remaining_components
        
        early_stopped = False
        stopped_at_component = None
        
        for component_idx, component in enumerate(active_components):
            if early_stopped and component != 'prompt_maker_generator' and self.optimizer != "grid":
                print(f"\n[Component-wise] Skipping {component} optimization due to early stopping at {stopped_at_component}")
                
                all_results['component_results'][component] = {
                    'component': component,
                    'best_config': {},
                    'best_score': all_results['component_results'][stopped_at_component]['best_score'],
                    'n_trials': 0,
                    'optimization_time': 0.0,
                    'skipped': True,
                    'skip_reason': f'Early stopped at {stopped_at_component}'
                }
                
                all_results['best_configs'][component] = {}
                all_results['component_order'].append(component)
                continue

            if component == 'passage_filter' and 'passage_reranker' in self.best_configs:
                reranker_config = self.best_configs['passage_reranker']
                if reranker_config.get('reranker_top_k') == 1:
                    print(f"\n[Component-wise] Skipping filter optimization because reranker_top_k=1")
                    all_results['best_configs'][component] = {}
                    all_results['component_order'].append(component)
                    
                    all_results['component_results'][component] = {
                        'component': component,
                        'best_config': {},
                        'best_score': self.component_results.get('passage_reranker', {}).get('best_score', 0.0),
                        'n_trials': 0,
                        'optimization_time': 0.0,
                        'skipped': True,
                        'skip_reason': 'reranker_top_k=1'
                    }
                    continue
            
            if self.use_wandb:
                if wandb.run is not None:
                    try:
                        wandb.finish()
                    except Exception as e:
                        print(f"[WARNING] Error finishing previous W&B run: {e}")

                    time.sleep(2)
                
                wandb_run_name = f"{self.study_name}_{component}"
                try:
                    wandb.init(
                        project=self.wandb_project,
                        entity=self.wandb_entity,
                        name=wandb_run_name,
                        config={
                            "component": component,
                            "stage": f"{component_idx + 1}/{len(active_components)}",
                            "optimizer": self.optimizer
                        },
                        reinit=True,
                        force=True
                    )
                    WandBLogger.reset_step_counter()
                except Exception as e:
                    print(f"[WARNING] Failed to initialize W&B for {component}: {e}")
                    print(f"[WARNING] Continuing without W&B logging for {component}")
                    self.wandb_enabled = False
            
            component_result = self._optimize_component(
                component, 
                component_idx,
                active_components
            )
            
            all_results['component_results'][component] = component_result
            all_results['best_configs'][component] = component_result['best_config']
            all_results['component_order'].append(component)
            
            self.component_results[component] = component_result
            self.best_configs[component] = component_result['best_config']

            if (component_result.get('best_score', 0) >= effective_early_stopping_threshold and 
            component != 'prompt_maker_generator' and 
            self.optimizer != "grid"):
                early_stopped = True
                stopped_at_component = component
                component_result['early_stopped'] = True
                print(f"\n[Component-wise] Early stopping triggered at {component} with score {component_result['best_score']:.4f}")
            
            if self.use_wandb and wandb.run is not None:
                try:
                    WandBLogger.log_component_summary(
                        component, 
                        component_result['best_config'],
                        component_result['best_score'],
                        component_result['n_trials'],
                        component_result.get('optimization_time', 0.0)
                    )
                    wandb.finish()
                except Exception as e:
                    print(f"[WARNING] Error logging W&B summary for {component}: {e}")

            if self.use_wandb and not self.wandb_enabled:
                self.wandb_enabled = True
        
        all_results['optimization_time'] = time.time() - start_time
        all_results['early_stopped'] = early_stopped
        all_results['stopped_at_component'] = stopped_at_component
        all_results['total_trials'] = sum(
            comp.get('n_trials', 0) for comp in all_results['component_results'].values()
        )
        
        all_results['study_name'] = self.study_name
        
        if self.use_wandb:
            if wandb.run is not None:
                try:
                    wandb.finish()
                except Exception as e:
                    print(f"[WARNING] Error finishing component W&B run: {e}")
                
                time.sleep(2)
            
            try:
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=f"{self.study_name}_summary",
                    config={
                        "optimization_mode": f"componentwise_{self.optimizer}",
                        "total_components": len(active_components),
                        "total_time": all_results['optimization_time'],
                        "early_stopped": early_stopped,
                        "stopped_at_component": stopped_at_component,
                        "optimizer": self.optimizer
                    },
                    reinit=True,
                    force=True
                )

                WandBLogger.log_final_componentwise_summary(all_results)
                wandb.finish()
            except Exception as e:
                print(f"[WARNING] Failed to log final W&B summary: {e}")
        
        self._save_final_results(all_results)
        self._print_final_summary(all_results)
        
        return all_results
    
    def _optimize_component(self, component: str, component_idx: int, 
                          active_components: List[str]) -> Dict[str, Any]:
        if self.optimizer in ["grid", "grid_search"]:
            return self.grid_optimizer._optimize_component(component, component_idx, active_components)
        else:
            return self._optimize_component_bayesian(component, component_idx, active_components)
    
    def _optimize_component_bayesian(self, component: str, component_idx: int, 
                                    active_components: List[str]) -> Dict[str, Any]:
        component_start_time = time.time()
        component_dir = os.path.join(self.result_dir, f"{component_idx}_{component}")
        os.makedirs(component_dir, exist_ok=True)
        
        fixed_config = self._get_fixed_config(component, active_components)
        search_space = self.search_space_builder.build_component_search_space(component, fixed_config)
        
        if not search_space:
            return {
                'component': component,
                'best_config': {},
                'best_score': 0.0,
                'n_trials': 0,
                'optimization_time': 0.0
            }

        storage_path = os.path.join(component_dir, f"{component}_optuna.db")
        storage_url = f"sqlite:///{storage_path}"
        
        result = self._run_bayesian_optimization(
            component, component_idx, component_dir,
            fixed_config, search_space, storage_url, active_components
        )
        
        result['optimization_time'] = time.time() - component_start_time
        
        result_serializable = self._convert_numpy_types(result.copy())
        with open(os.path.join(component_dir, "optimization_result.json"), 'w') as f:
            json.dump(result_serializable, f, indent=2)
        
        return result
    
    def _run_bayesian_optimization(self, component: str, component_idx: int, 
                                  component_dir: str, fixed_config: Dict[str, Any],
                                  search_space: Dict[str, Any], storage_url: str,
                                  active_components: List[str]) -> Dict[str, Any]:
        
        total_combinations, note = self.combination_calculator.calculate_component_combinations(
            component, search_space, fixed_config, self.best_configs
        )
        n_trials = self._calculate_component_trials(component, search_space)

        print(f"[{component}] Using {self.optimizer.upper()} sampler")
        print(f"  - Search space size: {total_combinations} combinations")
        print(f"  - Trials to run: {n_trials}")
        
        self.combination_calculator.print_combination_info(
            component, total_combinations, note, search_space
        )
        
        actual_sampler = self._create_sampler()
        
        self.current_component = component
        self.current_fixed_config = fixed_config
        self.component_trial_counter = 0
        self.component_trials = []
        self.component_detailed_metrics[component] = []
        
        if self.use_multi_objective:
            study = optuna.create_study(
                directions=["maximize", "minimize"],
                sampler=actual_sampler,
                study_name=f"{self.study_name}_{component}",
                storage=storage_url,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                direction="maximize", 
                sampler=actual_sampler,
                study_name=f"{self.study_name}_{component}",
                storage=storage_url,
                load_if_exists=True
            )
        
        early_stopping_callback = self._create_early_stopping_callback()
        
        def objective(trial: optuna.Trial):
            return self._component_objective(trial, component, fixed_config, search_space)
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=[early_stopping_callback] if self.early_stopping_threshold < 1.0 else [],
                show_progress_bar=True
            )
        except Exception as e:
            if hasattr(early_stopping_callback, 'should_stop') and early_stopping_callback.should_stop:
                print(f"[{component}] Early stopping triggered")
            else:
                print(f"[{component}] Optimization error: {e}")
        
        if self.use_multi_objective:
            best_trials = study.best_trials
            if best_trials:
                best_trial = max(best_trials, key=lambda t: t.values[0])
                best_config = dict(best_trial.params)
                best_score = best_trial.values[0]
            else:
                best_trial = None
                best_config = {}
                best_score = 0.0
        else:
            best_trial = study.best_trial if study.best_trial else None
            if best_trial:
                best_config = dict(best_trial.params)
                best_score = best_trial.value
            else:
                best_config = {}
                best_score = 0.0
        
        best_trial_data = Utils.find_best_trial_from_component(self.component_trials, component)

        if best_trial_data:
            best_score = best_trial_data['score']
            best_latency = best_trial_data.get('latency', 0.0)
            best_output_path = best_trial_data.get('output_parquet')
        else:
            best_score = 0.0
            best_latency = 0.0
            best_output_path = None
        
        if best_output_path and os.path.exists(best_output_path):
            self.component_dataframes[component] = best_output_path
            print(f"[{component}] Best configuration found with score {best_score:.4f} and latency {best_latency:.2f}s")
            print(f"[{component}] Best output saved at: {best_output_path}")
        
        final_best_config = {}
        if best_trial_data and 'config' in best_trial_data:
            final_best_config = best_trial_data['config']
        elif best_config:
            final_best_config = best_config
        
        return {
            'component': component,
            'best_config': final_best_config,
            'best_score': best_trial_data['score'] if best_trial_data else 0.0,
            'best_latency': best_trial_data.get('latency', 0.0) if best_trial_data else 0.0,
            'best_trial': best_trial_data,
            'all_trials': self.component_trials,
            'n_trials': len(self.component_trials),
            'fixed_config': fixed_config,
            'search_space_size': len(search_space),
            'total_combinations': total_combinations,
            'optimization_method': self.optimizer,
            'detailed_metrics': self.component_detailed_metrics.get(component, []),
            'best_output_path': best_output_path
        }
    
    def _component_objective(self, trial: optuna.Trial, component: str, 
                       fixed_config: Dict[str, Any], search_space: Dict[str, Any]):
        trial_config = self.search_space_builder.suggest_component_params(trial, component, search_space, fixed_config)
        
        full_config = trial_config.copy()
        full_config.update(fixed_config)
        trial_config, full_config = self._parse_composite_configs(component, trial_config, full_config)
        
        self._clean_config(full_config)
        
        if self.use_cache:
            cached_result = self.cache_manager.get_cached_result(component, full_config)
            if cached_result:
                self.component_trial_counter += 1
                
                trial_result = {
                    "trial": int(self.component_trial_counter),
                    "trial_number": int(self.component_trial_counter),
                    "global_trial_number": int(self.current_trial),
                    "component": component,
                    "config": self._convert_numpy_types(trial_config),
                    "full_config": self._convert_numpy_types(full_config),
                    "score": cached_result.get('score', 0.0),
                    "latency": cached_result.get('latency', 0.0),
                    "execution_time_s": 0.0,
                    "budget": cached_result.get('budget', 0),
                    "budget_percentage": cached_result.get('budget_percentage', 1.0),
                    "status": "COMPLETE",
                    "results": cached_result,
                    "output_parquet": cached_result.get('output_parquet'),
                    "timestamp": float(time.time()),
                    "from_cache": True,
                    "original_trial_id": cached_result.get('original_trial_id')
                }
                
                self.component_trials.append(trial_result)
                
                if cached_result.get('output_parquet') and os.path.exists(cached_result.get('output_parquet')):
                    if cached_result.get('score', 0.0) > self.component_results.get(component, {}).get('best_score', 0.0):
                        self.component_dataframes[component] = cached_result.get('output_parquet')
                
                if self.wandb_enabled:
                    WandBLogger.log_component_trial(component, self.component_trial_counter, 
                                                full_config, cached_result.get('score', 0.0), 0.0)
                    WandBLogger.log_dynamic_component_table(component, self.component_trials, self.wandb_enabled)
                
                print(f"\n[Trial {self.current_trial}] CACHED Score: {cached_result.get('score', 0.0):.4f} | Time: 0.00s")
                
                if self.use_multi_objective:
                    return cached_result.get('score', 0.0), cached_result.get('latency', 0.0)
                else:
                    return cached_result.get('score', 0.0)
        
        self.global_trial_counter += 1
        self.current_trial = self.global_trial_counter
        self.component_trial_counter += 1
        
        trial_id = f"trial_{self.global_trial_counter:04d}"
        trial_dir = os.path.join(self.result_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        
        start_time = time.time()
        
        try:
            if component == 'passage_compressor' and trial_config.get('passage_compressor_config') == 'pass_compressor':
                trial_config['passage_compressor_method'] = 'pass_compressor'
                full_config['passage_compressor_method'] = 'pass_compressor'

            qa_subset = self.pipeline_manager.prepare_component_data(
                component, self.qa_data, self.component_dataframes, trial_dir
            )
            
            self.pipeline_manager.copy_corpus_data(trial_dir)
            
            trial_config_yaml = self.config_generator.generate_trial_config(full_config)
            
            with open(os.path.join(trial_dir, "config.yaml"), 'w') as f:
                yaml.dump(trial_config_yaml, f)
            
            if hasattr(self.pipeline_runner, 'component_results'):
                self.pipeline_runner.component_results = self.component_results
            else:
                setattr(self.pipeline_runner, 'component_results', self.component_results)

            results = self.pipeline_runner.run_pipeline(
                full_config, 
                trial_dir, 
                qa_subset,
                is_local_optimization=True,
                current_component=component
            )
            
            if 'working_df' in results and isinstance(results['working_df'], pd.DataFrame):
                working_df = results['working_df']
                results.pop('working_df', None)
            else:
                working_df = qa_subset
            
            score = self.pipeline_manager.get_component_score(component, results)
            
            output_parquet_path = self.pipeline_manager.save_component_output(
                component, trial_dir, results, working_df
            )
            
            if score > self.component_results.get(component, {}).get('best_score', 0.0):
                self.component_dataframes[component] = output_parquet_path
            
            end_time = time.time()
            latency = end_time - start_time
            
            detailed_metrics = self.pipeline_manager.extract_detailed_metrics(component, results)
            
            if component not in self.component_detailed_metrics:
                self.component_detailed_metrics[component] = []
            
            self.component_detailed_metrics[component].append(detailed_metrics)
            
            results_to_cache = {
                'score': score,
                'latency': latency,
                'budget': len(qa_subset) if 'qa_subset' in locals() else 0,
                'budget_percentage': 1.0,
                'results': results
            }
            
            if self.use_cache:
                    self.cache_manager.save_to_cache(
                        component, full_config, results_to_cache, 
                        trial_id, output_parquet_path
                    )
            
            trial_result = self._create_trial_result(
                full_config, score, latency, 
                len(qa_subset) if 'qa_subset' in locals() else 0, 1.0,
                results, component, output_parquet_path
            )
            
            trial_result['component_trial_number'] = self.component_trial_counter
            trial_result['global_trial_number'] = self.global_trial_counter
            
            self.component_trials.append(trial_result)
            
            self._save_global_trial_state()
            
            if self.wandb_enabled:
                WandBLogger.log_component_trial(component, self.component_trial_counter, 
                                            full_config, score, latency)
                WandBLogger.log_dynamic_component_table(component, self.component_trials, self.wandb_enabled)
            
            print(f"\n[Trial {self.global_trial_counter}] Score: {score:.4f} | Time: {latency:.2f}s")
            
            if self.use_multi_objective:
                return score, latency
            else:
                return score
                
        except Exception as e:
            print(f"\n[ERROR] Trial {self.global_trial_counter} failed: {e}")
            import traceback
            traceback.print_exc()
            
            if self.use_multi_objective:
                return 0.0, float('inf')
            else:
                return 0.0
    
    def _create_sampler(self):
        if self.optimizer == "tpe":
            return TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=self.seed,
                multivariate=True,
                constant_liar=True
            )
        elif self.optimizer == "botorch":
            import warnings
            warnings.filterwarnings('ignore', message='qExpectedHypervolumeImprovement has known numerical issues')
            
            return BoTorchSampler(
                n_startup_trials=10,
                seed=self.seed
            )
        elif self.optimizer == "random":
            return RandomSampler(seed=self.seed)
        else:
            return TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=self.seed,
                multivariate=True,
                constant_liar=True
            )
            
    def _save_final_results(self, results: Dict[str, Any]):
        Utils.save_component_optimization_results(
            self.result_dir, results, self.config_generator
        )
        
        if self.use_cache:
            cache_stats = self.cache_manager.get_cache_stats()
            results['cache_stats'] = cache_stats
            
            cache_report_file = os.path.join(self.result_dir, "cache_report.json")
            with open(cache_report_file, 'w') as f:
                json.dump({
                    'cache_stats': cache_stats,
                    'cached_configs_by_component': {
                        comp: len(self.cache_manager.get_cached_configs_for_component(comp))
                        for comp in results.get('component_order', [])
                    }
                }, f, indent=2)
            
            print(f"\n[Cache] Final cache statistics:")
            print(f"  Total hits: {cache_stats['hits']}")
            print(f"  Total misses: {cache_stats['misses']}")
            print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
            print(f"  Cache size: {cache_stats['cache_size_mb']:.2f} MB")
            print(f"  Total cached configs: {cache_stats['total_cached_configs']}")
    
    def _create_early_stopping_callback(self):
        class EarlyStoppingCallback:
            def __init__(self, threshold: float, is_multi_objective: bool):
                self.threshold = threshold
                self.should_stop = False
                self.is_multi_objective = is_multi_objective
                
            def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    if hasattr(trial, 'values') and trial.values:
                        if self.is_multi_objective and isinstance(trial.values, list):
                            score = trial.values[0]
                        elif isinstance(trial.values, list):
                            score = trial.values[0]
                        else:
                            score = trial.values
                    else:
                        score = trial.value if hasattr(trial, 'value') else 0
                    
                    if score >= self.threshold:
                        self.should_stop = True
                        study.stop()
        
        return EarlyStoppingCallback(self.early_stopping_threshold, self.use_multi_objective)