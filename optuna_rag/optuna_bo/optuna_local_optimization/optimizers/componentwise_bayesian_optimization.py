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
from pipeline.wandb_logger import WandBLogger


class ComponentwiseBayesianOptimization(BaseComponentwiseOptimizer):
    
    def optimize(self) -> Dict[str, Any]:
        start_time = time.time()
        
        self._validate_all_components()
        
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
        
        active_components = self._get_active_components()
        
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
            if early_stopped and component != 'prompt_maker_generator':
                print(f"\n[Component-wise] Skipping {component} optimization due to early stopping at {stopped_at_component}")
                if component == 'passage_filter':
                    self.best_configs[component] = {'passage_filter_method': 'pass_passage_filter'}
                elif component == 'passage_compressor':
                    self.best_configs[component] = {'passage_compressor_method': 'pass_compressor'}
                
                all_results['component_results'][component] = {
                    'component': component,
                    'best_config': self.best_configs[component],
                    'best_score': all_results['component_results'][stopped_at_component]['best_score'],
                    'n_trials': 0,
                    'optimization_time': 0.0,
                    'skipped': True,
                    'skip_reason': f'Early stopped at {stopped_at_component}'
                }
                
                all_results['best_configs'][component] = self.best_configs[component]
                all_results['component_order'].append(component)
                continue
            
            if component == 'passage_filter' and 'passage_reranker' in self.best_configs:
                reranker_config = self.best_configs['passage_reranker']
                if reranker_config.get('reranker_top_k') == 1:
                    print(f"\n[Component-wise] Skipping filter optimization because reranker_top_k=1")
                    self.best_configs[component] = {'passage_filter_method': 'pass_passage_filter'}
                    all_results['best_configs'][component] = self.best_configs[component]
                    all_results['component_order'].append(component)
                    
                    all_results['component_results'][component] = {
                        'component': component,
                        'best_config': self.best_configs[component],
                        'best_score': self.component_results.get('passage_reranker', {}).get('best_score', 0.0),
                        'n_trials': 0,
                        'optimization_time': 0.0,
                        'skipped': True,
                        'skip_reason': 'reranker_top_k=1'
                    }
                    continue
            
            self._setup_wandb_for_component(component, component_idx, active_components)
            
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
            
            self._save_intermediate_state()
            
            if component_result.get('best_score', 0) >= self.early_stopping_threshold and component != 'prompt_maker_generator':
                early_stopped = True
                stopped_at_component = component
                component_result['early_stopped'] = True
                print(f"\n[Component-wise] Early stopping triggered at {component} with score {component_result['best_score']:.4f}")
            
            self._log_wandb_component_summary(component, component_result)
        
        all_results['optimization_time'] = time.time() - start_time
        all_results['early_stopped'] = early_stopped
        all_results['stopped_at_component'] = stopped_at_component
        all_results['total_trials'] = sum(
            comp.get('n_trials', 0) for comp in all_results['component_results'].values()
        )
        all_results['study_name'] = self.study_name
        
        self._log_wandb_final_summary(all_results, early_stopped, stopped_at_component)
        
        self._save_final_results(all_results)
        self._print_final_summary(all_results)
        
        return all_results
    
    def _get_active_components(self) -> List[str]:
        active_components = []
        has_active_query_expansion = False
        
        for comp in self.COMPONENT_ORDER:
            if comp == 'query_expansion' and self.config_generator.node_exists(comp):
                qe_config = self.config_generator.extract_node_config("query_expansion")
                qe_methods = []
                for module in qe_config.get("modules", []):
                    method = module.get("module_type")
                    if method and method != "pass_query_expansion":
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
        
        return active_components
    
    def _optimize_component(self, component: str, component_idx: int, 
                          active_components: List[str]) -> Dict[str, Any]:
        component_start_time = time.time()
        component_dir = os.path.join(self.result_dir, f"{component_idx}_{component}")
        os.makedirs(component_dir, exist_ok=True)
        
        fixed_config = self._get_fixed_config(component, active_components)
        
        if self.use_wandb:
            WandBLogger.log_component_optimization_start(
                component, component_idx, len(active_components), fixed_config
            )
        
        search_space = self.search_space_builder.build_component_search_space(component, fixed_config)
        
        if not search_space:
            return {
                'component': component,
                'best_config': {},
                'best_score': 0.0,
                'n_trials': 0,
                'optimization_time': 0.0
            }
        
        result = self._run_bayesian_optimization(
            component, component_idx, component_dir,
            fixed_config, search_space, active_components
        )
        
        result['optimization_time'] = time.time() - component_start_time
        
        result_serializable = self._convert_numpy_types(result.copy())
        with open(os.path.join(component_dir, "optimization_result.json"), 'w') as f:
            json.dump(result_serializable, f, indent=2)
        
        return result
    
    def _run_bayesian_optimization(self, component: str, component_idx: int, 
                                  component_dir: str, fixed_config: Dict[str, Any],
                                  search_space: Dict[str, Any], 
                                  active_components: List[str]) -> Dict[str, Any]:
        
        storage_path = os.path.join(component_dir, f"{component}_optuna.db")
        storage_url = f"sqlite:///{storage_path}"
        study_exists = os.path.exists(storage_path)
        
        total_combinations = self._calculate_total_combinations(component, search_space)
        n_trials = self._calculate_component_trials(component, search_space)
        
        print(f"\n[{component}] Search space has {total_combinations} combinations")
        print(f"[{component}] Using {self.optimizer.upper()} sampler for {n_trials} trials")
        
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
        
        if study_exists and len(study.trials) > 0:
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            running_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
            
            print(f"\n[{component}] Resuming existing study:")
            print(f"  - Completed trials: {completed_trials}")
            print(f"  - Failed trials: {failed_trials}")
            print(f"  - Running/interrupted trials: {running_trials}")
            
            self.component_trial_counter = completed_trials
            
            trial_results_file = os.path.join(component_dir, "trial_results.json")
            if os.path.exists(trial_results_file):
                with open(trial_results_file, 'r') as f:
                    self.component_trials = json.load(f)
                print(f"  - Loaded {len(self.component_trials)} previous trial results")
            
            metrics_file = os.path.join(component_dir, "detailed_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.component_detailed_metrics[component] = json.load(f)
            
            n_trials = max(0, n_trials - completed_trials)
            if n_trials > 0:
                print(f"  - Remaining trials to run: {n_trials}")
        
        early_stopping_callback = self._create_early_stopping_callback()
        
        def objective(trial: optuna.Trial):
            result = self._component_objective(trial, component, fixed_config, search_space)
            
            component_dir_obj = os.path.join(self.result_dir, f"{component_idx}_{component}")
            trial_results_file = os.path.join(component_dir_obj, "trial_results.json")
            with open(trial_results_file, 'w') as f:
                json.dump(self._convert_numpy_types(self.component_trials), f, indent=2)
            
            metrics_file = os.path.join(component_dir_obj, "detailed_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self._convert_numpy_types(self.component_detailed_metrics.get(component, [])), f, indent=2)
            
            return result
        
        if n_trials > 0:
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
        else:
            print(f"[{component}] No trials to run - optimization already complete")
        
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
        
        if best_config and component == 'retrieval' and 'retrieval_config' in best_config:
            parsed_config = self.pipeline_runner._parse_retrieval_config(best_config['retrieval_config'])
            best_config.update(parsed_config)
            best_config.pop('retrieval_config', None)
        
        best_trial_data = self._find_best_trial(self.component_trials)
        
        if best_trial_data:
            best_score = best_trial_data['score']
            best_latency = best_trial_data.get('latency', 0.0)
            best_output_path = best_trial_data.get('output_parquet')
            final_best_config = best_trial_data.get('config', {})
        else:
            best_latency = 0.0
            best_output_path = None
            final_best_config = best_config if best_config else {}
        
        if best_output_path and os.path.exists(best_output_path):
            self.component_dataframes[component] = best_output_path
        
        if self.use_wandb:
            detailed_metrics_dict = {}
            if component in self.component_detailed_metrics:
                for i, metrics in enumerate(self.component_detailed_metrics[component]):
                    detailed_metrics_dict[i] = metrics
            
            WandBLogger.log_component_optimization_table(
                component, 
                self.component_trials,
                detailed_metrics_dict
            )
        
        if component == 'passage_compressor' and not final_best_config:
            for trial in self.component_trials:
                if trial.get('score', 0) == best_score:
                    trial_config = trial.get('config', {})
                    if trial_config:
                        final_best_config = trial_config
                        break
                    else:
                        full_config = trial.get('full_config', {})
                        if full_config.get('passage_compressor_method') == 'pass_compressor':
                            final_best_config = {'passage_compressor_method': 'pass_compressor'}
                            break
        
        total_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"\n[{component}] Optimization complete:")
        print(f"  - Total trials completed: {total_trials}")
        print(f"  - Best score: {best_score:.4f}")
        
        return {
            'component': component,
            'best_config': final_best_config,
            'best_score': best_score,
            'best_latency': best_latency,
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
        
        self._clean_config(full_config)
        
        self.component_trial_counter += 1
        self.global_trial_counter += 1
        self.current_trial = self.global_trial_counter
        
        trial_id = f"trial_{self.current_trial:04d}"
        trial_dir = os.path.join(self.result_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        
        self._save_global_trial_state()
        
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
                
                if component == 'prompt_maker_generator' and 'generated_texts' not in results:
                    if 'generated_texts' in working_df.columns:
                        results['generated_texts'] = working_df['generated_texts'].tolist()
                
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
            
            trial_result = self._create_trial_result(
                full_config, score, latency, 
                len(qa_subset) if 'qa_subset' in locals() else 0, 1.0,
                results, component, output_parquet_path
            )
            
            self.component_trials.append(trial_result)
            
            if self.wandb_enabled:
                WandBLogger.log_component_trial(component, self.component_trial_counter, 
                                            full_config, score, latency)
                WandBLogger.log_dynamic_component_table(component, self.component_trials, self.wandb_enabled)
            
            print(f"\n[Trial {self.current_trial}] Score: {score:.4f} | Time: {latency:.2f}s")
            
            if self.use_multi_objective:
                return score, latency
            else:
                return score
                
        except Exception as e:
            print(f"\n[ERROR] Trial {self.current_trial} failed: {e}")
            import traceback
            traceback.print_exc()
            
            if self.use_multi_objective:
                return 0.0, float('inf')
            else:
                return 0.0
    
    def _create_sampler(self):
        if self.optimizer == "tpe":
            return TPESampler(
                n_startup_trials=5,
                n_ei_candidates=24,
                seed=self.seed,
                multivariate=True,
                constant_liar=True,
                warn_independent_sampling=False
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
                n_startup_trials=5,
                n_ei_candidates=24,
                seed=self.seed,
                multivariate=True,
                constant_liar=True,
                warn_independent_sampling=False
            )
    
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
    
    def _setup_wandb_for_component(self, component: str, component_idx: int, 
                                  active_components: List[str]):
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
                        "n_trials": self.n_trials_per_component,
                        "optimizer": self.optimizer,
                        "use_multi_objective": self.use_multi_objective
                    },
                    reinit=True,
                    force=True
                )
                WandBLogger.reset_step_counter()
            except Exception as e:
                print(f"[WARNING] Failed to initialize W&B for {component}: {e}")
                print(f"[WARNING] Continuing without W&B logging for {component}")
                self.wandb_enabled = False
    
    def _log_wandb_component_summary(self, component: str, component_result: Dict[str, Any]):
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
            
            if not self.wandb_enabled:
                self.wandb_enabled = True
    
    def _log_wandb_final_summary(self, all_results: Dict[str, Any], 
                                early_stopped: bool, stopped_at_component: Optional[str]):
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
                        "optimization_mode": "componentwise_bayesian",
                        "total_components": len(all_results['component_order']),
                        "total_time": all_results['optimization_time'],
                        "early_stopped": early_stopped,
                        "stopped_at_component": stopped_at_component,
                        "optimizer": self.optimizer,
                        "use_multi_objective": self.use_multi_objective
                    },
                    reinit=True,
                    force=True
                )
                
                WandBLogger.log_final_componentwise_summary(all_results)
                wandb.finish()
            except Exception as e:
                print(f"[WARNING] Failed to log final W&B summary: {e}")