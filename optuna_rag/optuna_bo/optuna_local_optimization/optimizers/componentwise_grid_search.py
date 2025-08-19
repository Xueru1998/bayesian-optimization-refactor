import os
import time
import json
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional

from ..base.base_componentwise_optimizer import BaseComponentwiseOptimizer
from pipeline.wandb_logger import WandBLogger


class ComponentwiseGridSearch(BaseComponentwiseOptimizer):
    
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
        
        for component_idx, component in enumerate(active_components):
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
            
            self._log_wandb_component_summary(component, component_result)
        
        all_results['optimization_time'] = time.time() - start_time
        all_results['total_trials'] = sum(
            comp.get('n_trials', 0) for comp in all_results['component_results'].values()
        )
        all_results['study_name'] = self.study_name
        
        self._log_wandb_final_summary(all_results)
        
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
        
        result = self._run_grid_search(
            component, component_idx, component_dir, 
            fixed_config, search_space, active_components
        )
        
        result['optimization_time'] = time.time() - component_start_time
        
        result_serializable = self._convert_numpy_types(result.copy())
        with open(os.path.join(component_dir, "optimization_result.json"), 'w') as f:
            json.dump(result_serializable, f, indent=2)
        
        return result
    
    def _run_grid_search(self, component: str, component_idx: int, 
                        component_dir: str, fixed_config: Dict[str, Any],
                        search_space: Dict[str, Any], 
                        active_components: List[str]) -> Dict[str, Any]:
        
        total_combinations = self.grid_search_helper.calculate_grid_search_combinations(
            component, search_space, fixed_config
        )
        
        self.grid_search_helper.print_grid_search_info(
            component, search_space, total_combinations, fixed_config
        )
        
        valid_combinations = self.grid_search_helper.get_valid_combinations(
            component, search_space, fixed_config
        )
        
        if len(valid_combinations) != total_combinations:
            print(f"[WARNING] Mismatch: expected {total_combinations}, got {len(valid_combinations)}")
        
        self.grid_search_helper.save_grid_search_sequence(
            component, valid_combinations, component_dir
        )
        
        print(f"[{component}] Using GRID SEARCH with {len(valid_combinations)} combinations")
        
        self.current_component = component
        self.current_fixed_config = fixed_config
        self.component_trial_counter = 0
        self.component_trials = []
        self.component_detailed_metrics[component] = []
        
        grid_state_file = os.path.join(component_dir, "grid_search_state.json")
        completed_configs = []
        last_incomplete_trial = None
        
        if os.path.exists(grid_state_file):
            completed_configs, last_incomplete_trial = self._load_grid_state(
                grid_state_file, component_dir, component
            )
        
        remaining_combinations = self._get_remaining_combinations(
            valid_combinations, completed_configs, last_incomplete_trial
        )
        
        print(f"[{component}] {len(remaining_combinations)} combinations remaining")
        
        if len(remaining_combinations) > 0:
            for i, trial_config in enumerate(remaining_combinations[:3]):
                print(f"  Next trial {i+1}: {trial_config}")
            if len(remaining_combinations) > 3:
                print(f"  ... and {len(remaining_combinations) - 3} more trials")
        
        for trial_idx, trial_config in enumerate(remaining_combinations):
            trial_completed = self._run_single_grid_trial(
                component, trial_config, fixed_config, valid_combinations,
                component_dir, grid_state_file, completed_configs,
                trial_idx, last_incomplete_trial
            )
            
            if trial_completed:
                last_incomplete_trial = None
        
        return self._finalize_grid_search(
            component, component_dir, fixed_config,
            search_space, total_combinations
        )
    
    def _load_grid_state(self, grid_state_file: str, component_dir: str, 
                         component: str) -> tuple:
        with open(grid_state_file, 'r') as f:
            grid_state = json.load(f)
            completed_configs = grid_state.get('completed_configs', [])
            self.component_trial_counter = grid_state.get('last_trial_number', 0)
            last_incomplete_trial = grid_state.get('last_incomplete_trial', None)
        
        print(f"[{component}] Resuming: found {len(completed_configs)} completed trials")
        
        trial_results_file = os.path.join(component_dir, "trial_results.json")
        if os.path.exists(trial_results_file):
            with open(trial_results_file, 'r') as f:
                self.component_trials = json.load(f)
        
        metrics_file = os.path.join(component_dir, "detailed_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.component_detailed_metrics[component] = json.load(f)
        
        if last_incomplete_trial:
            last_trial_dir = os.path.join(self.result_dir, f"trial_{self.global_trial_counter:04d}")
            last_trial_output = os.path.join(last_trial_dir, f"{component}_output.parquet")
            
            if not os.path.exists(last_trial_output):
                print(f"[{component}] Last trial {self.component_trial_counter} (global {self.global_trial_counter}) was incomplete")
                print(f"[{component}] Will retry this configuration")
                
                if self.component_trial_counter > 0:
                    self.component_trial_counter -= 1
                
                if last_incomplete_trial in completed_configs:
                    completed_configs.remove(last_incomplete_trial)
            else:
                print(f"[{component}] Last trial {self.component_trial_counter} was completed")
                last_incomplete_trial = None
        
        print(f"[{component}] Component trial will continue from: {self.component_trial_counter + 1}")
        print(f"[{component}] Global trial counter is currently: {self.global_trial_counter}")
        
        return completed_configs, last_incomplete_trial
    
    def _get_remaining_combinations(self, valid_combinations: List[Dict],
                                   completed_configs: List[Dict],
                                   last_incomplete_trial: Optional[Dict]) -> List[Dict]:
        remaining_combinations = []
        
        if last_incomplete_trial and isinstance(last_incomplete_trial, dict):
            remaining_combinations.append(last_incomplete_trial)
            print(f"[{self.current_component}] Retrying incomplete configuration first")
        
        for combo in valid_combinations:
            combo_str = json.dumps(combo, sort_keys=True)
            if combo_str not in [json.dumps(c, sort_keys=True) for c in completed_configs]:
                if not last_incomplete_trial or json.dumps(last_incomplete_trial, sort_keys=True) != combo_str:
                    remaining_combinations.append(combo)
        
        return remaining_combinations
    
    def _run_single_grid_trial(self, component: str, trial_config: Dict,
                              fixed_config: Dict, valid_combinations: List,
                              component_dir: str, grid_state_file: str,
                              completed_configs: List, trial_idx: int,
                              last_incomplete_trial: Optional[Dict]) -> bool:
        self.component_trial_counter += 1
        
        if trial_idx == 0 and last_incomplete_trial:
            print(f"\n[{component}] Retrying incomplete trial {self.component_trial_counter}/{len(valid_combinations)} " +
                f"(Global trial: {self.global_trial_counter})")
        else:
            self.global_trial_counter += 1
            self.current_trial = self.global_trial_counter
            print(f"\n[{component}] Running trial {self.component_trial_counter}/{len(valid_combinations)} " +
                f"(Global trial: {self.global_trial_counter})")
        
        full_config = trial_config.copy()
        full_config.update(fixed_config)
        
        self._clean_config(full_config)
        
        trial_id = f"trial_{self.current_trial:04d}"
        trial_dir = os.path.join(self.result_dir, trial_id)
        
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        
        grid_state_temp = {
            'completed_configs': completed_configs.copy(),
            'last_trial_number': self.component_trial_counter,
            'last_global_trial': self.global_trial_counter,
            'last_incomplete_trial': trial_config,
            'total_combinations': len(valid_combinations),
            'completed_count': len(completed_configs)
        }
        with open(grid_state_file, 'w') as f:
            json.dump(grid_state_temp, f, indent=2)
        
        self._save_global_trial_state()
        
        start_time = time.time()
        trial_completed = False
        
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
            
            end_time = time.time()
            latency = end_time - start_time
            
            detailed_metrics = self.pipeline_manager.extract_detailed_metrics(component, results)
            if component not in self.component_detailed_metrics:
                self.component_detailed_metrics[component] = []
            self.component_detailed_metrics[component].append(detailed_metrics)
            
            print(f"\n[Trial {self.current_trial}] Score: {score:.4f} | Time: {latency:.2f}s")
            trial_completed = True
            
        except Exception as e:
            print(f"\n[ERROR] Trial {self.current_trial} failed: {e}")
            import traceback
            traceback.print_exc()
            score = 0.0
            latency = float('inf')
            results = {}
            output_parquet_path = None
        
        if trial_completed or trial_idx == len(valid_combinations) - 1:
            trial_result = self._create_trial_result(
                {**trial_config, **fixed_config}, score, latency,
                len(self.qa_data), 1.0, results, component, output_parquet_path
            )
            
            trial_result['component_trial_number'] = self.component_trial_counter
            trial_result['global_trial_number'] = self.global_trial_counter
            
            self.component_trials.append(trial_result)
            
            if score > self.component_results.get(component, {}).get('best_score', 0.0):
                self.component_dataframes[component] = output_parquet_path
            
            completed_configs.append(trial_config)
            
            grid_state = {
                'completed_configs': completed_configs,
                'last_trial_number': self.component_trial_counter,
                'last_global_trial': self.global_trial_counter,
                'last_incomplete_trial': None,
                'total_combinations': len(valid_combinations),
                'completed_count': len(completed_configs)
            }
            with open(grid_state_file, 'w') as f:
                json.dump(grid_state, f, indent=2)
            
            with open(os.path.join(component_dir, "trial_results.json"), 'w') as f:
                json.dump(self._convert_numpy_types(self.component_trials), f, indent=2)
            
            with open(os.path.join(component_dir, "detailed_metrics.json"), 'w') as f:
                json.dump(self._convert_numpy_types(self.component_detailed_metrics.get(component, [])), f, indent=2)
            
            if self.wandb_enabled:
                WandBLogger.log_component_trial(component, self.component_trial_counter,
                                            {**trial_config, **fixed_config}, score, latency)
                WandBLogger.log_dynamic_component_table(component, self.component_trials, self.wandb_enabled)
        
        return trial_completed
    
    def _finalize_grid_search(self, component: str, component_dir: str,
                             fixed_config: Dict, search_space: Dict,
                             total_combinations: int) -> Dict[str, Any]:
        best_trial_data = self._find_best_trial(self.component_trials)
        
        if best_trial_data:
            best_score = best_trial_data['score']
            best_latency = best_trial_data.get('latency', 0.0)
            best_output_path = best_trial_data.get('output_parquet')
            final_best_config = best_trial_data.get('config', {})
            
            if best_output_path and os.path.exists(best_output_path):
                self.component_dataframes[component] = best_output_path
                print(f"[{component}] Best configuration found with score {best_score:.4f} and latency {best_latency:.2f}s")
                print(f"[{component}] Best output saved at: {best_output_path}")
                print(f"[{component}] Best config: {final_best_config}")
        else:
            best_score = 0.0
            best_latency = 0.0
            best_output_path = None
            final_best_config = {}
        
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
        
        print(f"\n[{component}] Grid search complete:")
        print(f"  - Total trials completed: {len(self.component_trials)}")
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
            'optimization_method': 'grid_search',
            'detailed_metrics': self.component_detailed_metrics.get(component, []),
            'best_output_path': best_output_path
        }
    
    def _setup_wandb_for_component(self, component: str, component_idx: int, 
                                  active_components: List[str]):
        if self.use_wandb:
            if hasattr(self, 'wandb') and self.wandb is not None:
                import wandb
                if wandb.run is not None:
                    try:
                        wandb.finish()
                    except Exception as e:
                        print(f"[WARNING] Error finishing previous W&B run: {e}")
                    time.sleep(2)
            
            wandb_run_name = f"{self.study_name}_{component}"
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=wandb_run_name,
                    config={
                        "component": component,
                        "stage": f"{component_idx + 1}/{len(active_components)}",
                        "optimizer": "grid"
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
        if self.use_wandb:
            import wandb
            if wandb.run is not None:
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
    
    def _log_wandb_final_summary(self, all_results: Dict[str, Any]):
        if self.use_wandb:
            import wandb
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
                        "optimization_mode": "componentwise_grid",
                        "total_components": len(all_results['component_order']),
                        "total_time": all_results['optimization_time'],
                        "optimizer": "grid"
                    },
                    reinit=True,
                    force=True
                )
                
                WandBLogger.log_final_componentwise_summary(all_results)
                wandb.finish()
            except Exception as e:
                print(f"[WARNING] Failed to log final W&B summary: {e}")