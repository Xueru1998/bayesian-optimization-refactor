import os
import time
import json
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional

from ..base.base_componentwise_optimizer import BaseComponentwiseOptimizer
from ..helpers.component_grid_search_helper import ComponentGridSearchHelper
from pipeline.utils import Utils
from pipeline.logging.wandb import WandBLogger


class ComponentwiseGridSearch(BaseComponentwiseOptimizer):
    """Grid Search specific implementation for componentwise optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_helper = ComponentGridSearchHelper()
    
    def _optimize_component(self, component: str, component_idx: int, 
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
    
        total_combinations, note = self.combination_calculator.calculate_component_combinations(
            component, search_space, fixed_config, self.best_configs
        )
        
        print(f"[{component}] Using GRID SEARCH for {total_combinations} combinations")
        self.combination_calculator.print_combination_info(
            component, total_combinations, note, search_space
        )
        
        valid_combinations = self.grid_helper.get_valid_combinations(component, search_space, fixed_config)
        
        if len(valid_combinations) != total_combinations:
            print(f"[WARNING] Mismatch: expected {total_combinations}, got {len(valid_combinations)}")
        
        self.current_component = component
        self.current_fixed_config = fixed_config
        self.component_trial_counter = 0
        self.component_trials = []
        self.component_detailed_metrics[component] = []
        
        grid_state_file = os.path.join(component_dir, "grid_search_state.json")
        completed_configs = []
        
        if os.path.exists(grid_state_file):
            completed_configs = self._load_grid_state(grid_state_file, component_dir)
        
        remaining_combinations = self._get_remaining_combinations(
            valid_combinations, completed_configs, component
        )
        
        print(f"[{component}] {len(remaining_combinations)} combinations remaining")
        
        if len(remaining_combinations) == 0:
            return self._handle_completed_grid_search(component, fixed_config, total_combinations)
        
        self._print_next_trials(remaining_combinations)
        
        for trial_idx, trial_config in enumerate(remaining_combinations):
            result = self._run_single_grid_trial(
                component, trial_config, fixed_config, 
                valid_combinations, component_dir, grid_state_file,
                completed_configs
            )
            
            if result is None:
                continue
        
        return self._finalize_grid_search_results(
            component, fixed_config, total_combinations
        )
    
    def _load_grid_state(self, grid_state_file: str, component_dir: str) -> List[Dict]:
        with open(grid_state_file, 'r') as f:
            grid_state = json.load(f)
            completed_configs = grid_state.get('completed_configs', [])
            self.component_trial_counter = len(completed_configs)
            
        print(f"[Grid Search] Resuming: found {len(completed_configs)} completed trials")
        print(f"[Grid Search] Component trial will continue from: {self.component_trial_counter + 1}")
        print(f"[Grid Search] Global trial counter is currently: {self.global_trial_counter}")
        
        trial_results_file = os.path.join(component_dir, "trial_results.json")
        if os.path.exists(trial_results_file):
            with open(trial_results_file, 'r') as f:
                self.component_trials = json.load(f)
        
        metrics_file = os.path.join(component_dir, "detailed_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.component_detailed_metrics[self.current_component] = json.load(f)
        
        return completed_configs
    
    def _normalize_config_for_comparison(self, config: Dict, component: str) -> Dict:
        normalized = config.copy()
        
        if component == 'passage_compressor':
            if 'passage_compressor_method' in config and 'passage_compressor_config' not in config:
                method = config.get('passage_compressor_method')
                if method == 'pass_compressor':
                    normalized = {'passage_compressor_config': 'pass_compressor'}
                elif method in ['tree_summarize', 'refine']:
                    gen_type = config.get('compressor_generator_module_type', '')
                    model = config.get('compressor_model', '')
                    if gen_type and model:
                        normalized = {'passage_compressor_config': f"{method}::{gen_type}::{model}"}
                elif method == 'lexrank':
                    normalized = {'passage_compressor_config': 'lexrank'}
                elif method == 'spacy':
                    spacy_model = config.get('spacy_model', '')
                    if spacy_model:
                        normalized = {'passage_compressor_config': f"spacy::{spacy_model}"}
                    else:
                        normalized = {'passage_compressor_config': 'spacy'}
        
        elif component == 'query_expansion':
            if 'query_expansion_method' in config and 'query_expansion_config' not in config:
                method = config.get('query_expansion_method')
                if method == 'pass_query_expansion':
                    normalized = {'query_expansion_config': 'pass_query_expansion'}
                else:
                    gen_type = config.get('query_expansion_generator_module_type', '')
                    model = config.get('query_expansion_model', '')
                    if gen_type and model:
                        normalized = {'query_expansion_config': f"{method}::{gen_type}::{model}"}
        
        elif component == 'prompt_maker_generator':
            if 'generator_module_type' in config and 'generator_config' not in config:
                gen_type = config.get('generator_module_type', '')
                model = config.get('generator_model', '')
                if gen_type and model:
                    normalized['generator_config'] = f"{gen_type}::{model}"
        
        return normalized
    
    def _get_remaining_combinations(self, valid_combinations: List[Dict], 
                                   completed_configs: List[Dict], 
                                   component: str) -> List[Dict]:
        remaining_combinations = []
        
        normalized_completed = set()
        for config in completed_configs:
            normalized = self._normalize_config_for_comparison(config, component)
            normalized_completed.add(json.dumps(normalized, sort_keys=True))
        
        for combo in valid_combinations:
            normalized_combo = self._normalize_config_for_comparison(combo, component)
            combo_str = json.dumps(normalized_combo, sort_keys=True)
            
            if combo_str not in normalized_completed:
                remaining_combinations.append(combo)
        
        return remaining_combinations
    
    def _print_next_trials(self, remaining_combinations: List[Dict]):
        if len(remaining_combinations) > 0:
            for i, trial_config in enumerate(remaining_combinations[:3]):
                print(f"  Next trial {i+1}: {trial_config}")
            if len(remaining_combinations) > 3:
                print(f"  ... and {len(remaining_combinations) - 3} more trials")
    
    def _run_single_grid_trial(self, component: str, trial_config: Dict, 
                              fixed_config: Dict, valid_combinations: List,
                              component_dir: str, grid_state_file: str,
                              completed_configs: List) -> Optional[Dict]:
        self.component_trial_counter += 1
        self.global_trial_counter += 1
        self.current_trial = self.global_trial_counter
        
        print(f"\n[{component}] Running trial {self.component_trial_counter}/{len(valid_combinations)} " +
              f"(Global trial: {self.global_trial_counter})")
        
        full_config = trial_config.copy()
        full_config.update(fixed_config)
        
        trial_config, full_config = self._parse_composite_configs(component, trial_config, full_config)
        
        self._clean_config(full_config)
        
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
            self.component_detailed_metrics[component].append(detailed_metrics)
            
            trial_result = self._create_trial_result(
                full_config, score, latency, 
                len(qa_subset) if 'qa_subset' in locals() else 0, 1.0,
                results, component, output_parquet_path
            )
            
            trial_result['component_trial_number'] = self.component_trial_counter
            trial_result['global_trial_number'] = self.global_trial_counter
            
            self.component_trials.append(trial_result)
            
            if self.wandb_enabled:
                WandBLogger.log_component_trial(component, self.component_trial_counter, 
                                              full_config, score, latency)
                WandBLogger.log_dynamic_component_table(component, self.component_trials, self.wandb_enabled)
            
            print(f"[Trial {self.global_trial_counter}] Score: {score:.4f} | Time: {latency:.2f}s")
            
            completed_configs.append(trial_config)
            self._save_grid_state(grid_state_file, component_dir, completed_configs, valid_combinations)
            
            self._save_global_trial_state()
            
            return trial_result
                
        except Exception as e:
            print(f"\n[ERROR] Trial {self.global_trial_counter} failed: {e}")
            import traceback
            traceback.print_exc()
            
            trial_result = self._create_trial_result(
                full_config, 0.0, float('inf'), 
                0, 1.0, {}, component, None
            )
            trial_result['status'] = 'FAILED'
            trial_result['error'] = str(e)
            self.component_trials.append(trial_result)
            
            return None
    
    def _save_grid_state(self, grid_state_file: str, component_dir: str, 
                         completed_configs: List, valid_combinations: List):
        grid_state = {
            'completed_configs': completed_configs,
            'last_trial_number': self.component_trial_counter,
            'last_global_trial': self.global_trial_counter,
            'total_combinations': len(valid_combinations),
            'completed_count': len(completed_configs)
        }
        with open(grid_state_file, 'w') as f:
            json.dump(grid_state, f, indent=2)
        
        with open(os.path.join(component_dir, "trial_results.json"), 'w') as f:
            json.dump(self._convert_numpy_types(self.component_trials), f, indent=2)
        
        with open(os.path.join(component_dir, "detailed_metrics.json"), 'w') as f:
            json.dump(self._convert_numpy_types(self.component_detailed_metrics.get(self.current_component, [])), f, indent=2)
    
    def _handle_completed_grid_search(self, component: str, fixed_config: Dict, 
                                 total_combinations: int) -> Dict[str, Any]:
        print(f"[{component}] All combinations already completed")
        
        best_trial_data = self._find_best_trial(self.component_trials)
        
        if best_trial_data:
            best_config = best_trial_data.get('config', {})
            best_output_path = best_trial_data.get('output_parquet')
            
            if best_output_path and os.path.exists(best_output_path):
                self.component_dataframes[component] = best_output_path
                print(f"[{component}] Best configuration found with score {best_trial_data['score']:.4f}")
                print(f"[{component}] Best output saved at: {best_output_path}")
                print(f"[{component}] Best config: {best_config}")
            
            final_best_config = self._parse_best_config_composite(component, best_config)
            
            self.best_configs[component] = final_best_config
            self.component_results[component] = {
                'best_score': best_trial_data.get('score', 0.0),
                'best_config': final_best_config
            }
            
            return {
                'component': component,
                'best_config': final_best_config,
                'best_score': best_trial_data.get('score', 0.0),
                'best_latency': best_trial_data.get('latency', 0.0),
                'best_trial': best_trial_data,
                'all_trials': self.component_trials,
                'n_trials': len(self.component_trials),
                'fixed_config': fixed_config,
                'search_space_size': len(self.search_space_builder.build_component_search_space(component, fixed_config)),
                'total_combinations': total_combinations,
                'optimization_method': 'grid',
                'detailed_metrics': self.component_detailed_metrics.get(component, []),
                'best_output_path': best_output_path
            }
        
        return {
            'component': component,
            'best_config': {},
            'best_score': 0.0,
            'n_trials': 0,
            'optimization_time': 0.0
        }

    
    def _finalize_grid_search_results(self, component: str, fixed_config: Dict, 
                                 total_combinations: int) -> Dict[str, Any]:
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
            
            if best_trial_data and best_trial_data['config']:
                print(f"[{component}] Best config: {best_trial_data['config']}")
        
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
        
        final_best_config = {}
        if best_trial_data and 'config' in best_trial_data:
            final_best_config = best_trial_data['config']
        
        final_best_config = self._parse_best_config_composite(component, final_best_config)
        
        self.best_configs[component] = final_best_config
        self.component_results[component] = {
            'best_score': best_score,
            'best_config': final_best_config
        }
        
        print(f"\n[{component}] Grid search complete:")
        print(f"  - Total trials completed: {len(self.component_trials)}")
        print(f"  - Best score: {best_score:.4f}")
        
        return {
            'component': component,
            'best_config': final_best_config,
            'best_score': best_trial_data['score'] if best_trial_data else 0.0,
            'best_latency': best_trial_data.get('latency', 0.0) if best_trial_data else 0.0,
            'best_trial': best_trial_data,
            'all_trials': self.component_trials,
            'n_trials': len(self.component_trials),
            'fixed_config': fixed_config,
            'search_space_size': len(self.search_space_builder.build_component_search_space(component, fixed_config)),
            'total_combinations': total_combinations,
            'optimization_method': 'grid',
            'detailed_metrics': self.component_detailed_metrics.get(component, []),
            'best_output_path': best_output_path
        }