import os
import time
import yaml
import shutil
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from ConfigSpace import Configuration

from pipeline.logging.wandb import WandBLogger


class TrialManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def component_target_function(self, config: Configuration, seed: int, component: str, 
                 fixed_components: Dict[str, Any], budget: float = None) -> float:
        trial_config = dict(config)
        
        trial_config, is_pass_component = self.optimizer.rag_processor.parse_trial_config(component, trial_config)
        
        full_config = {**fixed_components, **trial_config}

        print(f"[DEBUG] Is pass component: {is_pass_component}")
        
        cleaned_config = self.optimizer.config_space_builder.clean_trial_config(full_config)
        
        print(f"[DEBUG] Cleaned config: {cleaned_config}")
        
        self.optimizer.current_trial += 1
        self.optimizer.component_trial_counter += 1
        
        trial_id = f"trial_{self.optimizer.current_trial:04d}"
        trial_dir = os.path.join(self.optimizer.result_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "data"), exist_ok=True)
        
        start_time = time.time()
        
        try:
            if budget:
                qa_subset = self.sample_data(int(budget), seed)
            else:
                qa_subset = self.optimizer.qa_data.copy()
            
            qa_subset = self.optimizer.rag_processor.load_previous_outputs(
                component, qa_subset, self.optimizer.component_dataframes, trial_dir
            )
                    
            self.copy_corpus_data(trial_dir)
            
            trial_config_yaml = self.optimizer.config_generator.generate_trial_config(cleaned_config)
            
            with open(os.path.join(trial_dir, "config.yaml"), 'w') as f:
                yaml.dump(trial_config_yaml, f)

            results = self.optimizer.rag_processor.run_pipeline(
                cleaned_config,
                trial_dir,
                qa_subset,
                component, 
                self.optimizer.component_results 
            )
            
            working_df = results.pop('working_df', qa_subset)

            detailed_metrics = self.optimizer.rag_processor.extract_detailed_metrics(component, results)

            if component not in self.optimizer.component_detailed_metrics:
                self.optimizer.component_detailed_metrics[component] = []
            
            self.optimizer.component_detailed_metrics[component].append(detailed_metrics)

            score = self.optimizer.rag_processor.calculate_component_score(
                component, results, is_pass_component, self.optimizer.component_results
            )
            
            end_time = time.time()
            latency = end_time - start_time

            output_parquet_path = self.optimizer.rag_processor.save_component_output(
                component, trial_dir, results, working_df
            )
            
            current_best_score = self.optimizer.component_results.get(component, {}).get('best_score', -float('inf'))

            if len(self.optimizer.component_trials) == 0:
                current_best_score = -float('inf')

            print(f"[{component}] Trial score: {score:.4f}, Current best: {current_best_score:.4f}")

            if score > current_best_score:
                print(f"[{component}] New best score: {score:.4f}, updating best output to: {output_parquet_path}")
                self.optimizer.component_dataframes[component] = output_parquet_path

                if component not in self.optimizer.component_results:
                    self.optimizer.component_results[component] = {}
                self.optimizer.component_results[component]['best_score'] = score
                self.optimizer.component_results[component]['best_output_path'] = output_parquet_path
                self.optimizer.component_results[component]['best_config'] = cleaned_config.copy()

            trial_result = self.create_trial_result(
                cleaned_config, score, latency, 
                int(budget) if budget else len(qa_subset),
                budget / self.optimizer.total_samples if budget else 1.0,
                results, component, output_parquet_path
            )

            for metric_key in ['retrieval_score', 'query_expansion_score', 'reranker_score', 
                            'filter_score', 'compressor_score', 'compression_score',
                            'prompt_maker_score', 'generation_score', 'last_retrieval_score']:
                if metric_key in results:
                    trial_result[metric_key] = results[metric_key]
            
            self.optimizer.component_trials.append(trial_result)
            
            if self.optimizer.wandb_enabled:
                WandBLogger.log_component_trial(component, self.optimizer.component_trial_counter, 
                                            cleaned_config, score, latency)
            
            self.optimizer.results_manager.save_trial_results(trial_dir, results, cleaned_config, component, latency)
            
            print(f"\n[Trial {self.optimizer.current_trial}] Score: {score:.4f} | Time: {latency:.2f}s")
            
            return -score
            
        except Exception as e:
            print(f"\n[ERROR] Trial {self.optimizer.current_trial} failed:")
            print(f"  Component: {component}")
            print(f"  Config: {trial_config}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.optimizer.results_manager.save_error_results(trial_dir, cleaned_config, component, e)

            empty_output_df = qa_subset.copy() if 'qa_subset' in locals() else self.optimizer.qa_data.head(0).copy()
            output_path = os.path.join(trial_dir, f"{component}_output.parquet")
            empty_output_df.to_parquet(output_path)
            
            return 0.0
    
    def find_best_trial(self, component: str) -> Optional[Dict[str, Any]]:
        best_trial = None
        best_score = -float('inf')
        best_latency = float('inf')

        score_groups = {}
        for trial in self.optimizer.component_trials:
            score = trial['score']
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(trial)

        if score_groups:
            max_score = max(score_groups.keys())
            trials_with_max_score = score_groups[max_score]

            if len(trials_with_max_score) > 1:
                print(f"[{component}] Found {len(trials_with_max_score)} trials with score {max_score:.4f}, selecting by latency")
                best_trial = min(trials_with_max_score, key=lambda t: t.get('latency', float('inf')))
                print(f"[{component}] Selected trial {best_trial['trial_number']} with latency {best_trial['latency']:.2f}s")
            else:
                best_trial = trials_with_max_score[0]
            
            best_score = best_trial['score']
            best_latency = best_trial.get('latency', 0.0)

        for trial in self.optimizer.component_trials:
            trial_config = trial.get('config', {})
            is_pass = self.is_pass_configuration(component, trial_config)

            if is_pass and trial['score'] == best_score and trial.get('latency', 0) < best_latency:
                print(f"[{component}] Preferring pass configuration with same score but lower latency")
                best_trial = trial
                best_latency = trial.get('latency', 0.0)

        return best_trial
    
    def is_pass_configuration(self, component: str, config: Dict[str, Any]) -> bool:
        if component == 'passage_filter' and config.get('passage_filter_method') == 'pass_passage_filter':
            return True
        elif component == 'passage_reranker' and config.get('passage_reranker_method') == 'pass_reranker':
            return True
        elif component == 'passage_compressor' and config.get('passage_compressor_method') == 'pass_compressor':
            return True
        elif component == 'query_expansion' and config.get('query_expansion_method') == 'pass_query_expansion':
            return True
        return False
    
    def create_trial_result(self, config_dict, score, latency, budget, budget_percentage, 
                           results, component, output_parquet_path):
        trial_result = {
            "trial_number": int(self.optimizer.component_trial_counter),
            "component": component,
            "config": self.convert_numpy_types(config_dict),
            "full_config": self.convert_numpy_types({**self.optimizer.current_fixed_config, **config_dict}),
            "score": float(score),
            "latency": float(latency),
            "budget": int(budget),
            "budget_percentage": float(budget_percentage),
            "results": results,
            "output_parquet": output_parquet_path,
            "timestamp": float(time.time())
        }
        
        for k, v in results.items():
            if k.endswith('_score') or k.endswith('_metrics'):
                trial_result[k] = self.convert_numpy_types(v)
        
        return trial_result
    
    def sample_data(self, budget: int, seed: int) -> pd.DataFrame:
        actual_samples = min(budget, self.optimizer.total_samples)
        if actual_samples < self.optimizer.total_samples:
            return self.optimizer.qa_data.sample(n=actual_samples, random_state=seed)
        return self.optimizer.qa_data
    
    def copy_corpus_data(self, trial_dir: str):
        centralized_corpus_path = os.path.join(self.optimizer.project_dir, "data", "corpus.parquet")
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
    
    def convert_numpy_types(self, obj):    
        if isinstance(obj, pd.DataFrame):
            return None  
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                converted = self.convert_numpy_types(v)
                if converted is not None: 
                    result[k] = converted
            return result
        elif isinstance(obj, list):
            result = []
            for item in obj:
                converted = self.convert_numpy_types(item)
                if converted is not None:  
                    result.append(converted)
            return result
        else:
            return obj