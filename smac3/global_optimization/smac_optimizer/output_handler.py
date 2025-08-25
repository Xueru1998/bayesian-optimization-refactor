from pipeline.utils import Utils


class OutputHandler:
    
    def _print_initialization_summary(self):
        summary = self._get_search_space_summary()
        
        print(f"\n===== {self.optimizer.upper()} {'Multi-Fidelity ' if self.use_multi_fidelity else ''}RAG Pipeline Optimizer =====")
        print(f"Using {self.n_trials} trials")
        print(f"Total search space combinations: {summary['search_space_size']:,}")
        print(f"Note: {summary.get('combination_note', '')}")
        print(f"Objectives: maximize score (weight={self.generation_weight}), minimize latency (weight={self.retrieval_weight})")
        
        if self.use_multi_fidelity:
            print(f"\nMulti-fidelity settings:")
            print(f"  Min budget: {self.min_budget} samples ({self.min_budget_percentage:.1%})")
            print(f"  Max budget: {self.max_budget} samples ({self.max_budget_percentage:.1%})")
            print(f"  Eta: {self.eta}")
        
        if not self.disable_early_stopping:
            print("\nEarly stopping enabled with thresholds:")
            for component, threshold in self.early_stopping_thresholds.items():
                print(f"  {component}: < {threshold}")
        else:
            print("\nEarly stopping: DISABLED")
        
        for component, info in summary.items():
            if component not in ["search_space_size", "combination_note"] and info['combinations'] > 1:
                print(f"\n{component.title().replace('_', ' ')}:")
                print(f"  Combinations: {info['combinations']:,}")
        
        print(f"\nCPUs per trial: {self.cpu_per_trial}")
        print(f"Using cached embeddings: {self.use_cached_embeddings}")
        if self.walltime_limit is not None:
            print(f"Wall time limit: {self.walltime_limit}s")
        else:
            print(f"Wall time limit: No limit")
        print(f"Number of workers: {self.n_workers}")
    
    def _print_optimization_summary(self, optimization_results, pareto_front):
        time_str = Utils.format_time_duration(optimization_results['optimization_time'])
        
        print(f"\n{'='*60}")
        print(f"{self.optimizer.upper()} Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total optimization time: {time_str}")
        print(f"Total trials: {self.trial_counter}")
        print(f"Early stopped trials: {len(self.early_stopped_trials)}")
        
        if self.early_stopped_trials:
            print("\nEarly stopping summary:")
            component_counts = {}
            for trial in self.early_stopped_trials:
                component = trial.get('stopped_at', 'unknown')
                component_counts[component] = component_counts.get(component, 0) + 1
            
            for component, count in component_counts.items():
                print(f"  {component}: {count} trials stopped")
        
        if self.use_multi_fidelity and self.all_trials:
            self._print_multifidelity_summary()
        
        if optimization_results['early_stopped']:
            print("⚡ Optimization stopped early due to achieving target score!")
        
        self._print_best_config_summary(optimization_results['best_config'])
        self._print_pareto_front_summary(pareto_front)
        
        print(f"\nResults saved to: {self.result_dir}")
    
    def _print_multifidelity_summary(self):
        full_budget_count = len([
            t for t in self.all_trials 
            if t.get('budget_percentage', 1.0) >= 0.99
        ])
        unique_configs = len(set(
            self._get_config_hash(t.get('config', {})) 
            for t in self.all_trials
        ))
        print(f"Unique configurations tested: {unique_configs}")
        print(f"Fully evaluated configurations: {full_budget_count}")
    
    def _print_best_config_summary(self, best_config):
        if best_config and isinstance(best_config, dict):
            print("\nBest balanced configuration (high score with low latency):")
            print(f"  Score: {best_config.get('score', 'N/A')}")
            print(f"  Latency: {best_config.get('latency', 'N/A')}")
            if 'budget' in best_config:
                print(f"  Budget: {best_config['budget']} samples")
            if 'config' in best_config:
                print(f"  Config: {best_config['config']}")
    
    def _print_pareto_front_summary(self, pareto_front):
        print(f"\nPareto front contains {len(pareto_front)} solutions")
        if pareto_front:
            print("Top 5 Pareto optimal solutions:")
            for i, solution in enumerate(sorted(pareto_front, key=lambda x: -x['score'])[:5]):
                budget_str = f", Budget: {solution.get('budget', 'N/A')}" if 'budget' in solution else ""
                print(f"  {i+1}. Score: {solution['score']:.4f}, Latency: {solution['latency']:.2f}s{budget_str}")