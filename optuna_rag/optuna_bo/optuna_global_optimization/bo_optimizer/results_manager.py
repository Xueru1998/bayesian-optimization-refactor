import optuna
from typing import Dict, Any, Optional
from pipeline.utils import Utils
from optuna_rag.optuna_bo.optuna_global_optimization.plot_generator import PlotGenerator


class ResultsManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def save_study_results(self, study: optuna.Study):
        df = study.trials_dataframe()
        Utils.save_results_to_csv(self.optimizer.result_dir, "optuna_trials.csv", df)
        
        all_metrics = []
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
                
                all_metrics.append(metrics)
        
        Utils.save_results_to_csv(self.optimizer.result_dir, "all_metrics.csv", all_metrics)
        
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
            
            Utils.save_results_to_json(self.optimizer.result_dir, "best_params.json", best_metrics)
            
            print(f"\nBest trials found: {len(best_trials)}")
            for i, trial in enumerate(best_trials[:5]):
                print(f"\nBest trial #{i+1}: Trial {trial.number}")
                print(f"Score: {-trial.values[0]:.4f}, Latency: {trial.values[1]:.2f}s")
                print(f"Params: {trial.params}")

                gen_score = trial.user_attrs.get('generator_score', 
                                               trial.user_attrs.get('generation_score', 0.0))
                if gen_score > 0:
                    print(f"Generator Score: {gen_score:.4f}")
    
    def generate_plots(self, study):
        try:
            plot_generator = PlotGenerator(self.optimizer.result_dir)
            plot_generator.generate_all_plots(study, self.optimizer.all_trials)
        except ImportError:
            print("Matplotlib or Optuna visualization not available. Skipping plots.")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def find_best_config(self, study: optuna.Study) -> Optional[Dict[str, Any]]:
        high_score_trials = []

        for trial in self.optimizer.all_trials:
            if trial['score'] > 0.9:
                high_score_trials.append(trial)
        
        if high_score_trials:
            best_trial = min(high_score_trials, key=lambda x: x['latency'])
            return best_trial

        pareto_front = Utils.find_pareto_front(self.optimizer.all_trials)
        if pareto_front:
            for trial in sorted(pareto_front, key=lambda x: -x['score']):
                if trial['score'] > 0.8:
                    return trial
            return max(pareto_front, key=lambda x: x['score'])
        
        return None