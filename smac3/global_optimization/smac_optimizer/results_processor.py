import json
import hashlib
import numpy as np
from typing import Dict, Any, List
import wandb
from pipeline.utils import Utils
from pipeline.logging.wandb import WandBLogger


class ResultsProcessor:
    
    def _extract_pareto_front(self, smac, incumbents):
        pareto_front = []
        for incumbent in incumbents:
            try:
                cost = self._get_incumbent_cost(smac, incumbent)
                if cost is None:
                    continue
                
                score_val, latency_val = self._extract_cost_values(cost)
                pareto_solution = self._create_pareto_solution(incumbent, score_val, latency_val)
                pareto_front.append(pareto_solution)
                
            except Exception as e:
                print(f"Error processing incumbent: {e}")
                continue
        
        self._update_pareto_front_trial_numbers(pareto_front)
        return pareto_front
    
    def _get_incumbent_cost(self, smac, incumbent):
        if hasattr(smac, 'validate'):
            return smac.validate(incumbent)
        elif hasattr(smac, 'runhistory'):
            try:
                incumbent_cost = smac.runhistory.get_cost(incumbent)
                if isinstance(incumbent_cost, list):
                    return {"score": incumbent_cost[0], "latency": incumbent_cost[1]}
                else:
                    return {"score": incumbent_cost, "latency": 0.0}
            except:
                return None
        return None
    
    def _extract_cost_values(self, cost):
        if isinstance(cost, dict):
            score_val = cost.get("score", 0.0)
            latency_val = cost.get("latency", 0.0)
        elif isinstance(cost, (list, np.ndarray)):
            if len(cost) >= 2:
                score_val = float(cost[0])
                latency_val = float(cost[1])
            else:
                score_val = float(cost[0]) if len(cost) > 0 else 0.0
                latency_val = 0.0
        else:
            score_val = float(cost)
            latency_val = 0.0
        
        if hasattr(score_val, '__len__') and not isinstance(score_val, str):
            score_val = float(score_val[0] if len(score_val) > 0 else 0.0)
        if hasattr(latency_val, '__len__') and not isinstance(latency_val, str):
            latency_val = float(latency_val[0] if len(latency_val) > 0 else 0.0)
        
        return score_val, latency_val
    
    def _create_pareto_solution(self, incumbent, score_val, latency_val):
        config_dict = dict(incumbent)
        pareto_solution = {
            'config': config_dict,
            'score': -float(score_val) if score_val else 0.0,
            'latency': float(latency_val) if latency_val else 0.0,
            'trial_number': None
        }
        
        for trial in self.all_trials:
            if trial['config'] == config_dict:
                pareto_solution.update({
                    'score': trial['score'],
                    'latency': trial['latency'],
                    'trial_number': trial['trial_number'],
                    'budget': trial.get('budget', self.max_budget),
                    'budget_percentage': trial.get('budget_percentage', 1.0)
                })
                break
        
        return pareto_solution
    
    def _update_pareto_front_trial_numbers(self, pareto_front):
        for pf_solution in pareto_front:
            for trial in self.all_trials:
                if trial['config'] == pf_solution['config']:
                    pf_solution['trial_number'] = trial['trial_number']
                    break
    
    def _create_optimization_results(self, pareto_front, early_stopped, incumbents, total_time):
        best_configs = self._find_best_configurations(pareto_front)
        
        results = {
            'optimizer': self.optimizer,
            'use_multi_fidelity': bool(self.use_multi_fidelity),
            'min_budget': int(self.min_budget),
            'max_budget': int(self.max_budget),
            'best_config': self._convert_numpy_types(best_configs['best_balanced']),
            'best_score_config': self._convert_numpy_types(best_configs['best_score']['config']),
            'best_score': float(best_configs['best_score']['score']),
            'best_score_latency': float(best_configs['best_score']['latency']),
            'best_latency_config': self._convert_numpy_types(best_configs['best_latency']['config']),
            'best_latency': float(best_configs['best_latency']['latency']),
            'best_latency_score': float(best_configs['best_latency']['score']),
            'pareto_front': [self._convert_numpy_types(pf) for pf in pareto_front],
            'optimization_time': float(total_time),
            'n_trials': int(self.trial_counter),
            'total_trials': int(self.trial_counter),
            'early_stopped': bool(early_stopped),
            'incumbents': [self._convert_numpy_types(dict(inc)) for inc in incumbents],
            'all_trials': [self._convert_numpy_types(trial) for trial in self.all_trials]
        }
        
        return results
    
    def _find_best_configurations(self, pareto_front):
        if not self.all_trials:
            default_trial = {'config': {}, 'score': 0.0, 'latency': float('inf')}
            return {
                'best_score': default_trial,
                'best_latency': default_trial,
                'best_balanced': default_trial
            }
        
        if self.use_multi_fidelity:
            return self._find_best_multifidelity_configs(pareto_front)
        else:
            return self._find_best_standard_configs(pareto_front)
    
    def _find_best_multifidelity_configs(self, pareto_front):
        full_budget_trials = [
            t for t in self.all_trials 
            if t.get('budget_percentage', 1.0) >= 0.99
        ]
        
        if full_budget_trials:
            best_score_trial = max(full_budget_trials, key=lambda x: x['score'])
            best_latency_trial = min(full_budget_trials, key=lambda x: x['latency'])
            
            high_score_trials = [t for t in full_budget_trials if t['score'] > 0.9]
            if high_score_trials:
                best_balanced = min(high_score_trials, key=lambda x: x['latency'])
            else:
                best_balanced = max(full_budget_trials, key=lambda x: x['score'])
        else:
            best_score_trial = max(self.all_trials, key=lambda x: x['score'])
            best_latency_trial = min(self.all_trials, key=lambda x: x['latency'])
            best_balanced = best_score_trial
        
        return {
            'best_score': best_score_trial,
            'best_latency': best_latency_trial,
            'best_balanced': best_balanced
        }
    
    def _find_best_standard_configs(self, pareto_front):
        best_score_trial = max(self.all_trials, key=lambda x: x['score'])
        best_latency_trial = min(self.all_trials, key=lambda x: x['latency'])
        
        high_score_trials = [t for t in self.all_trials if t['score'] > 0.9]
        if high_score_trials:
            best_balanced = min(high_score_trials, key=lambda x: x['latency'])
        else:
            best_balanced = max(pareto_front, key=lambda x: x['score']) if pareto_front else best_score_trial
        
        return {
            'best_score': best_score_trial,
            'best_latency': best_latency_trial,
            'best_balanced': best_balanced
        }
    
    def _save_and_log_results(self, optimization_results, pareto_front):
        Utils.save_results_to_json(self.result_dir, "optimization_summary.json", optimization_results)
        
        if self.all_trials:
            converted_trials = self._convert_trials_for_csv(self.all_trials)
            Utils.save_results_to_csv(self.result_dir, "all_trials.csv", converted_trials)
        
        if self.use_wandb:
            WandBLogger.log_optimization_plots(None, self.all_trials, pareto_front, prefix=self.optimizer)
            WandBLogger.log_final_tables(self.all_trials, pareto_front, prefix="final")
            WandBLogger.log_summary(optimization_results)
            
            wandb.run.summary["tables_logged"] = True
            
            wandb.finish()
    
    def _convert_trials_for_csv(self, trials):
        converted_trials = []
        for trial in trials:
            converted_trial = {}
            for k, v in trial.items():
                if isinstance(v, np.integer):
                    converted_trial[k] = int(v)
                elif isinstance(v, np.floating):
                    converted_trial[k] = float(v)
                elif isinstance(v, np.ndarray):
                    converted_trial[k] = v.tolist()
                elif isinstance(v, dict):
                    converted_trial[k] = str(v)
                else:
                    converted_trial[k] = v
            converted_trials.append(converted_trial)
        return converted_trials
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        config_str = json.dumps(dict(sorted(config.items())), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]