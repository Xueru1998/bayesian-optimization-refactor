import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class PlotGenerator:
    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        self.figures_dir = os.path.join(result_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def generate_all_plots(self, study, all_trials: List[Dict[str, Any]]):
        self._generate_matplotlib_plots(all_trials)
        print(f"Saved plots to {self.figures_dir}")
    
    def _generate_matplotlib_plots(self, all_trials: List[Dict[str, Any]]):
        scores = [t["score"] for t in all_trials]
        latencies = [t["latency"] for t in all_trials if t["latency"] < float('inf')]
        trials = list(range(1, len(scores) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(trials, scores, 'b-', marker='o')
        plt.xlabel('Trial Number')
        plt.ylabel('Score')
        plt.title('Score Progression over Trials')
        plt.grid(True)
        plt.savefig(os.path.join(self.figures_dir, "score_progression.png"))
        plt.close()
        
        if latencies:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(latencies) + 1), latencies, 'r-', marker='o')
            plt.xlabel('Trial Number')
            plt.ylabel('Latency (s)')
            plt.title('Latency Progression over Trials')
            plt.grid(True)
            plt.savefig(os.path.join(self.figures_dir, "latency_progression.png"))
            plt.close()
        
        pareto_front = self._find_pareto_front(all_trials)
        if pareto_front:
            plt.figure(figsize=(10, 6))
            plt.scatter([t["latency"] for t in all_trials if t["latency"] < float('inf')], 
                      [t["score"] for t in all_trials if t["latency"] < float('inf')], 
                      c='blue', label='All Trials')
            
            plt.scatter([t["latency"] for t in pareto_front], 
                      [t["score"] for t in pareto_front], 
                      c='red', s=100, label='Pareto Optimal')
            
            plt.xlabel('Latency (s)')
            plt.ylabel('Score')
            plt.title('Pareto Front')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, "pareto_front_matplotlib.png"))
            plt.close()
        
        max_score_idx = scores.index(max(scores))
        max_score_trial = all_trials[max_score_idx]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(trials, scores, c=scores, cmap='viridis', s=50)
        plt.colorbar(label='Score')
        plt.scatter(max_score_idx + 1, max(scores), c='red', s=200, marker='*', 
                   label=f'Best Score: {max(scores):.4f}')
        plt.xlabel('Trial Number')
        plt.ylabel('Score')
        plt.title('Trial Performance Overview')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.figures_dir, "trial_performance.png"))
        plt.close()
        
        if len(all_trials) > 0 and 'config' in all_trials[0]:
            self._plot_parameter_importance(all_trials)
    
    def _plot_parameter_importance(self, all_trials: List[Dict[str, Any]]):
        params = {}
        for trial in all_trials:
            if 'config' in trial:
                for param, value in trial['config'].items():
                    if param not in params:
                        params[param] = []
                    params[param].append((value, trial['score']))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        param_items = list(params.items())[:6]
        
        for idx, (param_name, values) in enumerate(param_items):
            if idx < len(axes):
                ax = axes[idx]
                
                unique_values = list(set([v[0] for v in values]))
                if len(unique_values) <= 10:
                    value_scores = {}
                    for val, score in values:
                        if val not in value_scores:
                            value_scores[val] = []
                        value_scores[val].append(score)
                    
                    avg_scores = {val: sum(scores)/len(scores) 
                                for val, scores in value_scores.items()}
                    
                    sorted_items = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
                    x_labels = [str(item[0])[:20] for item in sorted_items]
                    y_values = [item[1] for item in sorted_items]
                    
                    ax.bar(range(len(x_labels)), y_values)
                    ax.set_xticks(range(len(x_labels)))
                    ax.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax.set_ylabel('Average Score')
                else:
                    x_vals = [v[0] for v in values if isinstance(v[0], (int, float))]
                    y_vals = [v[1] for v in values if isinstance(v[0], (int, float))]
                    if x_vals:
                        ax.scatter(x_vals, y_vals, alpha=0.6)
                        ax.set_xlabel(param_name)
                        ax.set_ylabel('Score')
                
                ax.set_title(f'{param_name}')
                ax.grid(True, alpha=0.3)
        
        for idx in range(len(param_items), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "parameter_analysis.png"))
        plt.close()
    
    def _find_pareto_front(self, all_trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not all_trials:
            return []
        
        valid_trials = [t for t in all_trials if t["score"] > 0 and t["latency"] < float('inf')]
        if not valid_trials:
            return []
        
        points = [(-t["score"], t["latency"]) for t in valid_trials]
        
        pareto_indices = []
        for i, point in enumerate(points):
            is_dominated = False
            for j, other_point in enumerate(points):
                if j != i:
                    if (other_point[0] <= point[0] and other_point[1] <= point[1]) and \
                       (other_point[0] < point[0] or other_point[1] < point[1]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return [valid_trials[i] for i in pareto_indices]