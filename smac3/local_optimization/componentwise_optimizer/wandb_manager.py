import time
import wandb
from typing import Dict, Any

from pipeline.logging.wandb import WandBLogger


class WandBManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def init_wandb_for_component(self, component: str, component_idx: int, total_components: int):
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARNING] Error finishing previous W&B run: {e}")

            time.sleep(2)
        
        wandb_run_name = f"{self.optimizer.study_name}_{component}"
        try:
            wandb.init(
                project=self.optimizer.wandb_project,
                entity=self.optimizer.wandb_entity,
                name=wandb_run_name,
                config={
                    "component": component,
                    "stage": f"{component_idx + 1}/{total_components}",
                    "n_trials": self.optimizer.n_trials_per_component,
                    "use_multi_fidelity": self.optimizer.use_multi_fidelity,
                    "use_multi_objective": self.optimizer.use_multi_objective
                },
                reinit=True,
                force=True
            )
            WandBLogger.reset_step_counter()
        except Exception as e:
            print(f"[WARNING] Failed to initialize W&B for {component}: {e}")
            print(f"[WARNING] Continuing without W&B logging for {component}")
            self.optimizer.wandb_enabled = False
    
    def log_wandb_component_summary(self, component: str, component_result: Dict[str, Any]):
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

        if not self.optimizer.wandb_enabled:
            self.optimizer.wandb_enabled = True
    
    def log_wandb_component_table(self, component: str):
        detailed_metrics_dict = {}
        if component in self.optimizer.component_detailed_metrics:
            for i, metrics in enumerate(self.optimizer.component_detailed_metrics[component]):
                detailed_metrics_dict[i] = metrics
        
        WandBLogger.log_component_optimization_table(
            component, 
            self.optimizer.component_trials,
            detailed_metrics_dict
        )
    
    def log_wandb_final_summary(self, all_results: Dict[str, Any]):
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"[WARNING] Error finishing component W&B run: {e}")
            
            time.sleep(2)
        
        try:
            wandb.init(
                project=self.optimizer.wandb_project,
                entity=self.optimizer.wandb_entity,
                name=f"{self.optimizer.study_name}_summary",
                config={
                    "optimization_mode": "componentwise",
                    "total_components": len(all_results['component_order']),
                    "total_time": all_results['optimization_time'],
                    "early_stopped": all_results.get('early_stopped', False),
                    "stopped_at_component": all_results.get('stopped_at_component'),
                    "use_multi_fidelity": self.optimizer.use_multi_fidelity,
                    "use_multi_objective": self.optimizer.use_multi_objective
                },
                reinit=True,
                force=True
            )

            WandBLogger.log_final_componentwise_summary(all_results)
            wandb.finish()
        except Exception as e:
            print(f"[WARNING] Failed to log final W&B summary: {e}")