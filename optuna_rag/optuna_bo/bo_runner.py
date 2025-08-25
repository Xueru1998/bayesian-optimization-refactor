import os
import sys
import time
import argparse
import pandas as pd
from typing import Optional, Dict, Any
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.utils import Utils
from optuna_global_optimization.bo_optimizer import BOPipelineOptimizer
from optuna_rag.optuna_bo.optuna_local_optimization.optimizers.componentwise_bayesian_optimization import ComponentwiseOptunaOptimizer
from pipeline.logging.email.email_notifier import ExperimentEmailNotifier, ExperimentNotificationWrapper

class UnifiedOptunaRunner:
    
    def __init__(self):
        self.parser = self._create_parser()
        self.project_root = Utils.find_project_root()
        self.wandb_project_global = "BO & AutoRAG"
        self.wandb_project_componentwise = "Componentwise Optimization"
        self.wandb_entity = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description='Run Bayesian Optimization for RAG pipeline using Optuna (Global or Component-wise)'
        )
        
        # Optimization mode
        parser.add_argument('--mode', type=str, default='global', 
                           choices=['global', 'componentwise', 'component-wise'],
                           help='Optimization mode: global or componentwise (default: global)')
        
        # Common arguments
        parser.add_argument('--config_path', type=str, default='config.yaml',
                           help='Path to the configuration YAML file (default: centralized config.yaml)')
        parser.add_argument('--project_dir', type=str, default='autorag_project',
                           help='Directory for the AutoRAG project (default: centralized autorag_project)')
        parser.add_argument('--cpu_per_trial', type=int, default=4,
                           help='CPUs allocated per trial (default: 4)')
        parser.add_argument('--retrieval_weight', type=float, default=0.5,
                           help='Weight for retrieval score (default: 0.5)')
        parser.add_argument('--generation_weight', type=float, default=0.5,
                           help='Weight for generation score (default: 0.5)')
        parser.add_argument('--use_cached_embeddings', action='store_true', default=True,
                           help='Use pre-generated embeddings (default: True)')
        parser.add_argument('--result_dir', type=str, default=None,
                           help='Directory to store results (default: auto-generated based on mode)')
        parser.add_argument('--study_name', type=str, default=None,
                           help='Name for the optimization study (default: auto-generated)')
        parser.add_argument('--no_wandb', action='store_true',
                           help='Disable Weights & Biases tracking')
        parser.add_argument('--sampler', type=str, default='tpe', 
                           choices=['tpe','botorch', 'random', 'grid'],
                           help='Sampler to use for optimization (default: tpe)')
        parser.add_argument('--wandb_run_name', type=str, default=None,
                           help='Custom name for the W&B run (default: same as study_name)')
        parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
        parser.add_argument('--early_stopping_threshold', type=float, default=0.9,
                           help='Score threshold for early stopping (default: 0.9)')
        
        # Global optimization specific arguments
        parser.add_argument('--n_trials', type=int, default=None,
                           help='Number of optimization trials for global mode (default: auto-calculated)')
        parser.add_argument('--sample_percentage', type=float, default=0.1,
                           help='Percentage of search space to sample when auto-calculating trials (default: 0.1)')
        parser.add_argument('--use_ragas', action='store_true', default=False,
                           help='Use RAGAS for evaluation (only available in global mode)')
        parser.add_argument('--ragas_llm_model', type=str, default='gpt-4o-mini',
                           help='LLM model to use for RAGAS evaluation (default: gpt-4o-mini)')
        parser.add_argument('--ragas_embedding_model', type=str, default='text-embedding-ada-002',
                           help='Embedding model to use for RAGAS evaluation (default: text-embedding-ada-002)')
        parser.add_argument('--ragas_metrics', type=str, nargs='+', 
                           choices=['context_precision', 'context_recall', 'answer_relevancy', 
                                   'faithfulness', 'factual_correctness', 'semantic_similarity'],
                           help='Specific RAGAS metrics to use (default: all)')
        
        # LLM Evaluator arguments 
        parser.add_argument('--use_llm_evaluator', action='store_true', default=False,
                           help='Use LLM-based evaluation for passage compressor')
        parser.add_argument('--llm_evaluator_model', type=str, default='gpt-4o',
                           help='LLM model to use for compressor evaluation (default: gpt-4o)')
        parser.add_argument('--llm_evaluator_temperature', type=float, default=0.0,
                           help='Temperature for LLM evaluator (default: 0.0)')
        
        # Component-wise optimization specific arguments
        parser.add_argument('--n_trials_per_component', type=int, default=20,
                           help='Number of trials per component for componentwise mode (default: 20)')
        parser.add_argument('--walltime_limit_per_component', type=int, default=None,
                           help='Walltime limit per component in seconds (default: None, no limit)')
        parser.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for parallel optimization (default: 1)')
        parser.add_argument('--use_multi_objective', action='store_true', default=False,
                           help='Use multi-objective optimization (score + latency) in componentwise mode')
        
        # Email notification arguments
        parser.add_argument('--send_email', action='store_true', default=False,
                           help='Send email notification when experiment completes (requires EMAIL_SENDER, EMAIL_PASSWORD env vars)')
        parser.add_argument('--disable_early_stopping', action='store_true',
                    help='Disable early stopping for low-scoring components')
        
        parser.add_argument('--resume', action='store_true', default=False,
                   help='Resume previous optimization from where it left off')
        
        return parser
    
    def _normalize_weights(self, retrieval_weight: float, generation_weight: float) -> tuple:
        total = retrieval_weight + generation_weight
        if total == 0:
            return 0.5, 0.5
        return retrieval_weight / total, generation_weight / total
    
    def _resolve_paths(self, args) -> None:
        args.config_path = Utils.get_centralized_config_path(args.config_path)
        args.project_dir = Utils.get_centralized_project_dir(args.project_dir)

        if args.result_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if args.mode in ['componentwise', 'component-wise']:
                args.result_dir = os.path.join(script_dir, 'componentwise_optuna_results')
            else:
                args.result_dir = os.path.join(script_dir, 'bo_optimization_results')
        elif not os.path.isabs(args.result_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.result_dir = os.path.abspath(os.path.join(script_dir, args.result_dir))
    
    def _validate_project_dir(self, project_dir: str) -> None:
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)
            print(f"Created project directory: {project_dir}")
    
    def _validate_config(self, config_path: str) -> bool:
        if not os.path.exists(config_path):
            print(f"ERROR: Config file not found at {config_path}")
            print("Please ensure config.yaml exists in the project root or specify --config_path")
            return False
        return True
    
    def _validate_data_files(self, qa_path: str, corpus_path: str) -> bool:
        if not os.path.exists(qa_path):
            print(f"ERROR: QA validation file not found at {qa_path}")
            print("Please ensure QA data exists in centralized location")
            return False
            
        if not os.path.exists(corpus_path):
            print(f"ERROR: Corpus file not found at {corpus_path}")
            print("Please ensure corpus data exists in centralized location")
            return False
            
        return True
    
    def _load_data(self, qa_path: str, corpus_path: str) -> tuple:
        try:
            qa_df = pd.read_parquet(qa_path)
            corpus_df = pd.read_parquet(corpus_path)
            
            print(f"Loaded QA dataset with {len(qa_df)} samples")
            print(f"Loaded Corpus with {len(corpus_df)} documents")
            
            return qa_df, corpus_df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _check_wandb_login(self) -> bool:
        try:
            if wandb.api.api_key is None:
                print("\nYou are not logged into Weights & Biases!")
                return False
            return True
        except ImportError:
            print("wandb package not installed!")
            return False
        except Exception:
            return False
    
    def _print_sampler_info(self, optimizer: str):
        sampler_info = {
            "tpe": {
                "name": "Tree-structured Parzen Estimator (TPE)",
                "description": "suitable for conditional parameters and mixed search spaces",
                "best_for": "RAG pipelines with many conditional parameters"
            },
            "botorch": {
                "name": "BoTorch (Gaussian Process)",
                "description": "State-of-the-art GP-based optimization with advanced acquisition functions",
                "best_for": "Multi-objective optimization with continuous parameters",
                "note": "Requires: pip install botorch gpytorch"
            },
            "random": {
                "name": "Random Search",
                "description": "Simple baseline for comparison",
                "best_for": "Quick exploration or baseline comparison"
            },
              "grid": {
            "name": "Grid Search",
            "description": "Exhaustive search through all parameter combinations",
            "best_for": "Small search spaces or when you need to explore all possibilities"
        }
        }
        
        info = sampler_info.get(optimizer, sampler_info["tpe"])
        print(f"\n Selected Sampler: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Best for: {info['best_for']}")
        if 'note' in info:
            print(f"   Note: {info['note']}")
    
    def _validate_arguments(self, args) -> bool:
        if args.mode in ['componentwise', 'component-wise']:
            if args.use_ragas:
                print("ERROR: RAGAS evaluation is not supported in component-wise mode")
                print("Component-wise optimization evaluates each component separately")
                return False

        if args.use_llm_evaluator:
            if args.use_ragas:
                print("WARNING: Both RAGAS and LLM evaluator are enabled. LLM evaluator will be used for compressor component.")
            print(f"LLM Evaluator enabled for compressor evaluation")
            print(f"  Model: {args.llm_evaluator_model}")
            print(f"  Temperature: {args.llm_evaluator_temperature}")
        
        return True
    
    def _load_config_template(self, config_path: str) -> Dict[str, Any]:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _run_global_optimization(self, args, qa_df, corpus_df) -> Dict[str, Any]:
        print("\n===== Starting Global Bayesian Optimization =====")
        self._print_sampler_info(args.sampler)
        
        ragas_metrics = None
        if args.use_ragas and args.ragas_metrics:
            retrieval_metrics = [m for m in args.ragas_metrics if m in ['context_precision', 'context_recall']]
            generation_metrics = [m for m in args.ragas_metrics if m in ['answer_relevancy', 'faithfulness', 
                                                                        'factual_correctness', 'semantic_similarity']]
            ragas_metrics = {
                'retrieval': retrieval_metrics,
                'generation': generation_metrics
            }
        
        if args.use_ragas:
            print("RAGAS evaluation enabled")
        
        if args.use_llm_evaluator:
            print("LLM Evaluator enabled for compressor component")
        
        wandb_run_name = args.wandb_run_name or args.study_name
        if not wandb_run_name:
            wandb_run_name = f"bo_{args.sampler}_{int(time.time())}"

        llm_evaluator_config = None
        if args.use_llm_evaluator:
            llm_evaluator_config = {
                "llm_model": args.llm_evaluator_model,
                "temperature": args.llm_evaluator_temperature
            }
        
        optimizer = BOPipelineOptimizer(
            config_path=args.config_path,
            qa_df=qa_df,
            corpus_df=corpus_df,
            project_dir=args.project_dir,
            n_trials=args.n_trials,
            sample_percentage=args.sample_percentage,
            cpu_per_trial=args.cpu_per_trial,
            retrieval_weight=args.retrieval_weight,
            generation_weight=args.generation_weight,
            use_cached_embeddings=args.use_cached_embeddings,
            result_dir=args.result_dir,
            study_name=args.study_name,
            continue_study=False,
            use_wandb=not args.no_wandb,
            wandb_project=self.wandb_project_global,
            wandb_entity=self.wandb_entity,
            wandb_run_name=wandb_run_name,
            optimizer=args.sampler,
            early_stopping_threshold=args.early_stopping_threshold,
            use_ragas=args.use_ragas,
            ragas_llm_model=args.ragas_llm_model,
            ragas_embedding_model=args.ragas_embedding_model,
            ragas_metrics=ragas_metrics,
            use_llm_compressor_evaluator=args.use_llm_evaluator,
            llm_evaluator_config=llm_evaluator_config,
            disable_early_stopping=args.disable_early_stopping 
        )
        
        if args.n_trials:
            print(f"\nStarting optimization with {args.n_trials} trials (user-specified)...")
        else:
            print(f"\nStarting optimization with auto-calculated number of trials...")
            print(f"Sample percentage: {args.sample_percentage * 100}% of search space")
        
        return optimizer
    
    def _run_componentwise_optimization(self, args, qa_df, corpus_df) -> ComponentwiseOptunaOptimizer:
        print("\n===== Starting Component-wise Bayesian Optimization =====")
        self._print_sampler_info(args.sampler)

        config_template = self._load_config_template(args.config_path)
        use_multi_objective = True if args.mode == 'componentwise' else args.use_multi_objective

        llm_evaluator_config = None
        if args.use_llm_evaluator:
            llm_evaluator_config = {
                "llm_model": args.llm_evaluator_model,
                "temperature": args.llm_evaluator_temperature
            }
        
        optimizer = ComponentwiseOptunaOptimizer(
            config_template=config_template,
            qa_data=qa_df,
            corpus_data=corpus_df,
            project_dir=args.project_dir,
            n_trials_per_component=args.n_trials_per_component,
            sample_percentage=args.sample_percentage,
            cpu_per_trial=args.cpu_per_trial,
            retrieval_weight=args.retrieval_weight,
            generation_weight=args.generation_weight,
            use_cached_embeddings=args.use_cached_embeddings,
            result_dir=args.result_dir,
            study_name=args.study_name,
            walltime_limit_per_component=args.walltime_limit_per_component,
            n_workers=args.n_workers,
            seed=args.seed,
            early_stopping_threshold=args.early_stopping_threshold,
            use_wandb=not args.no_wandb,
            wandb_project=self.wandb_project_componentwise,
            wandb_entity=self.wandb_entity,
            optimizer=args.sampler,
            use_multi_objective=use_multi_objective,
            use_llm_compressor_evaluator=args.use_llm_evaluator, 
            llm_evaluator_config=llm_evaluator_config,
            resume_study=args.resume   
        )
        
        print("\nComponent-wise optimization settings:")
        print(f"  Trials per component: {args.n_trials_per_component}")
        print(f"  Walltime limit per component: {args.walltime_limit_per_component if args.walltime_limit_per_component else 'None (no limit)'}")
        print(f"  Multi-objective: {use_multi_objective}")
        print(f"  Early stopping threshold: {args.early_stopping_threshold}")
        if args.use_llm_evaluator:
            print(f"  LLM Evaluator: Enabled for compressor component")
        print("\nNote: Component-wise optimization will:")
        print("  - Optimize each component sequentially")
        print("  - Skip retrieval if query expansion is active")
        print("  - Use saved outputs from previous components (no re-execution)")
        print("  - Not support RAGAS evaluation (component-specific metrics only)")
        
        return optimizer
    
    def run(self, argv=None) -> int:
        args = self.parser.parse_args(argv)

        if args.mode == 'component-wise':
            args.mode = 'componentwise'

        if not self._validate_arguments(args):
            return 1

        args.retrieval_weight, args.generation_weight = self._normalize_weights(
            args.retrieval_weight, args.generation_weight
        )
        
        self._resolve_paths(args)
        
        print(f"Using optimization mode: {args.mode.upper()}")
        
        self._validate_project_dir(args.project_dir)
        
        if not self._validate_config(args.config_path):
            return 1
        
        print(f"Results will be saved to: {args.result_dir}")

        qa_path, corpus_path = Utils.get_centralized_data_paths(args.project_dir)

        if not self._validate_data_files(qa_path, corpus_path):
            return 1

        if not args.no_wandb:            
            if not self._check_wandb_login():
                print("\nContinuing without W&B tracking...")
                args.no_wandb = True
            else:
                print("W&B authentication successful")
        
        print("\nAll required files found. Loading data...")

        qa_df, corpus_df = self._load_data(qa_path, corpus_path) 
        start_time = time.time()

        if args.mode == 'componentwise':
            optimizer = self._run_componentwise_optimization(args, qa_df, corpus_df)
        else:
            optimizer = self._run_global_optimization(args, qa_df, corpus_df)

        experiment_name = args.study_name or f"{args.mode}_{args.sampler}_{int(start_time)}"
        
        if args.send_email:
            try:
                email_notifier = ExperimentEmailNotifier()
                
                wrapper = ExperimentNotificationWrapper(optimizer, email_notifier)
                
                print("\n Email notifications enabled")
                print(f"Recipients: {', '.join(email_notifier.recipient_emails)}")
                
                best_results = wrapper.run_with_notification(experiment_name=experiment_name)
                
            except ValueError as e:
                print(f"\n Email notification setup failed: {e}")
                print("Continuing without email notifications...")
                best_results = optimizer.optimize()
            except Exception as e:
                print(f"\n Unexpected error in email setup: {e}")
                print("Continuing without email notifications...")
                best_results = optimizer.optimize()
        else:
            best_results = optimizer.optimize()

        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n===== {args.mode.title()} Optimization Complete =====")
        print(f"Total optimization time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        if args.mode == 'componentwise':
            print(f"Total trials completed: {best_results.get('total_trials', 0)}")
            print(f"Components optimized: {len(best_results.get('component_order', []))}")
            
            if best_results.get('validation_failed'):
                print("\n⚠️  Optimization failed due to insufficient search space")
                print(f"Invalid components: {best_results.get('invalid_components', [])}")
            else:
                print("\nOptimized components:")
                for component in best_results.get('component_order', []):
                    comp_result = best_results.get('component_results', {}).get(component, {})
                    print(f"  {component}: score={comp_result.get('best_score', 0):.4f}, "
                          f"trials={comp_result.get('n_trials', 0)}")

                final_config = {}
                for component in best_results.get('component_order', []):
                    best_cfg = best_results.get('best_configs', {}).get(component, {})
                    final_config.update(best_cfg)
                
                if final_config:
                    print(f"\nFinal optimized configuration:")
                    for key, value in sorted(final_config.items()):
                        print(f"  {key}: {value}")
        else:
            print(f"Total trials completed: {best_results.get('n_trials', 0)}")
            print(f"Sampler used: {args.sampler.upper()}")
            
            if best_results.get('early_stopped', False):
                print("⚡ Optimization stopped early due to achieving target score!")
            
            if best_results.get('best_config'):
                print(f"\nBest configuration found:")
                print(f"  Score: {best_results['best_config']['score']:.4f}")
                print(f"  Latency: {best_results['best_config']['latency']:.2f}s")
                print(f"  Config: {best_results['best_config']['config']}")
            elif best_results.get('best_score_config'):
                print(f"\nBest configuration by score:")
                print(f"  Score: {best_results['best_score']:.4f}")
                print(f"  Config: {best_results['best_score_config']}")
        
        print(f"\nResults saved to: {args.result_dir}")
        
        return 0


def main():
    runner = UnifiedOptunaRunner()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())