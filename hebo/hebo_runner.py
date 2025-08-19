import os
import sys
import time
import argparse
import pandas as pd
import wandb
from typing import Optional
from pipeline.utils import Utils
from hebo_optimizer import HEBORAGOptimizer
from pipeline.email_notifier import ExperimentEmailNotifier, ExperimentNotificationWrapper


class HEBORunner:
    
    def __init__(self):
        self.parser = self._create_parser()
        self.project_root = Utils.find_project_root()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description='Run HEBO Optimization for RAG pipeline',
            epilog='HEBO is a Heteroscedastic Evolutionary Bayesian Optimization algorithm'
        )
        
        parser.add_argument('--config_path', type=str, default='config.yaml',
                          help='Path to config.yaml file')
        parser.add_argument('--project_dir', type=str, default='autorag_project',
                          help='Project directory containing data')
        parser.add_argument('--n_trials', type=int, default=None,
                          help='Number of trials (auto-calculated if not provided)')
        parser.add_argument('--sample_percentage', type=float, default=0.1,
                          help='Percentage of search space to sample for auto-calculation')
        parser.add_argument('--cpu_per_trial', type=int, default=4,
                          help='Number of CPUs per trial')
        parser.add_argument('--retrieval_weight', type=float, default=0.5,
                          help='Weight for retrieval/latency objective (0-1)')
        parser.add_argument('--generation_weight', type=float, default=0.5,
                          help='Weight for generation/score objective (0-1)')
        parser.add_argument('--use_cached_embeddings', action='store_true', default=True,
                          help='Use cached embeddings if available')
        parser.add_argument('--result_dir', type=str, default=None,
                          help='Directory to save results')
        parser.add_argument('--study_name', type=str, default=None,
                          help='Name for this study')
        parser.add_argument('--walltime_limit', type=int, default=None,
                  help='Wall time limit in seconds (no limit if not specified)')
        parser.add_argument('--early_stopping_threshold', type=float, default=0.9,
                          help='Score threshold for early stopping')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed')
        
        parser.add_argument('--batch_size', type=int, default=1,
                          help='Batch size for parallel evaluations')
        parser.add_argument('--n_suggestions', type=int, default=1,
                          help='Number of suggestions per iteration')
        
        parser.add_argument('--no_wandb', action='store_true',
                          help='Disable Weights & Biases logging')
        parser.add_argument('--wandb_project', type=str, default='BO & AutoRAG',
                          help='W&B project name')
        parser.add_argument('--wandb_entity', type=str, default=None,
                          help='W&B entity name')
        parser.add_argument('--wandb_run_name', type=str, default=None,
                          help='W&B run name')
        
        parser.add_argument('--email_notifications', action='store_true',
                            help='Send email notification when experiment completes')
        parser.add_argument('--email_sender', type=str, default=None,
                            help='Sender email address (or set EMAIL_SENDER env var)')
        parser.add_argument('--email_password', type=str, default=None,
                            help='Email password/app password (or set EMAIL_PASSWORD env var)')
        parser.add_argument('--email_recipients', type=str, nargs='+', default=None,
                            help='Recipient email addresses (or set EMAIL_RECIPIENT env var)')
        parser.add_argument('--smtp_server', type=str, default='smtp.gmail.com',
                            help='SMTP server address')
        parser.add_argument('--smtp_port', type=int, default=587,
                            help='SMTP server port')
        
        return parser
    
    def _normalize_weights(self, retrieval_weight: float, generation_weight: float) -> tuple:
        total = retrieval_weight + generation_weight
        if total == 0:
            return 0.5, 0.5
        return retrieval_weight / total, generation_weight / total
    
    def _resolve_paths(self, args) -> None:
        args.config_path = Utils.get_centralized_config_path(args.config_path)
        args.project_dir = Utils.get_centralized_project_dir(args.project_dir)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if args.result_dir is None:
            args.result_dir = "hebo_optimization_results"
        
        if not os.path.isabs(args.result_dir):
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
    
    def _load_config_template(self, config_path: str):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_email_notifier(self, args) -> Optional[ExperimentEmailNotifier]:
        if not args.email_notifications:
            return None
        
        sender_email = args.email_sender or os.environ.get('EMAIL_SENDER')
        sender_password = args.email_password or os.environ.get('EMAIL_PASSWORD')
        
        if not sender_email or not sender_password:
            print("\n Email notifications requested but credentials not provided!")
            return None
        
        recipient_emails = args.email_recipients
        if not recipient_emails:
            recipient = os.environ.get('EMAIL_RECIPIENT', sender_email)
            recipient_emails = [recipient]
        
        try:
            notifier = ExperimentEmailNotifier(
                smtp_server=args.smtp_server,
                smtp_port=args.smtp_port,
                sender_email=sender_email,
                sender_password=sender_password,
                recipient_emails=recipient_emails,
                use_env_vars=False
            )
            print(f"\n Email notifications enabled. Will send to: {', '.join(recipient_emails)}")
            return notifier
        except Exception as e:
            print(f"\n Failed to setup email notifier: {e}")
            print("   Continuing without email notifications...")
            return None
    
    def _create_experiment_name(self, args) -> str:
        if args.study_name:
            return args.study_name
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"HEBO RAG Optimization - {timestamp}"
    
    def run(self, argv=None) -> int:
        args = self.parser.parse_args(argv)
        
        args.retrieval_weight, args.generation_weight = self._normalize_weights(
            args.retrieval_weight, args.generation_weight
        )
        
        self._resolve_paths(args)
        
        print(f"Using project root: {self.project_root}")
        print(f"Using centralized config file: {args.config_path}")
        print(f"Using centralized project directory: {args.project_dir}")
        print(f"Optimizer: HEBO (Heteroscedastic Evolutionary Bayesian Optimization)")
        
        self._validate_project_dir(args.project_dir)
        
        if not self._validate_config(args.config_path):
            return 1
        
        print(f"Results will be saved to: {args.result_dir}")
        
        qa_path, corpus_path = Utils.get_centralized_data_paths(args.project_dir)
        
        if not self._validate_data_files(qa_path, corpus_path):
            return 1
        
        print(f"\n===== Starting HEBO Optimization =====")
        print("All required files found. Loading data...")
        
        qa_df, corpus_df = self._load_data(qa_path, corpus_path)
        
        config_template = self._load_config_template(args.config_path)
        
        if not args.no_wandb:
            if not self._check_wandb_login():
                print("\nContinuing without W&B tracking...")
                args.no_wandb = True
            else:
                print("W&B authentication successful")
        
        email_notifier = self._setup_email_notifier(args)
        
        start_time = time.time()
        
        print(f"\nInitializing HEBO optimizer...")
        print(f"Batch size: {args.batch_size}")
        print(f"Suggestions per iteration: {args.n_suggestions}")
        
        optimizer = HEBORAGOptimizer(
            config_template=config_template,
            qa_data=qa_df,
            corpus_data=corpus_df,
            project_dir=args.project_dir,
            n_trials=args.n_trials,
            sample_percentage=args.sample_percentage,
            cpu_per_trial=args.cpu_per_trial,
            retrieval_weight=args.retrieval_weight,
            generation_weight=args.generation_weight,
            use_cached_embeddings=args.use_cached_embeddings,
            result_dir=args.result_dir,
            study_name=args.study_name,
            walltime_limit=args.walltime_limit,
            early_stopping_threshold=args.early_stopping_threshold,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            seed=args.seed,
            batch_size=args.batch_size,
            n_suggestions=args.n_suggestions
        )
        
        if args.n_trials:
            print(f"\nStarting optimization with {args.n_trials} trials (user-specified)...")
        else:
            print(f"\nStarting optimization with auto-calculated number of trials...")
            print(f"Sample percentage: {args.sample_percentage * 100}% of search space")
        
        print(f"Early stopping threshold: {args.early_stopping_threshold}")
        print(f"Objectives: maximize score (weight={args.generation_weight}), minimize latency (weight={args.retrieval_weight})")
        
        experiment_name = self._create_experiment_name(args)
        
        if email_notifier:
            wrapper = ExperimentNotificationWrapper(optimizer, email_notifier)
            best_results = wrapper.run_with_notification(experiment_name=experiment_name)
        else:
            best_results = optimizer.optimize()
        
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n===== HEBO Optimization Complete =====")
        print(f"Total optimization time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Total trials completed: {best_results.get('n_trials', 0)}")
        
        if best_results.get('early_stopped', False):
            print("Optimization stopped early due to achieving target score!")
        
        print(f"Results saved to: {args.result_dir}")
        
        if best_results.get('best_config'):
            print(f"\nBest configuration found:")
            print(f"  Score: {best_results['best_config']['score']:.4f}")
            print(f"  Latency: {best_results['best_config']['latency']:.2f}s")
            print(f"  Config: {best_results['best_config']['config']}")
        elif best_results.get('best_score_config'):
            print(f"\nBest configuration by score:")
            print(f"  Score: {best_results['best_score']:.4f}")
            print(f"  Config: {best_results['best_score_config']}")
        
        if best_results.get('pareto_front'):
            print(f"\nPareto front contains {len(best_results['pareto_front'])} solutions")
            print("Top 3 Pareto optimal solutions:")
            for i, solution in enumerate(best_results['pareto_front'][:3]):
                print(f"  {i+1}. Score: {solution['score']:.4f}, Latency: {solution['latency']:.2f}s")
        
        return 0


def main():
    runner = HEBORunner()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())