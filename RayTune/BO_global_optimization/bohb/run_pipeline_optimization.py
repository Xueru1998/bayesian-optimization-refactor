import os
import sys
import argparse
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RayTune.BO_global_optimization.bohb.bohb_rag_pipeline_optimizer import RAGPipelineOptimizer
from pipeline.utils import Utils
from pipeline.email_notifier import ExperimentEmailNotifier, ExperimentNotificationWrapper


def main():
    parser = argparse.ArgumentParser(
        description='Run Bayesian Optimization (BOHB) for the entire RAG pipeline'
    )
    parser.add_argument('--config_path', type=str, 
                        default="config.yaml",
                        help='Path to the configuration YAML file (default: centralized config.yaml)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of configurations to try (default: auto-calculate based on search space)')
    parser.add_argument('--sample_percentage', type=float, default=0.1,
                        help='Percentage of total combinations to sample when auto-calculating (default: 0.1 = 10%%)')
    parser.add_argument('--max_concurrent', type=int, default=4,
                        help='Maximum number of concurrent trials')
    parser.add_argument('--cpu_per_trial', type=int, default=None,
                        help='CPUs allocated per trial (default: no limit, use all available)')
    parser.add_argument('--gpu_per_trial', type=float, default=None,
                        help='GPUs allocated per trial (default: auto-detect and allocate)')
    parser.add_argument('--retrieval_weight', type=float, default=0.5,
                        help='Weight for retrieval score (0-1)')
    parser.add_argument('--generation_weight', type=float, default=0.5,
                        help='Weight for generation score (0-1)')
    parser.add_argument('--project_dir', type=str, default="autorag_project",
                        help='Directory for the AutoRAG project (default: centralized autorag_project)')
    parser.add_argument('--use_cached_embeddings', action='store_true', default=True,
                        help='Use pre-generated embeddings (default: True)')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory to store results (default: ./results/bohb_opt_<timestamp>)')
    parser.add_argument('--early_stop_threshold', type=float, default=0.9,
                        help='Early stopping threshold for combined score (default: 0.9)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the optimization study')
    
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases for logging (default: True)')
    parser.add_argument('--wandb_project', type=str, default="BO & AutoRAG",
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='WandB run name')
    
    parser.add_argument('--min_budget_percentage', type=float, default=0.33,
                        help='Minimum budget as percentage of total samples (default: 0.33)')
    parser.add_argument('--max_budget_percentage', type=float, default=1.0,
                        help='Maximum budget as percentage of total samples (default: 1.0)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Eta parameter for successive halving (default: 3)')
    
    parser.add_argument('--send_email', action='store_true', default=False,
                        help='Send email notification when experiment completes')
    parser.add_argument('--email_recipients', type=str, nargs='+', default=None,
                        help='Email recipients (space-separated list)')
    parser.add_argument('--email_sender', type=str, default=None,
                        help='Email sender address (overrides environment variable)')
    parser.add_argument('--email_password', type=str, default=None,
                        help='Email password (overrides environment variable)')
    parser.add_argument('--smtp_server', type=str, default="smtp.gmail.com",
                        help='SMTP server address')
    parser.add_argument('--smtp_port', type=int, default=587,
                        help='SMTP server port')
    
    args = parser.parse_args()

    project_root = Utils.find_project_root()
    config_path = Utils.get_centralized_config_path(args.config_path)
    project_dir = Utils.get_centralized_project_dir(args.project_dir)
    
    total_weight = args.retrieval_weight + args.generation_weight
    if total_weight != 1.0:
        print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
        args.retrieval_weight = args.retrieval_weight / total_weight
        args.generation_weight = args.generation_weight / total_weight

    print(f"Using project root: {project_root}")
    print(f"Using centralized config file: {config_path}")
    print(f"Using centralized project directory: {project_dir}")

    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)
        print(f"Created project directory: {project_dir}")

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}")
        print("Please ensure config.yaml exists in the project root or specify --config_path")
        return 1

    qa_path, corpus_path = Utils.get_centralized_data_paths(project_dir)

    if not os.path.exists(qa_path) or not os.path.exists(corpus_path):
        print(f"ERROR: Required data files not found.")
        print(f"  QA file: {qa_path} - exists: {os.path.exists(qa_path)}")
        print(f"  Corpus file: {corpus_path} - exists: {os.path.exists(corpus_path)}")
        return 1

    try:
        qa_df = pd.read_parquet(qa_path)
        corpus_df = pd.read_parquet(corpus_path)
        
        print(f"Loaded QA dataset with {len(qa_df)} samples")
        print(f"Loaded Corpus with {len(corpus_df)} documents")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    start_time = time.time()

    if args.result_dir:
        result_dir = args.result_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(script_dir, "results", f"bohb_opt_{int(start_time)}")
    
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}")

    print("\n===== Starting BOHB Multi-Fidelity Optimization =====")
    print("All required files found. Starting optimization...")

    if args.num_samples is not None:
        print(f"Using fixed sample size: {args.num_samples}")
    else:
        print(f"Using auto-calculated sample size: {args.sample_percentage*100}% of total combinations")

    optimizer = RAGPipelineOptimizer(
        config_path=config_path,
        qa_df=qa_df,
        corpus_df=corpus_df,
        project_dir=project_dir,
        num_samples=args.num_samples,  
        sample_percentage=args.sample_percentage,
        max_concurrent=args.max_concurrent,
        cpu_per_trial=args.cpu_per_trial,
        gpu_per_trial=args.gpu_per_trial,
        retrieval_weight=args.retrieval_weight,
        generation_weight=args.generation_weight,
        use_cached_embeddings=args.use_cached_embeddings,
        result_dir=result_dir,
        early_stop_threshold=args.early_stop_threshold,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        study_name=args.study_name,
        min_budget_percentage=args.min_budget_percentage,
        max_budget_percentage=args.max_budget_percentage,
        eta=args.eta
    )

    experiment_name = args.study_name or f"BOHB_RAG_{int(start_time)}"
    
    if args.send_email:
        try:
            email_notifier = ExperimentEmailNotifier(
                smtp_server=args.smtp_server,
                smtp_port=args.smtp_port,
                sender_email=args.email_sender,
                sender_password=args.email_password,
                recipient_emails=args.email_recipients,
                use_env_vars=True
            )
            
            wrapper = ExperimentNotificationWrapper(optimizer, email_notifier)
            
            print("\n Email notifications enabled")
            print(f"Recipients: {args.email_recipients or 'Using environment defaults'}")
            
            results = wrapper.run_with_notification(experiment_name=experiment_name)
            
            best_config = results.get('best_config', {})
            
        except ValueError as e:
            print(f"\n Email notification setup failed: {e}")
            print("Continuing without email notifications...")
            best_config = optimizer.optimize()
        except Exception as e:
            print(f"\nUnexpected error in email setup: {e}")
            print("Continuing without email notifications...")
            best_config = optimizer.optimize()
    else:
        best_config = optimizer.optimize()

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n===== BOHB Optimization Complete =====")
    print(f"Total run time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
    print(f"Best configuration: {best_config}")
    print(f"Results saved in: {result_dir}")

    timing_data = {
        "start_time": start_time,
        "end_time": end_time,
        "total_seconds": total_time,
        "formatted_time": f"{int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s"
    }
    Utils.save_results_to_json(result_dir, "timing_info.json", timing_data)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())