import os
import sys
import time
import argparse
import pandas as pd
from typing import Optional
from pipeline.utils import Utils
from pipeline.logging.email.email_notifier import ExperimentEmailNotifier, ExperimentNotificationWrapper
from smac3.global_optimization.smac_rag_optimizer import SMACRAGOptimizer


class UnifiedSMACRunner:
    
    def __init__(self):
        self.parser = self._create_parser()
        self.project_root = Utils.find_project_root()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description='Run SMAC/BOHB Optimization for RAG pipeline (Global or Component-wise)',
            epilog='Note: BOHB automatically enables multi-fidelity optimization.'
        )
        
        parser.add_argument('--optimization_mode', type=str, default='global', 
                            choices=['global', 'componentwise'],
                            help='Optimization mode: global (all components) or componentwise (sequential)')
        
        parser.add_argument('--config_path', type=str, default='config.yaml')
        parser.add_argument('--project_dir', type=str, default='autorag_project')
        parser.add_argument('--n_trials', type=int, default=None,
                            help='Total trials for global mode, or trials per component for componentwise')
        parser.add_argument('--sample_percentage', type=float, default=0.1)
        parser.add_argument('--cpu_per_trial', type=int, default=4)
        parser.add_argument('--retrieval_weight', type=float, default=0.5)
        parser.add_argument('--generation_weight', type=float, default=0.5)
        parser.add_argument('--use_cached_embeddings', action='store_true', default=True)
        parser.add_argument('--result_dir', type=str, default=None)
        parser.add_argument('--study_name', type=str, default=None)        
        parser.add_argument('--walltime_limit', type=int, default=None,
                            help='Total limit for global, per-component limit for componentwise')
        parser.add_argument('--n_workers', type=int, default=1)
        parser.add_argument('--early_stopping_threshold', type=float, default=0.9)
        parser.add_argument('--seed', type=int, default=42)
        
        parser.add_argument('--optimizer', type=str, default='smac', choices=['smac', 'bohb'],
                        help='Optimizer to use (global mode only)')
        parser.add_argument('--use_multi_fidelity', action='store_true', default=False,
                        help='Enable multi-fidelity optimization')
        parser.add_argument('--min_budget_percentage', type=float, default=0.1)
        parser.add_argument('--max_budget_percentage', type=float, default=1.0)
        parser.add_argument('--eta', type=int, default=3)
        
        parser.add_argument('--no_wandb', action='store_true')
        parser.add_argument('--wandb_project', type=str, default=None,
                            help='W&B project name (auto-set based on mode if not specified)')
        parser.add_argument('--wandb_entity', type=str, default=None)
        parser.add_argument('--wandb_run_name', type=str, default=None)
        
        parser.add_argument('--email_notifications', action='store_true',
                            help='Send email notification when experiment completes (requires EMAIL_SENDER, EMAIL_PASSWORD env vars)')
        
        parser.add_argument('--use_ragas', action='store_true', default=False,
                        help='Use RAGAS for evaluation instead of traditional metrics')
        parser.add_argument('--ragas_llm_model', type=str, default='gpt-4o-mini',
                            help='LLM model to use for RAGAS evaluation (default: gpt-4o-mini)')
        parser.add_argument('--ragas_embedding_model', type=str, default='text-embedding-ada-002',
                            help='Embedding model to use for RAGAS evaluation (default: text-embedding-ada-002)')
        parser.add_argument('--ragas_metrics', type=str, nargs='+', 
                            choices=['context_precision', 'context_recall', 'answer_relevancy', 
                                    'faithfulness', 'factual_correctness', 'semantic_similarity'],
                            help='Specific RAGAS metrics to use (default: all)')
        
        parser.add_argument('--use_llm_compressor_evaluator', action='store_true',
                    help='Use LLM to evaluate compression quality instead of token-based metrics')
        parser.add_argument('--llm_evaluator_model', type=str, default='gpt-4o',
                        help='LLM model to use for compression evaluation (default: gpt-4o)')
        parser.add_argument('--llm_compressor_temperature', type=float, default=0.0)
        
        parser.add_argument('--disable_early_stopping', action='store_true',
                            help='Disable early stopping for low-scoring components')
            
        return parser
    
    def _get_default_early_stopping_thresholds(self):
        return {
            'retrieval': 0.1,
            'query_expansion': 0.1,
            'reranker': 0.2,
            'filter': 0.25,
            'compressor': 0.3
        }
    
    
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
            if args.optimization_mode == 'componentwise':
                args.result_dir = "componentwise_optimization_results"
            else:
                args.result_dir = f"{args.optimizer}_optimization_results"
        
        if not os.path.isabs(args.result_dir):
            args.result_dir = os.path.abspath(os.path.join(script_dir, args.result_dir))
    
    def _validate_files(self, args) -> bool:
        if not os.path.exists(args.config_path):
            print(f"ERROR: Config file not found at {args.config_path}")
            return False
        
        qa_path, corpus_path = Utils.get_centralized_data_paths(args.project_dir)
        
        if not os.path.exists(qa_path):
            print(f"ERROR: QA validation file not found at {qa_path}")
            return False
            
        if not os.path.exists(corpus_path):
            print(f"ERROR: Corpus file not found at {corpus_path}")
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
    
    def _setup_wandb_project(self, args):
        if args.wandb_project is None:
            if args.optimization_mode == 'componentwise':
                args.wandb_project = "Component-wise RAG Optimization"
            else:
                args.wandb_project = "BO & AutoRAG"
    
    def _prepare_ragas_config(self, args):
        if not args.use_ragas:
            return None
            
        if args.ragas_metrics:
            retrieval_metrics = [m for m in args.ragas_metrics if m in ['context_precision', 'context_recall']]
            generation_metrics = [m for m in args.ragas_metrics if m in ['answer_relevancy', 'faithfulness', 
                                                                        'factual_correctness', 'semantic_similarity']]
            ragas_config = {
                'retrieval_metrics': retrieval_metrics or ["context_precision", "context_recall"],
                'generation_metrics': generation_metrics or ["answer_relevancy", "faithfulness", 
                                                            "factual_correctness", "semantic_similarity"],
                'llm_model': args.ragas_llm_model,
                'embedding_model': args.ragas_embedding_model
            }
        else:
            ragas_config = {
                'retrieval_metrics': ["context_precision", "context_recall"],
                'generation_metrics': ["answer_relevancy", "faithfulness", "factual_correctness", "semantic_similarity"],
                'llm_model': args.ragas_llm_model,
                'embedding_model': args.ragas_embedding_model
            }
        return ragas_config
    
    def _run_global_optimization(self, args, qa_df, corpus_df, config_template):
        if args.disable_early_stopping:
            print("\nEarly stopping disabled")
        else:
            print("\nEarly stopping enabled with default thresholds:")
            print("  retrieval/query_expansion < 0.1")
            print("  reranker < 0.2")
            print("  filter < 0.25")
            print("  compressor < 0.3")
        
        if args.use_ragas:
            ragas_config = {
                'retrieval_metrics': ["context_precision", "context_recall"],
                'generation_metrics': ["answer_relevancy", "faithfulness", "factual_correctness", "semantic_similarity"],
                'llm_model': "gpt-4o-mini",
                'embedding_model': "text-embedding-ada-002"
            }
        else:
            ragas_config = None

        use_llm_compressor_evaluator = getattr(args, 'use_llm_compressor_evaluator', False)
        if use_llm_compressor_evaluator:
            llm_model = getattr(args, "llm_evaluator_model", "gpt-4o")
            
            llm_compressor_config = {
                "llm_model": llm_model,
                "temperature": 0.0
            }
        else:
            llm_compressor_config = None

        n_trials = getattr(args, 'num_trials', None) or getattr(args, 'n_trials', None) or getattr(args, 'num_samples', None)

        optimizer = SMACRAGOptimizer(
            config_template=config_template,
            qa_data=qa_df,
            corpus_data=corpus_df,
            project_dir=args.project_dir,
            n_trials=n_trials,  
            sample_percentage=args.sample_percentage,
            cpu_per_trial=args.cpu_per_trial,
            retrieval_weight=args.retrieval_weight,
            generation_weight=args.generation_weight,
            use_cached_embeddings=getattr(args, 'use_cached_embeddings', True),
            result_dir=getattr(args, 'result_dir', None),
            study_name=getattr(args, 'study_name', None),
            walltime_limit=getattr(args, 'walltime_limit', 3600),
            n_workers=getattr(args, 'n_workers', 1),
            seed=getattr(args, 'seed', 42),
            early_stopping_threshold=getattr(args, 'early_stopping_threshold', 0.9),
            use_wandb=not args.no_wandb,
            wandb_project=getattr(args, 'wandb_project', "BO & AutoRAG"),
            wandb_entity=getattr(args, 'wandb_entity', None),
            wandb_run_name=getattr(args, 'wandb_run_name', None),
            optimizer=getattr(args, 'optimizer', 'smac'),
            use_multi_fidelity=getattr(args, 'use_multi_fidelity', False),
            min_budget_percentage=getattr(args, 'min_budget_percentage', 0.1),
            max_budget_percentage=getattr(args, 'max_budget_percentage', 1.0),
            eta=getattr(args, 'eta', 3),
            use_ragas=args.use_ragas,
            ragas_config=ragas_config,
            use_llm_compressor_evaluator=use_llm_compressor_evaluator,
            llm_evaluator_config=llm_compressor_config
        )
        
        return optimizer.optimize()

    def _run_componentwise_optimization(self, args, qa_df, corpus_df, config_template):
        from smac3.local_optimization.componentwise_smac_rag_optimizer import ComponentwiseSMACOptimizer

        use_llm_compressor_evaluator = getattr(args, 'use_llm_compressor_evaluator', False)
        
        print(f"[DEBUG] use_llm_compressor_evaluator from args: {use_llm_compressor_evaluator}")

        llm_compressor_config = {}
        if use_llm_compressor_evaluator:
            llm_model = getattr(args, "llm_evaluator_model",  "gpt-4o")
            
            llm_compressor_config = {
                "llm_model": llm_model,
                "temperature": getattr(args, "llm_compressor_temperature", 0.0)
            }
            print(f"[DEBUG] LLM compressor config: {llm_compressor_config}")

        optimizer = ComponentwiseSMACOptimizer(
            config_template=config_template,
            qa_data=qa_df,
            corpus_data=corpus_df,
            project_dir=args.project_dir,
            n_trials_per_component=args.n_trials,
            sample_percentage=args.sample_percentage,
            cpu_per_trial=args.cpu_per_trial,
            retrieval_weight=args.retrieval_weight,
            generation_weight=args.generation_weight,
            use_cached_embeddings=args.use_cached_embeddings,
            result_dir=args.result_dir,
            study_name=args.study_name,
            walltime_limit_per_component=args.walltime_limit,
            n_workers=args.n_workers,
            seed=args.seed,
            early_stopping_threshold=args.early_stopping_threshold,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            optimizer=args.optimizer,
            use_multi_fidelity=args.use_multi_fidelity,
            min_budget_percentage=args.min_budget_percentage,
            max_budget_percentage=args.max_budget_percentage,
            eta=args.eta,
            use_llm_compressor_evaluator=use_llm_compressor_evaluator,
            llm_evaluator_config=llm_compressor_config,
        )
        
        return optimizer.optimize()
    
    def run(self, argv=None) -> int:
        args = self.parser.parse_args(argv)
        
        if args.optimizer.lower() == 'bohb' and not args.use_multi_fidelity:
            print(f"\nNote: BOHB is a multi-fidelity algorithm. Auto-enabling multi-fidelity.")
            args.use_multi_fidelity = True
        
        args.retrieval_weight, args.generation_weight = self._normalize_weights(
            args.retrieval_weight, args.generation_weight
        )
        
        self._resolve_paths(args)
        
        self._setup_wandb_project(args)
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION MODE: {args.optimization_mode.upper()}")
        print(f"{'='*60}")
        print(f"Using project root: {self.project_root}")
        print(f"Using config: {args.config_path}")
        print(f"Using project dir: {args.project_dir}")
        print(f"Results will be saved to: {args.result_dir}")
        
        if args.optimization_mode == 'global':
            print(f"Optimizer: {args.optimizer.upper()}")
            print(f"Multi-fidelity: {'Enabled' if args.use_multi_fidelity else 'Disabled'}")
            if args.use_ragas:
                print(f"Evaluation: RAGAS with {args.ragas_llm_model}")
            else:
                print(f"Evaluation: Traditional component-wise metrics")
        else:
            print(f"Optimizing components sequentially")
            if args.n_trials:
                print(f"Trials per component: {args.n_trials}")
            else:
                print(f"Trials per component: Auto-calculated")
        
        if not self._validate_files(args):
            return 1
        
        os.makedirs(args.project_dir, exist_ok=True)
        
        qa_path, corpus_path = Utils.get_centralized_data_paths(args.project_dir)
        qa_df, corpus_df = self._load_data(qa_path, corpus_path)
        
        import yaml
        with open(args.config_path, 'r') as f:
            config_template = yaml.safe_load(f)
        
        if not args.no_wandb:
            try:
                import wandb
                if wandb.api.api_key is None:
                    print("\nYou are not logged into Weights & Biases!")
                    print("Continuing without W&B tracking...")
                    args.no_wandb = True
                else:
                    print("W&B authentication successful")
            except ImportError:
                print("wandb package not installed!")
                args.no_wandb = True
        
        email_notifier = self._setup_email_notifier(args) if args.email_notifications else None
        
        experiment_name = self._create_experiment_name(args)
        
        start_time = time.time()
        
        print(f"\nStarting {args.optimization_mode} optimization...")
        
        
        if args.optimization_mode == 'componentwise':
            if email_notifier:
                from smac3.local_optimization.componentwise_smac_rag_optimizer import ComponentwiseSMACOptimizer

                use_llm_compressor_evaluator = getattr(args, 'use_llm_compressor_evaluator', False)
                llm_compressor_config = {}
                if use_llm_compressor_evaluator:
                    llm_model = getattr(args, "llm_evaluator_model", "gpt-4o")
                    
                    llm_compressor_config = {
                        "llm_model": llm_model,
                        "temperature": getattr(args, "llm_compressor_temperature", 0.0)
                    }
                
                optimizer = ComponentwiseSMACOptimizer(
                    config_template=config_template,
                    qa_data=qa_df,
                    corpus_data=corpus_df,
                    project_dir=args.project_dir,
                    n_trials_per_component=args.n_trials,
                    sample_percentage=args.sample_percentage,
                    cpu_per_trial=args.cpu_per_trial,
                    retrieval_weight=args.retrieval_weight,
                    generation_weight=args.generation_weight,
                    use_cached_embeddings=args.use_cached_embeddings,
                    result_dir=args.result_dir,
                    study_name=args.study_name,
                    walltime_limit_per_component=args.walltime_limit,
                    n_workers=args.n_workers,
                    seed=args.seed,
                    early_stopping_threshold=args.early_stopping_threshold,
                    use_wandb=not args.no_wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    optimizer=args.optimizer,
                    use_multi_fidelity=args.use_multi_fidelity,
                    min_budget_percentage=args.min_budget_percentage,
                    max_budget_percentage=args.max_budget_percentage,
                    eta=args.eta,
                    use_llm_compressor_evaluator=use_llm_compressor_evaluator, 
                    llm_evaluator_config=llm_compressor_config 
                )
                wrapper = ExperimentNotificationWrapper(optimizer, email_notifier)
                best_results = wrapper.run_with_notification(experiment_name=experiment_name)
            else:
                best_results = self._run_componentwise_optimization(args, qa_df, corpus_df, config_template)
        else:
            if email_notifier:
                from smac3.global_optimization.smac_rag_optimizer import SMACRAGOptimizer
                
                use_llm_compressor_evaluator = getattr(args, 'use_llm_compressor_evaluator', False)
                llm_compressor_config = {}
                if use_llm_compressor_evaluator:
                    llm_model = getattr(args, "llm_evaluator_model", "gpt-4o")
                    
                    llm_compressor_config = {
                        "llm_model": llm_model,
                        "temperature": getattr(args, "llm_compressor_temperature", 0.0)
                    }
                    
                ragas_config = self._prepare_ragas_config(args)
                optimizer = SMACRAGOptimizer(
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
                    n_workers=args.n_workers,
                    seed=args.seed,
                    early_stopping_threshold=args.early_stopping_threshold,
                    use_wandb=not args.no_wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_run_name=args.wandb_run_name,
                    optimizer=args.optimizer,
                    use_multi_fidelity=args.use_multi_fidelity,
                    min_budget_percentage=args.min_budget_percentage,
                    max_budget_percentage=args.max_budget_percentage,
                    eta=args.eta,
                    use_ragas=args.use_ragas,
                    ragas_config=ragas_config,
                    use_llm_compressor_evaluator=use_llm_compressor_evaluator, 
                    llm_evaluator_config=llm_compressor_config
                )
                wrapper = ExperimentNotificationWrapper(optimizer, email_notifier)
                best_results = wrapper.run_with_notification(experiment_name=experiment_name)
            else:
                best_results = self._run_global_optimization(args, qa_df, corpus_df, config_template)
                
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{'='*60}")
        print(f"{args.optimization_mode.upper()} OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total optimization time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        if args.optimization_mode == 'componentwise':
            if 'component_results' in best_results:
                print("\nBest configuration for each component:")
                for component, result in best_results['component_results'].items():
                    print(f"\n{component.upper()}:")
                    print(f"  Best score: {result.get('best_score', 0.0):.4f}")
                    print(f"  Trials run: {result.get('n_trials', 0)}")
                    print(f"  Best config: {result.get('best_config', {})}")
        else:
            if best_results.get('best_config'):
                print(f"\nBest configuration found:")
                print(f"  Score: {best_results['best_config']['score']:.4f}")
                print(f"  Latency: {best_results['best_config']['latency']:.2f}s")
                if 'budget' in best_results['best_config']:
                    print(f"  Budget: {best_results['best_config']['budget']} samples")
                print(f"  Config: {best_results['best_config']['config']}")
        
        print(f"\nResults saved to: {args.result_dir}")
        
        return 0
    
    def _setup_email_notifier(self, args) -> Optional[ExperimentEmailNotifier]:
        try:
            notifier = ExperimentEmailNotifier()
            print(f"\nEmail notifications enabled. Will send to: {', '.join(notifier.recipient_emails)}")
            return notifier
        except ValueError as e:
            print(f"\n{'='*60}")
            print("ERROR: Email notification setup failed!")
            print(f"{'='*60}")
            print(f"{e}")
            print("\nTo use email notifications, please set the following environment variables:")
            print("  EMAIL_SENDER: Your Gmail address")
            print("  EMAIL_PASSWORD: Your Gmail app password (not regular password)")
            print("  EMAIL_RECIPIENTS: Comma-separated recipient emails (optional)")
            print("\nExample:")
            print("  export EMAIL_SENDER='your.email@gmail.com'")
            print("  export EMAIL_PASSWORD='your-app-password'")
            print("  export EMAIL_RECIPIENTS='recipient1@example.com,recipient2@example.com'")
            print(f"{'='*60}")
            sys.exit(1)
        except Exception as e:
            print(f"\n{'='*60}")
            print("ERROR: Unexpected error in email notification setup!")
            print(f"{'='*60}")
            print(f"{e}")
            print("\nPlease check your email configuration and try again.")
            print(f"{'='*60}")
            sys.exit(1)
    
    def _create_experiment_name(self, args) -> str:
        if args.study_name:
            return args.study_name
        
        if args.optimization_mode == 'componentwise':
            mode_name = "Component-wise"
        else:
            mode_name = args.optimizer.upper()
            if args.use_multi_fidelity:
                mode_name += " Multi-Fidelity"
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{mode_name} RAG Optimization - {timestamp}"


def main():
    runner = UnifiedSMACRunner()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
