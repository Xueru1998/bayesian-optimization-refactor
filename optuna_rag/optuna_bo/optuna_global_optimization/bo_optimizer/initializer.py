from pipeline.pipeline_runner.rag_pipeline_runner import RAGPipelineRunner


class PipelineInitializer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def initialize_metrics(self):
        opt = self.optimizer
        opt.retrieval_metrics = opt.config_generator.extract_retrieval_metrics_from_config()
        
        opt.query_expansion_metrics = []
        if opt.config_generator.node_exists("query_expansion"):
            opt.query_expansion_metrics = opt.config_generator.extract_query_expansion_metrics_from_config()
        
        opt.filter_metrics = []
        if opt.config_generator.node_exists("passage_filter"):
            opt.filter_metrics = opt.config_generator.extract_passage_filter_metrics_from_config()
        
        opt.compressor_metrics = []
        if opt.config_generator.node_exists("passage_compressor"):
            opt.compressor_metrics = opt.config_generator.extract_passage_compressor_metrics_from_config()
        
        opt.reranker_metrics = []
        if opt.config_generator.node_exists("passage_reranker"):
            opt.reranker_metrics = opt.config_generator.extract_metrics_from_config(node_type='passage_reranker')
        
        opt.generation_metrics = []
        if opt.config_generator.node_exists("generator"):
            opt.generation_metrics = opt.config_generator.extract_generation_metrics_from_config()
        
        opt.prompt_maker_metrics = []
        if opt.config_generator.node_exists("prompt_maker"):
            opt.prompt_maker_metrics = opt.config_generator.extract_generation_metrics_from_config(node_type='prompt_maker')

    def initialize_pipeline_runner(self):
        opt = self.optimizer
        early_stopping_thresholds = opt.component_early_stopping_thresholds if opt.component_early_stopping_enabled else None
        
        if opt.use_ragas:
            ragas_config = {
                'llm_model': opt.ragas_llm_model,
                'embedding_model': opt.ragas_embedding_model,
                'retrieval_metrics': opt.ragas_metrics.get('retrieval', []),
                'generation_metrics': opt.ragas_metrics.get('generation', [])
            }
            
            return RAGPipelineRunner(
                config_generator=opt.config_generator,
                retrieval_metrics=opt.retrieval_metrics,
                filter_metrics=opt.filter_metrics,
                compressor_metrics=opt.compressor_metrics,
                generation_metrics=opt.generation_metrics,
                prompt_maker_metrics=opt.prompt_maker_metrics,
                query_expansion_metrics=opt.query_expansion_metrics,
                reranker_metrics=opt.reranker_metrics,
                retrieval_weight=opt.retrieval_weight,
                generation_weight=opt.generation_weight,
                use_ragas=True,
                ragas_config=ragas_config,
                use_llm_compressor_evaluator=opt.use_llm_compressor_evaluator,
                llm_evaluator_model=opt.llm_evaluator_model,
                early_stopping_thresholds=early_stopping_thresholds  
            )
        else:
            return RAGPipelineRunner(
                config_generator=opt.config_generator,
                retrieval_metrics=opt.retrieval_metrics,
                filter_metrics=opt.filter_metrics,
                compressor_metrics=opt.compressor_metrics,
                generation_metrics=opt.generation_metrics,
                prompt_maker_metrics=opt.prompt_maker_metrics,
                query_expansion_metrics=opt.query_expansion_metrics,
                reranker_metrics=opt.reranker_metrics,
                retrieval_weight=opt.retrieval_weight,
                generation_weight=opt.generation_weight,
                use_llm_compressor_evaluator=opt.use_llm_compressor_evaluator,
                llm_evaluator_model=opt.llm_evaluator_model,
                early_stopping_thresholds=early_stopping_thresholds 
            )

    def print_initialization_summary(self):
        opt = self.optimizer
        summary = {}
        components = ['query_expansion', 'retrieval', 'passage_filter', 'passage_reranker', 
                    'passage_compressor', 'prompt_maker_generator']
        
        for component in components:
            combinations, note = opt.search_space_calculator.calculate_component_combinations(component)
            summary[component] = {
                'combinations': combinations,
                'note': note
            }
        
        total_combinations = opt.search_space_calculator.calculate_total_combinations()
        
        print("\n===== RAG Pipeline Optimizer Initialized =====")
        print(f"Using {opt.n_trials} trials with {opt.optimizer.upper()} sampler")

        if opt.optimizer == "random":
            print(f"\nNOTE: Early stopping is DISABLED for random sampler (both high-score and low-score)")
        else:
            if opt.component_early_stopping_enabled:
                print(f"\nComponent-level early stopping ENABLED with thresholds:")
                for component, threshold in opt.component_early_stopping_thresholds.items():
                    print(f"  {component}: {threshold}")
            else:
                print(f"\nComponent-level early stopping DISABLED")
            
            print(f"\nHigh-score early stopping threshold: {opt.early_stopping_threshold}")
        
        if opt.use_ragas:
            print(f"\nEvaluation Method: RAGAS")
            print(f"  LLM Model: {opt.ragas_llm_model}")
            print(f"  Embedding Model: {opt.ragas_embedding_model}")
            print(f"  Metrics: {list(opt.ragas_metrics.get('retrieval', [])) + list(opt.ragas_metrics.get('generation', []))}")
        else:
            print(f"\nEvaluation Method: Traditional (component-wise)")
        
        for component in components:
            if summary[component]['combinations'] > 0:
                print(f"\n{component.replace('_', ' ').title()}:")
                
                if component == "retrieval":
                    print(f"  Metrics: {opt.retrieval_metrics}")
                elif component == "query_expansion" and opt.query_expansion_metrics:
                    print(f"  Metrics: {opt.query_expansion_metrics}")
                elif component == "passage_filter" and opt.filter_metrics:
                    print(f"  Metrics: {opt.filter_metrics}")
                elif component == "passage_reranker" and opt.reranker_metrics:
                    print(f"  Metrics: {opt.reranker_metrics}")
                elif component == "passage_compressor" and opt.compressor_metrics:
                    print(f"  Metrics: {opt.compressor_metrics}")
                elif component == "prompt_maker_generator":
                    if opt.generation_metrics:
                        print(f"  Generator Metrics: {opt.generation_metrics}")
                        
        print(f"\nScore weights - Retrieval: {opt.retrieval_weight}, Generation: {opt.generation_weight}")
        
        print("\nSearch space summary:")
        for param, values in opt.search_space.items():
            if isinstance(values, list):
                print(f"  {param}: {len(values)} options (categorical)")
            elif isinstance(values, tuple):
                print(f"  {param}: ({values[0]}, {values[1]}) (continuous)")