from typing import Dict, Any
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
        runner_params = {
            'config_generator': opt.config_generator,
            'retrieval_metrics': opt.retrieval_metrics,
            'filter_metrics': opt.filter_metrics,
            'compressor_metrics': opt.compressor_metrics,
            'generation_metrics': opt.generation_metrics,
            'prompt_maker_metrics': opt.prompt_maker_metrics,
            'query_expansion_metrics': opt.query_expansion_metrics,
            'reranker_metrics': opt.reranker_metrics,
            'retrieval_weight': opt.retrieval_weight,
            'generation_weight': opt.generation_weight,
            'use_llm_evaluator': opt.use_llm_compressor_evaluator,
            'llm_evaluator_config': opt.llm_evaluator_config,
            'early_stopping_thresholds': opt.early_stopping_thresholds
        }
        
        if opt.use_ragas:
            ragas_config = {
                'llm_model': opt.ragas_llm_model,
                'embedding_model': opt.ragas_embedding_model,
                'retrieval_metrics': opt.ragas_metrics.get('retrieval', []),
                'generation_metrics': opt.ragas_metrics.get('generation', [])
            }
            runner_params['use_ragas'] = True
            runner_params['ragas_config'] = ragas_config
        
        return RAGPipelineRunner(**runner_params)

    def print_initialization_summary(self):
        opt = self.optimizer
        summary = self._get_search_space_summary_with_calculator()
        
        print("\n===== RAG Pipeline Optimizer Initialized =====")
        print(f"Using {opt.n_trials} trials with {opt.optimizer.upper()} sampler")
        print(f"Total search space combinations (estimated): {summary['search_space_size']}")
        print(f"Note: {summary['combination_note']}")
        
        if opt.use_ragas:
            print(f"\nEvaluation Method: RAGAS")
            print(f"  LLM Model: {opt.ragas_llm_model}")
            print(f"  Embedding Model: {opt.ragas_embedding_model}")
            print(f"  Metrics: {list(opt.ragas_metrics.get('retrieval', [])) + list(opt.ragas_metrics.get('generation', []))}")
        else:
            print(f"\nEvaluation Method: Traditional (component-wise)")

        for component, info in summary.items():
            if component not in ["search_space_size", "combination_note"] and info['combinations'] > 1:
                print(f"\n{component.title()}:")
                print(f"  Combinations (estimated): {info['combinations']}")
                
                if component == "retrieval":
                    print(f"  Metrics: {opt.retrieval_metrics}")
                elif component == "query_expansion" and opt.query_expansion_metrics:
                    print(f"  Metrics: {opt.query_expansion_metrics}")
                elif component == "filter" and opt.filter_metrics:
                    print(f"  Metrics: {opt.filter_metrics}")
                elif component == "reranker" and opt.reranker_metrics:
                    print(f"  Metrics: {opt.reranker_metrics}")
                elif component == "compressor" and opt.compressor_metrics:
                    print(f"  Metrics: {opt.compressor_metrics}")
                elif component == "prompt_maker" and opt.prompt_maker_metrics:
                    print(f"  Metrics: {opt.prompt_maker_metrics}")
                elif component == "generator" and opt.generation_metrics:
                    print(f"  Metrics: {opt.generation_metrics}")
        
        print(f"\nScore weights - Retrieval: {opt.retrieval_weight}, Generation: {opt.generation_weight}")

        print("\nSearch space summary:")
        continuous_params = []
        categorical_params = []
        
        for param, values in opt.search_space.items():
            if isinstance(values, list):
                categorical_params.append(f"  {param}: {len(values)} options (categorical)")
            elif isinstance(values, tuple):
                continuous_params.append(f"  {param}: ({values[0]}, {values[1]}) (continuous)")
        
        if categorical_params:
            print("Categorical parameters:")
            for param in categorical_params:
                print(param)
        
        if continuous_params:
            print("Continuous parameters (BO will explore within ranges):")
            for param in continuous_params:
                print(param)
        
        if not opt.disable_early_stopping:
            print("\nEarly stopping enabled with thresholds:")
            for component, threshold in opt.early_stopping_thresholds.items():
                print(f"  {component}: < {threshold}")
        else:
            print("\nEarly stopping: DISABLED")

    def _get_search_space_summary_with_calculator(self) -> Dict[str, Any]:
        opt = self.optimizer
        components = [
            'query_expansion', 'retrieval', 'passage_filter',
            'passage_reranker', 'passage_compressor', 'prompt_maker_generator'
        ]
        
        summary = {}
        total_combinations = 1
        combination_note = ""

        has_active_qe = False
        if opt.config_generator.node_exists("query_expansion"):
            qe_config = opt.config_generator.extract_node_config("query_expansion")
            if qe_config and qe_config.get("modules", []):
                for module in qe_config.get("modules", []):
                    if module.get("module_type") != "pass_query_expansion":
                        has_active_qe = True
                        break
        
        for component in components:
            if component == 'retrieval' and has_active_qe:
                summary[component] = {
                    'combinations': 0,
                    'config': None,
                    'skipped_when_qe_active': True
                }
                continue
            
            combos, note = opt.combination_calculator.calculate_component_combinations(component)
            combination_note = note
            
            config = None
            if component == 'query_expansion':
                config = opt.config_generator.extract_node_config("query_expansion")
            elif component == 'retrieval':
                config = opt.config_generator.extract_retrieval_options()
            else:
                config = opt.config_generator.extract_node_config(component.replace('_', '-'))
            
            summary[component] = {
                'combinations': combos,
                'config': config,
                'includes_retrieval': (component == 'query_expansion' and has_active_qe)
            }
            
            if combos > 0:
                total_combinations *= combos
        
        summary['search_space_size'] = total_combinations
        summary['combination_note'] = combination_note
        
        return summary