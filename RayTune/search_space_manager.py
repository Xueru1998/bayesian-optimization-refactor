from typing import Dict, Any, List
import numpy as np
from ray import tune
from pipeline.search_space_calculator import SearchSpaceCalculator
from pipeline.search_space_extractor import UnifiedSearchSpaceExtractor


class SearchSpaceManager:
    def __init__(self, config_generator):
        self.config_generator = config_generator
        self.calculator = SearchSpaceCalculator(config_generator)
        self.unified_extractor = UnifiedSearchSpaceExtractor(config_generator)
    
    def define_search_space(self) -> Dict[str, Any]:
        return self.unified_extractor.extract_search_space('bohb')
    
    def calculate_total_combinations(self) -> int:
        return self.calculator.calculate_total_combinations()
    
    def suggest_num_samples(self, sample_percentage: float = 0.1) -> Dict[str, Any]:
        return self.calculator.suggest_num_samples(sample_percentage)
    
    def get_search_space_summary(self) -> Dict[str, Dict[str, Any]]:
        return self.calculator.get_search_space_summary()
    
    def print_search_space_summary(self, search_space, num_samples, sample_percentage, max_concurrent, cpu_per_trial, retrieval_weight, generation_weight):
        has_active_qe = self._has_active_query_expansion()
        
        print(f"\n===== Starting optimization with {num_samples} samples =====")
        print(f"Search space parameters:")
        
        main_params = []
        conditional_params = []
        
        for param_name, param_value in search_space.items():
            if hasattr(param_value, "_sample_func"):
                conditional_params.append(param_name)
            else:
                main_params.append((param_name, param_value))
        
        for param_name, param_value in main_params:
            if hasattr(param_value, "categories"):
                categories = param_value.categories
                if param_name == "retriever_top_k" and len(categories) > 5:
                    print(f"  • {param_name}: range[{min(categories)}, {max(categories)}]")
                else:
                    print(f"  • {param_name}: {categories}")
            elif hasattr(param_value, "lower") and hasattr(param_value, "upper"):
                print(f"  • {param_name}: range[{param_value.lower}, {param_value.upper}]")
            else:
                print(f"  • {param_name}: {param_value}")

        if has_active_qe:
            print(f"\n⚠️  Query Expansion Configuration:")
            print("  • Query expansion has active methods - retrieval is included within QE")
            print("  • Separate retrieval node will be skipped when QE method != 'pass_query_expansion'")
        
        if conditional_params:
            print("\nConditional parameters:")
            
            if has_active_qe:
                qe_retrieval_options = self.config_generator.extract_query_expansion_retrieval_options()
                if qe_retrieval_options:
                    if "query_expansion_retrieval_method" in conditional_params:
                        print(f"  • query_expansion_retrieval_method: {qe_retrieval_options.get('methods', [])} (when QE is active)")
                    if "query_expansion_bm25_tokenizer" in conditional_params:
                        print(f"  • query_expansion_bm25_tokenizer: {qe_retrieval_options.get('bm25_tokenizers', [])} (when QE active & retrieval=bm25)")
                    if "query_expansion_vectordb_name" in conditional_params:
                        print(f"  • query_expansion_vectordb_name: {qe_retrieval_options.get('vectordb_names', [])} (when QE active & retrieval=vectordb)")
            
            if not has_active_qe:
                retrieval_options = self.config_generator.extract_retrieval_options()
                if "bm25_tokenizer" in conditional_params:
                    print(f"  • bm25_tokenizer: {retrieval_options.get('bm25_tokenizers', [])} (when retrieval_method=bm25)")
                if "vectordb_name" in conditional_params:
                    print(f"  • vectordb_name: {retrieval_options.get('vectordb_names', [])} (when retrieval_method=vectordb)")
            
            qe_options = self.config_generator.extract_query_expansion_options()
            if "query_expansion_llm" in conditional_params:
                print(f"  • query_expansion_llm: {qe_options.get('llms', [])} (when QE uses LLM methods)")
            if "query_expansion_model" in conditional_params:
                print(f"  • query_expansion_model: {qe_options.get('models', [])} (when QE uses LLM methods)")
            if "query_expansion_temperature" in conditional_params:
                print(f"  • query_expansion_temperature: {qe_options.get('temperatures', [0.5, 0.7, 0.9])} (when method=multi_query_expansion)")
            if "query_expansion_max_token" in conditional_params:
                print(f"  • query_expansion_max_token: {qe_options.get('max_tokens', [32, 64, 128])} (when method=hyde)")
            
            reranker_options = self.config_generator.extract_passage_reranker_options()
            if "reranker_model_name" in conditional_params:
                print(f"  • reranker_model_name: varies by reranker method (when using model-based rerankers)")
            if "reranker_top_k" in conditional_params:
                print(f"  • reranker_top_k: {reranker_options.get('top_k_values', [])} (when reranker != pass_reranker)")
            
            if "threshold" in conditional_params or "percentile" in conditional_params:
                filter_config = self.config_generator.extract_node_config("passage_filter")
                if filter_config:
                    filter_details = []
                    for module in filter_config.get("modules", []):
                        module_type = module.get("module_type")
                        if module_type in ["threshold_cutoff", "similarity_threshold_cutoff"] and "threshold" in module:
                            vals = module["threshold"] if isinstance(module["threshold"], list) else [module["threshold"]]
                            filter_details.append(f"{module_type}: [{min(vals)}, {max(vals)}]")
                        elif module_type in ["percentile_cutoff", "similarity_percentile_cutoff"] and "percentile" in module:
                            vals = module["percentile"] if isinstance(module["percentile"], list) else [module["percentile"]]
                            filter_details.append(f"{module_type}: [{min(vals)}, {max(vals)}]")
                    
                    if filter_details:
                        print(f"  • threshold/percentile: varies by filter method - {', '.join(filter_details)}")
            
            compressor_options = self.config_generator.extract_passage_compressor_options()
            if "compressor_llm" in conditional_params:
                print(f"  • compressor_llm: {compressor_options.get('llms', [])} (when using LLM-based compression)")
            if "compressor_model" in conditional_params:
                print(f"  • compressor_model: {compressor_options.get('models', [])} (when using LLM-based compression)")
            
            if "prompt_template_idx" in conditional_params:
                _, prompt_indices = self.config_generator.extract_prompt_maker_options()
                print(f"  • prompt_template_idx: {prompt_indices} (for fstring/long_context_reorder methods)")
        
        print("\nOptimization Configuration:")
        print(f"  • Primary metric: combined_score")
        print(f"  • Multi-fidelity: Yes (budget ranges from 33% to 100% of data)")
        print(f"  • Scheduler: AsyncHyperBand (max_t=3, grace_period=1, eta=3)")
        print(f"  • Search algorithm: BOHB (Bayesian Optimization HyperBand)")
        print(f"  • Concurrent trials: {max_concurrent}")
        print(f"  • CPUs per trial: {cpu_per_trial}")
        print(f"  • Score weights - Retrieval: {retrieval_weight}, Generation: {generation_weight}")
        print(f"  • Sample size: {num_samples} ({sample_percentage*100:.1f}% of estimated combinations)")
        
        if has_active_qe:
            print(f"\n Pipeline Flow: Query → Query Expansion (with Retrieval) → Reranker → Filter → Compressor → Prompt → Generator")
        else:
            print(f"\n Pipeline Flow: Query → Retrieval → Reranker → Filter → Compressor → Prompt → Generator")
    
    def _has_active_query_expansion(self) -> bool:
        query_expansion_config = self.config_generator.extract_node_config("query_expansion")
        if not query_expansion_config or not query_expansion_config.get("modules", []):
            return False
        
        for module in query_expansion_config.get("modules", []):
            if module.get("module_type") != "pass_query_expansion":
                return True
        return False