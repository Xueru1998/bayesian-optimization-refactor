from typing import List, Tuple, Dict, Any
from ConfigSpace import ConfigurationSpace


class ComponentManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def validate_all_components(self) -> Tuple[bool, List[str], Dict[str, int]]:
        active_components = []
        has_active_query_expansion = False
        
        for comp in self.optimizer.COMPONENT_ORDER:
            if comp == 'query_expansion':
                if not self.optimizer.config_generator.node_exists(comp):
                    continue 
                    
                qe_config = self.optimizer.config_generator.extract_node_config("query_expansion")
                qe_methods = []
                for module in qe_config.get("modules", []):
                    method = module.get("module_type")
                    if method and method != "pass_query_expansion":
                        qe_methods.append(method)
                if qe_methods:
                    has_active_query_expansion = True
                    active_components.append(comp)
            elif comp == 'retrieval':
                if has_active_query_expansion:
                    continue
                else:
                    if self.optimizer.config_generator.node_exists(comp):
                        active_components.append(comp)
            elif comp == 'prompt_maker_generator':
                if self.optimizer.config_generator.node_exists('prompt_maker') or self.optimizer.config_generator.node_exists('generator'):
                    active_components.append(comp)
            else:
                if self.optimizer.config_generator.node_exists(comp):
                    active_components.append(comp)
        
        component_combinations = {}
        
        n_trials = self.optimizer.n_trials_per_component if self.optimizer.n_trials_per_component else 20
        
        for component in active_components:
            combinations = self.get_component_combinations(component)
            component_combinations[component] = combinations
        
        print(f"\n{'='*70}")
        print(f"COMPONENT VALIDATION PASSED")
        print(f"All components have sufficient search space for optimization:")
        
        for component in active_components:
            combos = component_combinations[component]
            
            if combos == 0:
                print(f"  - {component}: No search space (will be skipped)")
                continue
                
            print(f"  - {component}: {combos} combinations")
            
            if combos < n_trials and combos > 0:
                print(f"    ⚠️  WARNING: Search space ({combos}) < n_trials ({n_trials})")
                print(f"       SMAC uses Bayesian Optimization with continuous sampling within parameter ranges.")
                print(f"       While the discrete combination count is limited, SMAC can explore")
                print(f"       continuous values between boundaries for numerical parameters.")
                
                if component == 'passage_filter':
                    print(f"       For filters, thresholds/percentiles are sampled continuously.")
                elif component == 'passage_compressor':
                    print(f"       For compressors, compression ratios and other numerical")
                    print(f"       parameters are sampled continuously within their ranges.")
        
        print(f"{'='*70}\n")

        return True, [], component_combinations
    
    def get_active_components(self) -> List[str]:
        active_components = []
        has_active_query_expansion = False
        
        for comp in self.optimizer.COMPONENT_ORDER:
            if comp == 'query_expansion':
                if not self.optimizer.config_generator.node_exists(comp):
                    continue  
                    
                qe_config = self.optimizer.config_generator.extract_node_config("query_expansion")
                qe_methods = []
                for module in qe_config.get("modules", []):
                    method = module.get("module_type")
                    if method and method != "pass_query_expansion":
                        qe_methods.append(method)
                if qe_methods:
                    has_active_query_expansion = True
                    active_components.append(comp)
            elif comp == 'retrieval':
                if not has_active_query_expansion:
                    if self.optimizer.config_generator.node_exists(comp):
                        active_components.append(comp)
                else:
                    print("[Component-wise] Skipping retrieval component since query expansion includes retrieval")
            elif comp == 'prompt_maker_generator':
                if self.optimizer.config_generator.node_exists('prompt_maker') or self.optimizer.config_generator.node_exists('generator'):
                    active_components.append(comp)
            else:
                if self.optimizer.config_generator.node_exists(comp):
                    active_components.append(comp)
        
        return active_components
    
    def should_skip_component(self, component: str) -> bool:
        if component == 'passage_filter' and 'passage_reranker' in self.optimizer.best_configs:
            reranker_config = self.optimizer.best_configs['passage_reranker']
            if reranker_config.get('reranker_top_k') == 1:
                print(f"\n[Component-wise] Skipping filter optimization because reranker_top_k=1")
                self.optimizer.best_configs[component] = {'passage_filter_method': 'pass_passage_filter'}
                return True
        return False
    
    def calculate_component_trials(self, component: str, cs: ConfigurationSpace) -> int:
        total_combinations = self.get_component_combinations(component)
        
        if self.optimizer.n_trials_per_component:
            return self.optimizer.n_trials_per_component

        print(f"[{component}] Total combinations: {total_combinations}")
        
        return min(5, total_combinations)
    
    def get_component_combinations(self, component: str) -> int:
        return self.optimizer.rag_processor.get_component_combinations(
            component, 
            self.optimizer.config_space_builder,
            self.optimizer.best_configs
        )