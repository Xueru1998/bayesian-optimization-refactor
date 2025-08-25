import math
from typing import Dict, Any, Tuple


class SearchSpaceHandler:
    
    def _calculate_trials_if_needed(self):
        if self.n_trials is None:
            search_space_size, note = self._calculate_total_search_space()
            
            if search_space_size > 500:
                log_combinations = math.log10(search_space_size)
                log_max = math.log10(500)
                suggested_samples = int(20 + (50 - 20) * min(log_combinations / log_max, 1.0))
                self.n_trials = 50
                reasoning = f"Large search space detected ({search_space_size:,} combinations), using max samples (50) for better coverage. {note}"
            else:
                suggested_samples = max(20, int(search_space_size * self.sample_percentage))
                self.n_trials = min(suggested_samples, 50)
                reasoning = f"Auto-calculated based on {self.sample_percentage*100}% of {search_space_size} total combinations. {note}"
            
            print(f"Auto-calculated num_trials: {self.n_trials}")
            print(f"Reasoning: {reasoning}")
        else:
            print(f"Using provided num_trials: {self.n_trials}")
            
    def _calculate_total_search_space(self) -> Tuple[int, str]:
        components = [
            'query_expansion', 'retrieval', 'passage_filter',
            'passage_reranker', 'passage_compressor', 'prompt_maker_generator'
        ]
        
        total_combinations = 1
        combination_note = ""
        
        has_active_qe = False
        if self.config_generator.node_exists("query_expansion"):
            qe_config = self.config_generator.extract_node_config("query_expansion")
            if qe_config and qe_config.get("modules", []):
                for module in qe_config.get("modules", []):
                    if module.get("module_type") != "pass_query_expansion":
                        has_active_qe = True
                        break
        
        for component in components:
            if component == 'retrieval' and has_active_qe:
                continue
            
            combos, note = self.combination_calculator.calculate_component_combinations(component)
            combination_note = note
            
            if combos > 0:
                total_combinations *= combos
        
        return total_combinations, combination_note
    
    def _get_search_space_summary(self) -> Dict[str, Any]:
        components = [
            'query_expansion', 'retrieval', 'passage_filter',
            'passage_reranker', 'passage_compressor', 'prompt_maker_generator'
        ]
        
        summary = {}
        total_combinations = 1
        combination_note = ""
        
        has_active_qe = False
        if self.config_generator.node_exists("query_expansion"):
            qe_config = self.config_generator.extract_node_config("query_expansion")
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
            
            combos, note = self.combination_calculator.calculate_component_combinations(component)
            combination_note = note
            
            config = None
            if component == 'query_expansion':
                config = self.config_generator.extract_node_config("query_expansion")
            elif component == 'retrieval':
                config = self.config_generator.extract_retrieval_options()
            elif component == 'prompt_maker_generator':
                config = {
                    'prompt_maker': self.config_generator.extract_node_config("prompt_maker"),
                    'generator': self.config_generator.extract_node_config("generator")
                }
            else:
                config = self.config_generator.extract_node_config(component.replace('_', '-'))
            
            summary[component] = {
                'combinations': combos,
                'config': config
            }
            
            if combos > 0:
                total_combinations *= combos
        
        summary['search_space_size'] = total_combinations
        summary['combination_note'] = combination_note
        
        return summary