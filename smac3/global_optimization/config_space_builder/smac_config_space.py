from typing import Dict, Any
from ConfigSpace import ConfigurationSpace
from pipeline.config_manager import ConfigGenerator

from .component_extractors import ComponentExtractors
from .parameter_and_conditions import ParameterAndConditions
from .config_utilities import ConfigUtilities


class SMACConfigSpaceBuilder(ComponentExtractors):
    
    def __init__(self, config_generator: ConfigGenerator, seed: int = 42):
        super().__init__(config_generator)
        self.config_generator = config_generator
        self.seed = seed
        
        class UnifiedExtractor:
            def __init__(self, parent):
                self.parent = parent
            
            def extract_search_space(self, format_type='smac'):
                return self.parent.get_unified_space()
        
        self.unified_extractor = UnifiedExtractor(self)
        self.query_expansion_retrieval_options = config_generator.extract_query_expansion_retrieval_options()
        
        self.param_condition_handler = ParameterAndConditions()
        self.config_utilities = ConfigUtilities()
    
    def build_configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        
        unified_space = self.unified_extractor.extract_search_space('smac')
        
        for param_name, param_info in unified_space.items():
            if param_info.get('type') == 'categorical' and 'values' in param_info:
                original_values = param_info['values']
                unique_values = list(dict.fromkeys(original_values))
                if len(unique_values) < len(original_values):
                    print(f"[WARNING] Removed duplicates from {param_name}: {original_values} -> {unique_values}")
                    param_info['values'] = unique_values

        if 'query_expansion_method' in unified_space:
            if 'retrieval_method' in unified_space:
                unified_space['retrieval_method']['condition'] = ('query_expansion_method', ['pass_query_expansion'])
            
            if 'bm25_tokenizer' in unified_space:
                unified_space['bm25_tokenizer']['condition'] = [
                    ('query_expansion_method', ['pass_query_expansion']),
                    ('retrieval_method', ['bm25'])
                ]
            
            if 'vectordb_name' in unified_space:
                unified_space['vectordb_name']['condition'] = [
                    ('query_expansion_method', ['pass_query_expansion']),
                    ('retrieval_method', ['vectordb'])
                ]
        
        if 'query_expansion_retrieval_method' not in unified_space:
            if 'query_expansion_method' in unified_space:
                qe_retrieval_options = self.query_expansion_retrieval_options
                if qe_retrieval_options and qe_retrieval_options.get('methods'):
                    unified_space['query_expansion_retrieval_method'] = {
                        'type': 'categorical',
                        'values': qe_retrieval_options['methods'],
                        'condition': ('query_expansion_method', [m for m in unified_space['query_expansion_method']['values'] if m != 'pass_query_expansion'])
                    }
                    
                    if 'bm25' in qe_retrieval_options['methods'] and qe_retrieval_options.get('bm25_tokenizers'):
                        unified_space['query_expansion_bm25_tokenizer'] = {
                            'type': 'categorical',
                            'values': qe_retrieval_options['bm25_tokenizers'],
                            'condition': [
                                ('query_expansion_method', [m for m in unified_space['query_expansion_method']['values'] if m != 'pass_query_expansion']),
                                ('query_expansion_retrieval_method', ['bm25'])
                            ]
                        }
                    
                    if 'vectordb' in qe_retrieval_options['methods'] and qe_retrieval_options.get('vectordb_names'):
                        unified_space['query_expansion_vectordb_name'] = {
                            'type': 'categorical',
                            'values': qe_retrieval_options['vectordb_names'],
                            'condition': [
                                ('query_expansion_method', [m for m in unified_space['query_expansion_method']['values'] if m != 'pass_query_expansion']),
                                ('query_expansion_retrieval_method', ['vectordb'])
                            ]
                        }
        
        for param_name, param_info in unified_space.items():
            param = self.param_condition_handler.create_parameter(param_name, param_info['type'], param_info)
            if param:
                cs.add(param)
        
        self.param_condition_handler.add_conditions(cs, unified_space)
        
        if 'retriever_top_k' in cs and 'reranker_top_k' in cs:
            self.param_condition_handler.add_forbidden_reranker_retriever_relation(cs)
        
        self.param_condition_handler.add_filter_reranker_constraints(cs, unified_space)
        
        return cs
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self.config_utilities.clean_trial_config(config)
    
    def get_search_space_info(self) -> Dict[str, Any]:
        unified_space = self.get_unified_space()
        return self.config_utilities.get_search_space_info(unified_space)
    
    def _create_parameter(self, name: str, param_type: str, param_info: Dict[str, Any]):
        return self.param_condition_handler.create_parameter(name, param_type, param_info)
    
    def _add_conditions(self, cs: ConfigurationSpace, unified_space: Dict[str, Any]):
        return self.param_condition_handler.add_conditions(cs, unified_space)
    
    def _add_forbidden_reranker_retriever_relation(self, cs: ConfigurationSpace):
        return self.param_condition_handler.add_forbidden_reranker_retriever_relation(cs)
    
    def _parse_reranker_config(self, reranker_config_str: str) -> Dict[str, Any]:
        return self.config_utilities._parse_reranker_config(reranker_config_str)