from typing import Dict, Any
from ConfigSpace import ConfigurationSpace, InCondition
from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_extractor import UnifiedSearchSpaceExtractor, OptimizationType

from .parameter_factory import ParameterFactory
from .condition_builder import ConditionBuilder
from .config_parser import ConfigParser
from .config_cleaner import ConfigCleaner


class SMACConfigSpaceBuilder:
    def __init__(self, config_generator: ConfigGenerator, seed: int = 42):
        self.config_generator = config_generator
        self.seed = seed
        self.unified_extractor = UnifiedSearchSpaceExtractor(config_generator)
        qe_params = config_generator.extract_unified_parameters('query_expansion')
        self.query_expansion_retrieval_options = qe_params.get('retrieval_options', {})
        
        self.parameter_factory = ParameterFactory()
        self.condition_builder = ConditionBuilder()
        self.config_parser = ConfigParser(config_generator)
        self.config_cleaner = ConfigCleaner(config_generator, self.query_expansion_retrieval_options)
    
    def build_configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        
        unified_space = self.unified_extractor.extract_search_space(OptimizationType.SMAC)

        added_params = set()
        non_pass_reranker_configs = []

        if 'passage_reranker_method' in unified_space:
            unified_params = self.config_generator.extract_unified_parameters('passage_reranker')
            models_by_method = unified_params.get('models', {})

            reranker_config_values = []
            
            for method in unified_space['passage_reranker_method']['values']:
                if method == 'pass_reranker':
                    reranker_config_values.append('pass_reranker')
                elif method == 'sap_api':
                    reranker_config_values.append('sap_api')
                    non_pass_reranker_configs.append('sap_api')
                elif method in models_by_method and models_by_method[method]:
                    for model in models_by_method[method]:
                        config_str = f"{method}::{model}"
                        reranker_config_values.append(config_str)
                        non_pass_reranker_configs.append(config_str)
                else:
                    reranker_config_values.append(method)
                    non_pass_reranker_configs.append(method)
                    print(f"    No models found for {method}, using method name only")

            unified_space['reranker_config'] = {
                'type': 'categorical',
                'values': reranker_config_values
            }

            del unified_space['passage_reranker_method']

            if 'reranker_top_k' in unified_space and unified_space['reranker_top_k'].get('condition'):
                condition = unified_space['reranker_top_k']['condition']
                if isinstance(condition, tuple) and condition[0] == 'passage_reranker_method':
                    unified_space['reranker_top_k']['condition'] = ('reranker_config', 'not_equals', 'pass_reranker')

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

        for param_name, param_info in unified_space.items():
            if param_name in added_params:
                print(f"  Skipping {param_name} - already added")
                continue
                
            param = self.parameter_factory.create_parameter(param_name, param_info['type'], param_info)
            if param:
                cs.add(param)
                added_params.add(param_name)

        self.condition_builder.add_conditions(cs, unified_space)

        if non_pass_reranker_configs and 'reranker_config' in cs and 'reranker_top_k' in cs:
            cs.add(InCondition(cs['reranker_top_k'], cs['reranker_config'], non_pass_reranker_configs))

        if 'retriever_top_k' in cs and 'reranker_top_k' in cs:
            self.condition_builder.add_forbidden_reranker_retriever_relation(cs)
        
        return cs
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self.config_cleaner.clean_trial_config(config, self.config_parser)
    
    def get_search_space_info(self) -> Dict[str, Any]:
        unified_space = self.unified_extractor.extract_search_space(OptimizationType.SMAC)
        
        n_hyperparameters = len(unified_space)
        n_categorical = 0
        n_continuous = 0
        n_integer = 0
        
        for param_name, param_info in unified_space.items():
            param_type = param_info.get('type', '')
            if param_type == 'categorical':
                n_categorical += 1
            elif param_type == 'float':
                n_continuous += 1
            elif param_type == 'int':
                n_integer += 1
        
        total_combinations = 1
        for param_name, param_info in unified_space.items():
            if param_info.get('type') == 'categorical':
                n_values = len(param_info.get('values', []))
                if n_values > 0:
                    total_combinations *= n_values
            elif param_info.get('type') == 'int':
                values = param_info.get('values', [])
                if len(values) == 2:
                    total_combinations *= (values[1] - values[0] + 1)
                elif len(values) > 2:
                    total_combinations *= len(values)
        
        return {
            'n_hyperparameters': n_hyperparameters,
            'n_categorical': n_categorical,
            'n_continuous': n_continuous,
            'n_integer': n_integer,
            'total_combinations': total_combinations,
            'hyperparameters': list(unified_space.keys())
        }