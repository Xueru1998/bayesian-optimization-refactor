from typing import Dict, Any
from ConfigSpace import ConfigurationSpace
from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_extractor import UnifiedSearchSpaceExtractor, OptimizationType
from smac3.global_optimization.config_space_builder import SMACConfigSpaceBuilder

from .component_builders import ComponentBuilders
from .parameter_factory import ParameterFactory


class ComponentwiseSMACConfigSpaceBuilder:
    
    def __init__(self, config_generator: ConfigGenerator, seed: int = 42):
        self.config_generator = config_generator
        self.seed = seed
        self.unified_extractor = UnifiedSearchSpaceExtractor(config_generator)

        qe_params = config_generator.extract_unified_parameters('query_expansion')
        self.query_expansion_retrieval_options = qe_params.get('retrieval_options', {})

        self.component_builders = ComponentBuilders(self)
        self.parameter_factory = ParameterFactory()
    
    def get_unified_space(self) -> Dict[str, Any]:
        return self.unified_extractor.extract_search_space(OptimizationType.SMAC)
    
    def build_component_space(self, component: str, fixed_params: Dict[str, Any] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=self.seed)
        
        if fixed_params is None:
            fixed_params = {}
        
        print(f"\n[DEBUG] Building config space for {component}")

        if component == 'retrieval':
            return self.component_builders.build_retrieval_space(cs, fixed_params)
        elif component == 'passage_filter':
            return self.component_builders.build_filter_space(cs, fixed_params)
        elif component == 'passage_reranker':
            return self.component_builders.build_reranker_space(cs, fixed_params)
        elif component == 'passage_compressor':
            return self.component_builders.build_compressor_space(cs, fixed_params)
        elif component == 'query_expansion':
            return self.component_builders.build_query_expansion_space(cs, fixed_params)
        elif component == 'generator':
            return self.component_builders.build_generator_space(cs, fixed_params)
        elif component == 'prompt_maker_generator':
            return self.component_builders.build_prompt_generator_space(cs, fixed_params)
        
        return cs
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        builder = SMACConfigSpaceBuilder(self.config_generator, self.seed)
        return builder.clean_trial_config(config)
    
    def _create_parameter(self, name: str, param_type: str, param_info: Dict[str, Any]):
        return self.parameter_factory.create_parameter(name, param_type, param_info)
    
    def _get_default_value(self, param_name: str, values: list) -> Any:
        return self.parameter_factory.get_default_value(param_name, values)
    
    def _add_single_condition(self, cs: ConfigurationSpace, child_name: str, condition):
        return self.parameter_factory.add_single_condition(cs, child_name, condition)
    
    def _create_parameter_from_values(self, name: str, values: list, param_type: str):
        return self.parameter_factory.create_parameter_from_values(name, values, param_type)
    
    def _extract_all_values(self, method_values: Dict[str, Any]) -> list:
        return self.parameter_factory.extract_all_values(method_values)