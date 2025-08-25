from typing import Dict, Any
from ConfigSpace import ConfigurationSpace
from pipeline.config_manager import ConfigGenerator
from .extractor import ParameterExtractor
from .component_builders import ComponentSpaceBuilder
from .parameter_factory import ParameterFactory
from .utils import ConfigCleaner


class ComponentwiseSMACConfigSpaceBuilder:
    
    def __init__(self, config_generator: ConfigGenerator, seed: int = 42):
        self.config_generator = config_generator
        self.seed = seed
        self._unified_space = None
        
        class UnifiedExtractor:
            def __init__(self, parent):
                self.parent = parent
            
            def extract_search_space(self, format_type='smac'):
                return self.parent.get_unified_space()
        
        self.unified_extractor = UnifiedExtractor(self)
        self.query_expansion_retrieval_options = config_generator.extract_query_expansion_retrieval_options()
        self.extractor = ParameterExtractor(config_generator)
        self.component_builder = ComponentSpaceBuilder(config_generator, self, seed)
        self.param_factory = ParameterFactory()
        self.config_cleaner = ConfigCleaner()
    
    def get_unified_space(self) -> Dict[str, Any]:
        if self._unified_space is None:
            self._unified_space = self._extract_all_hyperparameters()
        return self._unified_space
    
    def _extract_all_hyperparameters(self) -> Dict[str, Any]:
        params = {}
        
        if self.config_generator.node_exists("query_expansion"):
            params.update(self.extractor.extract_query_expansion_params())
        
        if self.config_generator.node_exists("retrieval"):
            params.update(self.extractor.extract_retrieval_params())
        
        if self.config_generator.node_exists("passage_reranker"):
            params.update(self.extractor.extract_reranker_params())
        
        if self.config_generator.node_exists("passage_filter"):
            params.update(self.extractor.extract_filter_params())
        
        if self.config_generator.node_exists("passage_compressor"):
            params.update(self.extractor.extract_compressor_params())
        
        if self.config_generator.node_exists("prompt_maker"):
            params.update(self.extractor.extract_prompt_maker_params())
        
        if self.config_generator.node_exists("generator"):
            params.update(self.extractor.extract_generator_params())
        
        return params
    
    def build_component_space(self, component: str, fixed_params: Dict[str, Any] = None) -> ConfigurationSpace:
        return self.component_builder.build_component_space(component, fixed_params)
    
    def clean_trial_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self.config_cleaner.clean_trial_config(config)
    
    def _create_parameter(self, name: str, param_type: str, param_info: Dict[str, Any]):
        return self.param_factory.create_parameter(name, param_type, param_info)
    
    def _get_default_value(self, param_name: str, values: list) -> Any:
        return self.param_factory.get_default_value(param_name, values)
    
    def _add_single_condition(self, cs: ConfigurationSpace, child_name: str, condition):
        return self.param_factory.add_single_condition(cs, child_name, condition)