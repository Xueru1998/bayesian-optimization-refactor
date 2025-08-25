from typing import Dict, Any, List
from ConfigSpace import Categorical, Float, Integer, ConfigurationSpace
from ConfigSpace import InCondition, EqualsCondition, AndConjunction
from ConfigSpace import ForbiddenEqualsClause, ForbiddenAndConjunction


class ParameterAndConditions:
    
    def create_parameter(self, name: str, param_type: str, param_info: Dict[str, Any]):
        if param_type == 'categorical':
            values = param_info.get('values', [])
            if not values:
                return None
            return Categorical(name, values, default=self._get_default_value(name, values))
        
        elif param_type == 'int':
            values = param_info.get('values', [])
            if len(values) == 2:
                return Integer(name, bounds=(min(values), max(values)), default=min(values))
            elif len(values) > 2:
                return Categorical(name, values, default=values[0])
            else:
                return None
        
        elif param_type == 'float':
            if 'method_values' in param_info:
                all_values = self._extract_all_values(param_info['method_values'])
                if all_values:
                    return self._create_parameter_from_values(name, all_values, param_type)
            else:
                values = param_info.get('values', [])
                if len(values) == 2:
                    return Float(name, bounds=(min(values), max(values)), default=min(values))
                elif len(values) > 2:
                    return Categorical(name, values, default=values[0])
        
        return None
    
    def _extract_all_values(self, method_values: Dict[str, Any]) -> List[Any]:
        all_values = []
        for method, values in method_values.items():
            if isinstance(values, list):
                all_values.extend(values)
            else:
                all_values.append(values)
        return sorted(list(set(all_values)))
    
    def _create_parameter_from_values(self, name: str, values: List[Any], param_type: str):
        if not values:
            return None
        
        if param_type == 'float':
            if len(values) == 2:
                return Float(name, bounds=(min(values), max(values)), default=min(values))
            else:
                return Categorical(name, values, default=values[0])
        else:
            return Categorical(name, values, default=values[0])
    
    def _get_default_value(self, param_name: str, values: List[Any]) -> Any:
        default_map = {
            'retrieval_method': 'bm25',
            'bm25_tokenizer': 'space',
            'passage_filter_method': 'pass_passage_filter',
            'passage_compressor_method': 'pass_compressor',
            'passage_reranker_method': 'pass_reranker',
            'prompt_maker_method': 'fstring',
            'query_expansion_method': 'pass_query_expansion'
        }
        
        if param_name in default_map and default_map[param_name] in values:
            return default_map[param_name]
        
        return values[0] if values else None
    
    def add_conditions(self, cs: ConfigurationSpace, unified_space: Dict[str, Any]):
        for param_name, param_info in unified_space.items():
            if 'condition' in param_info and param_name in cs:
                self._add_single_condition(cs, param_name, param_info['condition'])
    
    def _add_single_condition(self, cs: ConfigurationSpace, child_name: str, condition):
        if isinstance(condition, list) and len(condition) > 0:
            if isinstance(condition[0], tuple):
                conjunctions = []
                for parent_name, parent_values in condition:
                    if parent_name in cs and isinstance(parent_values, list):
                        if len(parent_values) == 1:
                            conjunctions.append(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                        else:
                            conjunctions.append(InCondition(cs[child_name], cs[parent_name], parent_values))
                
                if len(conjunctions) == 1:
                    cs.add(conjunctions[0])
                elif len(conjunctions) > 1:
                    cs.add(AndConjunction(*conjunctions))
            else:
                parent_name, parent_values = condition
                if parent_name in cs and isinstance(parent_values, list):
                    if len(parent_values) == 1:
                        cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                    else:
                        cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))
        elif isinstance(condition, tuple):
            parent_name, parent_values = condition
            if parent_name in cs and isinstance(parent_values, list):
                if len(parent_values) == 1:
                    cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                else:
                    cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))
    
    def add_forbidden_reranker_retriever_relation(self, cs: ConfigurationSpace):
        retriever_param = cs['retriever_top_k']
        reranker_param = cs['reranker_top_k']

        retriever_bounds = retriever_param.lower, retriever_param.upper
        reranker_bounds = reranker_param.lower, reranker_param.upper

        for retriever_k in range(retriever_bounds[0], retriever_bounds[1] + 1):
            for reranker_k in range(retriever_k + 1, reranker_bounds[1] + 1):
                clause = ForbiddenAndConjunction(
                    ForbiddenEqualsClause(retriever_param, retriever_k),
                    ForbiddenEqualsClause(reranker_param, reranker_k)
                )
                cs.add_forbidden_clause(clause)
    
    def add_filter_reranker_constraints(self, cs: ConfigurationSpace, unified_space: Dict[str, Any]):
        if 'reranker_top_k' in cs and 'passage_filter_method' in cs:
            filter_methods = [m for m in unified_space.get('passage_filter_method', {}).get('values', []) 
                            if m != 'pass_passage_filter']
            
            for filter_method in filter_methods:
                cs.add_forbidden_clause(
                    ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs['reranker_top_k'], 1),
                        ForbiddenEqualsClause(cs['passage_filter_method'], filter_method)
                    )
                )