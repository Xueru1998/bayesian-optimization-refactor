from typing import Dict, Any, List
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, InCondition, EqualsCondition, AndConjunction


class ParameterFactory:
    
    def create_parameter(self, name: str, param_type: str, param_info: Dict[str, Any]):
        if param_type == 'categorical':
            values = param_info.get('values', [])
            if not values:
                return None
            return Categorical(name, values, default=self.get_default_value(name, values))
        
        elif param_type == 'int':
            values = param_info.get('values', [])
            if len(values) == 2:
                return Integer(name, bounds=(min(values), max(values)), default=min(values))
            elif len(values) > 2:
                return Categorical(name, values, default=values[0])
            else:
                return None
        
        elif param_type == 'float':
            values = param_info.get('values', [])
            if len(values) == 2:
                return Float(name, bounds=(min(values), max(values)), default=min(values))
            elif len(values) > 2:
                return Categorical(name, values, default=values[0])
        
        return None
    
    def create_parameter_from_values(self, name: str, values: List[Any], param_type: str):
        if not values:
            return None
        
        if param_type == 'float':
            if len(values) == 2:
                return Float(name, bounds=(min(values), max(values)), default=min(values))
            else:
                return Categorical(name, values, default=values[0])
        else:
            return Categorical(name, values, default=values[0])
    
    def extract_all_values(self, method_values: Dict[str, Any]) -> List[Any]:
        all_values = []
        for method, values in method_values.items():
            if isinstance(values, list):
                all_values.extend(values)
            else:
                all_values.append(values)
        return sorted(list(set(all_values)))
    
    def get_default_value(self, param_name: str, values: List[Any]) -> Any:
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
    
    def add_single_condition(self, cs: ConfigurationSpace, child_name: str, condition):
        if isinstance(condition, list) and len(condition) > 0:
            if isinstance(condition[0], tuple):
                conjunctions = []
                for cond_item in condition:
                    if isinstance(cond_item, tuple) and len(cond_item) == 3:
                        parent_name, op, parent_values = cond_item
                        if parent_name in cs:
                            if op == 'equals':
                                conjunctions.append(EqualsCondition(cs[child_name], cs[parent_name], parent_values))
                            elif op == 'in' and isinstance(parent_values, list):
                                if len(parent_values) == 1:
                                    conjunctions.append(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                                else:
                                    conjunctions.append(InCondition(cs[child_name], cs[parent_name], parent_values))
                            elif op == 'contains':
                                parent_param = cs[parent_name]
                                if hasattr(parent_param, 'choices'):
                                    matching_choices = [choice for choice in parent_param.choices if parent_values in str(choice)]
                                    if matching_choices:
                                        if len(matching_choices) == 1:
                                            conjunctions.append(EqualsCondition(cs[child_name], cs[parent_name], matching_choices[0]))
                                        else:
                                            conjunctions.append(InCondition(cs[child_name], cs[parent_name], matching_choices))
                    elif isinstance(cond_item, tuple) and len(cond_item) == 2:
                        parent_name, parent_values = cond_item
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
            if len(condition) == 3:
                parent_name, op, parent_values = condition
                if parent_name in cs:
                    if op == 'equals':
                        cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values))
                    elif op == 'in' and isinstance(parent_values, list):
                        if len(parent_values) == 1:
                            cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                        else:
                            cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))
                    elif op == 'contains':
                        parent_param = cs[parent_name]
                        if hasattr(parent_param, 'choices'):
                            matching_choices = [choice for choice in parent_param.choices if parent_values in str(choice)]
                            if matching_choices:
                                if len(matching_choices) == 1:
                                    cs.add(EqualsCondition(cs[child_name], cs[parent_name], matching_choices[0]))
                                else:
                                    cs.add(InCondition(cs[child_name], cs[parent_name], matching_choices))
            elif len(condition) == 2:
                parent_name, parent_values = condition
                if parent_name in cs and isinstance(parent_values, list):
                    if len(parent_values) == 1:
                        cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                    else:
                        cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))