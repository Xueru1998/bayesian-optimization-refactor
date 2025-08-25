from typing import Dict, Any
from ConfigSpace import ConfigurationSpace, InCondition, EqualsCondition, AndConjunction, ForbiddenAndConjunction, ForbiddenEqualsClause


class ConditionBuilder:
    
    def add_conditions(self, cs: ConfigurationSpace, unified_space: Dict[str, Any]):
        for param_name, param_info in unified_space.items():
            if 'condition' in param_info and param_name in cs:
                self._add_single_condition(cs, param_name, param_info['condition'])
    
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
    
    def _add_single_condition(self, cs: ConfigurationSpace, child_name: str, condition):
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
                    elif op == 'not_equals':
                        non_pass_values = [v for v in cs[parent_name].choices if v != parent_values]
                        if non_pass_values:
                            cs.add(InCondition(cs[child_name], cs[parent_name], non_pass_values))
            elif len(condition) == 2:
                parent_name, parent_values = condition
                if parent_name in cs and isinstance(parent_values, list):
                    if len(parent_values) == 1:
                        cs.add(EqualsCondition(cs[child_name], cs[parent_name], parent_values[0]))
                    else:
                        cs.add(InCondition(cs[child_name], cs[parent_name], parent_values))