import pandas as pd
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseQueryExpansion(ABC):
    
    def __init__(self, project_dir: Union[str, Path], **kwargs):
        logger.info(f"Initialize query expansion node - {self.__class__.__name__} module...")
        
        from pipeline_component.nodes.generator import make_generator_callable_param
        
        generator_class, generator_params = make_generator_callable_param(kwargs)
        self.generator = generator_class(project_dir, **generator_params)
        self.project_dir = project_dir
    
    def __del__(self):
        if hasattr(self, 'generator'):
            del self.generator
            logger.info(f"Delete query expansion node - {self.__class__.__name__} module...")
    
    @abstractmethod
    def _pure(self, queries: List[str], **kwargs) -> List[List[str]]:
        pass
    
    def pure(self, previous_result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logger.info(f"Running query expansion node - {self.__class__.__name__} module...")
        
        assert "query" in previous_result.columns, "previous_result must have query column."
        queries = previous_result["query"].tolist()
        
        expanded_queries = self._pure(queries, **kwargs)
        expanded_queries = self._check_expanded_query(queries, expanded_queries)
        
        result_df = previous_result.copy()
        result_df["queries"] = expanded_queries
        return result_df
    
    def run_evaluator(self, project_dir: str, previous_result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.pure(previous_result, **kwargs)
    
    @staticmethod
    def _check_expanded_query(queries: List[str], expanded_queries: List[List[str]]) -> List[List[str]]:
        return [
            [expanded.strip() if expanded.strip() else query for expanded in expanded_list]
            for query, expanded_list in zip(queries, expanded_queries)
        ]


class PassQueryExpansion(BaseQueryExpansion):
    
    def _pure(self, queries: List[str], **kwargs) -> List[List[str]]:
        return [[query] for query in queries]


class HyDE(BaseQueryExpansion):
    
    DEFAULT_PROMPT = "Please write a passage to answer the question"
    
    def _pure(self, queries: List[str], **kwargs) -> List[List[str]]:
        prompt_template = kwargs.get("prompt", self.DEFAULT_PROMPT)
        
        full_prompts = [
            f"{prompt_template}\nQuestion: {query}\nPassage:"
            for query in queries
        ]
        
        generation_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_token", kwargs.get("max_tokens", 256))
        }
        
        input_df = pd.DataFrame({"prompts": full_prompts})
        result_df = self.generator.pure(input_df, **generation_params)
        generated_passages = result_df["generated_texts"].tolist()
        
        return [[passage] for passage in generated_passages]


class QueryDecompose(BaseQueryExpansion):
    
    DEFAULT_PROMPT = """Decompose a question or a statement into self-contained sub-questions. These sub-questions should help in retrieving relevant factual information to understand or verify the original input. Use "The input needs no decomposition" if it is already atomic.

    Example 1:
    Input: Is Hamlet more common on IMDB than Comedy of Errors?
    Decompositions:
    1: How many listings of Hamlet are there on IMDB?
    2: How many listings of Comedy of Errors are there on IMDB?

    Example 2:
    Input: Treatment with a protein named FN impairs regenerative abilities of aged muscles.
    Decompositions:
    1: What is the protein named FN?
    2: How does FN affect muscle regeneration?
    3: What is known about regenerative abilities in aged muscles?
    4: Does treatment with FN impair regeneration in aged muscle tissue?

    Example 3:
    Input: Are birds important to badminton?
    Decompositions:
    The input needs no decomposition

    Example 4:
    Input: A licensed child driving a Mercedes-Benz is employed in the US.
    Decompositions:
    1: What is the minimum legal driving age in the US?
    2: What is the legal age to be employed in the US?

    Example 5:
    Input: {question}
    Decompositions:
    """
    
    def _pure(self, queries: List[str], **kwargs) -> List[List[str]]:
        prompt_template = kwargs.get("prompt", self.DEFAULT_PROMPT)
        
        full_prompts = []
        for query in queries:
            if "{question}" in prompt_template:
                prompt = prompt_template.format(question=query)
            else:
                prompt = f"{prompt_template}\n\nQuestion: {query}\nDecompositions:"
            full_prompts.append(prompt)
        
        generation_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 256)
        }
        
        input_df = pd.DataFrame({"prompts": full_prompts})
        result_df = self.generator.pure(input_df, **generation_params)
        answers = result_df["generated_texts"].tolist()
        
        return [
            self._parse_decomposition(query, answer)
            for query, answer in zip(queries, answers)
        ]
    
    @staticmethod
    def _parse_decomposition(query: str, answer: str) -> List[str]:
        if "the question needs no decomposition" in answer.lower():
            return [query]
        
        try:
            lines = [line.strip() for line in answer.splitlines() if line.strip()]
            
            if lines and lines[0].startswith("Decompositions:"):
                lines.pop(0)
            
            questions = []
            for line in lines:
                if ":" in line and line[0].isdigit():
                    _, question = line.split(":", 1)
                    questions.append(question.strip())
                elif not line[0].isdigit() and line not in ["Decompositions:", ""]:
                    questions.append(line)
            
            return questions if questions else [query]
            
        except Exception as e:
            logger.warning(f"Failed to parse decomposition: {e}")
            return [query]


class MultiQueryExpansion(BaseQueryExpansion):
    
    DEFAULT_PROMPT = """You are an AI language model assistant.
    Your task is to generate 3 different versions of the given user
    question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions separated by newlines. Original question: {query}"""
    
    def _pure(self, queries: List[str], **kwargs) -> List[List[str]]:
        prompt_template = kwargs.get("prompt", self.DEFAULT_PROMPT)
        
        full_prompts = [
            prompt_template.format(query=query)
            for query in queries
        ]
        
        generation_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 256)
        }
        
        input_df = pd.DataFrame({"prompts": full_prompts})
        result_df = self.generator.pure(input_df, **generation_params)
        answers = result_df["generated_texts"].tolist()
        
        return [
            self._parse_multi_query(query, answer)
            for query, answer in zip(queries, answers)
        ]
    
    @staticmethod
    def _parse_multi_query(query: str, answer: str) -> List[str]:
        try:
            queries = answer.strip().split("\n")
            queries = [q.strip() for q in queries if q.strip()]
            
            cleaned_queries = []
            for q in queries:
                if q.startswith(("1.", "2.", "3.", "-", "*", "•")):
                    q = q[2:].strip()
                cleaned_queries.append(q)
            
            cleaned_queries.insert(0, query)
            
            return cleaned_queries[:4]
            
        except Exception as e:
            logger.warning(f"Failed to parse multi-query expansion: {e}")
            return [query]


def create_query_expansion(module_type: str, project_dir: str, **kwargs) -> BaseQueryExpansion:
    
    expansion_classes = {
        "pass_query_expansion": PassQueryExpansion,
        "pass": PassQueryExpansion,
        "hyde": HyDE,
        "query_decompose": QueryDecompose,
        "multi_query_expansion": MultiQueryExpansion
    }
    
    if module_type not in expansion_classes:
        raise ValueError(f"Unknown query expansion method: {module_type}. "
                        f"Available methods: {list(expansion_classes.keys())}")
    
    return expansion_classes[module_type](project_dir, **kwargs)


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "query": [
            "What are the benefits of exercise?",
            "How does climate change affect biodiversity?"
        ]
    })
    
    hyde_config = {
        "generator_module_type": "sap_api",
        "llm": "mistralai",
        "model": "mistralai-large-instruct",
        "api_url": "https://api.ai.internalprod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d2c244844b703dc4/chat/completions",
        "bearer_token": "eyJhbGciOiJSUzI1NiIsImprdSI6Imh0dHBzOi8vc2FwdHVtZW50ZXJwcmlzZWFpLTV2OHkyZjk0LmF1dGhlbnRpY2F0aW9uLnNhcC5oYW5hLm9uZGVtYW5kLmNvbS90b2tlbl9rZXlzIiwia2lkIjoiZGVmYXVsdC1qd3Qta2V5LTM3N2M0MDBiYTkiLCJ0eXAiOiJKV1QiLCJqaWQiOiAiOGZxamh0ZEpiNHVPMW0zUGtZYTZuM3BnK3ZoS21GQXlrQy8vYVhYaStGdz0ifQ.eyJqdGkiOiIzZjJiNjg5NTE0YjE0MjkxYmNlMThkMDc3ZGFhMmRhMiIsImV4dF9hdHRyIjp7ImVuaGFuY2VyIjoiWFNVQUEiLCJzdWJhY2NvdW50aWQiOiI4YjBkZGQ5Ny1mZTZiLTQ1ZjQtODc4Zi04NjMzNjc4MTFjM2IiLCJ6ZG4iOiJzYXB0dW1lbnRlcnByaXNlYWktNXY4eTJmOTQiLCJzZXJ2aWNlaW5zdGFuY2VpZCI6ImIyMTI2ZjZlLTBlZTUtNDk3My1iNmMxLWMzYmZjNGQzODg0OCJ9LCJzdWIiOiJzYi1iMjEyNmY2ZS0wZWU1LTQ5NzMtYjZjMS1jM2JmYzRkMzg4NDghYjE5NTc2Mnx4c3VhYV9zdGQhYjc3MDg5IiwiYXV0aG9yaXRpZXMiOlsieHN1YWFfc3RkIWI3NzA4OS5kb2NrZXJyZWdpc3RyeXNlY3JldC5jcmVkZW50aWFscy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LmRvY2tlcnJlZ2lzdHJ5c2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5Lm5vZGVzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuYXJ0aWZhY3RzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuZGVwbG95bWVudHMud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5Lm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnJlc291cmNlZ3JvdXAud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5LmRlcGxveW1lbnRzLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LmFwcGxpY2F0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmV4ZWN1dGlvbnMud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5wcm9tcHRUZW1wbGF0ZXMucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnByZWRpY3QiLCJ4c3VhYV9zdGQhYjc3MDg5LmFwcGxpY2F0aW9ucy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuZXhlY3V0aW9ucy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuYXJ0aWZhY3RzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5leGVjdXRhYmxlcy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zZXJ2aWNlcy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zZWNyZXRzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5vYmplY3RzdG9yZXNlY3JldC5jcmVkZW50aWFscy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmV4ZWN1dGlvbnNjaGVkdWxlcy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuZXhlY3V0aW9ucy5sb2dzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5tZXRyaWNzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zZWNyZXRzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5tZXRyaWNzLnJlYWQiLCJ1YWEucmVzb3VyY2UiLCJ4c3VhYV9zdGQhYjc3MDg5LmtwaXMucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkucmVwb3NpdG9yaWVzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5yZXBvc2l0b3JpZXMucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkuZGF0YXNldHMud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5Lm5vZGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5Lm1ldGEucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkubG9ncy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5yZXNvdXJjZWdyb3VwLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5jb25maWd1cmF0aW9ucy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5kYXRhc2V0cy5kb3dubG9hZCIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLnByb21wdFRlbXBsYXRlcy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmNvbmZpZ3VyYXRpb25zLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuZXhlY3V0aW9uc2NoZWR1bGVzLnJlYWQiXSwic2NvcGUiOlsieHN1YWFfc3RkIWI3NzA4OS5kb2NrZXJyZWdpc3RyeXNlY3JldC5jcmVkZW50aWFscy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LmRvY2tlcnJlZ2lzdHJ5c2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5Lm5vZGVzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuYXJ0aWZhY3RzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuZGVwbG95bWVudHMud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5Lm9iamVjdHN0b3Jlc2VjcmV0LmNyZWRlbnRpYWxzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnJlc291cmNlZ3JvdXAud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5LmRlcGxveW1lbnRzLmxvZ3MucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LmFwcGxpY2F0aW9ucy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmV4ZWN1dGlvbnMud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5wcm9tcHRUZW1wbGF0ZXMucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmRlcGxveW1lbnRzLnByZWRpY3QiLCJ4c3VhYV9zdGQhYjc3MDg5LmFwcGxpY2F0aW9ucy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuZXhlY3V0aW9ucy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuYXJ0aWZhY3RzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5leGVjdXRhYmxlcy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zZXJ2aWNlcy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5zZWNyZXRzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5vYmplY3RzdG9yZXNlY3JldC5jcmVkZW50aWFscy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmV4ZWN1dGlvbnNjaGVkdWxlcy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuZXhlY3V0aW9ucy5sb2dzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5tZXRyaWNzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zZWNyZXRzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5tZXRyaWNzLnJlYWQiLCJ1YWEucmVzb3VyY2UiLCJ4c3VhYV9zdGQhYjc3MDg5LmtwaXMucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkucmVwb3NpdG9yaWVzLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5yZXBvc2l0b3JpZXMucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkuZGF0YXNldHMud3JpdGUiLCJ4c3VhYV9zdGQhYjc3MDg5Lm5vZGVzLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5Lm1ldGEucmVhZCIsInhzdWFhX3N0ZCFiNzcwODkubG9ncy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5yZXNvdXJjZWdyb3VwLnJlYWQiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5jb25maWd1cmF0aW9ucy5yZWFkIiwieHN1YWFfc3RkIWI3NzA4OS5kYXRhc2V0cy5kb3dubG9hZCIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLnByb21wdFRlbXBsYXRlcy53cml0ZSIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmNvbmZpZ3VyYXRpb25zLndyaXRlIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuZXhlY3V0aW9uc2NoZWR1bGVzLnJlYWQiXSwiY2xpZW50X2lkIjoic2ItYjIxMjZmNmUtMGVlNS00OTczLWI2YzEtYzNiZmM0ZDM4ODQ4IWIxOTU3NjJ8eHN1YWFfc3RkIWI3NzA4OSIsImNpZCI6InNiLWIyMTI2ZjZlLTBlZTUtNDk3My1iNmMxLWMzYmZjNGQzODg0OCFiMTk1NzYyfHhzdWFhX3N0ZCFiNzcwODkiLCJhenAiOiJzYi1iMjEyNmY2ZS0wZWU1LTQ5NzMtYjZjMS1jM2JmYzRkMzg4NDghYjE5NTc2Mnx4c3VhYV9zdGQhYjc3MDg5IiwiZ3JhbnRfdHlwZSI6ImNsaWVudF9jcmVkZW50aWFscyIsInJldl9zaWciOiJhYjVlYjAwIiwiaWF0IjoxNzUyMjM4MTMxLCJleHAiOjE3NTIyODEzMzEsImlzcyI6Imh0dHBzOi8vc2FwdHVtZW50ZXJwcmlzZWFpLTV2OHkyZjk0LmF1dGhlbnRpY2F0aW9uLnNhcC5oYW5hLm9uZGVtYW5kLmNvbS9vYXV0aC90b2tlbiIsInppZCI6IjhiMGRkZDk3LWZlNmItNDVmNC04NzhmLTg2MzM2NzgxMWMzYiIsImF1ZCI6WyJ4c3VhYV9zdGQhYjc3MDg5LmtwaXMiLCJ4c3VhYV9zdGQhYjc3MDg5LmRhdGFzZXRzIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5jb25maWd1cmF0aW9ucyIsInhzdWFhX3N0ZCFiNzcwODkubWV0YSIsInhzdWFhX3N0ZCFiNzcwODkucmVwb3NpdG9yaWVzIiwieHN1YWFfc3RkIWI3NzA4OS5ub2RlcyIsInhzdWFhX3N0ZCFiNzcwODkuZXhlY3V0aW9ucy5sb2dzIiwieHN1YWFfc3RkIWI3NzA4OS5hcHBsaWNhdGlvbnMiLCJ1YWEiLCJ4c3VhYV9zdGQhYjc3MDg5LnJlc291cmNlZ3JvdXAiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5leGVjdXRpb25zY2hlZHVsZXMiLCJ4c3VhYV9zdGQhYjc3MDg5LnNlcnZpY2VzIiwieHN1YWFfc3RkIWI3NzA4OS5sb2dzIiwieHN1YWFfc3RkIWI3NzA4OS5vYmplY3RzdG9yZXNlY3JldC5jcmVkZW50aWFscyIsInhzdWFhX3N0ZCFiNzcwODkuZGVwbG95bWVudHMubG9ncyIsInNiLWIyMTI2ZjZlLTBlZTUtNDk3My1iNmMxLWMzYmZjNGQzODg0OCFiMTk1NzYyfHhzdWFhX3N0ZCFiNzcwODkiLCJ4c3VhYV9zdGQhYjc3MDg5LmRvY2tlcnJlZ2lzdHJ5c2VjcmV0LmNyZWRlbnRpYWxzIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MuZGVwbG95bWVudHMiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5wcm9tcHRUZW1wbGF0ZXMiLCJ4c3VhYV9zdGQhYjc3MDg5LnNlY3JldHMiLCJ4c3VhYV9zdGQhYjc3MDg5LnNjZW5hcmlvcy5leGVjdXRpb25zIiwieHN1YWFfc3RkIWI3NzA4OS5zY2VuYXJpb3MubWV0cmljcyIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmFydGlmYWN0cyIsInhzdWFhX3N0ZCFiNzcwODkuc2NlbmFyaW9zLmV4ZWN1dGFibGVzIl19.siY40299UCcuDQXLUbnuS3zIM0la_iOcBQnbb8ZvWEx6Oxro4sB-6Z-M8lq-zATGVZMSu77iXuUusvJPDZpjrNzl7EEY5fx84DfGYGbLwc6AzDIA7_8yYkJEFZlY0lhh-YpWQUEPwG4gs6Utib7JejXbvLUP0Te561-295lpFlAOEBNkCvGovfVgFNsf0KusuGfiG0cQI0aStLw-VuGUtPqVo-8fsQGRGyq2ZSioytaAPKbrg6m8fJnfHxdG9DncMbdfOy1AP7GdCvdSPjwZV8QVZYChHlRJjOrLz1pK8jSad2BNsYBa9IqraJv7NTMEDQNC_-IsBbvUdRZFAWQ6uw",
        "max_token": 128,
        "temperature": 0.7
    }
    
    hyde_expander = create_query_expansion(
        module_type="query_decompose",
        project_dir="./",
        **hyde_config
    )
    
    result = hyde_expander.pure(sample_df)
    print("HyDE Results:")
    for idx, queries in enumerate(result["queries"]):
        print(f"Original: {sample_df.iloc[idx]['query']}")
        print(f"Expanded: {queries}")
        print("-" * 50)