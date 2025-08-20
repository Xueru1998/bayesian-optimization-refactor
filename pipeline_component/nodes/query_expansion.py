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
    
    def __init__(self, project_dir: Union[str, Path] = "", **kwargs):
        self.project_dir = project_dir
        logger.info(f"Initialize query expansion node - {self.__class__.__name__} module...")
    
    def __del__(self):
        logger.info(f"Delete query expansion node - {self.__class__.__name__} module...")
    
    def _pure(self, queries: List[str], **kwargs) -> List[List[str]]:
        return [[query] for query in queries]
    
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
    
    DEFAULT_PROMPT = """Decompose a question into sub-questions. Output ONLY numbered sub-questions, no explanations.

    Example 1:
    Input: Is Hamlet more common on IMDB than Comedy of Errors?
    1: How many listings of Hamlet are there on IMDB?
    2: How many listings of Comedy of Errors are there on IMDB?

    Example 2:
    Input: Treatment with a protein named FN impairs regenerative abilities of aged muscles.
    1: What is the protein named FN?
    2: How does FN affect muscle regeneration?
    3: What is known about regenerative abilities in aged muscles?
    4: Does treatment with FN impair regeneration in aged muscle tissue?

    Example 3:
    Input: Are birds important to badminton?
    The input needs no decomposition

    Example 4:
    Input: A licensed child driving a Mercedes-Benz is employed in the US.
    1: What is the minimum legal driving age in the US?
    2: What is the legal age to be employed in the US?

    Input: {question}
    Output:""" 
        
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
        if "the question needs no decomposition" in answer.lower() or "the input needs no decomposition" in answer.lower():
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
question to retrieve relevant documents from a hybrid retrieval system
that uses both semantic search and keyword matching (BM25).
By generating multiple perspectives on the user question,
your goal is to help overcome limitations of both search methods:
- Include variations with synonyms and related terms for better keyword matching
- Rephrase the question from different angles for semantic understanding
- Consider both specific keywords and conceptual meanings

Provide these alternative questions separated by newlines. Generate EXACTLY 3 alternative versions of the given question. Output ONLY the 3 questions, one per line, no explanations or numbering.
Original question: {query}"""

    
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


class QueryExpansionModule:
    def __init__(self, project_dir: str = ""):
        self.project_dir = project_dir
        
        self.expanders = {
            "pass_query_expansion": PassQueryExpansion,
            "query_decompose": QueryDecompose,
            "hyde": HyDE,
            "multi_query_expansion": MultiQueryExpansion
        }
        
        self.default_prompts = {
            "hyde": "Please write a passage to answer the question. Output ONLY the answer passage, no explanations.",
            "query_decompose": QueryDecompose.DEFAULT_PROMPT,
            "multi_query_expansion": MultiQueryExpansion.DEFAULT_PROMPT
        }
    
    def create_expander(self, method: str, **kwargs) -> BaseQueryExpansion:
        if method not in self.expanders:
            raise ValueError(f"Unknown query expansion method: {method}. Available methods: {list(self.expanders.keys())}")
        
        expander_class = self.expanders[method]
        return expander_class(project_dir=self.project_dir, **kwargs)
    
    def expand_queries(self, 
                       df: pd.DataFrame, 
                       method: str = "pass_query_expansion", 
                       **kwargs) -> pd.DataFrame:
        if "query" not in df.columns:
            raise ValueError("DataFrame must have 'query' column")

        init_params = {}
        operation_params = {}
        
        if 'generator_module_type' in kwargs:
            init_params['generator_module_type'] = kwargs.pop('generator_module_type')
        if 'llm' in kwargs:
            init_params['llm'] = kwargs.pop('llm')
        if 'model' in kwargs:
            init_params['model'] = kwargs.pop('model')
        if 'batch' in kwargs:
            init_params['batch'] = kwargs.pop('batch')

        queries = df['query'].tolist()

        expander = self.create_expander(method, **init_params)
        
        try:
            if method == "pass_query_expansion":
                expanded_queries = [[query] for query in queries]
                
            elif method == "query_decompose":
                prompt = kwargs.pop('prompt', self.default_prompts.get('query_decompose'))
                expanded_queries = expander._pure(queries, prompt=prompt, **kwargs)
                
            elif method == "hyde":
                prompt = kwargs.pop('prompt', self.default_prompts.get('hyde'))
                if 'max_token' in kwargs:
                    kwargs['max_tokens'] = kwargs.pop('max_token')
                expanded_queries = expander._pure(queries, prompt=prompt, **kwargs)
                
            elif method == "multi_query_expansion":
                prompt = kwargs.pop('prompt', self.default_prompts.get('multi_query_expansion'))
                expanded_queries = expander._pure(queries, prompt=prompt, **kwargs)
            
            result_df = df.copy()
            result_df['queries'] = expanded_queries
            result_df['expanded_queries'] = expanded_queries
            
            return result_df
        except Exception as e:
            print(f"Error during query expansion: {e}")
            import traceback
            traceback.print_exc()

            result_df = df.copy()
            result_df['queries'] = [[query] for query in queries]
            result_df['expanded_queries'] = [[query] for query in queries]
            return result_df
    
    def apply_expansion(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        method = config.get('module_type', 'pass_query_expansion')
        
        expansion_params = {}

        if 'generator_module_type' in config:
            expansion_params['generator_module_type'] = config['generator_module_type']
        if 'llm' in config:
            expansion_params['llm'] = config['llm']
        if 'model' in config:
            expansion_params['model'] = config['model']
        if 'batch' in config:
            expansion_params['batch'] = config['batch']
        
        if method == 'hyde' and 'max_token' in config:
            expansion_params['max_token'] = config['max_token']
        if method == 'multi_query_expansion' and 'temperature' in config:
            expansion_params['temperature'] = config['temperature']
        
        print(f"Applying {method} query expansion")
        if expansion_params.get('model'):
            print(f"Using model: {expansion_params['model']}")
        elif expansion_params.get('llm'):
            print(f"Using LLM: {expansion_params['llm']}")

        if method == 'pass_query_expansion':
            print("Using pass-through query expansion (no expansion)")
            result_df = df.copy()
            result_df['queries'] = [[query] for query in df['query'].tolist()]
            result_df['expanded_queries'] = [[query] for query in df['query'].tolist()]
            return result_df

        try:
            expanded_df = self.expand_queries(
                df,
                method=method,
                **expansion_params
            )
            
            print(f"Successfully expanded queries for {len(df)} queries")
            return expanded_df
            
        except Exception as e:
            print(f"Error during query expansion: {e}")
            import traceback
            traceback.print_exc()
            print("Returning original queries due to error")
            
            result_df = df.copy()
            result_df['queries'] = [[query] for query in df['query'].tolist()]
            result_df['expanded_queries'] = [[query] for query in df['query'].tolist()]
            return result_df
    
    def perform_query_expansion(self, df: pd.DataFrame, config: Dict[str, Any]) -> tuple:
        expanded_df = self.apply_expansion(df, config)
        expanded_queries = expanded_df['expanded_queries'].tolist()
        
        return expanded_df, expanded_queries
        
    def perform_retrieval_with_expanded_queries(self, df: pd.DataFrame, retrieval_func, config: Dict[str, Any]) -> pd.DataFrame:
        expanded_df, expanded_queries = self.perform_query_expansion(df, config)

        all_queries = []
        query_indices = []
        
        for i, query_list in enumerate(expanded_queries):
            all_queries.extend(query_list)
            query_indices.extend([i] * len(query_list))
        
        retrieval_df = pd.DataFrame({'query': all_queries})
        
        retrieved_df = retrieval_func(retrieval_df)
        
        result_df = df.copy()
        result_df['queries'] = expanded_queries
        result_df['expanded_queries'] = expanded_queries

        result_df['retrieved_contents'] = [[] for _ in range(len(df))]
        result_df['retrieved_ids'] = [[] for _ in range(len(df))]
        result_df['retrieve_scores'] = [[] for _ in range(len(df))]
        
        for i, idx in enumerate(query_indices):
            if 'retrieved_contents' in retrieved_df.columns and i < len(retrieved_df):
                if isinstance(retrieved_df['retrieved_contents'].iloc[i], list):
                    result_df['retrieved_contents'].iloc[idx].extend(retrieved_df['retrieved_contents'].iloc[i])
            
            if 'retrieved_ids' in retrieved_df.columns and i < len(retrieved_df):
                if isinstance(retrieved_df['retrieved_ids'].iloc[i], list):
                    result_df['retrieved_ids'].iloc[idx].extend(retrieved_df['retrieved_ids'].iloc[i])
            
            if 'retrieve_scores' in retrieved_df.columns and i < len(retrieved_df):
                if isinstance(retrieved_df['retrieve_scores'].iloc[i], list):
                    result_df['retrieve_scores'].iloc[idx].extend(retrieved_df['retrieve_scores'].iloc[i])
        
        return result_df