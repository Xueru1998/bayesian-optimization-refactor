import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import asyncio
import logging
from pathlib import Path
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from pipeline_component.nodes.generator import make_generator_callable_param


try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

try:
    from llama_index.core import PromptTemplate
    from llama_index.core.prompts import PromptType
    from llama_index.core.prompts.utils import is_chat_model
    from llama_index.core.response_synthesizers import TreeSummarize as ts
    from llama_index.core.response_synthesizers import Refine as rf
    from llama_index.core.llms import LLM
    from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.base.llms.generic_utils import completion_response_to_chat_response
    from llama_index.core.llms.callbacks import llm_completion_callback
except ImportError:
    raise ImportError("Please install llama-index-core: pip install llama-index-core")

logger = logging.getLogger(__name__)


class GeneratorLLMWrapper(LLM):
    generator: Any = None
    generator_kwargs: Dict[str, Any] = {}
    _model: str = "custom"
    
    model_config = {"protected_namespaces": ()}
    
    def __init__(self, generator_instance, model_name: str = "custom", **kwargs):
        super().__init__()

        self.generator = generator_instance
        self._model = model_name
        self.generator_kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 500)
        }
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=self.generator_kwargs.get("max_tokens", 500),
            model_name=self._model,
            is_chat_model=True,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        df = pd.DataFrame({"prompts": [prompt]})
        generation_kwargs = {
            "temperature": kwargs.get("temperature", self.generator_kwargs.get("temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", self.generator_kwargs.get("max_tokens", 500))
        }
        result_df = self.generator.pure(df, **generation_kwargs)
        response_text = result_df["generated_texts"].iloc[0]
        return CompletionResponse(text=response_text)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, **kwargs)
        yield response
    
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs: Any):
        response = await self.acomplete(prompt, **kwargs)
        yield response
    
    def chat(self, messages, **kwargs: Any):
        prompt = self._flatten_messages(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    def stream_chat(self, messages, **kwargs: Any):
        prompt = messages[-1].content if messages else ""
        for response in self.stream_complete(prompt, **kwargs):
            yield completion_response_to_chat_response(response)
    
    async def achat(self, messages, **kwargs: Any):
        prompt = self._flatten_messages(messages)
        completion_response = await self.acomplete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)
    
    async def astream_chat(self, messages, **kwargs: Any):
        prompt = messages[-1].content if messages else ""
        async for response in self.astream_complete(prompt, **kwargs):
            yield completion_response_to_chat_response(response)
            
    def _flatten_messages(self, messages: List) -> str:
        return "\n\n".join(m.content for m in messages if m.content)


def make_llm_from_generator(project_dir: str, **kwargs) -> LLM:
    generator_class, generator_params = make_generator_callable_param(kwargs)
    generator_instance = generator_class(project_dir, **generator_params)
    
    model_name = kwargs.get("model", kwargs.get("llm", "custom"))
    is_chat_model = kwargs.get("is_chat_model", True)
    
    return GeneratorLLMWrapper(
        generator_instance,
        model_name=model_name,
        is_chat_model=is_chat_model,
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 500)
    )


class BasePassageCompressor(ABC):
    
    def __init__(self, project_dir: Union[str, Path], **kwargs):
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        logger.info(f"Initialize passage compressor - {self.__class__.__name__} module...")
        self.project_dir = project_dir
        self.kwargs = kwargs
    
    @abstractmethod
    def _pure(self, queries: List[str], contents: List[List[str]], **kwargs) -> List[List[str]]:
        pass
    
    def pure(self, previous_result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logger.info(f"Running passage compressor - {self.__class__.__name__} module...")
        
        assert "query" in previous_result.columns, "previous_result must have query column."
        assert "retrieved_contents" in previous_result.columns, "previous_result must have retrieved_contents column."
        
        queries = previous_result["query"].tolist()
        contents = previous_result["retrieved_contents"].tolist()
        
        compressed_contents = self._pure(queries, contents, **kwargs)
        
        result_df = previous_result.copy()
        result_df["retrieved_contents"] = compressed_contents
        return result_df
    
    def __del__(self):
        try:
            if logger:
                logger.info(f"Delete passage compressor - {self.__class__.__name__} module...")
        except:
            pass 


class LlamaIndexCompressor(BasePassageCompressor):
    
    def __init__(self, project_dir: Union[str, Path], **kwargs):
        super().__init__(project_dir, **kwargs)
        self.llm = make_llm_from_generator(project_dir, **kwargs)
        self.batch_size = kwargs.get("batch", 16)
    
    async def _process_batch_async(self, tasks: List, batch_size: int) -> List:
        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
        return results
    
    def _run_async_tasks(self, tasks: List, batch_size: int) -> List:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self._process_batch_async(tasks, batch_size))
            return results
        finally:
            loop.close()
    
    def __del__(self):
        if hasattr(self, 'llm'):
            del self.llm
        super().__del__()


class PassCompressor(BasePassageCompressor):
    
    def _pure(self, queries: List[str], contents: List[List[str]], **kwargs) -> List[List[str]]:
        return contents


class TreeSummarizeCompressor(LlamaIndexCompressor):
    
    def _pure(self, queries: List[str], contents: List[List[str]], **kwargs) -> List[List[str]]:
        prompt = kwargs.get("prompt", None)
        chat_prompt = kwargs.get("chat_prompt", None)
        batch_size = kwargs.get("batch", self.batch_size)
        
        if prompt is not None and not is_chat_model(self.llm):
            summary_template = PromptTemplate(prompt, prompt_type=PromptType.SUMMARY)
        elif chat_prompt is not None and is_chat_model(self.llm):
            summary_template = PromptTemplate(chat_prompt, prompt_type=PromptType.SUMMARY)
        else:
            summary_template = None
        
        summarizer = ts(
            llm=self.llm, 
            summary_template=summary_template,
            use_async=True
        )
        
        tasks = [
            summarizer.aget_response(query, content)
            for query, content in zip(queries, contents)
        ]
        
        results = self._run_async_tasks(tasks, batch_size)

        return [[str(result)] for result in results]


class RefineCompressor(LlamaIndexCompressor):
    
    def _pure(self, queries: List[str], contents: List[List[str]], **kwargs) -> List[List[str]]:
        prompt = kwargs.get("prompt", None)
        chat_prompt = kwargs.get("chat_prompt", None)
        batch_size = kwargs.get("batch", self.batch_size)
        
        if prompt is not None and not is_chat_model(self.llm):
            refine_template = PromptTemplate(prompt, prompt_type=PromptType.REFINE)
        elif chat_prompt is not None and is_chat_model(self.llm):
            refine_template = PromptTemplate(chat_prompt, prompt_type=PromptType.REFINE)
        else:
            refine_template = None
        
        summarizer = rf(
            llm=self.llm,
            refine_template=refine_template,
            verbose=kwargs.get("verbose", False)
        )
        
        tasks = [
            summarizer.aget_response(query, content)
            for query, content in zip(queries, contents)
        ]
        
        results = self._run_async_tasks(tasks, batch_size)
        
        return [[str(result)] for result in results]


class LexRankCompressor(BasePassageCompressor):
    
    def __init__(self, project_dir: Union[str, Path], **kwargs):
        super().__init__(project_dir, **kwargs)
        self.compression_ratio = kwargs.get("compression_ratio", 0.5)
        self.threshold = kwargs.get("threshold", 0.1)
        self.damping = kwargs.get("damping", 0.85)
        self.max_iterations = kwargs.get("max_iterations", 30)
    
    def _compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.array([])
        
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            similarity_matrix[similarity_matrix < self.threshold] = 0
            
            row_sums = similarity_matrix.sum(axis=1)
            similarity_matrix = similarity_matrix / (row_sums[:, np.newaxis] + 1e-10)
            
            return similarity_matrix
        except:
            return np.eye(len(sentences))
    
    def _lexrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        n = similarity_matrix.shape[0]
        scores = np.ones(n) / n
        
        for _ in range(self.max_iterations):
            scores = self.damping * similarity_matrix.T.dot(scores) + (1 - self.damping) / n
        
        return scores
    
    def _pure(self, queries: List[str], contents: List[List[str]], **kwargs) -> List[List[str]]:
        compressed_contents = []
        
        for query, content_list in zip(queries, contents):
            compressed_list = []
            
            for content in content_list:
                sentences = sent_tokenize(content)
                
                if len(sentences) <= 2:
                    compressed_list.append(content)
                    continue
                
                sim_matrix = self._compute_similarity_matrix(sentences)
                
                scores = self._lexrank(sim_matrix)
                
                n_keep = max(1, int(len(sentences) * self.compression_ratio))
                top_indices = np.argsort(scores)[-n_keep:]
                
                top_indices = sorted(top_indices)
                
                compressed_text = " ".join([sentences[i] for i in top_indices])
                compressed_list.append(compressed_text)
            
            compressed_contents.append(compressed_list)
        
        return compressed_contents


class SpacyCompressor(BasePassageCompressor):
    
    def __init__(self, project_dir: Union[str, Path], **kwargs):
        super().__init__(project_dir, **kwargs)
        self.compression_ratio = kwargs.get("compression_ratio", 0.5)
        self.model_name = kwargs.get("spacy_model", "en_core_web_sm")
        
        try:
            self.nlp = spacy.load(self.model_name)
            self.has_vectors = self.nlp.vocab.vectors_length > 0
            if not self.has_vectors:
                logger.info(f"SpaCy model {self.model_name} doesn't have word vectors. Query similarity will be based on linguistic features.")
        except:
            logger.warning(f"SpaCy model {self.model_name} not found. Please install with: python -m spacy download {self.model_name}")
            self.nlp = None
            self.has_vectors = False
    
    def _score_sentence_importance(self, sent, query_doc=None) -> float:
        if self.nlp is None:
            return 1.0
        
        score = 0.0
        
        entities = [ent for ent in sent.ents]
        score += len(entities) * 2
        
        noun_phrases = [chunk for chunk in sent.noun_chunks]
        score += len(noun_phrases)
        
        if query_doc and self.has_vectors:
            try:
                similarity = sent.similarity(query_doc)
                score += similarity * 5
            except:
                query_tokens = set(token.text.lower() for token in query_doc if not token.is_stop)
                sent_tokens = set(token.text.lower() for token in sent if not token.is_stop)
                overlap = len(query_tokens.intersection(sent_tokens))
                score += overlap * 2
        elif query_doc:
            query_tokens = set(token.text.lower() for token in query_doc if not token.is_stop)
            sent_tokens = set(token.text.lower() for token in sent if not token.is_stop)
            overlap = len(query_tokens.intersection(sent_tokens))
            score += overlap * 2
        
        optimal_length = 15
        length_penalty = abs(len(sent) - optimal_length) / optimal_length
        score *= (1 - 0.3 * min(length_penalty, 1))
        
        return score
    
    def _pure(self, queries: List[str], contents: List[List[str]], **kwargs) -> List[List[str]]:
        if self.nlp is None:
            logger.warning("SpaCy not available, falling back to LexRank compression")
            return LexRankCompressor(self.project_dir, **self.kwargs)._pure(queries, contents, **kwargs)
        
        compressed_contents = []
        
        for query, content_list in zip(queries, contents):
            query_doc = self.nlp(query)
            compressed_list = []
            
            for content in content_list:
                doc = self.nlp(content)
                sentences = list(doc.sents)
                
                if not sentences:
                    compressed_list.append(content)
                    continue
                
                sentence_scores = [
                    (sent, self._score_sentence_importance(sent, query_doc))
                    for sent in sentences
                ]
                
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                
                n_keep = max(1, int(len(sentences) * self.compression_ratio))
                top_sentences = sentence_scores[:n_keep]
                
                top_sentences.sort(key=lambda x: x[0].start)
                
                compressed_text = " ".join([str(sent[0]) for sent in top_sentences])
                compressed_list.append(compressed_text)
            
            compressed_contents.append(compressed_list)
        
        return compressed_contents


def create_passage_compressor(method: str, project_dir: str = "./", **kwargs) -> BasePassageCompressor:
    
    compressor_classes = {
        "pass": PassCompressor,
        "pass_compressor": PassCompressor,
        "tree_summarize": TreeSummarizeCompressor,
        "refine": RefineCompressor,
        "lexrank": LexRankCompressor,
        "spacy": SpacyCompressor,
    }
    
    if method not in compressor_classes:
        raise ValueError(f"Unknown compression method: {method}. "
                        f"Available methods: {list(compressor_classes.keys())}")
    
    return compressor_classes[method](project_dir, **kwargs)


class PassageCompressorModule:
    
    def __init__(self, project_dir: str = "", **kwargs):
        self.project_dir = project_dir
        self.kwargs = kwargs
        
        self.compressors = {
            "pass": PassCompressor,
            "pass_compressor": PassCompressor,
            "tree_summarize": TreeSummarizeCompressor,
            "refine": RefineCompressor,
            "lexrank": LexRankCompressor,
            "spacy": SpacyCompressor,
        }
    
    def create_compressor(self, method: str, **kwargs) -> BasePassageCompressor:
        if method not in self.compressors:
            raise ValueError(f"Unknown compression method: {method}. Available methods: {list(self.compressors.keys())}")
        
        compressor_class = self.compressors[method]
        merged_kwargs = {**self.kwargs, **kwargs}
        
        return compressor_class(project_dir=self.project_dir, **merged_kwargs)
    
    def compress_passages(self, 
                         df: pd.DataFrame, 
                         method: str = "tree_summarize", 
                         **kwargs) -> pd.DataFrame:
        if "query" not in df.columns or "retrieved_contents" not in df.columns:
            raise ValueError("DataFrame must have 'query' and 'retrieved_contents' columns")
        
        compressor = self.create_compressor(method, **kwargs)
        result_df = compressor.pure(df, **kwargs)
        
        return result_df
    
    def apply_compression(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        method = config.get('module_type', 'pass_compressor')
        
        if method == 'pass_compressor' or method == 'pass':
            print("Using pass-through compression (no actual compression)")
            return df
        
        try:
            compressed_df = self.compress_passages(
                df,
                method=method,
                **config
            )
            
            print(f"Successfully compressed {len(df)} passages using {method}")
            return compressed_df
            
        except Exception as e:
            print(f"Error during compression: {e}")
            import traceback
            traceback.print_exc()
            print("Returning uncompressed passages due to error")
            return df


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "query": ["What method has been developed and validated for the quantification of six sex hormones in milk?"],
        "retrieved_contents": [
            ["file_name: corpus.json\n contents: Hormones work in harmony in the body, and this status must be maintained to avoid metabolic disequilibrium and the subsequent illness. Besides, it has been reported that exogenous steroids (presence in the environment and food products) influence the development of several important illnesses in humans. Endogenous steroid hormones in food of animal origin are unavoidable as they occur naturally in these products. The presence of hormones in food has been connected with several human health problems. Bovine milk contains considerable quantities of hormones and it is of particular concern. A liquid chromatography-tandem mass spectrometry (LC-MS/MS) method, based on hydroxylamine derivatisation, has been developed and validated for the quantification of six sex hormones in milk [pregnenolone (P₅), progesterone (P₄), estrone (E₁), testosterone (T), androstenedione (A) and dehydroepiandrosterone (DHEA)]. This method has been applied to real raw milk samples and the existence of differences between milk from pregnant and non-pregnant cows has been statistically confirmed. Basing on a revision of existing published data, it could be concluded that maximum daily intakes for hormones are not reached through milk ingestion. Although dairy products are an important source of hormones, other products of animal origin must be considered as well for intake calculations."],
        ]
    })
    
    print("=" * 50)
    print("Testing LexRank Compression")
    print("=" * 50)
    
    compressor = create_passage_compressor(
        method="lexrank",
        project_dir="./",
        compression_ratio=0.5,
        threshold=0.1,
        damping=0.85
    )
    
    compressed_df = compressor.pure(sample_df)
    
    print("Compressed passages:")
    for i, row in compressed_df.iterrows():
        print(f"Query: {row.get('query', 'N/A')}")
        print(f"Compressed: {row.get('retrieved_contents', 'N/A')}")
        print("-" * 50)
    
    print("\n" + "=" * 50)
    print("Testing SpaCy Compression")
    print("=" * 50)
    
    compressor = create_passage_compressor(
        method="spacy",
        project_dir="./",
        compression_ratio=0.4,
        spacy_model="en_core_web_sm"
    )
    
    compressed_df = compressor.pure(sample_df)
    
    print("Compressed passages:")
    for i, row in compressed_df.iterrows():
        print(f"Query: {row.get('query', 'N/A')}")
        print(f"Compressed: {row.get('retrieved_contents', 'N/A')}")
        print("-" * 50)