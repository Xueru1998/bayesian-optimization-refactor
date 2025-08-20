import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from autorag.nodes.passagecompressor.base import (
    BasePassageCompressor,
    LlamaIndexCompressor,
    make_llm
)
from autorag.utils.util import (
    get_event_loop,
    process_batch,
    empty_cuda_cache
)
from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core.response_synthesizers import TreeSummarize as ts
from llama_index.core.response_synthesizers import Refine as rf
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class PassageCompressorModule:
    
    def __init__(self, project_dir: str = "", llm_name: str = "openai", llm_params: Dict = None):
        self.project_dir = project_dir
        self.llm_name = llm_name
        self.llm_params = llm_params or {}
        
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
        if issubclass(compressor_class, LlamaIndexCompressor):
            kwargs["llm"] = self.llm_name
            kwargs.update(self.llm_params)
        
        return compressor_class(project_dir=self.project_dir, **kwargs)
    
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
            compression_kwargs = {}
            
            if method in ['tree_summarize', 'refine']:
                compression_kwargs['llm'] = config.get('llm', 'openai')
                compression_kwargs['model'] = config.get('model', 'gpt-4o-mini')
                compression_kwargs['batch'] = config.get('batch', 16)
                print(f"Applying {method} compression using {compression_kwargs['llm']}/{compression_kwargs['model']}")
            
            elif method == 'lexrank':
                compression_kwargs['compression_ratio'] = config.get('compression_ratio', 0.5)
                compression_kwargs['threshold'] = config.get('threshold', 0.1)
                compression_kwargs['damping'] = config.get('damping', 0.85)
                compression_kwargs['max_iterations'] = config.get('max_iterations', 30)
                print(f"Applying {method} compression with ratio={compression_kwargs['compression_ratio']}")
            
            elif method == 'spacy':
                compression_kwargs['compression_ratio'] = config.get('compression_ratio', 0.5)
                compression_kwargs['spacy_model'] = config.get('spacy_model', 'en_core_web_sm')
                print(f"Applying {method} compression with model={compression_kwargs['spacy_model']}")
            
            compressed_df = self.compress_passages(
                df,
                method=method,
                **compression_kwargs
            )
            
            print(f"Successfully compressed {len(df)} passages using {method}")
            return compressed_df
            
        except Exception as e:
            print(f"Error during compression: {e}")
            import traceback
            traceback.print_exc()
            print("Returning uncompressed passages due to error")
            return df


class PassCompressor(BasePassageCompressor):
    def _pure(self, contents: List[List[str]]) -> List[List[str]]:
        return contents
    
    def pure(self, previous_result: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        _, contents = self.cast_to_run(previous_result)
        result = self._pure(contents)
        
        result_df = previous_result.copy()
        result_df["retrieved_contents"] = result
        return result_df


class TreeSummarizeCompressor(LlamaIndexCompressor):
    def _pure(
        self,
        queries: List[str],
        contents: List[List[str]],
        prompt: Optional[str] = None,
        chat_prompt: Optional[str] = None,
        batch: int = 16,
    ) -> List[str]:
        if prompt is not None and not is_chat_model(self.llm):
            summary_template = PromptTemplate(prompt, prompt_type=PromptType.SUMMARY)
        elif chat_prompt is not None and is_chat_model(self.llm):
            summary_template = PromptTemplate(chat_prompt, prompt_type=PromptType.SUMMARY)
        else:
            summary_template = None
            
        summarizer = ts(llm=self.llm, summary_template=summary_template, use_async=True)
        tasks = [
            summarizer.aget_response(query, content)
            for query, content in zip(queries, contents)
        ]
        
        loop = get_event_loop()
        results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
        
        empty_cuda_cache()
        
        return results

    def pure(self, previous_result: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        queries, retrieved_contents = self.cast_to_run(previous_result)

        param_dict = dict(filter(lambda x: x[0] in self.param_list, kwargs.items()))
        compressed_texts = self._pure(queries, retrieved_contents, **param_dict)
        
        result_df = previous_result.copy()
        result_df["retrieved_contents"] = [[text] for text in compressed_texts]
        
        return result_df


class RefineCompressor(LlamaIndexCompressor):
    def _pure(
        self,
        queries: List[str],
        contents: List[List[str]],
        prompt: Optional[str] = None,
        chat_prompt: Optional[str] = None,
        batch: int = 16,
    ) -> List[str]:
        if prompt is not None and not is_chat_model(self.llm):
            refine_template = PromptTemplate(prompt, prompt_type=PromptType.REFINE)
        elif chat_prompt is not None and is_chat_model(self.llm):
            refine_template = PromptTemplate(chat_prompt, prompt_type=PromptType.REFINE)
        else:
            refine_template = None
            
        summarizer = rf(llm=self.llm, refine_template=refine_template, verbose=True)
        tasks = [
            summarizer.aget_response(query, content)
            for query, content in zip(queries, contents)
        ]
        
        loop = get_event_loop()
        results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
        
        empty_cuda_cache()
        
        return results
        
    def pure(self, previous_result: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        queries, retrieved_contents = self.cast_to_run(previous_result)
        
        param_dict = dict(filter(lambda x: x[0] in self.param_list, kwargs.items()))
        compressed_texts = self._pure(queries, retrieved_contents, **param_dict)
        
        result_df = previous_result.copy()
        result_df["retrieved_contents"] = [[text] for text in compressed_texts]
        
        return result_df


class LexRankCompressor(BasePassageCompressor):
    def __init__(self, project_dir: str = "", **kwargs):
        super().__init__(project_dir, **kwargs)
        self.compression_ratio = kwargs.get('compression_ratio', 0.5)
        self.threshold = kwargs.get('threshold', 0.1)
        self.damping = kwargs.get('damping', 0.85)
        self.max_iterations = kwargs.get('max_iterations', 30)

        self.param_list = ['compression_ratio', 'threshold', 'damping', 'max_iterations']
    
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
    
    def _pure(
        self,
        queries: List[str],
        contents: List[List[str]],
        compression_ratio: float = None,
        threshold: float = None,
        damping: float = None,
        max_iterations: int = None
    ) -> List[List[str]]:
        compression_ratio = compression_ratio or self.compression_ratio
        threshold = threshold or self.threshold
        damping = damping or self.damping
        max_iterations = max_iterations or self.max_iterations
        
        compressed_contents = []
        
        for query, content_list in zip(queries, contents):
            compressed_list = []
            
            for content in content_list:
                sentences = sent_tokenize(content)
                
                if len(sentences) <= 2:
                    compressed_list.append(content)
                    continue

                self.threshold = threshold
                self.damping = damping
                self.max_iterations = max_iterations
                
                sim_matrix = self._compute_similarity_matrix(sentences)
                scores = self._lexrank(sim_matrix)
                
                n_keep = max(1, int(len(sentences) * compression_ratio))
                top_indices = np.argsort(scores)[-n_keep:]
                top_indices = sorted(top_indices)
                
                compressed_text = " ".join([sentences[i] for i in top_indices])
                compressed_list.append(compressed_text)
            
            compressed_contents.append(compressed_list)
        
        return compressed_contents
    
    def pure(self, previous_result: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        queries, retrieved_contents = self.cast_to_run(previous_result)
        
        param_dict = dict(filter(lambda x: x[0] in self.param_list, kwargs.items()))
        compressed_contents = self._pure(queries, retrieved_contents, **param_dict)
        
        result_df = previous_result.copy()
        result_df["retrieved_contents"] = compressed_contents
        
        return result_df


class SpacyCompressor(BasePassageCompressor):
    def __init__(self, project_dir: str = "", **kwargs):
        super().__init__(project_dir, **kwargs)
        self.compression_ratio = kwargs.get('compression_ratio', 0.5)
        self.model_name = kwargs.get('spacy_model', 'en_core_web_sm')
        
        # Add parameters that can be tuned
        self.param_list = ['compression_ratio', 'spacy_model']
        
        try:
            self.nlp = spacy.load(self.model_name)
            self.has_vectors = self.nlp.vocab.vectors_length > 0
        except:
            print(f"SpaCy model {self.model_name} not found. Please install with: python -m spacy download {self.model_name}")
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
        
        if query_doc:
            if self.has_vectors:
                try:
                    similarity = sent.similarity(query_doc)
                    score += similarity * 5
                except:
                    query_tokens = set(token.text.lower() for token in query_doc if not token.is_stop)
                    sent_tokens = set(token.text.lower() for token in sent if not token.is_stop)
                    overlap = len(query_tokens.intersection(sent_tokens))
                    score += overlap * 2
            else:
                query_tokens = set(token.text.lower() for token in query_doc if not token.is_stop)
                sent_tokens = set(token.text.lower() for token in sent if not token.is_stop)
                overlap = len(query_tokens.intersection(sent_tokens))
                score += overlap * 2

        optimal_length = 15
        length_penalty = abs(len(sent) - optimal_length) / optimal_length
        score *= (1 - 0.3 * min(length_penalty, 1))
        
        return score
    
    def _pure(
        self,
        queries: List[str],
        contents: List[List[str]],
        compression_ratio: float = None,
        spacy_model: str = None
    ) -> List[List[str]]:
        compression_ratio = compression_ratio or self.compression_ratio

        if spacy_model and spacy_model != self.model_name:
            try:
                self.nlp = spacy.load(spacy_model)
                self.model_name = spacy_model
                self.has_vectors = self.nlp.vocab.vectors_length > 0
            except:
                print(f"Failed to load SpaCy model {spacy_model}, using {self.model_name}")
        
        if self.nlp is None:
            print("SpaCy not available, returning original content")
            return contents
        
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

                n_keep = max(1, int(len(sentences) * compression_ratio))
                top_sentences = sentence_scores[:n_keep]

                top_sentences.sort(key=lambda x: x[0].start)
                
                compressed_text = " ".join([str(sent[0]) for sent in top_sentences])
                compressed_list.append(compressed_text)
            
            compressed_contents.append(compressed_list)
        
        return compressed_contents
    
    def pure(self, previous_result: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        queries, retrieved_contents = self.cast_to_run(previous_result)

        param_dict = dict(filter(lambda x: x[0] in self.param_list, kwargs.items()))
        compressed_contents = self._pure(queries, retrieved_contents, **param_dict)
        
        result_df = previous_result.copy()
        result_df["retrieved_contents"] = compressed_contents
        
        return result_df


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "query": ["What method has been developed and validated for the quantification of six sex hormones in milk?"],
        "retrieved_contents": [
            ["file_name: corpus.json\n contents: Hormones work in harmony in the body, and this status must be maintained to avoid metabolic disequilibrium and the subsequent illness. Besides, it has been reported that exogenous steroids (presence in the environment and food products) influence the development of several important illnesses in humans. Endogenous steroid hormones in food of animal origin are unavoidable as they occur naturally in these products. The presence of hormones in food has been connected with several human health problems. Bovine milk contains considerable quantities of hormones and it is of particular concern. A liquid chromatography-tandem mass spectrometry (LC-MS/MS) method, based on hydroxylamine derivatisation, has been developed and validated for the quantification of six sex hormones in milk [pregnenolone (P₅), progesterone (P₄), estrone (E₁), testosterone (T), androstenedione (A) and dehydroepiandrosterone (DHEA)]. This method has been applied to real raw milk samples and the existence of differences between milk from pregnant and non-pregnant cows has been statistically confirmed. Basing on a revision of existing published data, it could be concluded that maximum daily intakes for hormones are not reached through milk ingestion. Although dairy products are an important source of hormones, other products of animal origin must be considered as well for intake calculations."],
        ]
    })
    
    # Test LexRank
    print("=" * 50)
    print("Testing LexRank Compression")
    print("=" * 50)
    
    compressor_module = PassageCompressorModule()
    compressed_df = compressor_module.compress_passages(
        sample_df, 
        method="lexrank",
        compression_ratio=0.5,
        threshold=0.1
    )
    
    print("Compressed passages:")
    for i, row in compressed_df.iterrows():
        print(f"Query: {row.get('query', 'N/A')}")
        print(f"Compressed: {row.get('retrieved_contents', 'N/A')}")
        print("-" * 50)
    
    # Test SpaCy
    print("\n" + "=" * 50)
    print("Testing SpaCy Compression")
    print("=" * 50)
    
    compressed_df = compressor_module.compress_passages(
        sample_df, 
        method="spacy",
        compression_ratio=0.4
    )
    
    print("Compressed passages:")
    for i, row in compressed_df.iterrows():
        print(f"Query: {row.get('query', 'N/A')}")
        print(f"Compressed: {row.get('retrieved_contents', 'N/A')}")
        print("-" * 50)