import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
import asyncio

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    ContextRecall,
    FactualCorrectness,
    SemanticSimilarity
)
from ragas.dataset_schema import SingleTurnSample
from ragas.evaluation import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RagasEvaluatorModule:
    
    def __init__(self, 
                 retrieval_metrics=None, 
                 generation_metrics=None,
                 llm_model="gpt-4o-mini",
                 embedding_model="text-embedding-ada-002"):
        
        if retrieval_metrics is None:
            retrieval_metrics = ["context_precision", "context_recall"]
        if generation_metrics is None:
            generation_metrics = ["answer_relevancy", "faithfulness", "factual_correctness", "semantic_similarity"]
            
        self.retrieval_metrics = retrieval_metrics
        self.generation_metrics = generation_metrics
        self.all_metrics = retrieval_metrics + generation_metrics
        
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        self.llm_wrapper = LangchainLLMWrapper(self.llm)
        self.embeddings_wrapper = LangchainEmbeddingsWrapper(self.embeddings)
        
        self.metric_instances = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        for metric_name in self.all_metrics:
            if metric_name == "context_precision":
                self.metric_instances[metric_name] = LLMContextPrecisionWithoutReference(llm=self.llm_wrapper)
            elif metric_name == "context_recall":
                self.metric_instances[metric_name] = ContextRecall(llm=self.llm_wrapper)
            elif metric_name == "answer_relevancy":
                self.metric_instances[metric_name] = ResponseRelevancy(
                    llm=self.llm_wrapper, 
                    embeddings=self.embeddings_wrapper
                )
            elif metric_name == "faithfulness":
                self.metric_instances[metric_name] = Faithfulness(llm=self.llm_wrapper)
            elif metric_name == "factual_correctness":
                self.metric_instances[metric_name] = FactualCorrectness(llm=self.llm_wrapper)
            elif metric_name == "semantic_similarity":
                self.metric_instances[metric_name] = SemanticSimilarity(embeddings=self.embeddings_wrapper)
    
    def prepare_ragas_dataset(self, 
                              queries: List[str],
                              retrieved_contexts: List[List[str]],
                              generated_texts: List[str],
                              ground_truths: List[str],
                              retrieval_ground_truths: Optional[List[List[str]]] = None) -> EvaluationDataset:
        
        samples = []
        
        min_len = min(len(queries), len(retrieved_contexts), len(generated_texts), len(ground_truths))
        
        for i in range(min_len):
            sample_dict = {
                "user_input": queries[i] if queries[i] else "",
                "retrieved_contexts": retrieved_contexts[i] if retrieved_contexts[i] else [""],
                "response": generated_texts[i] if generated_texts[i] else "",
                "reference": ground_truths[i] if ground_truths[i] else ""
            }
            
            if retrieval_ground_truths and i < len(retrieval_ground_truths):
                sample_dict["reference_contexts"] = retrieval_ground_truths[i]
            
            sample = SingleTurnSample(**sample_dict)
            samples.append(sample)
        
        return EvaluationDataset(samples=samples)
    
    def extract_ground_truths(self, qa_df: pd.DataFrame, eval_df: pd.DataFrame, corpus_df: Optional[pd.DataFrame] = None):        
        queries_in_eval = eval_df['query'].tolist() if 'query' in eval_df.columns else []
        
        generation_gts = []
        retrieval_gt_contents = []
        
        for idx, query in enumerate(queries_in_eval):
            matching_rows = qa_df[qa_df['query'] == query]
            
            if matching_rows.empty:
                print(f"[DEBUG] No matching row found for query at index {idx}: {query[:50]}...")
                generation_gts.append("")
                retrieval_gt_contents.append([])
                continue
            
            row = matching_rows.iloc[0]
            
            if 'generation_gt' in qa_df.columns:
                gt_value = row['generation_gt']
                if pd.notna(gt_value):
                    if isinstance(gt_value, (list, np.ndarray)):
                        if len(gt_value) > 0 and gt_value[0]:
                            generation_gts.append(str(gt_value[0]))
                        else:
                            generation_gts.append("")
                            print(f"[DEBUG] Empty array/list generation_gt for query at index {idx}")
                    elif isinstance(gt_value, str) and gt_value.strip():
                        generation_gts.append(gt_value)
                    else:
                        generation_gts.append("")
                        print(f"[DEBUG] Empty or invalid generation_gt for query at index {idx}")
                else:
                    generation_gts.append("")
                    print(f"[DEBUG] NaN generation_gt for query at index {idx}")
            else:
                generation_gts.append("")
            
            if 'retrieval_gt' in qa_df.columns and corpus_df is not None:
                rt_value = row['retrieval_gt']
                if isinstance(rt_value, (list, np.ndarray)):
                    if isinstance(rt_value, np.ndarray):
                        gt_list = rt_value.tolist()
                    else:
                        gt_list = rt_value
                    
                    contents = []
                    for doc_id in gt_list:
                        doc_rows = corpus_df[corpus_df['doc_id'] == doc_id]
                        if not doc_rows.empty:
                            content = doc_rows.iloc[0]['contents']
                            contents.append(content)
                    retrieval_gt_contents.append(contents)
                else:
                    retrieval_gt_contents.append([])
            else:
                retrieval_gt_contents.append([])
        
        print(f"[DEBUG] Extracted {len(generation_gts)} generation ground truths")
        non_empty_gts = sum(1 for gt in generation_gts if gt and gt.strip())
        print(f"[DEBUG] Non-empty generation ground truths: {non_empty_gts}/{len(generation_gts)}")
        
        if non_empty_gts == 0:
            print(f"[DEBUG] WARNING: All generation ground truths are empty!")
            print(f"[DEBUG] Sample queries from eval_df: {queries_in_eval[:3]}")
            print(f"[DEBUG] Sample queries from qa_df: {qa_df['query'].head(3).tolist()}")
        
        return generation_gts, retrieval_gt_contents if corpus_df is not None else None
    
    def evaluate_all(self,
                     queries: List[str],
                     retrieved_contexts: List[List[str]],
                     generated_texts: List[str],
                     ground_truths: List[str],
                     retrieval_ground_truths: Optional[List[List[str]]] = None) -> Dict[str, float]:
        
        print(f"\n[DEBUG] Starting RAGAS evaluation with:")
        print(f"  - {len(queries)} queries")
        print(f"  - {len(retrieved_contexts)} retrieved contexts")
        print(f"  - {len(generated_texts)} generated texts")
        print(f"  - {len(ground_truths)} ground truths")
        
        dataset = self.prepare_ragas_dataset(
            queries=queries,
            retrieved_contexts=retrieved_contexts,
            generated_texts=generated_texts,
            ground_truths=ground_truths,
            retrieval_ground_truths=retrieval_ground_truths
        )
        
        print(f"\n[DEBUG] Dataset created with {len(dataset.samples)} samples")
        
        metrics_to_use = []
        for metric_name in self.all_metrics:
            if metric_name in self.metric_instances:
                metrics_to_use.append(self.metric_instances[metric_name])
        
        print(f"\n[DEBUG] Using {len(metrics_to_use)} metrics")
        
        try:
            print(f"\n[DEBUG] Starting RAGAS evaluate...")
            
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_use
            )
            
            print(f"\n[DEBUG] RAGAS evaluate completed")
            print(f"[DEBUG] Results type: {type(results)}")
            
            all_results = {}
            
            if hasattr(results, 'to_pandas'):
                df_results = results.to_pandas()
                print(f"[DEBUG] Results as DataFrame shape: {df_results.shape}")
                print(f"[DEBUG] Results columns: {list(df_results.columns)}")
                
                column_mapping = {
                    "context_precision": "llm_context_precision_without_reference",
                    "factual_correctness": "factual_correctness(mode=f1)",
                }
                
                for metric_name in self.all_metrics:
                    actual_column = column_mapping.get(metric_name, metric_name)
                    
                    if actual_column in df_results.columns:
                        try:
                            if df_results[actual_column].dtype in ['float64', 'int64'] or pd.api.types.is_numeric_dtype(df_results[actual_column]):
                                metric_values = df_results[actual_column].dropna()
                                if len(metric_values) > 0:
                                    all_results[metric_name] = float(metric_values.mean())
                                    print(f"[DEBUG] {metric_name}: {all_results[metric_name]:.4f}")
                            else:
                                print(f"[DEBUG] Skipping non-numeric column {actual_column}")
                        except Exception as e:
                            print(f"[DEBUG] Error processing {actual_column}: {e}")
                    else:
                        metric_class_name = type(self.metric_instances.get(metric_name, None)).__name__
                        for col in df_results.columns:
                            if col in ['user_input', 'retrieved_contexts', 'reference_contexts', 'response', 'reference']:
                                continue  
                            
                            if metric_class_name.lower() in col.lower() or col.lower() in metric_class_name.lower():
                                try:
                                    if df_results[col].dtype in ['float64', 'int64'] or pd.api.types.is_numeric_dtype(df_results[col]):
                                        metric_values = df_results[col].dropna()
                                        if len(metric_values) > 0:
                                            all_results[metric_name] = float(metric_values.mean())
                                            print(f"[DEBUG] {metric_name} (from column {col}): {all_results[metric_name]:.4f}")
                                            break
                                except Exception as e:
                                    print(f"[DEBUG] Error processing {col}: {e}")
            
            elif hasattr(results, 'scores') and isinstance(results.scores, list):
                print(f"[DEBUG] Processing scores list of length {len(results.scores)}")
                if results.scores:
                    for metric_name in self.all_metrics:
                        metric_scores = []
                        for score in results.scores:
                            if isinstance(score, dict) and metric_name in score:
                                metric_scores.append(score[metric_name])
                        if metric_scores:
                            all_results[metric_name] = float(np.mean(metric_scores))
            
            elif isinstance(results, dict):
                for metric_name in self.all_metrics:
                    if metric_name in results:
                        all_results[metric_name] = float(results[metric_name])
            
            for metric_name in self.all_metrics:
                if metric_name not in all_results:
                    all_results[metric_name] = 0.0
                    print(f"[DEBUG] WARNING: {metric_name} not found in results, defaulting to 0.0")
            
            retrieval_scores = [all_results[m] for m in self.retrieval_metrics if m in all_results and all_results[m] > 0]
            generation_scores = [all_results[m] for m in self.generation_metrics if m in all_results and all_results[m] > 0]
            
            if retrieval_scores:
                all_results["retrieval_mean_score"] = float(np.mean(retrieval_scores))
            if generation_scores:
                all_results["generation_mean_score"] = float(np.mean(generation_scores))
            
            all_scores = retrieval_scores + generation_scores
            if all_scores:
                all_results["overall_mean_score"] = float(np.mean(all_scores))
                all_results["ragas_mean_score"] = all_results["overall_mean_score"]
            else:
                all_results["overall_mean_score"] = 0.0
                all_results["ragas_mean_score"] = 0.0
                
            return all_results
            
        except Exception as e:
            print(f"\n[DEBUG] Error in RAGAS evaluation: {e}")
            print(f"[DEBUG] Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            result_dict = {metric: 0.0 for metric in self.all_metrics}
            result_dict["overall_mean_score"] = 0.0
            result_dict["ragas_mean_score"] = 0.0
            return result_dict
    
    def evaluate_from_dataframe(self, df: pd.DataFrame, qa_df: pd.DataFrame, 
                                corpus_df: Optional[pd.DataFrame] = None,
                                evaluate_retrieval_only: bool = False,
                                evaluate_generation_only: bool = False) -> Dict[str, float]:
        
        if 'query' not in df.columns:
            print("Error: DataFrame must contain 'query' column")
            return {"overall_mean_score": 0.0, "ragas_mean_score": 0.0}
        
        queries = df['query'].tolist()
        
        retrieved_contexts = []
        if 'retrieved_contents' in df.columns:
            for contents in df['retrieved_contents'].tolist():
                if isinstance(contents, list):
                    retrieved_contexts.append(contents)
                elif isinstance(contents, str):
                    retrieved_contexts.append([contents])
                else:
                    retrieved_contexts.append([])
        else:
            retrieved_contexts = [[] for _ in queries]
        
        generated_texts = []
        if 'generated_texts' in df.columns:
            generated_texts = df['generated_texts'].tolist()
        else:
            generated_texts = [""] * len(queries)
        
        generation_gts, retrieval_gt_contents = self.extract_ground_truths(qa_df, df, corpus_df)
                
        empty_queries = sum(1 for q in queries if not q or q.strip() == "")
        empty_contexts = sum(1 for c in retrieved_contexts if not c or all(not ctx.strip() for ctx in c))
        empty_generated = sum(1 for g in generated_texts if not g or g.strip() == "")
        empty_gen_gts = sum(1 for g in generation_gts if not g or g.strip() == "")
        
        print(f"  - Empty queries: {empty_queries}/{len(queries)}")
        print(f"  - Empty context lists: {empty_contexts}/{len(retrieved_contexts)}")
        print(f"  - Empty generated texts: {empty_generated}/{len(generated_texts)}")
        print(f"  - Empty generation ground truths: {empty_gen_gts}/{len(generation_gts)}")
        
        return self.evaluate_all(queries, retrieved_contexts, generated_texts, 
                                 generation_gts, retrieval_gt_contents)
    
    def evaluate_pipeline_results(self, result_df: pd.DataFrame, qa_df: pd.DataFrame, 
                                  corpus_path: Optional[str] = None) -> Dict[str, float]:
        
        corpus_df = None
        if corpus_path and os.path.exists(corpus_path):
            corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
        elif corpus_path:
            print(f"Warning: Corpus file not found at {corpus_path}")
        
        return self.evaluate_from_dataframe(result_df, qa_df, corpus_df)