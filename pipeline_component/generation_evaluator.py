import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
import asyncio

from autorag.evaluation.metric.generation import (
    bleu,
    meteor,
    rouge,
    sem_score,
    bert_score,
    g_eval,
)
from autorag.schema.metricinput import MetricInput
from autorag.evaluation.util import cast_metrics
from autorag.embedding.base import embedding_models
from llama_index.embeddings.openai import OpenAIEmbedding
from autorag.utils.util import (
    empty_cuda_cache, 
    get_event_loop, 
    process_batch
)


class GenerationEvaluatorModule:
    
    def __init__(self, metrics=None, embedding_model_name="openai"):

        if metrics is None:
            metrics = ["bleu", "meteor", "rouge"]
            
            if embedding_model_name == "openai":
                metrics.append("sem_score")
                
        self.metrics = metrics
        self.embedding_model_name = embedding_model_name
        
        self.metric_functions = {
            "bleu": bleu,
            "meteor": meteor,
            "rouge": rouge,
            "sem_score": sem_score,
            "bert_score": bert_score,
            "g_eval": g_eval,
        }
    
    def prepare_metric_inputs(self, generated_texts: List[str], ground_truths: List[List[str]], 
                              retrieval_contents: Optional[List[List[str]]] = None) -> List[MetricInput]:
        """Prepare MetricInput objects for evaluation."""
        metric_inputs = []
        
        for i in range(len(generated_texts)):
            gt = ground_truths[i] if i < len(ground_truths) else []
            
            # Ensure ground_truths is a list of lists of strings
            if isinstance(gt, str):
                gt = [gt]
            elif isinstance(gt, list):
                gt = [str(item) for item in gt]  # Ensure all items are strings
            else:
                gt = []
            
            input_kwargs = {
                "generated_texts": generated_texts[i],
                "generation_gt": gt
            }
            
            if retrieval_contents and i < len(retrieval_contents):
                input_kwargs["retrieval_gt_contents"] = retrieval_contents[i]
                            
            metric_inputs.append(MetricInput(**input_kwargs))
        
        return metric_inputs
    
    def extract_generation_ground_truths(self, qa_df: pd.DataFrame) -> List[List[str]]:
        ground_truths = []
        
        if 'generation_gt' not in qa_df.columns:
            print("Warning: 'generation_gt' column not found in QA dataframe.")
            return [[] for _ in range(len(qa_df))]
            
        for gt_item in qa_df['generation_gt'].tolist():
            if isinstance(gt_item, list):
                ground_truths.append(gt_item)
            elif isinstance(gt_item, np.ndarray):
                ground_truths.append(gt_item.tolist())
            elif isinstance(gt_item, str):
                ground_truths.append([gt_item])
            else:
                ground_truths.append([])
        
        print(f"Found {len(ground_truths)} generation ground truth items")
        empty_gt_count = sum(1 for gt in ground_truths if not gt)
        if empty_gt_count > 0:
            print(f"Warning: {empty_gt_count}/{len(ground_truths)} generation ground truths are empty")
        
        return ground_truths
    
    def evaluate_generation(self, generated_texts: List[str], ground_truths: List[List[str]], 
                            retrieval_contents: Optional[List[List[str]]] = None) -> Dict[str, float]:
        
        if not generated_texts or not ground_truths:
            print("Error: Empty generated texts or ground truths")
            return {"mean_score": 0.0}
        
        # Ensure lists are of equal length for evaluation
        min_len = min(len(generated_texts), len(ground_truths))
        generated_texts = generated_texts[:min_len]
        ground_truths = ground_truths[:min_len]
        if retrieval_contents:
            retrieval_contents = retrieval_contents[:min_len]
        
        # Prepare metric inputs
        metric_inputs = self.prepare_metric_inputs(
            generated_texts, ground_truths, retrieval_contents
        )
        
        # Calculate metrics
        metric_results = {}
        normalized_scores = {}  # Store normalized scores for mean calculation
        metric_names, metric_params = cast_metrics([{"metric_name": name} for name in self.metrics])
        
        for metric_name, metric_param in zip(metric_names, metric_params):
            if metric_name not in self.metric_functions:
                print(f"Warning: Metric {metric_name} not supported. Skipping.")
                continue
                
            # Handle special case for semantic similarity with embedding model
            if metric_name == "sem_score":
                try:
                    if self.embedding_model_name == "openai":
                        embedding_model = OpenAIEmbedding()
                    else:
                        # Use HuggingFace embedding models
                        try:
                            embedding_model = embedding_models.get("huggingface_all_mpnet_base_v2")()
                        except (ImportError, AttributeError):
                            print("Warning: HuggingFace embedding models not available. Skipping sem_score.")
                            continue
                    
                    metric_param["embedding_model"] = embedding_model
                except Exception as e:
                    print(f"Error initializing embedding model: {e}")
                    print(f"Skipping {metric_name} metric")
                    continue
            
            try:
                print(f"Calculating {metric_name}...")
                
                if metric_name == "rouge":
                    if "batch" not in metric_param:
                        metric_param["batch"] = os.cpu_count() or 4
                
                scores = self.metric_functions[metric_name](
                    metric_inputs=metric_inputs,
                    **metric_param
                )

                if isinstance(scores, list):
                    valid_scores = [s for s in scores if isinstance(s, (int, float))]
                    if valid_scores:
                        metric_results[metric_name] = float(np.mean(valid_scores))
                    else:
                        metric_results[metric_name] = 0.0
                else:
                    metric_results[metric_name] = float(scores)
                
                # Normalize BLEU scores to 0-1 range for fair averaging
                if metric_name == "bleu" and metric_results[metric_name] > 1.0:
                    normalized_scores[metric_name] = metric_results[metric_name] / 100.0
                    print(f"  Normalized {metric_name}: {normalized_scores[metric_name]:.4f} (original: {metric_results[metric_name]:.4f})")
                else:
                    normalized_scores[metric_name] = metric_results[metric_name]
                
            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")
                metric_results[metric_name] = 0.0
                normalized_scores[metric_name] = 0.0
                
            if metric_name == "sem_score" and "embedding_model" in metric_param:
                del metric_param["embedding_model"]
                empty_cuda_cache()
        
        # Calculate mean score across all metrics using normalized values
        if normalized_scores:
            mean_score = float(np.mean(list(normalized_scores.values())))
            metric_results["mean_score"] = mean_score
            
            print(f"Generation evaluation results:")
            for metric, score in metric_results.items():
                if metric != "mean_score":
                    print(f"  {metric}: {score:.4f}")
            print(f"  mean_score (with normalized BLEU): {mean_score:.4f}")
        else:
            metric_results["mean_score"] = 0.0
        
        return metric_results
    
    def evaluate_from_dataframe(self, df: pd.DataFrame, qa_df: pd.DataFrame) -> Dict[str, float]:
        if 'generated_texts' not in df.columns:
            print("Error: DataFrame must contain 'generated_texts' column")
            return {"mean_score": 0.0}

        generated_texts = df['generated_texts'].tolist()
        ground_truths = self.extract_generation_ground_truths(qa_df)
        
        retrieval_contents = None
        if 'retrieved_contents' in df.columns:
            retrieval_contents = df['retrieved_contents'].tolist()
        
        return self.evaluate_generation(
            generated_texts=generated_texts,
            ground_truths=ground_truths,
            retrieval_contents=retrieval_contents
        )
