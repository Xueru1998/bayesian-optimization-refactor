import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
import ast
import json

from autorag.evaluation.metric import (
    retrieval_token_f1,
    retrieval_token_precision,
    retrieval_token_recall,
)
from autorag.schema.metricinput import MetricInput
from autorag.evaluation.util import cast_metrics


class TokenEvaluatorModule:
    
    def __init__(self, metrics=None, corpus_data=None, fetch_contents_fn=None):
        if metrics is None:
            metrics = ["retrieval_token_f1", "retrieval_token_precision", "retrieval_token_recall"]
                
        self.metrics = metrics
        self.corpus_data = corpus_data
        self.fetch_contents = fetch_contents_fn if fetch_contents_fn is not None else self.default_fetch_contents
        
        self.corpus_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../..", 
            "autorag_project", 
            "corpus.parquet"
        ))
        
        self.metric_functions = {
            "retrieval_token_f1": retrieval_token_f1,
            "retrieval_token_precision": retrieval_token_precision,
            "retrieval_token_recall": retrieval_token_recall,
        }

    def default_fetch_contents(self, corpus_data, doc_ids):
        if corpus_data is None:
            try:
                if os.path.exists(self.corpus_path):
                    print(f"Loading corpus data from {self.corpus_path}")
                    corpus_data = pd.read_parquet(self.corpus_path)
                    self.corpus_data = corpus_data  
                else:
                    print(f"Warning: Corpus file not found at {self.corpus_path}")
                    return []
            except Exception as e:
                print(f"Error loading corpus data: {e}")
                return []
        
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]
        
        processed_ids = []
        for item in doc_ids:
            if isinstance(item, str):
                if item.startswith("[") and item.endswith("]"):
                    try:
                        parsed = ast.literal_eval(item)
                        if isinstance(parsed, list):
                            processed_ids.extend([str(x) for x in parsed])
                        else:
                            processed_ids.append(str(item))
                    except:
                        processed_ids.append(str(item))
                else:
                    processed_ids.append(str(item))
            else:
                processed_ids.append(str(item))
        
        doc_ids = processed_ids

        contents = []
        if isinstance(corpus_data, pd.DataFrame):
            for doc_id in doc_ids:
                matches = corpus_data[corpus_data['doc_id'] == doc_id]
                if len(matches) > 0:
                    if 'contents' in matches.columns:
                        content = matches.iloc[0]['contents']
                        if isinstance(content, str) and '\n contents:' in content:
                            content = content.split('\n contents:', 1)[1].strip()
                        contents.append(content)
                    else:
                        print(f"Warning: Found document {doc_id} but no 'contents' column available")
                        contents.append("")
                else:
                    print(f"Warning: Document ID {doc_id} not found in corpus DataFrame")
                    contents.append("")
        
        elif isinstance(corpus_data, dict):
            for doc_id in doc_ids:
                if doc_id in corpus_data:
                    doc = corpus_data[doc_id]
                    if isinstance(doc, dict) and 'contents' in doc:
                        content = doc['contents']
                        if isinstance(content, str) and '\n contents:' in content:
                            content = content.split('\n contents:', 1)[1].strip()
                        contents.append(content)
                    else:
                        contents.append(str(doc))
                else:
                    print(f"Warning: Document ID {doc_id} not found in corpus data")
                    contents.append("")
        
        elif isinstance(corpus_data, list):
            corpus_dict = {doc.get('doc_id', i): doc for i, doc in enumerate(corpus_data) if isinstance(doc, dict)}
            for doc_id in doc_ids:
                if doc_id in corpus_dict:
                    doc = corpus_dict[doc_id]
                    if 'contents' in doc:
                        content = doc['contents']
                        if isinstance(content, str) and '\n contents:' in content:
                            content = content.split('\n contents:', 1)[1].strip()
                        contents.append(content)
                    else:
                        contents.append(str(doc))
                else:
                    print(f"Warning: Document ID {doc_id} not found in corpus list")
                    contents.append("")
        
        else:
            print(f"Warning: Unsupported corpus_data type: {type(corpus_data)}")
        
        return contents

    
    def prepare_metric_inputs(self, retrieved_contents: List[List[str]], 
                              retrieval_gt_contents: List[List[str]]) -> List[MetricInput]:
        metric_inputs = []
        
        for i in range(len(retrieved_contents)):
            gt_contents = retrieval_gt_contents[i] if i < len(retrieval_gt_contents) else []

            contents = retrieved_contents[i] if i < len(retrieved_contents) else []
            if isinstance(contents, str):
                contents = [contents]
            elif isinstance(contents, np.ndarray):
                contents = [str(c) if not isinstance(c, str) else c for c in contents]
            
            if isinstance(gt_contents, str):
                gt_contents = [gt_contents]
            elif isinstance(gt_contents, np.ndarray):
                gt_contents = [str(c) if not isinstance(c, str) else c for c in gt_contents]
            
            input_kwargs = {
                "retrieved_contents": contents,
                "retrieval_gt_contents": [gt_contents]  
            }
                            
            metric_inputs.append(MetricInput(**input_kwargs))
        
        return metric_inputs
    
    def extract_retrieval_ground_truths(self, qa_df: pd.DataFrame) -> List[List[str]]:
        ground_truths = []
        
        if 'retrieval_gt' in qa_df.columns:
            column_name = 'retrieval_gt'
        elif 'retrieval_gt_contents' in qa_df.columns:
            column_name = 'retrieval_gt_contents'
        else:
            print("Warning: Neither 'retrieval_gt' nor 'retrieval_gt_contents' column found in QA dataframe.")
            return [[] for _ in range(len(qa_df))]
                
        if self.corpus_data is None:
            try:
                if os.path.exists(self.corpus_path):
                    print(f"Loading corpus data from {self.corpus_path} for ground truth resolution")
                    self.corpus_data = pd.read_parquet(self.corpus_path)
                    print(f"Loaded corpus data with {len(self.corpus_data)} documents")
                else:
                    print(f"Warning: Corpus file not found at {self.corpus_path}")
            except Exception as e:
                print(f"Error loading corpus data: {e}")
        
        for gt_item in qa_df[column_name].tolist():
            if isinstance(gt_item, list):
                ground_truths.append([str(item) if not isinstance(item, str) else item for item in gt_item])
            elif isinstance(gt_item, np.ndarray):
                ground_truths.append([str(item) if not isinstance(item, str) else item for item in gt_item.tolist()])
            elif isinstance(gt_item, str):
                ground_truths.append([gt_item])
            else:
                ground_truths.append([str(gt_item)] if gt_item is not None else [])
        
        print(f"Retrieved {len(ground_truths)} ground truth items")
        empty_gt_count = sum(1 for gt in ground_truths if not gt)
        if empty_gt_count > 0:
            print(f"Warning: {empty_gt_count}/{len(ground_truths)} ground truths are empty")
                
        try:
            print("Resolving document IDs to content...")
            resolved_gt_contents = []
            for gt_list in ground_truths:
                if gt_list:
                    contents = self.fetch_contents(self.corpus_data, gt_list)
                    if contents and isinstance(contents, list):
                        resolved_gt_contents.append(contents)
                    else:
                        resolved_gt_contents.append(gt_list)
                else:
                    resolved_gt_contents.append(gt_list)
            
            if any(gt != original_gt for gt, original_gt in zip(resolved_gt_contents, ground_truths) if gt and original_gt):
                print("Successfully resolved document IDs to content")
                ground_truths = resolved_gt_contents
            else:
                print("No document IDs were resolved to content")
        except Exception as e:
            print(f"Error resolving document IDs to content: {e}")
                
        if ground_truths and len(ground_truths) > 0 and ground_truths[0]:
            print(f"First processed ground truth content: {ground_truths[0][0][:100]}...")
                
        return ground_truths
    
    def evaluate_retrieval(self, retrieved_contents: List[List[str]],     
                           retrieval_gt_contents: List[List[str]]) -> Dict[str, float]:
        
        if not retrieved_contents or not retrieval_gt_contents:
            print("Error: Empty retrieved contents or ground truths")            
            return {"mean_score": 0.0}

        min_len = min(len(retrieved_contents), len(retrieval_gt_contents))
        retrieved_contents = retrieved_contents[:min_len]                   
        retrieval_gt_contents = retrieval_gt_contents[:min_len]

        metric_inputs = self.prepare_metric_inputs(
            retrieved_contents, retrieval_gt_contents
        )

        metric_results = {}
        metric_names, metric_params = cast_metrics([{"metric_name": name} for name in self.metrics])
        
        for metric_name, metric_param in zip(metric_names, metric_params):
            if metric_name not in self.metric_functions:
                print(f"Warning: Metric {metric_name} not supported. Skipping.")
                continue
            
            try:
                print(f"Calculating {metric_name}...")
                for input_item in metric_inputs:
                    if hasattr(input_item, 'retrieved_contents'):
                        input_item.retrieved_contents = [
                            str(content) if not isinstance(content, str) else content
                            for content in input_item.retrieved_contents
                        ]
                    if hasattr(input_item, 'retrieval_gt_contents') and input_item.retrieval_gt_contents:
                        input_item.retrieval_gt_contents = [
                            [str(content) if not isinstance(content, str) else content 
                             for content in gt_list]
                            for gt_list in input_item.retrieval_gt_contents
                        ]
                
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
            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")
                metric_results[metric_name] = 0.0

        if metric_results:
            mean_score = float(np.mean(list(metric_results.values())))        
            metric_results["mean_score"] = mean_score
            print(f"Token evaluation results:")
            for metric, score in metric_results.items():
                print(f"  {metric}: {score:.4f}")
        else:
            metric_results["mean_score"] = 0.0

        return metric_results
    
    def evaluate_from_dataframe(self, df: pd.DataFrame, qa_df: pd.DataFrame, corpus_data=None, fetch_contents_fn=None) -> Dict[str, float]:

        if corpus_data is not None:
            self.corpus_data = corpus_data
        if fetch_contents_fn is not None:
            self.fetch_contents = fetch_contents_fn
            
        if self.corpus_data is None:
            try:
                if os.path.exists(self.corpus_path):
                    print(f"Loading corpus data from {self.corpus_path}")
                    self.corpus_data = pd.read_parquet(self.corpus_path)
                    print(f"Loaded corpus with {len(self.corpus_data)} documents")
                else:
                    print(f"Warning: Corpus file not found at {self.corpus_path}")
            except Exception as e:
                print(f"Error loading corpus data: {e}")

        if 'retrieved_contents' not in df.columns:
            print("Error: DataFrame must contain 'retrieved_contents' column")
            return {"mean_score": 0.0}
        
        retrieved_contents = df['retrieved_contents'].tolist()

        if retrieved_contents:
            processed_contents = []
            for content in retrieved_contents:
                if isinstance(content, list):
                    processed_contents.append([str(item) if not isinstance(item, str) else item for item in content])
                elif isinstance(content, np.ndarray):
                    processed_contents.append([str(item) if not isinstance(item, str) else item for item in content.tolist()])
                else:
                    processed_contents.append([str(content) if not isinstance(content, str) else content])
            retrieved_contents = processed_contents

        retrieval_gt_contents = self.extract_retrieval_ground_truths(qa_df)
        
        return self.evaluate_retrieval(
            retrieved_contents=retrieved_contents,
            retrieval_gt_contents=retrieval_gt_contents
        )
