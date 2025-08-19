import numpy as np
import pandas as pd
from typing import Dict, Any, List


class MetricInput:
    def __init__(self, retrieval_gt, retrieved_ids):
        self.retrieval_gt = retrieval_gt
        self.retrieved_ids = retrieved_ids


def retrieval_recall(metric_inputs):
    results = []
    for metric_input in metric_inputs:
        gt = metric_input.retrieval_gt
        pred = metric_input.retrieved_ids
        
        if not gt or not pred:
            results.append(0.0)
            continue
        
        gt_flat = set()
        for sublist in gt:
            if isinstance(sublist, list):
                gt_flat.update(sublist)
            else:
                gt_flat.add(sublist)
        
        if not gt_flat:
            results.append(0.0)
            continue
        
        pred_set = set(pred)
        hits = len(gt_flat.intersection(pred_set))
        recall = hits / len(gt_flat) if gt_flat else 0.0
        results.append(recall)
    
    return results


def retrieval_precision(metric_inputs):
    results = []
    for metric_input in metric_inputs:
        gt = metric_input.retrieval_gt
        pred = metric_input.retrieved_ids
        
        if not pred:
            results.append(0.0)
            continue
        
        gt_flat = set()
        for sublist in gt:
            if isinstance(sublist, list):
                gt_flat.update(sublist)
            else:
                gt_flat.add(sublist)
        
        pred_set = set(pred)
        hits = len(gt_flat.intersection(pred_set))
        precision = hits / len(pred_set) if pred_set else 0.0
        results.append(precision)
    
    return results


def retrieval_f1(metric_inputs):
    recalls = retrieval_recall(metric_inputs)
    precisions = retrieval_precision(metric_inputs)
    
    results = []
    for recall, precision in zip(recalls, precisions):
        if recall + precision == 0:
            results.append(0.0)
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
            results.append(f1)
    
    return results


def retrieval_ndcg(metric_inputs):
    results = []
    for metric_input in metric_inputs:
        gt = metric_input.retrieval_gt
        pred = metric_input.retrieved_ids
        
        if not gt or not pred:
            results.append(0.0)
            continue
        
        gt_flat = set()
        for sublist in gt:
            if isinstance(sublist, list):
                gt_flat.update(sublist)
            else:
                gt_flat.add(sublist)
        
        dcg = 0.0
        for i, doc_id in enumerate(pred):
            if doc_id in gt_flat:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_flat), len(pred))))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        results.append(ndcg)
    
    return results


def retrieval_mrr(metric_inputs):
    results = []
    for metric_input in metric_inputs:
        gt = metric_input.retrieval_gt
        pred = metric_input.retrieved_ids
        
        if not gt or not pred:
            results.append(0.0)
            continue
        
        gt_flat = set()
        for sublist in gt:
            if isinstance(sublist, list):
                gt_flat.update(sublist)
            else:
                gt_flat.add(sublist)
        
        for i, doc_id in enumerate(pred):
            if doc_id in gt_flat:
                results.append(1.0 / (i + 1))
                break
        else:
            results.append(0.0)
    
    return results


def retrieval_map(metric_inputs):
    results = []
    for metric_input in metric_inputs:
        gt = metric_input.retrieval_gt
        pred = metric_input.retrieved_ids
        
        if not gt or not pred:
            results.append(0.0)
            continue
        
        gt_flat = set()
        for sublist in gt:
            if isinstance(sublist, list):
                gt_flat.update(sublist)
            else:
                gt_flat.add(sublist)
        
        if not gt_flat:
            results.append(0.0)
            continue
        
        num_hits = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(pred):
            if doc_id in gt_flat:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                precision_sum += precision_at_i
        
        avg_precision = precision_sum / len(gt_flat) if gt_flat else 0.0
        results.append(avg_precision)
    
    return results


class EvaluationModule:
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = ["retrieval_f1", "retrieval_recall", "retrieval_precision", 
                      "retrieval_ndcg", "retrieval_map", "retrieval_mrr"]
        self.metrics = metrics
    
    def extract_ground_truths(self, qa_df):
        ground_truths = []
        for gt_item in qa_df['retrieval_gt'].tolist():
            if isinstance(gt_item, (list, np.ndarray)):
                if isinstance(gt_item, np.ndarray):
                    gt_list = gt_item.tolist()
                else:
                    gt_list = gt_item
                    
                if gt_list and not isinstance(gt_list[0], (list, np.ndarray)):
                    ground_truths.append([gt_list])
                else:
                    ground_truths.append(gt_list)
            elif isinstance(gt_item, str):
                if ',' in gt_item:
                    gt_list = [item.strip() for item in gt_item.split(',') if item.strip()]
                    ground_truths.append([gt_list])
                else:
                    ground_truths.append([[gt_item]])
            else:
                ground_truths.append([[]])
        
        print(f"Found {len(ground_truths)} ground truth items")
        empty_gt_count = sum(1 for gt in ground_truths if not gt[0])
        if empty_gt_count > 0:
            print(f"Warning: {empty_gt_count}/{len(ground_truths)} ground truths are empty")
        if ground_truths and ground_truths[0] and ground_truths[0][0]:
            print(f"Sample ground truth format: {ground_truths[0]}")
            
        return ground_truths
    
    def evaluate(self, retrieved_results, ground_truths, retrieved_scores=None, retrieved_contents=None):
        if not retrieved_results or not ground_truths:
            print("Error: Retrieved results or ground truths are empty.")
            return {metric: 0.0 for metric in self.metrics}
        
        print("\n--- DEBUG: first 10 retrieved–GT pairs ---------------------")  
        for i in range(min(10, len(ground_truths))):                           
            print(f"{i:2d}:  pred={retrieved_results[i][:10]}  |  gt={ground_truths[i]}")

        retrieval_results = {}
        for metric_name in self.metrics:
            metric_fn = {
                "retrieval_recall": retrieval_recall,
                "retrieval_precision": retrieval_precision,
                "retrieval_f1": retrieval_f1,
                "retrieval_ndcg": retrieval_ndcg,
                "retrieval_mrr": retrieval_mrr,
                "retrieval_map": retrieval_map,
            }.get(metric_name)

            if not metric_fn:
                continue

            metric_scores = []
            for i, gt in enumerate(ground_truths):
                pred = retrieved_results[i]
                if isinstance(pred, np.ndarray):
                    pred = pred.tolist()
                if isinstance(gt, np.ndarray):
                    gt = gt.tolist()

                if not pred or not gt:
                    metric_scores.append(0.0)
                    continue

                try:
                    metric_input = [MetricInput(retrieval_gt=gt, retrieved_ids=pred)]
                    score = metric_fn(metric_input)
                    metric_scores.append(float(score[0]) if isinstance(score, list) else float(score))
                except Exception as e:
                    print(f"Error calculating {metric_name} for query {i}: {e}")
                    metric_scores.append(0.0)

            if metric_scores:
                numeric_scores = [float(score) for score in metric_scores if isinstance(score, (int, float))]
                if numeric_scores:
                    avg_score = sum(numeric_scores) / len(numeric_scores)
                    retrieval_results[metric_name] = float(avg_score)
                else:
                    retrieval_results[metric_name] = 0.0
                    print(f"Warning: No valid numeric scores for {metric_name}")
            else:
                retrieval_results[metric_name] = 0.0
                print(f"Warning: Empty metric scores list for {metric_name}")
        
        if retrieval_results:
            composite_score = float(np.mean(list(retrieval_results.values())))
            print(f"Composite score: {composite_score}")
            print(f"Individual metrics: {retrieval_results}")
            
            return {
                "mean_accuracy": composite_score,
                **retrieval_results
            }
        else:
            return {"mean_accuracy": 0.0}