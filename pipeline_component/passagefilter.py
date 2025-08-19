import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

from autorag.nodes.passagefilter.percentile_cutoff import PercentileCutoff
from autorag.nodes.passagefilter.threshold_cutoff import ThresholdCutoff
from autorag.nodes.passagefilter.similarity_threshold_cutoff import SimilarityThresholdCutoff
from autorag.nodes.passagefilter.similarity_percentile_cutoff import SimilarityPercentileCutoff


class PassageFilterModule:
    
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
    
    def _validate_and_clean_for_embedding(self, contents_list, ids_list, scores_list):
        cleaned_contents_list = []
        cleaned_ids_list = []
        cleaned_scores_list = []
        
        for contents, ids, scores in zip(contents_list, ids_list, scores_list):
            cleaned_contents = []
            cleaned_ids = []
            cleaned_scores = []
            
            for content, id_, score in zip(contents, ids, scores):
                if content and isinstance(content, str) and content.strip():
                    content = ' '.join(content.split())
                    # Truncate very long content 
                    if len(content) > 8000:
                        content = content[:8000] + "..."
                    cleaned_contents.append(content)
                    cleaned_ids.append(id_)
                    cleaned_scores.append(score)
            
            if not cleaned_contents:
                cleaned_contents.append("No valid content")
                cleaned_ids.append("placeholder")
                cleaned_scores.append(0.0)
                
            cleaned_contents_list.append(cleaned_contents)
            cleaned_ids_list.append(cleaned_ids)
            cleaned_scores_list.append(cleaned_scores)
        
        return cleaned_contents_list, cleaned_ids_list, cleaned_scores_list

    def apply_filter_directly(self, filter_type: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        contents_list = df['retrieved_contents'].tolist()
        scores_list = df['retrieve_scores'].tolist()
        ids_list = df['retrieved_ids'].tolist()
        
        if filter_type == 'threshold_cutoff':
            threshold = kwargs.get('threshold', 0.85)
            reverse = kwargs.get('reverse', False)
            
            threshold_filter = ThresholdCutoff(project_dir=self.project_dir)
            
            contents, ids, scores = threshold_filter._pure(
                contents_list=contents_list,
                scores_list=scores_list,
                ids_list=ids_list,
                threshold=threshold,
                reverse=reverse
            )
            
        elif filter_type == 'percentile_cutoff':
            percentile = kwargs.get('percentile', 0.6)
            reverse = kwargs.get('reverse', False)
            
            percentile_filter = PercentileCutoff(project_dir=self.project_dir)

            queries = df['query'].tolist()
            contents, ids, scores = percentile_filter._pure(
                queries=queries,
                contents_list=contents_list,
                scores_list=scores_list,
                ids_list=ids_list,
                percentile=percentile,
                reverse=reverse
            )
        
        elif filter_type == 'similarity_percentile_cutoff':
            percentile = kwargs.get('percentile', 0.6)
            batch = kwargs.get('batch', 128)
            embedding_model = kwargs.get('embedding_model', 'openai')
            
            cleaned_contents_list, cleaned_ids_list, cleaned_scores_list = self._validate_and_clean_for_embedding(
                contents_list, ids_list, scores_list
            )

            similarity_percentile_filter = SimilarityPercentileCutoff(
                project_dir=self.project_dir,
                embedding_model=embedding_model
            )

            queries = df['query'].tolist()
            ids, contents, scores = similarity_percentile_filter._pure(
                queries=queries,
                contents_list=cleaned_contents_list,
                scores_list=cleaned_scores_list,
                ids_list=cleaned_ids_list,
                percentile=percentile,
                batch=batch
            )
        
        elif filter_type == 'similarity_threshold_cutoff':
            threshold = kwargs.get('threshold', 0.7)
            batch = kwargs.get('batch', 128)
            embedding_model = kwargs.get('embedding_model', 'openai')

            cleaned_contents_list, cleaned_ids_list, cleaned_scores_list = self._validate_and_clean_for_embedding(
                contents_list, ids_list, scores_list
            )

            similarity_filter = SimilarityThresholdCutoff(
                project_dir=self.project_dir,
                embedding_model=embedding_model
            )

            queries = df['query'].tolist()
            contents, ids, scores = similarity_filter._pure(
                queries=queries,
                contents_list=cleaned_contents_list,
                scores_list=cleaned_scores_list,
                ids_list=cleaned_ids_list,
                threshold=threshold,
                batch=batch
            )
        
        else:
            return df

        result_df = df.copy()
        result_df['retrieved_contents'] = contents
        result_df['retrieved_ids'] = ids
        result_df['retrieve_scores'] = scores
        
        return result_df