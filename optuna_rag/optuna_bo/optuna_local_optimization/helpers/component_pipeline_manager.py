import os
import shutil
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class ComponentPipelineManager:
    
    def __init__(self, config_generator, project_dir: str, corpus_data: pd.DataFrame, qa_data: pd.DataFrame):
        self.config_generator = config_generator
        self.project_dir = project_dir
        self.corpus_data = corpus_data
        self.qa_data = qa_data
    
    def prepare_component_data(self, component: str, qa_data: pd.DataFrame, 
                      component_dataframes: Dict[str, str], trial_dir: str) -> pd.DataFrame:
        qa_subset = qa_data.copy()
        
        if component not in ['query_expansion', 'retrieval']:
            component_order = [
                'query_expansion', 'retrieval', 'passage_reranker', 
                'passage_filter', 'passage_compressor', 'prompt_maker_generator'
            ]
            
            current_idx = component_order.index(component)

            for prev_idx in range(current_idx - 1, -1, -1):
                prev_comp = component_order[prev_idx]
                if prev_comp in component_dataframes:
                    best_parquet_path = component_dataframes[prev_comp]
                    if os.path.exists(best_parquet_path):
                        print(f"[{component}] Loading output from {prev_comp}: {best_parquet_path}")
                        best_prev_df = pd.read_parquet(best_parquet_path)
                        
                        if len(best_prev_df) == len(qa_subset):
                            for col in best_prev_df.columns:
                                if col not in ['query', 'retrieval_gt', 'generation_gt', 'qid']:
                                    qa_subset[col] = best_prev_df[col]

                            qa_subset.attrs['last_component'] = prev_comp
                            qa_subset.attrs['last_component_path'] = best_parquet_path
                        break
        
        return qa_subset
        
    def copy_corpus_data(self, trial_dir: str):
        centralized_corpus_path = os.path.join(self.project_dir, "data", "corpus.parquet")
        trial_corpus_path = os.path.join(trial_dir, "data", "corpus.parquet")
        if os.path.exists(centralized_corpus_path) and not os.path.exists(trial_corpus_path):
            shutil.copy2(centralized_corpus_path, trial_corpus_path)
    
    def save_component_output(self, component: str, trial_dir: str, 
                              results: Dict[str, Any], qa_data: pd.DataFrame) -> str:
        output_df = qa_data.copy()
        
        if component == 'query_expansion':
            if 'retrieval_df' in results:
                retrieval_df = results['retrieval_df']
                if isinstance(retrieval_df, pd.DataFrame):
                    for col in ['retrieved_ids', 'retrieved_contents', 'retrieve_scores']:
                        if col in retrieval_df.columns:
                            output_df[col] = retrieval_df[col]
                    if 'queries' in retrieval_df.columns:
                        output_df['queries'] = retrieval_df['queries']
            
            for col in ['retrieved_ids', 'retrieved_contents', 'retrieve_scores', 'queries']:
                if col in qa_data.columns and col not in output_df.columns:
                    output_df[col] = qa_data[col]
        
        elif component == 'retrieval':
            retrieval_columns = ['retrieved_ids', 'retrieved_contents', 'retrieve_scores']
            for col in retrieval_columns:
                if col in qa_data.columns:
                    output_df[col] = qa_data[col]
            
            if 'queries' in qa_data.columns:
                output_df['queries'] = qa_data['queries']
        
        elif component in ['passage_reranker', 'passage_filter', 'passage_compressor']:
            base_columns = ['query', 'retrieval_gt', 'generation_gt', 'qid']
            for col in qa_data.columns:
                if col not in base_columns and col not in output_df.columns:
                    output_df[col] = qa_data[col]
        
        elif component == 'prompt_maker_generator':
            for col in qa_data.columns:
                if col not in output_df.columns:
                    output_df[col] = qa_data[col]
            
            if 'eval_df' in results and isinstance(results['eval_df'], pd.DataFrame):
                eval_df = results['eval_df']
                for col in ['generated_texts', 'prompts']:
                    if col in eval_df.columns:
                        output_df[col] = eval_df[col]
        
        output_path = os.path.join(trial_dir, f"{component}_output.parquet")
        output_df.to_parquet(output_path)
        
        return output_path
    
    def get_component_score(self, component: str, results: Dict[str, Any]) -> float:
        if component == 'retrieval':
            return results.get('last_retrieval_score', 0.0)
        elif component == 'query_expansion':
            if results.get('query_expansion_score', 0.0) > 0:
                return results.get('query_expansion_score', 0.0)
            else:
                return results.get('retrieval_score', results.get('last_retrieval_score', 0.0))
        elif component == 'passage_reranker':
            return results.get('reranker_score', results.get('last_retrieval_score', 0.0))
        elif component == 'passage_filter':
            return results.get('filter_score', results.get('last_retrieval_score', 0.0))
        elif component == 'passage_compressor':
            return results.get('compression_score', results.get('last_retrieval_score', 0.0))
        elif component == 'prompt_maker_generator':
            if 'generation_score' in results and results['generation_score'] > 0:
                return results['combined_score']
            else:
                return results.get('prompt_maker_score', results.get('last_retrieval_score', 0.0))
        else:
            return results.get('combined_score', 0.0)
    
    def is_pass_component(self, component: str, config: Dict[str, Any]) -> bool:
        if component == 'passage_filter' and config.get('passage_filter_method') == 'pass_passage_filter':
            return True
        elif component == 'passage_reranker' and config.get('passage_reranker_method') == 'pass_reranker':
            return True
        elif component == 'passage_compressor' and config.get('passage_compressor_method') == 'pass_compressor':
            return True
        elif component == 'query_expansion' and config.get('query_expansion_method') == 'pass_query_expansion':
            return True
        return False
    
    def extract_detailed_metrics(self, component: str, results: Dict[str, Any]) -> Dict[str, Any]:
        detailed_metrics = {}
        
        if component == 'retrieval' or component == 'query_expansion':
            if 'retrieval_metrics' in results:
                detailed_metrics.update(results['retrieval_metrics'])
            if 'query_expansion_metrics' in results:
                detailed_metrics.update(results['query_expansion_metrics'])
        elif component == 'passage_reranker':
            if 'reranker_metrics' in results:
                detailed_metrics.update(results['reranker_metrics'])
        elif component == 'passage_filter':
            if 'filter_metrics' in results:
                detailed_metrics.update(results['filter_metrics'])
        elif component == 'passage_compressor':
            if 'compression_metrics' in results:
                detailed_metrics.update(results['compression_metrics'])
            if 'compressor_metrics' in results:
                detailed_metrics.update(results['compressor_metrics'])
        elif component == 'prompt_maker_generator':
            if 'prompt_maker_metrics' in results:
                detailed_metrics.update(results['prompt_maker_metrics'])
            if 'generation_metrics' in results:
                detailed_metrics.update(results['generation_metrics'])
        
        return detailed_metrics