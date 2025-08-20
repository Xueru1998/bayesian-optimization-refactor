import os
import pandas as pd
from typing import List, Dict, Any, Union, Optional

from autorag.nodes.promptmaker.base import BasePromptMaker
from autorag.nodes.promptmaker.fstring import Fstring
from autorag.nodes.promptmaker.long_context_reorder import LongContextReorder
from autorag.nodes.promptmaker.window_replacement import WindowReplacement
from pipeline.config_manager import ConfigGenerator


class PromptMakerModule:
    
    def __init__(self, project_dir: str, config_manager: ConfigGenerator):
        self.project_dir = project_dir
        self.config_manager = config_manager  
        self.method_type = 'fstring'
        self.template_idx = 0

        self.corpus_data = None
        try:
            data_dir = os.path.join(project_dir, "data")
            corpus_path = os.path.join(data_dir, "corpus.parquet")
            if os.path.exists(corpus_path):
                self.corpus_data = pd.read_parquet(corpus_path, engine="pyarrow")
        except Exception as e:
            print(f"Warning: Could not load corpus data for window replacement: {e}")
        
    def create_prompts(self, 
                       queries: List[str],
                       retrieved_contents: List[List[str]],
                       retrieve_scores: Optional[List[List[float]]] = None,
                       retrieved_ids: Optional[List[List[str]]] = None) -> List[str]:
       
        print("Creating prompts from retrieved contents...")
        
        prompt_methods, prompt_indices = self.config_manager.extract_prompt_maker_options()
        
        module_type = prompt_methods[0] if prompt_methods else 'fstring'
        prompt_template_idx = prompt_indices[0] if prompt_indices else 0
        
        prompt_templates = self.config_manager.get_prompt_templates_from_config(
            self.config_manager.config_template, module_type
        )
        
        prompt_template = prompt_templates[prompt_template_idx] if prompt_templates else \
            "Answer this question: {query}\n\nContext: {retrieved_contents}\n\nAnswer:"
        
        print(f"Using prompt module: {module_type}, template index: {prompt_template_idx}")
        
        if module_type == 'fstring':
            return self.apply_fstring_prompt(queries, retrieved_contents, prompt_template)
        elif module_type == 'long_context_reorder':
            if not retrieve_scores:
                print("Warning: retrieve_scores is required for long_context_reorder. Falling back to fstring.")
                return self.apply_fstring_prompt(queries, retrieved_contents, prompt_template)
            return self.apply_long_context_reorder(queries, retrieved_contents, retrieve_scores, prompt_template)
        elif module_type == 'window_replacement':
            if not retrieved_ids or not self.corpus_data:
                print("Warning: retrieved_ids and corpus data are required for window_replacement. Falling back to fstring.")
                return self.apply_fstring_prompt(queries, retrieved_contents, prompt_template)
            return self.apply_window_replacement(queries, retrieved_contents, retrieved_ids, prompt_template)
        else:
            print(f"Warning: Unknown prompt maker module type '{module_type}'. Using default fstring.")
            return self.apply_fstring_prompt(queries, retrieved_contents, prompt_template)
    
    def apply_fstring_prompt(self, 
                              queries: List[str], 
                              retrieved_contents: List[List[str]], 
                              prompt_template: str) -> List[str]:
        fstring_module = Fstring(project_dir=self.project_dir)
        prompts = fstring_module._pure(prompt_template, queries, retrieved_contents)
        return prompts
    
    def apply_long_context_reorder(self, 
                                   queries: List[str], 
                                   retrieved_contents: List[List[str]], 
                                   retrieve_scores: List[List[float]],
                                   prompt_template: str) -> List[str]:
        reorder_module = LongContextReorder(project_dir=self.project_dir)
        prompts = reorder_module._pure(prompt_template, queries, retrieved_contents, retrieve_scores)
        return prompts
    
    def apply_window_replacement(self,
                                queries: List[str],
                                retrieved_contents: List[List[str]],
                                retrieved_ids: List[List[str]],
                                prompt_template: str) -> List[str]:
        window_module = WindowReplacement(project_dir=self.project_dir)
        
        retrieved_metadata = []
        for ids_list in retrieved_ids:
            metadata_list = []
            for doc_id in ids_list:
                corpus_row = self.corpus_data[self.corpus_data['doc_id'] == doc_id]
                if not corpus_row.empty and 'metadata' in corpus_row.columns:
                    metadata = corpus_row['metadata'].iloc[0]
                    if isinstance(metadata, str):
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                else:
                    metadata = {}
                metadata_list.append(metadata)
            retrieved_metadata.append(metadata_list)
        
        prompts = window_module._pure(prompt_template, queries, retrieved_contents, retrieved_metadata)
        return prompts
    
    def create_prompts_from_dataframe(self, df, method_type=None, template_idx=None):
        if method_type is not None:
            self.method_type = method_type
            
        if template_idx is not None:
            self.template_idx = template_idx
            
        print(f"Creating prompts using method={self.method_type}, template_idx={self.template_idx}")
        
        templates = self.config_manager.get_prompt_templates_from_config(
            self.config_manager.config_template, 
            self.method_type
        )
        
        if self.template_idx >= len(templates):
            self.template_idx = 0
            
        template = templates[self.template_idx]
        print(f"Using template: {template}")
        
        result_df = df.copy()
        
        prompts = []
        for _, row in df.iterrows():
            query = row['query']
            retrieved_contents = row['retrieved_contents']
            
            if isinstance(retrieved_contents, list):
                retrieved_text = "\n\n".join(retrieved_contents)
            else:
                retrieved_text = str(retrieved_contents)
            
            if self.method_type == 'window_replacement' and 'retrieved_ids' in row and self.corpus_data is not None:
                retrieved_ids = row['retrieved_ids']
                if isinstance(retrieved_ids, list):
                    metadata_list = []
                    for doc_id in retrieved_ids:
                        corpus_row = self.corpus_data[self.corpus_data['doc_id'] == doc_id]
                        if not corpus_row.empty and 'metadata' in corpus_row.columns:
                            metadata = corpus_row['metadata'].iloc[0]
                            if isinstance(metadata, str):
                                try:
                                    import json
                                    metadata = json.loads(metadata)
                                except:
                                    metadata = {}
                        else:
                            metadata = {}
                        metadata_list.append(metadata)
                    
                    window_list = []
                    if isinstance(retrieved_contents, list):
                        for content, metadata in zip(retrieved_contents, metadata_list):
                            if "window" in metadata:
                                window_list.append(metadata["window"])
                            else:
                                window_list.append(content)
                    else:
                        window_list = [retrieved_text]
                    
                    retrieved_text = "\n\n".join(window_list)
                
            if self.method_type in ['fstring', 'long_context_reorder', 'window_replacement']:
                prompt = template.format(
                    query=query,
                    retrieved_contents=retrieved_text
                )
            else:
                prompt = f"Answer this question based on the context: {query}\n\nContext: {retrieved_text}"
            
            prompts.append(prompt)
            
        result_df['prompts'] = prompts
        return result_df
