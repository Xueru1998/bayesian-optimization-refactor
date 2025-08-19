import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from pipeline.rag_pipeline_runner import RAGPipelineRunner
from pipeline.pipeline_executor import RAGPipelineExecutor
from pipeline.config_manager import ConfigGenerator
from pipeline.search_space_calculator import SearchSpaceCalculator


class ComponentwiseRAGProcessor:
    
    def __init__(
        self,
        config_generator: ConfigGenerator,
        retrieval_weight: float,
        generation_weight: float,
        project_dir: str,
        corpus_data: pd.DataFrame,
        qa_data: pd.DataFrame,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o"
    ):
        self.config_generator = config_generator
        self.config_template = config_generator.config_template
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
        self.project_dir = project_dir
        self.corpus_data = corpus_data
        self.qa_data = qa_data
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
        self._setup_metrics()
        self._setup_pipeline_runner()
        self.executor = RAGPipelineExecutor(config_generator)
    
    def _setup_metrics(self):
        self.retrieval_metrics = self.config_generator.extract_retrieval_metrics_from_config()
        self.filter_metrics = self.config_generator.extract_passage_filter_metrics_from_config()
        self.compressor_metrics = self.config_generator.extract_passage_compressor_metrics_from_config()
        self.reranker_metrics = self.config_generator.extract_passage_reranker_metrics_from_config()
        self.generation_metrics = self.config_generator.extract_generation_metrics_from_config()
        self.prompt_maker_metrics = self.config_generator.extract_generation_metrics_from_config('prompt_maker')
        self.query_expansion_metrics = self.config_generator.extract_query_expansion_metrics_from_config()
    
    def _setup_pipeline_runner(self):
        self.pipeline_runner = RAGPipelineRunner(
            config_generator=self.config_generator,
            retrieval_metrics=self.retrieval_metrics,
            filter_metrics=self.filter_metrics,
            compressor_metrics=self.compressor_metrics,
            generation_metrics=self.generation_metrics,
            prompt_maker_metrics=self.prompt_maker_metrics,
            query_expansion_metrics=self.query_expansion_metrics,
            reranker_metrics=self.reranker_metrics,
            retrieval_weight=self.retrieval_weight,
            generation_weight=self.generation_weight,
            use_llm_compressor_evaluator=self.use_llm_compressor_evaluator,
            llm_evaluator_model=self.llm_evaluator_model
        )
        
    def get_component_order(self) -> List[str]:
        component_order = []
        if 'node_lines' in self.config_template:
            for node_line in self.config_template['node_lines']:
                nodes = node_line.get('nodes', [])
                for node in nodes:
                    node_type = node.get('node_type')
                    if node_type and node_type not in component_order:
                        component_order.append(node_type)

        if not component_order:
            return [
                'query_expansion',
                'retrieval',
                'passage_reranker',
                'passage_filter', 
                'passage_compressor',
                'prompt_maker',
                'generator'
            ]
        
        return component_order
    
    def parse_trial_config(self, component: str, trial_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        is_pass_component = False
        
        config_mapping = {
            'query_expansion': ('query_expansion_config', self.executor._parse_query_expansion_config),
            'retrieval': ('retrieval_config', self.executor._parse_retrieval_config),
            'passage_reranker': ('reranker_config', self.executor._parse_reranker_config),
            'passage_filter': ('passage_filter_config', self.executor._parse_filter_config),
            'passage_compressor': ('compressor_config', self.executor._parse_compressor_config),
            'prompt_maker_generator': ('prompt_config', self.executor._parse_prompt_config)
        }
        
        if component in config_mapping:
            config_key, parse_func = config_mapping[component]
            
            if config_key in trial_config:
                config_str = trial_config[config_key]
                
                if component == 'retrieval' and config_str.startswith('bm25_'):
                    tokenizer = config_str.replace('bm25_', '')
                    trial_config['retrieval_method'] = 'bm25'
                    trial_config['bm25_tokenizer'] = tokenizer
                elif component == 'retrieval' and config_str.startswith('vectordb_'):
                    vdb_name = config_str.replace('vectordb_', '')
                    trial_config['retrieval_method'] = 'vectordb'
                    trial_config['vectordb_name'] = vdb_name
                else:
                    parsed_config = parse_func(config_str)
                    trial_config.update(parsed_config)
                    
                    pass_methods = {
                        'query_expansion': 'pass_query_expansion',
                        'passage_reranker': 'pass_reranker',
                        'passage_filter': 'pass_passage_filter',
                        'passage_compressor': 'pass_compressor',
                        'prompt_maker_generator': 'pass_prompt_maker'
                    }
                    
                    if component in pass_methods and config_str == pass_methods[component]:
                        is_pass_component = True
                
                trial_config.pop(config_key, None)
        
        if component == 'passage_compressor':
            if 'passage_compressor_config' in trial_config:
                comp_config_str = trial_config['passage_compressor_config']
                if comp_config_str == 'pass_compressor':
                    is_pass_component = True
                    trial_config['passage_compressor_method'] = 'pass_compressor'
                else:
                    parts = comp_config_str.split('::')
                    if len(parts) >= 2:
                        method = parts[0]
                        trial_config['passage_compressor_method'] = method
                        
                        if method in ['tree_summarize', 'refine']:
                            llm_model = parts[1]
                            trial_config['compressor_llm_model'] = llm_model
                            if '_' in llm_model:
                                llm, model = llm_model.split('_', 1)
                                trial_config['compressor_llm'] = llm
                                trial_config['compressor_model'] = model
                        
                        elif method == 'lexrank':
                            trial_config['compressor_compression_ratio'] = float(parts[1])
                            trial_config['compressor_threshold'] = 0.1
                            trial_config['compressor_damping'] = 0.85
                            trial_config['compressor_max_iterations'] = 30
                        
                        elif method == 'spacy':
                            trial_config['compressor_compression_ratio'] = float(parts[1])
                            if len(parts) >= 3:
                                trial_config['compressor_spacy_model'] = parts[2]
                            else:
                                trial_config['compressor_spacy_model'] = 'en_core_web_sm'
                
                trial_config.pop('passage_compressor_config', None)
            
            elif 'compressor_llm_model' in trial_config:
                llm_model = trial_config.pop('compressor_llm_model', '')
                if '_' in llm_model:
                    llm, model = llm_model.split('_', 1)
                    trial_config['compressor_llm'] = llm
                    trial_config['compressor_model'] = model
        
        if component in ['generator', 'prompt_maker_generator'] and 'generator_config' in trial_config:
            gen_config_str = trial_config['generator_config']
            parts = gen_config_str.split('::', 1)
            if len(parts) == 2:
                module_type, model = parts
                trial_config['generator_module_type'] = module_type
                trial_config['generator_model'] = model
                
                unified_params = self.config_generator.extract_unified_parameters('generator')
                for module_config in unified_params.get('module_configs', []):
                    if module_config['module_type'] == module_type and model in module_config['models']:
                        trial_config['generator_llm'] = module_config.get('llm', 'openai')
                        break
            
            trial_config.pop('generator_config', None)
        
        if not is_pass_component:
            pass_method_keys = {
                'passage_filter': ('passage_filter_method', 'pass_passage_filter'),
                'passage_reranker': ('passage_reranker_method', 'pass_reranker'),
                'passage_compressor': ('passage_compressor_method', 'pass_compressor'),
                'query_expansion': ('query_expansion_method', 'pass_query_expansion')
            }
            
            if component in pass_method_keys:
                method_key, pass_value = pass_method_keys[component]
                if trial_config.get(method_key) == pass_value:
                    is_pass_component = True
        
        return trial_config, is_pass_component
    
    def parse_component_config(self, component: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if component == 'retrieval' and 'retrieval_config' in config:
            parsed_config = self.executor._parse_retrieval_config(config['retrieval_config'])
            config.update(parsed_config)
            config.pop('retrieval_config', None)
        return config
    
    def get_fixed_config(self, component: str, best_configs: Dict[str, Any], 
                        component_order: List[str]) -> Dict[str, Any]:
        fixed_config = {}
        
        for prev_component in component_order[:component_order.index(component)]:
            if prev_component in best_configs:
                fixed_config.update(best_configs[prev_component])
        
        if component == 'retrieval' and 'query_expansion_method' in fixed_config:
            if fixed_config['query_expansion_method'] != 'pass_query_expansion':
                fixed_config['retrieval_method'] = 'pass_retrieval'
        
        if 'query_expansion' in best_configs and best_configs['query_expansion']:
            qe_best = best_configs['query_expansion']
            if qe_best.get('query_expansion_method') != 'pass_query_expansion':
                if 'retrieval_config' in qe_best:
                    parsed = self.executor._parse_retrieval_config(qe_best['retrieval_config'])
                    fixed_config.update(parsed)
                if 'retriever_top_k' in qe_best:
                    fixed_config['retriever_top_k'] = qe_best['retriever_top_k']
        
        return fixed_config
    
    def load_previous_outputs(self, component: str, qa_subset: pd.DataFrame, 
                    component_dataframes: Dict[str, str], 
                    trial_dir: str) -> pd.DataFrame:
        prev_components = ['query_expansion', 'retrieval', 'passage_reranker', 
                        'passage_filter', 'passage_compressor', 'prompt_maker_generator']
        
        best_output_loaded = False
        
        if component not in ['query_expansion', 'retrieval']:
            current_idx = prev_components.index(component) if component in prev_components else len(prev_components)
            
            for prev_comp in reversed(prev_components[:current_idx]):
                if prev_comp in component_dataframes:
                    best_parquet_path = component_dataframes[prev_comp]
                    if os.path.exists(best_parquet_path):
                        print(f"[{component}] Using best {prev_comp} output from: {best_parquet_path}")
                        best_prev_df = pd.read_parquet(best_parquet_path)
                        
                        if len(best_prev_df) == len(qa_subset):
                            for col in best_prev_df.columns:
                                if col not in ['query', 'retrieval_gt', 'generation_gt', 'qid']:
                                    qa_subset[col] = best_prev_df[col]
                            best_output_loaded = True
                        
                        trial_prev_path = os.path.join(trial_dir, f"{prev_comp}_output.parquet")
                        best_prev_df.to_parquet(trial_prev_path)
                        break
                    
        return qa_subset
    
    def run_pipeline(self, config: Dict[str, Any], trial_dir: str, qa_subset: pd.DataFrame,
                component: str, component_results: Dict[str, Any]) -> Dict[str, Any]:

        if self.pipeline_runner is None:
            self._initialize_pipeline_runner()

        self.pipeline_runner.component_results = component_results

        full_config = {}

        component_order = self.get_component_order()
        for prev_component in component_order:
            if prev_component == component:
                full_config.update(config)
                break
            if prev_component in component_results and 'best_config' in component_results[prev_component]:
                full_config.update(component_results[prev_component]['best_config'])

        full_config.update(config)

        results = self.pipeline_runner.run_pipeline(
            config=full_config, 
            trial_dir=trial_dir, 
            qa_subset=qa_subset,
            is_local_optimization=True,
            current_component=component
        )
        
        if results is None:
            print(f"[WARNING] Pipeline returned None for component {component}")
            results = {}
        
        if 'working_df' not in results:
            results['working_df'] = qa_subset
        
        return results
    
    def extract_detailed_metrics(self, component: str, results: Dict[str, Any]) -> Dict[str, Any]:
        detailed_metrics = {}
        
        metrics_mapping = {
            'retrieval': ['retrieval_metrics'],
            'query_expansion': ['retrieval_metrics', 'query_expansion_metrics'],
            'passage_reranker': ['reranker_metrics'],
            'passage_filter': ['filter_metrics'],
            'passage_compressor': ['compression_metrics', 'compressor_metrics'],
            'prompt_maker_generator': ['prompt_maker_metrics', 'generation_metrics']
        }
        
        for metric_key in metrics_mapping.get(component, []):
            if metric_key in results:
                detailed_metrics.update(results[metric_key])
        
        return detailed_metrics
    
    def calculate_component_score(self, component: str, results: Dict[str, Any], 
                            is_pass_component: bool, 
                            component_results: Dict[str, Any]) -> float:
        if is_pass_component:
            if component == 'query_expansion':
                print(f"[{component}] Pass query expansion detected, will run retrieval only")
                score = results.get('last_retrieval_score', 0.0)
            elif component == 'retrieval':
                print(f"[{component}] Retrieval component - no inheritance needed")
                score = results.get('last_retrieval_score', 0.0)
            else:
                print(f"[{component}] Pass component detected, using pipeline results")
                score = results.get('last_retrieval_score', 0.0)

                if score == 0.0:
                    prev_best_scores = []
                    for prev_comp in reversed(list(component_results.keys())):
                        if prev_comp in component_results and component_results[prev_comp].get('best_score', 0) > 0:
                            prev_best_scores.append(component_results[prev_comp]['best_score'])
                            print(f"[{component}] Using score from {prev_comp}: {component_results[prev_comp]['best_score']}")
                            break
                    
                    if prev_best_scores:
                        score = prev_best_scores[0]
                    else:
                        score = results.get('score', 0.0)
                    
                print(f"[{component}] Pass component final score: {score}")
        else:
            score_mapping = {
                'retrieval': 'last_retrieval_score',
                'query_expansion': 'query_expansion_score',
                'passage_reranker': 'reranker_score',
                'passage_filter': 'filter_score',
                'passage_compressor': 'compressor_score',
                'prompt_maker_generator': 'combined_score'
            }
            
            if component == 'query_expansion':
                score = results.get('query_expansion_score', 0.0)
                if score == 0.0:
                    score = results.get('retrieval_score', results.get('last_retrieval_score', 0.0))
            elif component == 'passage_compressor':
                score = results.get('compressor_score', 0.0) 
                if score == 0.0:
                    score = results.get('compression_score', 0.0) 
                if score == 0.0:
                    score = results.get('compressor_llm_score', 0.0)
                if score == 0.0:
                    score = results.get('last_retrieval_score', 0.0)
                print(f"[{component}] Compressor score: {score} (from results: compressor_score={results.get('compressor_score', 'N/A')}, compression_score={results.get('compression_score', 'N/A')}, compressor_llm_score={results.get('compressor_llm_score', 'N/A')})")
            elif component == 'prompt_maker_generator':
                has_generator = 'generation_score' in results
                if has_generator:
                    score = results.get('combined_score', 0.0)
                    print(f"[{component}] Generator exists, using combined score: {score:.4f}")
                else:
                    score = results.get('last_retrieval_score', 0.0)
                    print(f"[{component}] No generator, using retrieval score: {score:.4f}")
            else:
                primary_key = score_mapping.get(component, 'combined_score')
                score = results.get(primary_key, results.get('last_retrieval_score', 0.0))
        
        return score
    
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
                else:
                    if col in results:
                        output_df[col] = results[col]
                    elif f'{col[:-1]}_list' in results: 
                        output_df[col] = results[f'{col[:-1]}_list']

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

        print(f"[{component}] Saved output with columns: {list(output_df.columns)}")
        
        return output_path
    
    def get_component_combinations(self, component: str, config_space_builder, 
                             best_configs: Dict[str, Any]) -> int:
        
        if not hasattr(self, 'search_space_calculator'):
            self.search_space_calculator = SearchSpaceCalculator(self.config_generator)
        
        fixed_config = self.get_fixed_config(component, best_configs, 
                                            ['query_expansion', 'retrieval', 'passage_reranker', 
                                            'passage_filter', 'passage_compressor', 'prompt_maker_generator'])
        
        cs = config_space_builder.build_component_space(component, fixed_config)
        
        if cs is None or len(cs.get_hyperparameters()) == 0:
            return 0
        
        search_space = {}
        
        for hp in cs.get_hyperparameters():
            param_name = hp.name
            
            if param_name == 'retrieval_config':
                methods = []
                tokenizers = []
                vdb_names = []
                
                for config in hp.choices:
                    if config.startswith('bm25_'):
                        methods.append('bm25')
                        tokenizers.append(config.replace('bm25_', ''))
                    elif config.startswith('vectordb_'):
                        methods.append('vectordb')
                        vdb_names.append(config.replace('vectordb_', ''))
                
                if methods:
                    search_space['retrieval_method'] = list(set(methods))
                if tokenizers:
                    search_space['bm25_tokenizer'] = list(set(tokenizers))
                if vdb_names:
                    search_space['vectordb_name'] = list(set(vdb_names))
                    
            elif param_name == 'reranker_config':
                methods = []
                method_models = {}
                
                for config in hp.choices:
                    if config == 'pass_reranker':
                        methods.append('pass_reranker')
                    elif config in ['upr', 'colbert_reranker']:
                        methods.append(config)
                    else:
                        for method_prefix in ['colbert_reranker', 'sentence_transformer_reranker',
                                            'flag_embedding_reranker', 'flag_embedding_llm_reranker',
                                            'openvino_reranker', 'flashrank_reranker', 'monot5']:
                            if config.startswith(method_prefix + '_'):
                                methods.append(method_prefix)
                                model = config[len(method_prefix) + 1:]
                                if method_prefix not in method_models:
                                    method_models[method_prefix] = []
                                method_models[method_prefix].append(model)
                                break
                
                if methods:
                    search_space['passage_reranker_method'] = list(set(methods))
                
                for method, models in method_models.items():
                    search_space[f'{method}_models'] = list(set(models))
            
            elif component == 'passage_filter':
                if param_name == 'passage_filter_method':
                    search_space[param_name] = hp.choices
                elif param_name == 'threshold':
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        search_space['similarity_threshold_cutoff_threshold'] = [hp.lower, hp.upper]
                    elif hasattr(hp, 'choices'):
                        search_space['similarity_threshold_cutoff_threshold'] = hp.choices
                elif param_name == 'percentile':
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        search_space['percentile_cutoff_percentile'] = [hp.lower, hp.upper]
                        search_space['similarity_percentile_cutoff_percentile'] = [hp.lower, hp.upper]
                    elif hasattr(hp, 'choices'):
                        search_space['percentile_cutoff_percentile'] = hp.choices
                        search_space['similarity_percentile_cutoff_percentile'] = hp.choices
            
            elif component == 'passage_compressor':
                if param_name == 'passage_compressor_method':
                    search_space[param_name] = hp.choices
                elif param_name == 'compressor_llm_model':
                    models = []
                    for llm_model in hp.choices:
                        if '_' in llm_model:
                            _, model = llm_model.split('_', 1)
                            models.append(model)
                        else:
                            models.append(llm_model)
                    search_space['compressor_model'] = list(set(models))
                    search_space['compressor_llm'] = ['openai']
                elif param_name == 'compressor_compression_ratio':
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        search_space['compressor_compression_ratio'] = [hp.lower, hp.upper]
                    elif hasattr(hp, 'choices'):
                        search_space['compressor_compression_ratio'] = hp.choices
                elif param_name == 'compressor_threshold':
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        search_space['compressor_threshold'] = [hp.lower, hp.upper]
                    elif hasattr(hp, 'choices'):
                        search_space['compressor_threshold'] = hp.choices
                elif param_name == 'compressor_damping':
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        search_space['compressor_damping'] = [hp.lower, hp.upper]
                    elif hasattr(hp, 'choices'):
                        search_space['compressor_damping'] = hp.choices
                elif param_name == 'compressor_max_iterations':
                    if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                        search_space['compressor_max_iterations'] = [hp.lower, hp.upper]
                    elif hasattr(hp, 'choices'):
                        search_space['compressor_max_iterations'] = hp.choices
                elif param_name == 'compressor_spacy_model':
                    search_space['compressor_spacy_model'] = hp.choices
            
            elif hasattr(hp, 'choices'):
                search_space[param_name] = hp.choices
                
            elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                if isinstance(hp.lower, int) and isinstance(hp.upper, int):
                    search_space[param_name] = list(range(hp.lower, hp.upper + 1))
                else:
                    search_space[param_name] = (hp.lower, hp.upper)
        
        if component == 'prompt_maker_generator':
            if 'prompt_template_idx' not in search_space:
                prompt_config = self.config_generator.extract_node_config("prompt_maker")
                prompt_count = 0
                if prompt_config and prompt_config.get("modules"):
                    for module in prompt_config.get("modules", []):
                        prompts = module.get('prompt', [])
                        if isinstance(prompts, list):
                            prompt_count += len(prompts)
                        else:
                            prompt_count += 1
                search_space['prompt_template_idx'] = list(range(prompt_count)) if prompt_count > 0 else [0]
            
            if 'generator_model' not in search_space:
                gen_config = self.config_generator.extract_node_config("generator")
                models = []
                if gen_config and gen_config.get("modules"):
                    for module in gen_config.get("modules", []):
                        module_models = module.get("model", module.get("llm", []))
                        if isinstance(module_models, list):
                            models.extend(module_models)
                        else:
                            models.append(module_models)
                if models:
                    search_space['generator_model'] = list(set(models))
        
        combinations, note = self.search_space_calculator.calculate_component_combinations(
            component, 
            search_space, 
            fixed_config, 
            best_configs
        )
        
        return combinations