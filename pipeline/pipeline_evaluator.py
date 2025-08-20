import pandas as pd
from typing import Dict, Any, List, Optional

from pipeline_component.evaluation.retrieval_evaluation import EvaluationModule
from pipeline_component.evaluation.token_evaluation import TokenEvaluatorModule
from pipeline_component.evaluation.generation_evaluator import GenerationEvaluatorModule
from pipeline_component.evaluation.ragas_evaluator import RagasEvaluatorModule
from pipeline_component.evaluation.llm_evaluator import CompressorLLMEvaluator
from pipeline.config_manager import ConfigGenerator
from pipeline.utils import Utils


class RAGPipelineEvaluator:
    def __init__(
        self,
        config_generator: ConfigGenerator,
        retrieval_metrics: List, 
        filter_metrics: List,
        compressor_metrics: List,
        generation_metrics: List,
        prompt_maker_metrics: List,
        query_expansion_metrics: List = [],
        reranker_metrics: List = [], 
        retrieval_weight: float = 0.5,
        generation_weight: float = 0.5,
        use_ragas: bool = False, 
        ragas_config: Optional[Dict[str, Any]] = None,
        use_llm_compressor_evaluator: bool = False,
        llm_evaluator_model: str = "gpt-4o",
    ):
        self.config_generator = config_generator
        self.retrieval_metrics = retrieval_metrics
        self.filter_metrics = filter_metrics
        self.compressor_metrics = compressor_metrics
        self.generation_metrics = generation_metrics
        self.prompt_maker_metrics = prompt_maker_metrics
        self.query_expansion_metrics = query_expansion_metrics
        self.reranker_metrics = reranker_metrics 
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight

        self.use_ragas = use_ragas
        self.ragas_config = ragas_config or {}
        
        if self.use_ragas:
            self.ragas_evaluator = RagasEvaluatorModule(
                retrieval_metrics=self.ragas_config.get('retrieval_metrics', ["context_precision", "context_recall"]),
                generation_metrics=self.ragas_config.get('generation_metrics', 
                    ["answer_relevancy", "faithfulness", "factual_correctness", "semantic_similarity"]), 
                llm_model=self.ragas_config.get('llm_model', "gpt-4o-mini"),
                embedding_model=self.ragas_config.get('embedding_model', "text-embedding-ada-002")
            )
        self.use_llm_compressor_evaluator = use_llm_compressor_evaluator
        self.llm_evaluator_model = llm_evaluator_model
        
        if self.use_llm_compressor_evaluator:
            self.compressor_llm_evaluator = CompressorLLMEvaluator(
                llm_model=self.llm_evaluator_model,
                temperature=0.0
            )
    
    def evaluate_with_ragas(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame, 
                           corpus_path: Optional[str] = None) -> Dict[str, Any]:

        if not self.use_ragas:
            raise ValueError("RAGAS evaluation is not enabled")
        
        ragas_results = self.ragas_evaluator.evaluate_pipeline_results(
            eval_df, qa_subset, corpus_path
        )

        all_metric_scores = []
        for metric, score in ragas_results.items():
            if metric.endswith('_score') or metric in ['context_precision', 'context_recall', 
                                                    'answer_relevancy', 'faithfulness', 
                                                    'factual_correctness', 'semantic_similarity']: 
                if isinstance(score, (int, float)):
                    all_metric_scores.append(score)
        
        if all_metric_scores:
            mean_ragas_score = sum(all_metric_scores) / len(all_metric_scores)
        else:
            mean_ragas_score = 0.0
        
        return {
            'ragas_mean_score': mean_ragas_score,
            'ragas_metrics': ragas_results,
            **ragas_results
        }

    def evaluate_query_expansion(self, retrieval_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        if not self.query_expansion_metrics:
            return {}
            
        print(f"[Trial] Evaluating query expansion with metrics: {self.query_expansion_metrics}")
        query_expansion_evaluator = EvaluationModule(metrics=self.query_expansion_metrics)
        
        expansion_ground_truths = query_expansion_evaluator.extract_ground_truths(qa_subset)
        retrieved_ids = retrieval_df['retrieved_ids'].tolist()
        expanded_results = query_expansion_evaluator.evaluate(retrieved_ids, expansion_ground_truths)
        
        query_expansion_results = {
            'mean_score': expanded_results.get('mean_accuracy', 0.0),
            'metrics': Utils.json_serializable(expanded_results)
        }
        
        print(f"Query expansion evaluation score: {expanded_results.get('mean_accuracy', 0.0):.4f}")
        return query_expansion_results
    
    def evaluate_retrieval(self, retrieval_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        retrieval_evaluator = EvaluationModule(metrics=self.retrieval_metrics)
        ground_truths = retrieval_evaluator.extract_ground_truths(qa_subset)
        retrieved_ids = retrieval_df['retrieved_ids'].tolist()
        retrieval_results = retrieval_evaluator.evaluate(retrieved_ids, ground_truths)
        
        return retrieval_results
    
    def evaluate_reranker(self, reranked_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        evaluator = EvaluationModule(metrics=self.reranker_metrics)
        ground_truths = evaluator.extract_ground_truths(qa_subset)
        reranked_ids = reranked_df['retrieved_ids'].tolist()
        reranker_results = evaluator.evaluate(reranked_ids, ground_truths)
        return reranker_results
    
    def evaluate_filter(self, filtered_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        retrieval_evaluator = EvaluationModule(metrics=self.filter_metrics)
        ground_truths = retrieval_evaluator.extract_ground_truths(qa_subset)
        filtered_ids = filtered_df['retrieved_ids'].tolist()
        filter_results = retrieval_evaluator.evaluate(filtered_ids, ground_truths)
        return filter_results
    
    def evaluate_compressor(self, compressed_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        if self.use_llm_compressor_evaluator: 
            print(f"[DEBUG] Evaluating compressor using LLM evaluator")
            
            batch_size = 5
            scored_df = self.compressor_llm_evaluator.evaluate_batch(
                df=compressed_df, 
                qa_df=qa_subset,
                context_col="retrieved_contents",
                query_col="query", 
                batch_size=batch_size
            )
            
            scores_list = scored_df["compressor_llm_score"].tolist()
            
            numeric_scores = []
            for score in scores_list:
                if isinstance(score, str):
                    try:
                        numeric_scores.append(float(score))
                    except (ValueError, TypeError):
                        print(f"[WARNING] Could not convert score to float: {score}")
                        numeric_scores.append(0.0)
                else:
                    numeric_scores.append(float(score))
            
            mean_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
            
            compressed_df["compressor_score"] = numeric_scores
            
            return {
                "mean_score": mean_score,
                "compressor_score": mean_score,  
                "compressor_llm_score": mean_score,
                "scores": numeric_scores,  
                "compression_metrics": {
                    "llm_mean_score": mean_score,
                    "llm_scores": numeric_scores,  
                    "evaluation_method": "llm_batch",
                    "batch_size": batch_size
                }
            }

        if not self.compressor_metrics:
            return {}

        token_evaluator = TokenEvaluatorModule(metrics=self.compressor_metrics)
        token_eval_results = token_evaluator.evaluate_from_dataframe(compressed_df, qa_subset)
        return token_eval_results
    
    def evaluate_generation(self, eval_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        embedding_model_name = None
        for metric in self.generation_metrics:
            if isinstance(metric, dict) and metric.get("metric_name") == "sem_score":
                embedding_model_name = metric.get("embedding_model")
                break
                
        generation_evaluator = GenerationEvaluatorModule(
            metrics=[m.get("metric_name") if isinstance(m, dict) else m for m in self.generation_metrics],
            embedding_model_name=embedding_model_name
        )
        
        generation_results = generation_evaluator.evaluate_from_dataframe(eval_df, qa_subset)
        return generation_results
    
    def evaluate_prompts(self, config: Dict[str, Any], trial_dir: str, prompts_df: pd.DataFrame,
                       working_df: pd.DataFrame, qa_subset: pd.DataFrame) -> Dict[str, Any]:
        print(f"[Trial] Evaluating prompts with model: {config.get('prompt_maker_generator_model')}")
        
        prompt_maker_embedding_model = None
        for metric in self.prompt_maker_metrics:
            if isinstance(metric, dict) and metric.get("metric_name") == "sem_score":
                prompt_maker_embedding_model = metric.get("embedding_model")
                break
                
        prompt_evaluator = GenerationEvaluatorModule(
            metrics=[m.get("metric_name") if isinstance(m, dict) else m for m in self.prompt_maker_metrics],
            embedding_model_name=prompt_maker_embedding_model
        )
        
        prompt_gen_config = Utils.find_generator_config(
            self.config_generator, 
            "prompt_maker", 
            config.get('prompt_maker_generator_model')
        )
        
        prompt_generator = Utils.create_generator_from_config(
            config.get('prompt_maker_generator_model'),
            prompt_gen_config
        )

        temperature = Utils.get_temperature_from_config(config, prompt_gen_config, 'prompt_maker_temperature')
        
        prompt_eval_df = prompt_generator.generate_from_dataframe(
            df=prompts_df,
            prompt_column='prompts',
            output_column='prompt_generated_texts',
            temperature=temperature
        )
        
        prompt_eval_data = pd.DataFrame()
        prompt_eval_data['query'] = qa_subset['query'].values
        prompt_eval_data['generated_texts'] = prompt_eval_df['prompt_generated_texts'].values
        
        if 'generation_gt' in qa_subset.columns:
            prompt_eval_data['generation_gt'] = qa_subset['generation_gt'].values
        elif 'ground_truth' in qa_subset.columns:
            prompt_eval_data['generation_gt'] = qa_subset['ground_truth'].values
        
        prompt_eval_data['retrieved_contents'] = working_df['retrieved_contents'].values
        prompt_eval_data['prompts'] = prompts_df['prompts'].values
        
        prompt_results = prompt_evaluator.evaluate_from_dataframe(prompt_eval_data, qa_subset)
        print(f"Prompt evaluation results: {prompt_results}")
    
        return prompt_results
    
    def calculate_combined_score(self, last_retrieval_score: float, generation_score: float, 
                               has_generator: bool, use_ragas_score: bool = False, 
                               ragas_score: float = 0.0) -> float:
  
        if use_ragas_score and ragas_score > 0:
            return ragas_score
        
        if has_generator and self.generation_metrics and generation_score > 0:
            combined_score = (
                self.retrieval_weight * last_retrieval_score + 
                self.generation_weight * generation_score
            )
        else:
            combined_score = last_retrieval_score
        return combined_score