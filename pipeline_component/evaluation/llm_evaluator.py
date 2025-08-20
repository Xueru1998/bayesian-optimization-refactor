import os
import pandas as pd
from typing import List, Optional, Union
from langchain_openai import ChatOpenAI
import numpy as np
import re
import time
import json


class CompressorLLMEvaluator:
    def __init__(self, llm_model="gpt-4o", temperature: float = 0.0):        
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        self.judge_prompt = """\
        Question: {question}
        Ground Truth Answer: {ground_truth}
        Compressed Context: {context}

        Score this compressed context (0.0–1.0) based on:

        **1. ATOMIC FACT PRESERVATION (50%)**
        - List all atomic facts in the ground truth (specific terms, numbers, methods, comparisons, measurements).
        - Check if EACH fact exists in the compressed context.
        - Missing fact: −0.15
        - Missing critical fact (methods, measurements, specific comparator): −0.25
        - Replaced with vague/general term: −0.20

        **2. COMPLETENESS (15%)**
        - Can someone write the EXACT ground truth answer using ONLY this context?
        - If NO due to missing specifics: cap total score at 0.5

        **3. RELEVANCE & ACCURACY (20%)**
        - Irrelevant, off-topic, or wrong facts: −0.15 each
        - If >25% of content is unrelated: cap at 0.5
        - If unrelated content changes meaning or causes confusion: max 0.3

        **4. EFFICIENCY & PRECISION (15%)**
        - Brevity bonus (+0.1) if ALL atomic facts are preserved AND context is under 70 words
        - Excessive length with unrelated filler: −0.1 to −0.3 depending on severity

        Return only a number between 0.0 and 1.0
        """

        self.batch_judge_prompt = """\
        Evaluate {num_samples} compressed contexts using the same criteria for each:

        **SCORING CRITERIA (0.0–1.0 for each sample):**
        **1. ATOMIC FACT PRESERVATION (50%)**
        - List all atomic facts in the ground truth (specific terms, numbers, methods, comparisons, measurements).
        - Check if EACH fact exists in the compressed context.
        - Missing fact: −0.15
        - Missing critical fact (methods, measurements, specific comparator): −0.25
        - Replaced with vague/general term: −0.20

        **2. COMPLETENESS (15%)**
        - Can someone write the EXACT ground truth answer using ONLY this context?
        - If NO due to missing specifics: cap total score at 0.5

        **3. RELEVANCE & ACCURACY (20%)**
        - Irrelevant, off-topic, or wrong facts: −0.15 each
        - If >25% of content is unrelated: cap at 0.5
        - If unrelated content changes meaning or causes confusion: max 0.3

        **4. EFFICIENCY & PRECISION (15%)**
        - Brevity bonus (+0.1) if ALL atomic facts are preserved AND context is under 70 words
        - Excessive length with unrelated filler: −0.1 to −0.3 depending on severity

        **SAMPLES TO EVALUATE:**
        {samples}

        Return ONLY a JSON array of {num_samples} scores in the exact same order as the samples above: [score1, score2, score3, ...]
        """

    def _format_prompt(self, query: str, ground_truth: str, context: Union[str, List[str], np.ndarray]) -> str:
        if isinstance(context, (list, np.ndarray)):
            context = "\n".join(map(str, context))
        if isinstance(ground_truth, (list, np.ndarray)) and len(ground_truth) > 0:
            ground_truth = str(ground_truth[0])
        elif not isinstance(ground_truth, str):
            ground_truth = str(ground_truth) if ground_truth is not None else ""
        if not isinstance(query, str):
            query = str(query) if query is not None else ""

        return self.judge_prompt.format(
            question=query.strip(),
            ground_truth=ground_truth.strip(),
            context=context.strip()
        )

    def _format_batch_prompt(self, samples: List[dict]) -> str:
        formatted_samples = []
        for i, sample in enumerate(samples, 1):
            query = str(sample['query']) if sample['query'] is not None else ""
            ground_truth = str(sample['ground_truth']) if sample['ground_truth'] is not None else ""
            context = str(sample['context']) if sample['context'] is not None else ""
            
            formatted_sample = f"""Sample {i}:
Question: {query.strip()}
Ground Truth Answer: {ground_truth.strip()}
Compressed Context: {context.strip()}
---"""
            formatted_samples.append(formatted_sample)
        
        samples_text = "\n".join(formatted_samples)
        
        return self.batch_judge_prompt.format(
            num_samples=len(samples),
            samples=samples_text
        )

    def evaluate_single(self, query: str, context: Union[str, List[str]], ground_truth: str) -> float:
        if isinstance(context, list):
            context = "\n".join(context)

        prompt = self._format_prompt(query, ground_truth, context)
        response = self.llm.invoke(prompt)
        
        score_str = response.content.strip()

        try:
            float_pattern = r"([01](?:\.\d+)?)"
            match = re.search(float_pattern, score_str)
            
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))
            else:
                try:
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    raise ValueError("No valid float between 0.0 and 1.0 found.")
        except Exception as e:
            print(f"[WARN] Could not parse score from: {score_str} - Error: {e}")
            score = 0.0

        return score

    def evaluate_batch(self, df: pd.DataFrame, qa_df: pd.DataFrame, context_col: str = "retrieved_contents", query_col: str = "query", batch_size: int = 5) -> pd.DataFrame:
        queries = df[query_col].tolist()
        contexts = df[context_col].tolist()

        ground_truths = []
        for i, query in enumerate(queries):
            gt_row = qa_df[qa_df[query_col] == query]
            if not gt_row.empty:
                gt_val = gt_row.iloc[0].get("generation_gt", None)
                if gt_val is not None and not (isinstance(gt_val, float) and pd.isna(gt_val)):
                    if isinstance(gt_val, np.ndarray):
                        if len(gt_val) > 0:
                            ground_truths.append(str(gt_val[0]))
                        else:
                            ground_truths.append("")
                    elif isinstance(gt_val, list) and len(gt_val) > 0:
                        ground_truths.append(str(gt_val[0]))
                    elif isinstance(gt_val, str):
                        ground_truths.append(gt_val)
                    else:
                        ground_truths.append(str(gt_val))
                else:
                    ground_truths.append("")
            else:
                ground_truths.append("")
        if ground_truths:
            print(f"  First ground truth (truncated): {ground_truths[0][:100]}...")

        print(f"[INFO] Running batch evaluation on {len(queries)} samples with batch_size={batch_size}")

        all_scores = []
        total_api_calls = 0
        
        for i in range(0, len(queries), batch_size):
            end_idx = min(i + batch_size, len(queries))
            batch_samples = []
            
            for j in range(i, end_idx):
                query_val = queries[j]
                context_val = contexts[j]
                gt_val = ground_truths[j]
                
                if not isinstance(query_val, str):
                    query_val = str(query_val) if query_val is not None else ""
                
                if isinstance(context_val, (list, np.ndarray)):
                    if isinstance(context_val, np.ndarray):
                        context_val = "\n".join(map(str, context_val))
                    else:
                        context_val = "\n".join(map(str, context_val))
                elif not isinstance(context_val, str):
                    context_val = str(context_val) if context_val is not None else ""
                
                if not isinstance(gt_val, str):
                    gt_val = str(gt_val) if gt_val is not None else ""
                
                batch_samples.append({
                    'query': query_val,
                    'ground_truth': gt_val,
                    'context': context_val
                })

            
            print(f"[BATCH] Processing batch {i//batch_size + 1}: samples {i+1}-{end_idx}")
            
            try:
                prompt = self._format_batch_prompt(batch_samples)
                response = self.llm.invoke(prompt)
                total_api_calls += 1
                batch_scores = self._parse_batch_scores(response.content.strip(), len(batch_samples))
                all_scores.extend(batch_scores)
                
            except Exception as e:
                print(f"[ERROR] Batch {i//batch_size + 1} failed: {e}")
                print(f"[FALLBACK] Using single evaluation for batch {i//batch_size + 1}")
                for sample in batch_samples:
                    score = self.evaluate_single(sample['query'], sample['context'], sample['ground_truth'])
                    all_scores.append(score)
                    total_api_calls += 1
            
            time.sleep(0.2)

        df_result = df.copy()
        df_result["compressor_llm_score"] = all_scores

        return df_result

    def _parse_batch_scores(self, response_str: str, expected_count: int) -> List[float]:
        try:
            json_match = re.search(r'\[[\d\.,\s]+\]', response_str)
            if json_match:
                scores_json = json_match.group()
                scores = json.loads(scores_json)
                
                validated_scores = []
                for score in scores:
                    if isinstance(score, (int, float)):
                        validated_scores.append(max(0.0, min(1.0, float(score))))
                    else:
                        print(f"[WARN] Invalid score type: {score}, using 0.0")
                        validated_scores.append(0.0)
                
                if len(validated_scores) == expected_count:
                    return validated_scores
                else:
                    print(f"[WARN] Expected {expected_count} scores, got {len(validated_scores)}")
            
            float_pattern = r'([01](?:\.\d+)?)'
            matches = re.findall(float_pattern, response_str)
            
            if len(matches) >= expected_count:
                scores = [max(0.0, min(1.0, float(match))) for match in matches[:expected_count]]
                return scores
            
            print(f"[ERROR] Could not extract {expected_count} scores from batch response")
            return [0.0] * expected_count
            
        except Exception as e:
            print(f"[ERROR] Failed to parse batch scores: {e}")
            return [0.0] * expected_count


class CompressorLLMEvaluationExample:
    def __init__(self):
        self.evaluator = CompressorLLMEvaluator(llm_model="gpt-4o", temperature=0.0)

    def run(self):
        compressed_df = pd.DataFrame({
            "query": [
                "Are 'per trade' fees charged on every order or just once per stock?",
            #    "Scapular stabilizer exercises are more effective than general exercise therapy in reducing pain and improving function of the shoulder."
            ],
            "retrieved_contents": [
                "Per trade' fees are charged on every order, not just once per stock.",
            #    "Effect of specific exercise strategy on need for surgery in patients with subacromial impingement syndrome: randomised controlled study OBJECTIVE To evaluate if a specific exercise strategy, targeting the rotator cuff and scapula stabilisers, improves shoulder function and pain more than unspecific exercises in patients with subacromial impingement syndrome, thereby decreasing the need for arthroscopic subacromial decompression.   \n DESIGN Randomised, participant and single assessor blinded, controlled study.   \n SETTING Department of orthopaedics in a Swedish university hospital.   \n PARTICIPANTS 102 patients with long standing (over six months) persistent subacromial impingement syndrome in whom earlier conservative treatment had failed, recruited through orthopaedic specialists.   \n INTERVENTIONS The specific exercise strategy consisted of strengthening eccentric exercises for the rotator cuff and concentric/eccentric exercises for the scapula stabilisers in combination with manual mobilisation. The control exercise programme consisted of unspecific movement exercises for the neck and shoulder. Patients in both groups received five to six individual guided treatment sessions during 12 weeks. In between these supervised sessions the participants performed home exercises once or twice a day for 12 weeks.   \n MAIN OUTCOME MEASURES The primary outcome was the Constant-Murley shoulder assessment score evaluating shoulder function and pain. Secondary outcomes were patients' global impression of change because of treatment and decision regarding surgery.   \n RESULTS Most (97, 95%) participants completed the 12 week study. There was a significantly greater improvement in the Constant-Murley score in the specific exercise group than in the control exercise group (24 points (95% confidence interval 19 to 28.0) v 9 points (5 to 13); mean difference between group: 15 points (8.5 to 20.6)). Significantly more patients in the specific exercise group reported successful outcome (defined as large improvement or recovered) in the patients' global assessment of change because of treatment: 69% (35/51) v 24% (11/46); odds ratio 7.6, 3.1 to 18.9; P<0.001. A significantly lower proportion of patients in the specific exercise group subsequently chose to undergo surgery: 20% (10/51) v 63% (29/46); odds ratio 7.7, 3.1 to 19.4; P<0.001).   \n CONCLUSION A specific exercise strategy, focusing on strengthening eccentric exercises for the rotator cuff and concentric/eccentric exercises for the scapula stabilisers, is effective in reducing pain and improving shoulder function in patients with persistent subacromial impingement syndrome. By extension, this exercise strategy reduces the need for arthroscopic subacromial decompression within the three month timeframe used in the study.   \n TRIAL REGISTRATION Clinical trials NCT01037673.Children's Exercise Physiology The reorganized and newly revised \"Children's Exercise Physiology, Second Edition, \" presents the most up-to-date research, methodology, and approaches related to children's physiologic responses to exercise. The book examines not only the current major issues that separate children from adults, but also the underlying mechanisms of these differences. Readers will learn what makes children different from adults physiologically--such as size, biochemical differences, neuromuscular differences, and lack of sexual and hormonal maturation--and the reasons for these differences. Those involved with young athletes, disease management, and health promotion will gain valuable insight into the physiologic determinants of exercise performance. Children's exercise physiology is a fast-moving field. In the eight years since the first edition of this book was published, much new information has surfaced. This streamlined new edition contains 13 instead of 15 chapters, an introduction, and updated features: -Chapter objectives, discussion questions and research directions, and a glossary of terms promote learning.-A reorganized table of contents improves the flow from chapter to chapter.-A new final chapter covers the role of the central nervous system. Also included is in-depth discussion of the determinants of aerobic fitness and VO2 kinetics and the significance of maximal aerobic power in children. With improved chapters on thermoregulation and metabolic and endocrinologic responses to exercise, you can be confident you're getting the latest information with \"Children's Exercise Physiology, Second Edition."
            ]
        })

        qa_df = pd.DataFrame({
            "query": [
                "Are 'per trade' fees charged on every order or just once per stock?",
               # "Scapular stabilizer exercises are more effective than general exercise therapy in reducing pain and improving function of the shoulder.",
            ],
            "generation_gt": [
                "'Per trade' fees are charged on every order, not just once per stock. This means you pay a commission each time you buy and each time you sell a stock. For example, if a brokerage firm advertises a $7 commission per trade, you would pay $7 when you buy the stock and another $7 when you sell it. Additionally, if you place a Good-til-Canceled (GTC) order that is partially filled on different days, you may be charged multiple commissions for each partial fill.",
           #     "Yes, the study found that a specific exercise strategy focusing on strengthening eccentric exercises for the rotator cuff and concentric/eccentric exercises for the scapula stabilizers was more effective in reducing pain and improving shoulder function in patients with persistent subacromial impingement syndrome compared to unspecific movement exercises for the neck and shoulder. This was evidenced by a significantly greater improvement in the Constant-Murley score in the specific exercise group compared to the control group."
            ]
        })

        print("[INFO] Running LLM-based compression evaluation...")
        print("[INFO] Testing with first example only to debug...")
        
        test_score = self.evaluator.evaluate_single(
            compressed_df.iloc[0]["query"],
            compressed_df.iloc[0]["retrieved_contents"],
            qa_df.iloc[0]["generation_gt"]
        )
        print(f"\n[DEBUG] Test score: {test_score}")
        
        scored_df = self.evaluator.evaluate_batch(compressed_df, qa_df, batch_size=2)
        print("\n[RESULTS] Batch Scores:")
        print(scored_df[["query", "compressor_llm_score"]])


def main():
    runner = CompressorLLMEvaluationExample()
    runner.run()


if __name__ == "__main__":
    main()