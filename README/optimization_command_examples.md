# Optimization Command Examples

This document lists commonly used command-line invocations for running optimization experiments with **SMAC**, **Bayesian Optimization (BO)**, and **Optuna**.
The examples below use an LLM as the evaluator for the node compressor. If a non-LLM evaluator is desired, simply remove this option.
---

## SMAC Optimization

### Global Optimization (200 trials)
```bash
mkdir -p datasetname_results_smac_global_optimization_200

python smac_runner.py \
  --optimization_mode global \
  --use_llm_compressor_evaluator \
  --llm_evaluator_model gpt-4o \
  --n_trials 190 \
  --email_notifications \
  --study_name datasetname_results_smac_global_optimization_200 \
  --result_dir ./datasetname_results_smac_global_optimization_200 \
  2>&1 | tee ~/autorag_trial_1.log
```

### Local (Componentwise) Optimization
```bash
mkdir -p datasetname_results_smac_local_optimization

python smac_runner.py \
  --optimization_mode componentwise \
  --use_llm_compressor_evaluator \
  --llm_evaluator_model gpt-4o \
  --email_notifications \
  --study_name datasetname_results_smac_local_optimization \
  --result_dir ./datasetname_results_smac_local_optimization \
  2>&1 | tee ~/autorag_trial_local.log
```

---

## Optuna

### Random Search 
```bash
mkdir -p datasetname_results_random_trial_200

python bo_runner.py \
  --sampler random \
  ----use_llm_compressor_evaluator \
  --llm_evaluator_model gpt-4o \
  --n_trials 200 \
  --study_name datasetname_results_random_trial_200 \
  --send_email \
  --result_dir ./datasetname_results_random_trial_200 \
  2>&1 | tee ~/autorag_trial_random_200.log
```

### TPE (Tree-structured Parzen Estimator) with Componentwise Mode
```bash
python bo_runner.py \
  --sampler tpe \
  --mode componentwise \
  --use_llm_compressor_evaluator \
  --n_trials_per_component 20 \
  --send_email \
  --llm_evaluator_model gpt-4o \
  --study_name fiqa_sap_Optuna \
  2>&1 | tee ~/autorag_trial_tpe.log
```

---

## Optuna Optimization

### Global Optimization with TPE (100 trials, Ragas enabled)
```bash
python unified_optuna_runner.py \
  --mode global \
  --n_trials 100 \
  --sampler tpe \
  --use_ragas \
  --ragas_llm_model gpt-4o-mini
```

**optuna optimization**

### Local (Componentwise) Optimization with TPE
```bash
mkdir -p datasetname_results_tpe_local_optimization

python bo_runner.py \
  --mode componentwise \
  --sampler tpe \
  --result_dir ./datasetname_results_tpe_local_optimization \
  --use_llm_compressor_evaluator \
  --send_email \
  --study_name datasetname_results_tpe_local_optimization \
  2>&1 | tee ./autorag_trial_tpe.log
```

---

## Notes
- `--study_name` uniquely identifies each experiment run.  
- `--result_dir` specifies where optimization results are saved.  
- `--email_notifications` / `--send_email` enables email alerts when trials complete.  
- `tee` logs outputs to both console and a file.  
