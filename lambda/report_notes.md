# Lambda Regularization Experiment Report

## 1. Executive Summary

This experiment extends the objective from `L(theta) = L_NPO,beta(theta) + K_RT(theta)` to `L(theta) = L_NPO,beta(theta) + lambda K_RT(theta)`, sweeping `lambda in {0.0, 0.1, 0.3, 1.0, 3.0, 10.0}` on the fixed 2x2 setup. It covers Phi-3 and LLaMA-3-3B on OpenBookQA and StrategyQA, 24 full combinations in total; all results are written under `lambda/results/lambda=<value>/...` and do not overwrite `final_results`.

Conclusion: lambda does control the stability-plasticity trade-off, but the curve is not strictly monotonic. Lower lambda tends to improve answer-level faithfulness or answer-change behavior; higher lambda tends to better preserve specificity; the optimum clearly depends on model and dataset. Under the pre-registered constraint that specificity may not drop more than 3 points below lambda=1.0, the recommended values are: LLaMA/openbook `0.3`, LLaMA/sqa `3.0`, Phi/openbook `0.0`, Phi/sqa `10.0`.

## 2. Experiment Purpose And Hypotheses

Purpose: examine how the retain KL regularization weight `lambda` affects unlearning efficacy, answer-level faithfulness, specificity, and answer-change rate, and whether the current implicit `lambda=1.0` baseline sits in a good trade-off region.

Hypotheses:

- H1: Lower lambda weakens retain regularization, raising local plasticity; it may strengthen faithfulness/answer-change but risks specificity.
- H2: Higher lambda strengthens retained-behavior stability; it may improve specificity but suppresses unlearning's effect on the final answer.
- H3: The current implicit baseline `lambda=1.0` is not necessarily optimal.
- H4: Phi-3 and LLaMA-3-3B differ in their sensitivity to lambda.

The results broadly support these hypotheses, especially H3/H4: the recommended lambda differs across the four model/dataset combinations.

## 3. Objects, Scope, And Workflow

Objects and scope:

- Models: Phi-3, LLaMA-3-3B
- Datasets: OpenBookQA (`openbook`), StrategyQA (`sqa`)
- Method: `npo_KL`, sentence-level stepwise unlearning
- Optimization: POS filter enabled, FF2-only optimization enabled
- Scale: `N_UNLEARN=40`, `N_VERIFY=10`, seed `1001`
- Lambda grid: `0.0, 0.1, 0.3, 1.0, 3.0, 10.0`

Workflow: first expose `--rt_lambda` and ensure result paths include lambda; then smoke-test schema, paths, and GPU memory; then run the full sweep on 2xA100; finally generate JSON/CSV/PNG/SVG and this merged report.

## 4. Deliverables

- Raw results: `lambda/results/`
- Metric summary: `lambda/analysis_summary.json`
- Answer-change summary: `lambda/answer_change_rate_summary.json`
- Main table: `lambda/summary_table.csv`
- Pareto summary table: `lambda/pareto_frontier_summary.csv`
- Report: `lambda/report_notes.md`
- Existing plots: `lambda/figures/*.png`
- Added Pareto/scatter plots:
  - `lambda/figures/pareto_faithfulness_specificity.svg`
  - `lambda/figures/pareto_efficacy_specificity.svg`
  - `lambda/figures/pareto_answer_change_rate_specificity.svg`

## 5. Core Metrics

| Lambda | Model | Dataset | Faithfulness | Efficacy | Specificity | Answer Change |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | LLaMA-3-3B | openbook | 62.50 | 82.35 | 89.26 | 28.43 |
| 0.1 | LLaMA-3-3B | openbook | 65.00 | 82.32 | 89.21 | 28.93 |
| 0.3 | LLaMA-3-3B | openbook | 65.00 | 82.32 | 89.59 | 28.93 |
| 1.0 | LLaMA-3-3B | openbook | 62.50 | 82.32 | 90.30 | 28.43 |
| 3.0 | LLaMA-3-3B | openbook | 62.50 | 82.30 | 91.06 | 28.43 |
| 10.0 | LLaMA-3-3B | openbook | 60.00 | 82.20 | 91.94 | 26.90 |
| 0.0 | LLaMA-3-3B | sqa | 67.50 | 82.91 | 92.14 | 31.68 |
| 0.1 | LLaMA-3-3B | sqa | 67.50 | 82.91 | 92.23 | 32.30 |
| 0.3 | LLaMA-3-3B | sqa | 67.50 | 82.91 | 92.17 | 31.68 |
| 1.0 | LLaMA-3-3B | sqa | 67.50 | 82.91 | 92.29 | 31.06 |
| 3.0 | LLaMA-3-3B | sqa | 67.50 | 82.91 | 92.42 | 30.43 |
| 10.0 | LLaMA-3-3B | sqa | 62.50 | 82.91 | 92.15 | 30.43 |
| 0.0 | Phi-3 | openbook | 40.00 | 82.85 | 93.86 | 13.38 |
| 0.1 | Phi-3 | openbook | 37.50 | 82.85 | 94.10 | 12.74 |
| 0.3 | Phi-3 | openbook | 35.00 | 82.85 | 94.40 | 12.10 |
| 1.0 | Phi-3 | openbook | 35.00 | 82.85 | 94.57 | 12.10 |
| 3.0 | Phi-3 | openbook | 37.50 | 82.85 | 94.94 | 12.74 |
| 10.0 | Phi-3 | openbook | 37.50 | 82.85 | 94.70 | 12.74 |
| 0.0 | Phi-3 | sqa | 25.00 | 81.64 | 96.80 | 10.71 |
| 0.1 | Phi-3 | sqa | 25.00 | 81.44 | 96.86 | 11.43 |
| 0.3 | Phi-3 | sqa | 25.00 | 81.76 | 96.88 | 12.14 |
| 1.0 | Phi-3 | sqa | 25.00 | 81.65 | 97.19 | 12.14 |
| 3.0 | Phi-3 | sqa | 25.00 | 81.13 | 97.52 | 12.14 |
| 10.0 | Phi-3 | sqa | 25.00 | 80.39 | 97.96 | 12.14 |

## 6. Pareto Frontier Analysis

Here specificity and a second metric are both treated as objectives to maximize. A lambda is dominated if some other lambda is at least as good on both objectives. Black rings and connecting lines in the figures mark the nondominated lambda values.

| Model | Dataset | Recommended lambda | Faithfulness/Specificity | Faithfulness frontier | Efficacy frontier | Answer-change frontier |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3-3B | openbook | 0.3 | 65.00/89.59 | 0.3, 3.0, 10.0 | 0.0, 0.3, 1.0, 3.0, 10.0 | 0.3, 3.0, 10.0 |
| LLaMA-3-3B | sqa | 3.0 | 67.50/92.42 | 3.0 | 0.1, 1.0, 3.0 | 0.1, 1.0, 3.0 |
| Phi-3 | openbook | 0.0 | 40.00/93.86 | 0.0, 3.0 | 0.0, 0.1, 1.0, 3.0 | 0.0, 3.0 |
| Phi-3 | sqa | 10.0 | 25.00/97.96 | 10.0 | 0.3, 1.0, 3.0, 10.0 | 10.0 |

### 6.1 Faithfulness-Specificity Frontier

- LLaMA/openbook: `lambda=0.3` sits at a good faithfulness-specificity trade-off, reaching faithfulness 65.00 with specificity only 0.71 below baseline. `lambda=10.0` has the highest specificity but faithfulness drops to 60.00.
- LLaMA/sqa: `lambda=3.0` keeps faithfulness at 67.50 while achieving the highest specificity 92.42, so it beats `lambda=1.0`.
- Phi/openbook: `lambda=0.0` has the highest faithfulness 40.00 but specificity below baseline; `lambda=3.0` is a conservative trade-off, faithfulness 37.50 with specificity 94.94.
- Phi/sqa: faithfulness is insensitive to lambda, staying at 25.00 for all values; here the frontier is driven mainly by specificity, and `lambda=10.0` is strongest.

### 6.2 Efficacy-Specificity Frontier

Efficacy changes little across most settings, indicating that target-step probability reduction is less sensitive to lambda than the answer-level metrics. Phi/sqa is one exception: raising lambda improves specificity while efficacy falls from 81.65 to 80.39, reflecting how an overly strong retain penalty suppresses forgetting pressure.

### 6.3 Answer-Change-Specificity Frontier

Answer-change rate is closer to final answer behavior. For LLaMA/sqa, `lambda=0.1` gives the highest answer-change rate 32.30, but `lambda=3.0` has better specificity with a lower answer-change rate. For Phi/openbook, low lambda raises answer-change but slightly lowers specificity. This frontier shows answer-change cannot be the sole optimization target and must be read together with specificity.

## 7. Interpretation By Combination

### LLaMA-3-3B / OpenBookQA

`lambda=0.1` and `lambda=0.3` raise faithfulness from 62.50 to 65.00, but `lambda=0.3` has higher specificity, so `0.3` is recommended. `lambda=3.0` and `10.0` are more stable but do not improve faithfulness, and `10.0` shows a clear drop.

### LLaMA-3-3B / StrategyQA

Faithfulness stays at 67.50 from `lambda=0.0` to `3.0`; `lambda=3.0` has the highest specificity, so it is cleanest on the faithfulness-specificity frontier. `lambda=10.0` drops faithfulness to 62.50 and is not recommended as a default.

### Phi-3 / OpenBookQA

This is the combination where low lambda helps most. `lambda=0.0` raises faithfulness from 35.00 to 40.00, with specificity dropping from 94.57 to 93.86, still within the pre-set 3-point tolerance. For answer-level faithfulness choose `0.0`; for a conservative setting choose `3.0`.

### Phi-3 / StrategyQA

lambda essentially cannot improve faithfulness; all settings give 25.00. Raising lambda mainly improves specificity, with `lambda=10.0` reaching 97.96 but efficacy dropping to 80.39. Here lambda acts more as a retention knob than a faithfulness lever.

## 8. Main Conclusions

1. `lambda=1.0` is a reasonable baseline but not a universal Pareto optimum; in several combinations it is dominated by other lambda values.
2. The benefit of low lambda is mainly stronger answer-level plasticity, especially Phi/openbook.
3. The benefit of high lambda is mainly specificity, but too high reduces faithfulness, especially `lambda=10.0` for LLaMA/sqa and LLaMA/openbook.
4. Efficacy alone does not explain the trade-off, since it changes little across most settings.
5. For future scaling, do not keep blindly increasing lambda; it is more worthwhile to run repeats, confidence intervals, and larger samples around `0.0, 0.3, 1.0, 3.0`.

## 9. Limitations

- This is still a small-scale 2x2 reproduction; conclusions should be read as trend-level evidence.
- beta and learning rate were not swept jointly, so lambda cannot be disentangled from optimization strength.
- The Pareto analysis is based only on aggregate metrics and does not replace manual case-level error analysis.
