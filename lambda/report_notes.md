# Lambda Sweep Report

Generated: 2026-05-24T01:04:35

## Experiment Design

- Objective: evaluate how scaling the retain regularizer in L(theta)=L_NPO,beta(theta)+lambda*K_RT(theta) changes the faithfulness/specificity tradeoff.
- Hypothesis: lambda=0.0 should maximize forgetting pressure but may damage retained behavior; larger lambda values should better preserve specificity while potentially weakening step-level intervention.
- Scope: the fixed 2x2 reproduction grid from repro/config.py, namely Phi-3 and LLaMA-3-3B on OpenBookQA and StrategyQA.
- Lambda grid: 0.0, 0.1, 0.3, 1.0, 3.0, 10.0.

## Completeness

- Expected result groups: 24
- Observed result groups: 24
- Missing result groups: 0

## Core Metrics

| Lambda | Model | Dataset | Faithfulness | Efficacy | Specificity | Answer Change |
| --- | --- | --- | ---: | ---: | ---: | ---: |
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

## Baseline-Relative Interpretation

Deltas are computed against lambda=1.0 within the same model and dataset.
Positive faithfulness/efficacy deltas indicate stronger step-level intervention; positive specificity deltas indicate better retention on held-out examples.

| Lambda | Model | Dataset | Delta Faithfulness | Delta Efficacy | Delta Specificity | Delta Answer Change |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0.0 | LLaMA-3-3B | openbook | 0.00 | 0.03 | -1.04 | 0.00 |
| 0.1 | LLaMA-3-3B | openbook | 2.50 | 0.00 | -1.09 | 0.51 |
| 0.3 | LLaMA-3-3B | openbook | 2.50 | 0.01 | -0.71 | 0.51 |
| 1.0 | LLaMA-3-3B | openbook | 0.00 | 0.00 | 0.00 | 0.00 |
| 3.0 | LLaMA-3-3B | openbook | 0.00 | -0.01 | 0.76 | 0.00 |
| 10.0 | LLaMA-3-3B | openbook | -2.50 | -0.12 | 1.64 | -1.52 |
| 0.0 | LLaMA-3-3B | sqa | 0.00 | -0.00 | -0.14 | 0.62 |
| 0.1 | LLaMA-3-3B | sqa | 0.00 | 0.00 | -0.06 | 1.24 |
| 0.3 | LLaMA-3-3B | sqa | 0.00 | -0.00 | -0.11 | 0.62 |
| 1.0 | LLaMA-3-3B | sqa | 0.00 | 0.00 | 0.00 | 0.00 |
| 3.0 | LLaMA-3-3B | sqa | 0.00 | -0.00 | 0.13 | -0.62 |
| 10.0 | LLaMA-3-3B | sqa | -5.00 | -0.00 | -0.13 | -0.62 |
| 0.0 | Phi-3 | openbook | 5.00 | 0.00 | -0.71 | 1.27 |
| 0.1 | Phi-3 | openbook | 2.50 | 0.00 | -0.47 | 0.64 |
| 0.3 | Phi-3 | openbook | 0.00 | -0.00 | -0.17 | 0.00 |
| 1.0 | Phi-3 | openbook | 0.00 | 0.00 | 0.00 | 0.00 |
| 3.0 | Phi-3 | openbook | 2.50 | -0.00 | 0.37 | 0.64 |
| 10.0 | Phi-3 | openbook | 2.50 | -0.00 | 0.13 | 0.64 |
| 0.0 | Phi-3 | sqa | 0.00 | -0.00 | -0.39 | -1.43 |
| 0.1 | Phi-3 | sqa | 0.00 | -0.21 | -0.33 | -0.71 |
| 0.3 | Phi-3 | sqa | 0.00 | 0.11 | -0.31 | 0.00 |
| 1.0 | Phi-3 | sqa | 0.00 | 0.00 | 0.00 | 0.00 |
| 3.0 | Phi-3 | sqa | 0.00 | -0.52 | 0.33 | 0.00 |
| 10.0 | Phi-3 | sqa | 0.00 | -1.26 | 0.78 | 0.00 |

## Recommended Lambda By Combination

- LLaMA-3-3B/openbook: lambda=0.3 (faithfulness=65.00, specificity=89.59).
- LLaMA-3-3B/sqa: lambda=3.0 (faithfulness=67.50, specificity=92.42).
- Phi-3/openbook: lambda=0.0 (faithfulness=40.00, specificity=93.86).
- Phi-3/sqa: lambda=10.0 (faithfulness=25.00, specificity=97.96).

## Interpretation Guide

- Low lambda weakens retain regularization; it should be treated as a plasticity stress test, not automatically as the best setting.
- High lambda strengthens retain regularization; it is useful when specificity is the primary constraint.
- The selected trade-off uses the pre-declared rule: maximize faithfulness while staying within 3 specificity points of lambda=1.0.
- Answer-change rate is diagnostic rather than the sole target metric; it can rise when unlearning disrupts final-answer behavior.
- Missing groups indicate incomplete runs, failed jobs, or results written outside the expected lambda/results/lambda=<value>/ tree.
