# Lambda Regularization Experiment Report

## 1. Executive Summary

本实验将目标函数从 `L(theta) = L_NPO,beta(theta) + K_RT(theta)` 扩展为 `L(theta) = L_NPO,beta(theta) + lambda K_RT(theta)`，在固定 2x2 设置上扫描 `lambda in {0.0, 0.1, 0.3, 1.0, 3.0, 10.0}`。实验覆盖 Phi-3 与 LLaMA-3-3B，在 OpenBookQA 与 StrategyQA 上共 24 个正式组合；全部结果已写入 `lambda/results/lambda=<value>/...`，没有覆盖 `final_results`。

结论是：lambda 确实控制 stability-plasticity trade-off，但曲线并不严格单调。较低 lambda 通常更容易提高 answer-level faithfulness 或 answer-change 行为，较高 lambda 通常更保护 specificity；不过最优点明显依赖模型和数据集。以预先设定的“specificity 不比 lambda=1.0 低超过 3 个百分点”为约束，推荐值为：LLaMA/openbook 取 `0.3`，LLaMA/sqa 取 `3.0`，Phi/openbook 取 `0.0`，Phi/sqa 取 `10.0`。

## 2. Experiment Purpose And Hypotheses

实验目的：检验 retain KL regularization 权重 `lambda` 如何影响 unlearning efficacy、answer-level faithfulness、specificity 与 answer change rate，并判断当前隐式 `lambda=1.0` baseline 是否处在较优 trade-off 区间。

实验假设：

- H1: 较低 lambda 减弱 retain regularization，会提高局部可塑性，可能增强 faithfulness/answer change，但带来 specificity 风险。
- H2: 较高 lambda 增强 retained behavior 稳定性，可能提高 specificity，但压制 unlearning 对 final answer 的影响。
- H3: 当前 implicit baseline `lambda=1.0` 不一定是最优值。
- H4: Phi-3 与 LLaMA-3-3B 对 lambda 的敏感性不同。

结果总体支持这些假设，尤其 H3/H4：四个 model/dataset 组合的推荐 lambda 并不一致。

## 3. Objects, Scope, And Workflow

实验对象与范围：

- Models: Phi-3, LLaMA-3-3B
- Datasets: OpenBookQA (`openbook`), StrategyQA (`sqa`)
- Method: `npo_KL`, sentence-level stepwise unlearning
- Optimization: POS filter enabled, FF2-only optimization enabled
- Scale: `N_UNLEARN=40`, `N_VERIFY=10`, seed `1001`
- Lambda grid: `0.0, 0.1, 0.3, 1.0, 3.0, 10.0`

流程：先暴露 `--rt_lambda` 并确保结果路径包含 lambda；再 smoke test 验证 schema、路径和显存；随后在 2xA100 上完成完整 sweep；最后生成 JSON/CSV/PNG/SVG 和本综合报告。

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

这里把 specificity 和另一个指标同时视为要最大化的目标。如果某个 lambda 在两个目标上都不优于另一个 lambda，则它被视为 dominated。图中的黑色圆环和连线表示 nondominated lambda。

| Model | Dataset | Recommended lambda | Faithfulness/Specificity | Faithfulness frontier | Efficacy frontier | Answer-change frontier |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3-3B | openbook | 0.3 | 65.00/89.59 | 0.3, 3.0, 10.0 | 0.0, 0.3, 1.0, 3.0, 10.0 | 0.3, 3.0, 10.0 |
| LLaMA-3-3B | sqa | 3.0 | 67.50/92.42 | 3.0 | 0.1, 1.0, 3.0 | 0.1, 1.0, 3.0 |
| Phi-3 | openbook | 0.0 | 40.00/93.86 | 0.0, 3.0 | 0.0, 0.1, 1.0, 3.0 | 0.0, 3.0 |
| Phi-3 | sqa | 10.0 | 25.00/97.96 | 10.0 | 0.3, 1.0, 3.0, 10.0 | 10.0 |

### 6.1 Faithfulness-Specificity Frontier

- LLaMA/openbook: `lambda=0.3` 位于较好的 faithfulness-specificity 折中点，faithfulness 达到 65.00，specificity 只比 baseline 低 0.71。`lambda=10.0` specificity 最高但 faithfulness 下降到 60.00。
- LLaMA/sqa: `lambda=3.0` 在保持 faithfulness 67.50 的同时获得最高 specificity 92.42，因此比 `lambda=1.0` 更优。
- Phi/openbook: `lambda=0.0` faithfulness 最高 40.00，但 specificity 低于 baseline；`lambda=3.0` 是保守折中，faithfulness 37.50 且 specificity 94.94。
- Phi/sqa: faithfulness 对 lambda 不敏感，所有 lambda 都是 25.00；此时前沿主要由 specificity 决定，`lambda=10.0` 最强。

### 6.2 Efficacy-Specificity Frontier

Efficacy 在大部分设置中变化很小，说明 target-step probability reduction 对 lambda 不如 answer-level 指标敏感。Phi/sqa 是例外之一，高 lambda 提高 specificity 的同时 efficacy 从 81.65 降到 80.39，体现了过强 retain penalty 对 forgetting pressure 的抑制。

### 6.3 Answer-Change-Specificity Frontier

Answer-change rate 更接近最终 answer behavior。LLaMA/sqa 的 `lambda=0.1` 带来最高 answer-change rate 32.30，但 `lambda=3.0` specificity 更好且 answer-change rate 较低。Phi/openbook 的低 lambda 提高 answer-change，但同时略降 specificity。这个前沿说明 answer-change 不能单独作为最优目标，需要和 specificity 联合解释。

## 7. Interpretation By Combination

### LLaMA-3-3B / OpenBookQA

`lambda=0.1` 和 `lambda=0.3` 将 faithfulness 从 62.50 提高到 65.00，但 `lambda=0.3` 的 specificity 更高，因此推荐 `0.3`。`lambda=3.0` 和 `10.0` 更稳定，但 faithfulness 没有提升，`10.0` 还出现明显下降。

### LLaMA-3-3B / StrategyQA

`lambda=0.0` 到 `3.0` 的 faithfulness 都保持 67.50；`lambda=3.0` specificity 最高，所以它在 faithfulness-specificity 前沿上最干净。`lambda=10.0` faithfulness 降到 62.50，不建议作为默认设置。

### Phi-3 / OpenBookQA

这是低 lambda 最有用的组合。`lambda=0.0` 把 faithfulness 从 35.00 提高到 40.00，specificity 从 94.57 降到 93.86，仍在预设 3 个百分点容忍范围内。如果追求 answer-level faithfulness，可以选 `0.0`；如果偏保守，可以选 `3.0`。

### Phi-3 / StrategyQA

lambda 基本不能提高 faithfulness，所有设置都是 25.00。提高 lambda 主要提升 specificity，`lambda=10.0` 达到 97.96，但 efficacy 下降到 80.39。因此这里 lambda 更像 retention knob，而不是 faithfulness 改善手段。

## 8. Main Conclusions

1. `lambda=1.0` 是合理 baseline，但不是 Pareto 意义下的普遍最优点；在多个组合中它被其他 lambda 支配。
2. 低 lambda 的收益主要体现在更强 answer-level 可塑性，尤其 Phi/openbook。
3. 高 lambda 的收益主要体现在 specificity，但过高会降低 faithfulness，尤其 LLaMA/sqa 和 LLaMA/openbook 的 `lambda=10.0`。
4. Efficacy 本身不足以解释 trade-off，因为它在多数设置中变化很小。
5. 后续如果扩大实验，不建议继续盲目加大 lambda；更值得围绕 `0.0, 0.3, 1.0, 3.0` 做重复实验、置信区间和更大样本。

## 9. Limitations

- 本实验仍是小规模 2x2 reproduction，结论应视为趋势性证据。
- 没有同时 sweep beta 和 learning rate，因此无法分离 lambda 与优化强度的交互。
- Pareto 分析只基于汇总指标，没有替代人工 case-level error analysis。
