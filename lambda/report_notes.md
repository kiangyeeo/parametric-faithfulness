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

## 9. CoT-Level Interpretability Analysis

这一节把前面 aggregate-level 的 lambda 结论落到逐 CoT step。所有比较都使用跨 lambda 完全对齐的 `(id, step_idx)` 记录，并以 `lambda=1.0` 作为 baseline。这里最关键的局部量是 target-step logprob delta：如果它为正，说明该目标 CoT step 在当前 lambda 下比 `lambda=1.0` 更容易被模型继续生成，也就是 retain KL 对这个 step 的 forgetting 有抑制作用；如果它为负，则说明该 step 比 baseline 更容易被遗忘或替换。

### 9.1 高 lambda 的局部效应

| Model | Dataset | Lambda | Higher logprob share | Avg logprob delta | Prediction diff rate | Avg specificity drift delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LLaMA-3-3B | openbook | 10.0 | 96.12% | 5.2188 | 3.45% | -0.2155 |
| LLaMA-3-3B | openbook | 3.0 | 71.12% | 1.5237 | 1.72% | -0.0819 |
| LLaMA-3-3B | sqa | 10.0 | 96.19% | 4.7429 | 2.38% | 0.0476 |
| LLaMA-3-3B | sqa | 3.0 | 69.05% | 1.1976 | 0.48% | 0.0000 |
| Phi-3 | openbook | 10.0 | 93.60% | 5.7224 | 1.16% | -0.0349 |
| Phi-3 | openbook | 3.0 | 85.47% | 3.1250 | 1.16% | -0.0872 |
| Phi-3 | sqa | 10.0 | 82.94% | 1.2074 | 0.59% | -0.0647 |
| Phi-3 | sqa | 3.0 | 75.88% | 0.7191 | 0.59% | -0.0235 |

主要结论是：高 lambda 通常会让目标 CoT step 比 `lambda=1.0` 更高概率地保留下来。这个现象在 LLaMA/openbook 和 LLaMA/sqa 上最明显，`lambda=10.0` 的平均 final target-step logprob delta 分别约为 5.22 和 4.74；Phi/openbook 也很强，约为 5.72；Phi/sqa 较弱但方向仍为正。换句话说，调高 lambda 主要不是创造新的推理路径，而是增强对原始目标 step 的保留压力。这可以解释为什么高 lambda 往往提高 specificity、降低或抑制 answer-change，但也可能削弱 unlearning 对 final answer 的影响。

### 9.2 差异集中在 CoT 的哪些位置

| Model | Dataset | Lambda | Position | Avg logprob delta | Prediction diff rate | Avg specificity drift delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| LLaMA-3-3B | openbook | 10.0 | early | 7.1519 | 2.53% | -0.3291 |
| Phi-3 | openbook | 10.0 | late | 5.8904 | 2.74% | 0.0411 |
| Phi-3 | openbook | 10.0 | middle | 5.6827 | 0.00% | -0.1346 |
| Phi-3 | openbook | 10.0 | early | 5.6033 | 0.00% | -0.0217 |
| LLaMA-3-3B | openbook | 10.0 | middle | 5.4167 | 4.76% | -0.1429 |
| LLaMA-3-3B | sqa | 10.0 | early | 5.2586 | 6.90% | 0.1552 |
| LLaMA-3-3B | sqa | 10.0 | middle | 5.2348 | 1.52% | -0.0152 |
| LLaMA-3-3B | sqa | 10.0 | late | 4.0174 | 0.00% | 0.0233 |

差异并不是均匀分布在所有 CoT step 上。高 lambda 的最大 logprob 差异常出现在早期事实铺垫、中段因果/定义判断、以及带有排除或答案承诺的 step。比如 LLaMA/sqa 的 early step 在 `lambda=10.0` 下 prediction diff rate 达到 6.90%，说明一些早期判断一旦被保留或替换，可能更容易传播到最终答案。Phi/openbook 则在 middle 和 late step 上都有较强的 logprob 保留效应，但 final answer 分歧率很低，说明它更多表现为局部 CoT 稳定性变化，而不是答案翻转。

### 9.3 对“调高比例导致哪里差异”的解释

调高到 `lambda=3.0/10.0` 后，最直接的变化是目标 step 的 final logprob 相对 `lambda=1.0` 上升：四个 model/dataset 组合中，`lambda=10.0` 有 82.94% 到 96.19% 的 step 高于 baseline。也就是说，高 lambda 会在具体 step 层面抑制 NPO 对目标推理片段的删除。若该 step 是事实定义、因果桥接或答案选择相关判断，它被保留下来后 final answer 就更可能维持 baseline；若该 step 只是局部解释性文本，它可能只改变 new CoT 的措辞或 overlap，而不改变答案。

低 lambda 的行为正好提供对照：`lambda=0.0/0.1/0.3` 更容易降低目标 step logprob，代表更强 plasticity 和更强 forgetting pressure。这个方向有时会带来更高 answer-level faithfulness 或 answer-change，但也更容易引入 specificity drift。因此，lambda 的可解释作用可以概括为：它调节“目标 step 被替换的强度”和“retained probes 被扰动的强度”之间的平衡，而不是简单单调地提高所有指标。

### 9.4 具体案例证据

`lambda/cot_level_analysis/representative_cases.md` 列出了每个 model/dataset 的代表样本，包含高 lambda 保留目标 step、低 lambda 强化遗忘、final-answer 分歧、specificity drift 四类案例。每个案例都给出 question、target CoT step、step position、prediction change、target-step logprob delta、specificity drift delta 和解释，用来定位“到底是哪一个局部 CoT 判断发生了差异”。

### 9.5 新增 CoT-Level 产物

- `lambda/cot_level_analysis/cot_step_effects.csv`: 每一行是一个对齐后的 lambda/CoT step 比较。
- `lambda/cot_level_analysis/step_position_summary.csv`: 按 step index、early/middle/late 位置和启发式 target-step tag 聚合。
- `lambda/cot_level_analysis/high_lambda_effect_summary.csv`: 专门比较 `lambda=3.0/10.0` 相对 `lambda=1.0` 的局部影响。
- `lambda/cot_level_analysis/representative_cases.md`: 支撑数值结论的具体样本。
- `lambda/figures/cot_level_high_lambda_logprob_delta.svg`
- `lambda/figures/cot_level_high_lambda_prediction_diff.svg`
- `lambda/figures/cot_level_position_effects.svg`

## 10. Limitations

- 本实验仍是小规模 2x2 reproduction，结论应视为趋势性证据。
- 没有同时 sweep beta 和 learning rate，因此无法分离 lambda 与优化强度的交互。
- Pareto 分析只基于汇总指标，没有替代人工 case-level error analysis。
