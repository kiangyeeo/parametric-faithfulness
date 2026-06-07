## 核心观点

NPO 的 unlearning criterion 是 **sequence-level likelihood ratio**，不是 token-level confidence removal。

它优化的是降低整段 forget CoT 的 likelihood：

```text
log p_theta(y_forget | x) < log p_ref(y_forget | x)
```

由于自回归序列概率是 token log-prob 的求和，NPO 可以只压低少数 token，使整段 likelihood 显著下降；但其他关键 content tokens 仍可能保留高 logit margin、高 max-softmax confidence 或低 entropy。

因此，NPO 的缺陷不是“忘得不够”，而是 **忘的证据粒度太粗**。

## 方法动机

LMF 不应作为 blanket flattening 使用。直接压平 logic tokens 会伤害 reasoning，也解释了：

```text
LMF + KL: 缺少 directional forgetting signal
naive NPO + KL + LMF: 无差别压 confidence，可能伤害 retain/reasoning
```

更合理的版本是：

```text
NPO + KL + Saturation-Gated Hinge LMF
```

即：

```text
loss = NPO + KL + mu * gate_NPO_saturated * hinge_LMF
```

其中：

```text
gate_NPO_saturated = stopgrad(sigmoid(beta * (NLL_current - NLL_ref)))
hinge_LMF = max(margin - tau, 0)^2
```

含义：

- 当 NPO 还没有完成 sequence-level forgetting 时，主要由 NPO 提供方向性遗忘信号。
- 当 `NLL_current - NLL_ref` 已经较大、NPO 梯度进入低梯度区时，再检查 token-level residual confidence。
- 只有 forget tokens 的 margin 仍过高时，LMF 才 clipping 这部分 residual confidence。

## 理论依据

代码中的 NPO 近似为：

```python
neg_log_ratios = forget_loss_current - forget_loss_oracle
forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
```

令：

```text
Delta = NLL_current(y|x) - NLL_ref(y|x)
```

则：

```text
L_NPO = -2 / beta * log sigmoid(beta * Delta)
dL/dDelta = -2 * sigmoid(-beta * Delta)
```

所以当 Delta 变大时，NPO 梯度会指数下降。默认 `beta = 0.1` 时：

```text
Delta = 20 nats -> gradient factor ≈ 0.238
Delta = 40 nats -> gradient factor ≈ 0.036
Delta = 60 nats -> gradient factor ≈ 0.005
```

问题在于：

```text
Delta = sum_i [NLL_current_i - NLL_ref_i]
```

这是 token-level NLL change 的总和。只要部分 token 被拉高，整体 Delta 就可能进入饱和区；其他 token 仍可能保持 high confidence。NPO 不会单独检查这些局部 token 是否已经被 flatten。

更极端地，NPO/CE-ascent 类更新还可能产生 confidently wrong：目标 token 概率下降，但概率质量集中到某个 alternative token，使 target CoT likelihood 下降而 max-softmax 仍然很高。

## 需要验证的关键比例

为了把这个理论风险变成强论点，需要估计：

```text
P(problem) = P(NPO saturated AND residual confidence high)
```

推荐指标：

```text
npo_grad_factor = 2 * sigmoid(-beta * Delta)
forget_margin_mean
forget_max_softmax_prob_mean
forget_entropy_mean
target_token_prob_mean
```

判定条件可用：

```text
npo_grad_factor < 0.05
AND forget_max_softmax_prob_mean > 0.5
```

或：

```text
npo_grad_factor < 0.05
AND forget_margin_mean remains high
```

如果该比例在某些组合中超过约 20%，即可支持：

```text
NPO often satisfies sequence-level forgetting while leaving token-level confidence residuals.
```

## 当前 diagnostics 结果

已完成的脚本：

```powershell
python new_lmf/analyze_npo_saturation_proxy.py
```

输出：

```text
new_lmf/npo_saturation_proxy_summary.csv
```

该脚本基于已有 `.out` 文件，用 `cot_step_prob` 估计：

```text
delta_proxy = log p_epoch0(target) - log p_epoch_t(target)
grad_factor_proxy = 2 * sigmoid(-beta * delta_proxy)
```

注意：`cot_step_prob` 是 length-penalized log probability，因此这是保守 proxy，不等同于训练时 NPO 使用的 unnormalized sequence NLL gap。

epoch 5 结果：

| dataset | model | records | delta_proxy mean | grad_factor_proxy mean | saturated proxy rate |
|---|---:|---:|---:|---:|---:|
| openbook | LLaMA-3-3B | 1308 | 72.027 | 0.034 | 0.907 |
| openbook | Phi-3 | 959 | 0.751 | 0.962 | 0.000 |
| sqa | LLaMA-3-3B | 1228 | 0.600 | 0.970 | 0.000 |
| sqa | Phi-3 | 999 | 27.660 | 0.217 | 0.230 |

结论：

- `openbook/LLaMA-3-3B` 中 sequence-level NPO saturation 很强，epoch 5 约 90.7% 进入低梯度区。
- `sqa/Phi-3` 中现象较弱但不可忽略，约 23.0% 进入低梯度区。
- `openbook/Phi-3` 和 `sqa/LLaMA-3-3B` 中 proxy 几乎没有 saturation，说明收益可能是 dataset/model dependent。

该实验支持“sequence-level NPO saturation 在部分主实验组合中真实且高频发生”，但尚不能证明 saturation 后仍有 high-margin tokens。

## 尚需运行的 logits 级 diagnostics

已编写脚本：

```powershell
python new_lmf/run_npo_residual_diagnostics.py --short_model Phi-3 --dataset openbook --lr 1e-4 --n_unlearn 5 --epochs 5 --trust_remote_code
```

它会重新跑小规模 `NPO+KL`，直接从 logits 记录：

```text
npo_delta
npo_grad_factor
forget_margin_mean / max
forget_max_softmax_prob_mean / max
forget_entropy_mean
target_token_prob_mean
positive_delta_top1_share
positive_delta_top20pct_share
saturated_and_high_conf
retain_kl
```

这一步用于真正检验：

```text
P(NPO saturated AND residual confidence high)
```

当前可写入 report 的谨慎表述是：

The proxy results show that sequence-level NPO saturation occurs frequently in some model-dataset settings. This motivates a residual, saturation-gated LMF term, but logits-level diagnostics are still required to confirm whether saturated examples retain high token-level confidence.
