# Lambda Regularization Experiment Plan

## 1. Background

This repository reproduces and extends FUR, Faithfulness by Unlearning Reasoning steps, for measuring parametric faithfulness in Chain-of-Thought reasoning. The implemented reproduction focuses on a scaled-down 2x2 setting:

- Models: `Phi-3-mini-4k-instruct` and `Llama-3.2-3B-Instruct`
- Datasets: OpenBookQA and StrategyQA
- Current main method: stepwise sentence-level unlearning with `npo_KL`
- Current evaluation metrics: faithfulness, efficacy, specificity, and answer change rate

The current implementation of `npo_KL` in `unlearn.py` computes:

```text
L(theta) = L_NPO,beta(theta) + K_RT(theta)
```

where the NPO term is the forget loss on the target reasoning step, and `K_RT` is implemented as a retain-set KL regularization term that keeps the current model close to the frozen oracle model on retained CoT samples. In code, the final line is already structurally parameterized:

```python
loss = npo_coeff * forget_loss + KL_coeff * retain_loss
```

However, `KL_coeff` currently defaults to `1.0` and is not exposed through the command-line interface, `repro/config.py`, result filenames, or analysis scripts. The planned lambda experiment makes this coefficient explicit:

```text
L(theta) = L_NPO,beta(theta) + lambda * K_RT(theta)
```

The existing baseline corresponds to `lambda = 1.0`.

## 2. Experiment Objective

The primary objective is to measure how the retain regularization weight `lambda` affects the trade-off between unlearning efficacy, answer-level faithfulness, and model specificity.

Concretely, the experiment should answer:

1. Does reducing `lambda` improve faithfulness by allowing stronger local removal of the target reasoning step?
2. Does increasing `lambda` preserve specificity better but weaken the unlearning effect?
3. Is there a stable middle range of `lambda` that improves the faithfulness-specificity trade-off relative to the current implicit `lambda = 1.0` baseline?
4. Are the trends consistent across model families and datasets, or are they model/dataset dependent?

## 3. Experiment Hypotheses

### H1: Lower lambda increases efficacy but may hurt specificity

When `lambda < 1`, the retain KL penalty becomes weaker. The optimizer should more aggressively reduce the probability of the targeted CoT step, so efficacy is expected to increase. However, weaker retain regularization may also increase collateral changes on held-out examples, lowering specificity.

Expected pattern:

```text
lambda down -> efficacy up, faithfulness possibly up, specificity down
```

### H2: Higher lambda improves specificity but may suppress unlearning

When `lambda > 1`, the model is more strongly constrained to remain close to the oracle model on retained samples. This should protect specificity, but may reduce the probability shift on the unlearned step and reduce answer changes caused by unlearning.

Expected pattern:

```text
lambda up -> specificity up, efficacy down, faithfulness possibly down
```

### H3: The best lambda is not necessarily 1.0

The current FUR reproduction uses an implicit `lambda = 1.0`. Because this repository runs a smaller local reproduction setting, the best regularization strength may shift. A smaller or larger `lambda` may produce a better faithfulness-specificity frontier under the 2x2 setup.

### H4: LLaMA-3-3B and Phi-3 may respond differently

Existing reproduction results show stronger faithfulness for LLaMA-3-3B than Phi-3 under `lambda = 1.0`. The lambda sweep should test whether this gap is robust, or whether Phi-3 is more sensitive to the regularization coefficient.

## 4. Experiment Object and Scope

### Main scope

Use the existing 2x2 reproduction setup:

| Model | Dataset | Baseline LR |
| --- | --- | --- |
| Phi-3 | openbook | `1e-4` |
| Phi-3 | sqa | `5e-5` |
| LLaMA-3-3B | openbook | `3e-5` |
| LLaMA-3-3B | sqa | `3e-5` |

Keep all other settings aligned with the current reproduction:

- Method: `npo_KL`
- Strategy: `sentencize`
- Stepwise unlearning: `True`
- POS filter: `True`
- FF2-only optimization: `True`
- Epochs: `5`
- Seed: `1001`
- CoT cache: existing `final_cot`
- Dataset subset: local OpenBookQA and StrategyQA samples under `data/`
- Main sample scale: `N_UNLEARN = 40`, `N_VERIFY = 10`

### Lambda sweep range

Recommended first sweep:

```text
lambda in {0.0, 0.1, 0.3, 1.0, 3.0, 10.0}
```

Rationale:

- `0.0`: removes retain regularization and gives a lower-bound control for collateral damage.
- `0.1`, `0.3`: weak regularization region.
- `1.0`: current baseline, must be rerun or reused as the anchor.
- `3.0`, `10.0`: stronger regularization region.

If compute is limited, run a pilot sweep first:

```text
lambda in {0.0, 0.3, 1.0, 3.0}
```

If the pilot shows a turning point, refine locally around that region.

### Out of scope for the first lambda experiment

- Do not change beta unless the lambda trend is understood.
- Do not change learning rates during the first pass.
- Do not mix the lambda sweep with SimNPO, Gradient Ascent, MMLU, GSM8K, or LLM-as-judge unless the main 2x2 sweep is complete.
- Do not regenerate CoTs unless cache inconsistency is discovered.

## 5. Current Environment Assessment

The repository is ready for experiment planning, but the current environment needs preparation before running the sweep.

Observed local state:

- Repository has existing `final_cot`, `final_results`, `mistake_results`, `mistake_stats`, `simnpo_results`, and `evaluate_and_visualize` outputs.
- `final_cot` already contains cached CoTs for the 2x2 reproduction setup.
- Local data subsets exist under `data/openbookqa` and `data/strategyqa`.
- Current default Python is `D:\Users\hp\anaconda3\python.exe`.
- CUDA is available with one `NVIDIA GeForce RTX 4060 Ti`.
- The `llm-26-gpu` conda environment has `torch`, `transformers`, `datasets`, `spacy`, `matplotlib`, `scipy`, and `huggingface_hub`.
- The `llm-26-gpu` environment is missing `nltk`; the default base environment has `nltk` but lacks `datasets`.
- `HF_TOKEN` is not currently set.
- Hugging Face cache paths for Phi-3 and LLaMA-3.2-3B were not found in the default user cache.

Practical implication:

Before experiments, use `llm-26-gpu` or another complete environment, install/enable missing `nltk`, set `HF_TOKEN` or provide local model paths, and run a smoke test. Because `unlearn_single` loads both the trainable model and frozen oracle model, GPU memory should be checked during the first smoke run.

## 6. Required Code Changes Before Running

No experiment should be launched until the following small code changes are made and reviewed.

### 6.1 Expose lambda in the loss path

`compute_loss` already accepts `KL_coeff`, but `unlearn_single` currently calls:

```python
loss = compute_loss(model, oracle_model, batch, loss_type=args.method)
```

The experiment needs a runtime parameter, for example:

```python
loss = compute_loss(
    model,
    oracle_model,
    batch,
    loss_type=args.method,
    KL_coeff=args.rt_lambda,
)
```

Because `lambda` is a Python keyword, use a name such as `rt_lambda`, `lambda_rt`, or `retain_lambda` in code.

### 6.2 Add CLI/config support

Add the same parameter to:

- `unlearn.make_parser`
- `repro/config.py`
- `repro/run_repro.py::build_args`
- optional batch script or future lambda-specific runner

Default value should be `1.0`, preserving existing behavior.

### 6.3 Encode lambda in result paths

Result filenames or directories must include lambda to prevent collisions. A safe layout is:

```text
lambda/results/lambda={value}/{dataset}/{model}/{method}_sentencize_s=True_lr={lr}_rs=1001_pos=True_ff2=True.out
```

or:

```text
lambda/results/{dataset}/{model}/{method}_sentencize_s=True_lr={lr}_lambda={value}_rs=1001_pos=True_ff2=True.out
```

The first layout is easier for cross-lambda analysis.

### 6.4 Update analysis scripts

The existing metrics functions in `stats.py` can be reused. A lambda analysis script should aggregate by:

```text
lambda, dataset, model
```

and produce JSON/CSV summaries under `lambda/`.

## 7. Experiment Workflow

### Step 0: Preparation only

Complete the code changes above, but do not start full experiments. Verify that `lambda = 1.0` gives the same behavior and output schema as the current baseline.

### Step 1: Smoke test

Run one small smoke test before the sweep:

```text
model = Phi-3
dataset = openbook
lambda in {0.0, 1.0, 3.0}
N_UNLEARN = 5
epochs = 2
```

Acceptance criteria:

- Output files are written to lambda-specific paths.
- No result files overwrite existing `final_results`.
- Result JSONL schema matches current `final_results`.
- `lambda = 1.0` is compatible with current analysis functions.
- GPU memory behavior is acceptable.

### Step 2: Pilot sweep

Run:

```text
lambda in {0.0, 0.3, 1.0, 3.0}
```

for the 2x2 combinations. This gives an initial view of monotonicity and compute cost.

### Step 3: Full sweep

Run:

```text
lambda in {0.0, 0.1, 0.3, 1.0, 3.0, 10.0}
```

for all 2x2 combinations with the main reproduction scale.

### Step 4: Aggregate metrics

For every `(lambda, dataset, model)` combination, compute:

- `n_instances`
- `n_cot_steps`
- faithfulness
- efficacy
- specificity
- answer change rate among no-CoT/CoT agreeing instances
- average mass shift
- max mass shift

Also retain per-epoch curves where possible:

- step probability reduction curve
- answer probability mass shift curve
- specificity curve over epochs

### Step 5: Select candidate lambda values

Define selection rules before looking at final plots:

- Primary candidate: highest faithfulness subject to specificity not dropping below the baseline by more than an acceptable tolerance.
- Conservative candidate: highest specificity among values that improve faithfulness over `lambda = 1.0`.
- Diagnostic candidate: strongest efficacy value, even if specificity drops, used only to explain the trade-off.

Suggested tolerance:

```text
specificity(lambda) >= specificity(lambda=1.0) - 3 percentage points
```

This can be adjusted, but should be fixed before reporting.

## 8. Core Results to Produce

### 8.1 Summary table

One table with rows:

```text
lambda x model x dataset
```

Columns:

- faithfulness
- efficacy
- specificity
- answer change rate
- average mass shift
- max mass shift
- number of unique instances
- number of CoT steps

### 8.2 Baseline-relative table

For each lambda, report differences from `lambda = 1.0`:

```text
Delta faithfulness
Delta efficacy
Delta specificity
Delta answer change rate
```

This is important because absolute values vary across model and dataset.

### 8.3 Trade-off plots

Recommended plots:

- `lambda` vs faithfulness
- `lambda` vs efficacy
- `lambda` vs specificity
- faithfulness vs specificity frontier
- efficacy vs specificity frontier
- answer change rate vs specificity

Use log-scale x-axis for lambda, except represent `lambda = 0.0` separately or as a leftmost categorical point.

### 8.4 Per-combination diagnostics

For each of the 2x2 combinations, include:

- Which lambda maximizes faithfulness?
- Which lambda maximizes specificity?
- Which lambda gives the best trade-off under the pre-defined tolerance?
- Does the curve look monotonic, U-shaped, or noisy?

### 8.5 Failure case samples

For selected lambdas, inspect a small number of examples:

- A case where lower lambda changes the answer but damages specificity.
- A case where higher lambda preserves specificity but fails to unlearn.
- A case where an intermediate lambda changes the target answer while retaining held-out behavior.

Do not rely only on aggregate metrics; qualitative examples are needed to explain what lambda is doing.

## 9. Required Result Interpretation

The final analysis should explicitly separate three concepts:

### 9.1 Efficacy is not enough

If `lambda = 0.0` produces the largest target step probability reduction, that does not automatically mean it is best. The result must be interpreted together with specificity and answer-level behavior.

### 9.2 Faithfulness must be read with specificity

A higher answer change rate after unlearning can indicate stronger parametric dependence on the removed reasoning step, but it can also reflect general model disruption. The key question is whether answer changes happen while unrelated held-out predictions remain stable.

### 9.3 Lambda controls a stability-plasticity trade-off

The expected interpretation is:

```text
low lambda: more plastic, stronger unlearning, higher collateral risk
high lambda: more stable, weaker unlearning, lower collateral risk
intermediate lambda: possible best trade-off
```

The report should determine whether this expected trade-off actually appears in the 2x2 reproduction setting.

### 9.4 Compare across models and datasets

The report should not average everything too early. First interpret:

- OpenBookQA vs StrategyQA
- Phi-3 vs LLaMA-3-3B
- model-specific sensitivity to lambda
- dataset-specific sensitivity to lambda

Only then provide an overall recommendation.

### 9.5 Relate back to current repository results

Use the current `lambda = 1.0` reproduction results as the anchor:

| Setting | Faithfulness | Efficacy | Specificity |
| --- | ---: | ---: | ---: |
| openbook / LLaMA-3-3B | 62.5 | 82.34 | 90.44 |
| openbook / Phi-3 | 35.0 | 82.87 | 94.71 |
| sqa / LLaMA-3-3B | 67.5 | 82.91 | 92.32 |
| sqa / Phi-3 | 25.0 | 81.71 | 97.05 |

Also compare against the current answer change-rate summary:

| Setting | Mistake change rate | Unlearning change rate |
| --- | ---: | ---: |
| openbook / LLaMA-3-3B | 11.76 | 28.28 |
| openbook / Phi-3 | 9.90 | 12.88 |
| sqa / LLaMA-3-3B | 7.85 | 31.06 |
| sqa / Phi-3 | 9.88 | 12.41 |

The lambda experiment should explain whether these numbers are robust to the retain regularization strength or are artifacts of the implicit `lambda = 1.0`.

## 10. Deliverables

The lambda experiment should eventually produce:

- `lambda/experiment_plan.md`: this planning document.
- `lambda/results/`: raw JSONL outputs from lambda-specific runs.
- `lambda/analysis_summary.json`: metric summary by lambda/model/dataset.
- `lambda/answer_change_rate_summary.json`: answer change summary by lambda/model/dataset.
- `lambda/summary_table.csv`: table-ready numerical summary.
- `lambda/figures/`: plots for lambda curves and trade-off frontiers.
- `lambda/report_notes.md`: concise interpretation notes for the final report or presentation.

Only the planning document is created at this stage. No experiment run is started here.
