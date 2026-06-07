# new_lmf: NPO + KL + small LMF

This folder contains the extension that treats LMF as an auxiliary
forget-token regularizer, not as a replacement for NPO.

Loss:

```text
loss = NPO + rt_lambda * KL + lmf_coeff * LMF
```

Default pilot:

```powershell
python new_lmf/run_ablation.py --dry_run
python new_lmf/run_ablation.py --short_model Phi-3 --dataset openbook --n_unlearn 30
```

Single run:

```powershell
python new_lmf/run_new_lmf.py --short_model Phi-3 --dataset openbook --lr 1e-4 --lmf_coeff 0.03 --n_unlearn 30
```

Compute metrics:

```powershell
python new_lmf/compute_metrics.py --results-dir new_lmf/ablation --output-csv new_lmf/ablation_metrics.csv --output-json new_lmf/ablation_metrics.json
```

Recommended first grid:

```text
lr: use repro/config.py baseline lr for each model/dataset
lmf_coeff: 0.01, 0.03, 0.1, 0.3
n_unlearn: 30 for pilot, then 230 for final
```
