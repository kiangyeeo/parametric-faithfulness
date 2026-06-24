#!/usr/bin/env bash
# repro/run_all.sh
#
# Run all 4 cells of the 2x2 reproduction serially. With PRE=1 only the
# initial CoT/noCoT cache is generated (no unlearning); unset PRE for the
# real run.
#
# Usage:
#   export HF_TOKEN=hf_xxxxx       # required: Llama is gated
#   PRE=1 bash repro/run_all.sh    # generate initial CoT/noCoT cache only
#   bash repro/run_all.sh          # full run (~hours per cell, GPU-dependent)

set -e
cd "$(dirname "$0")/.."   # cd to repo root

EXTRA_ARGS=()
if [[ "${PRE:-0}" == "1" ]]; then
    EXTRA_ARGS=(--n_unlearn 0)
    echo ">>> PRE mode: generate initial CoT/noCoT cache only, skip unlearning"
fi

# 4 cells (short_model, dataset, best_lr); lr from upstream const.py:dataset_model_best_lr
RUNS=(
    "Phi-3       openbook  1e-4"
    "Phi-3       sqa       5e-5"
    "LLaMA-3-3B  openbook  3e-5"
    "LLaMA-3-3B  sqa       3e-5"
)

for r in "${RUNS[@]}"; do
    read -r model dataset lr <<< "$r"
    echo "==============================================="
    echo ">>> $model × $dataset (lr=$lr)"
    echo "==============================================="
    python -m repro.run_repro \
        --short_model "$model" \
        --dataset "$dataset" \
        --lr "$lr" \
        "${EXTRA_ARGS[@]}"
done

echo ">>> Done. Initial CoT/noCoT cache in final_cot/, full results in final_results/"
