#!/usr/bin/env bash
# repro/run_all.sh
#
# 串行跑完 2x2 复现的全部 4 组合。PRE=1 时只生成初始 CoT/noCoT
# 缓存结果，不执行 unlearning；正式实验 unset PRE。
#
# 用法：
#   export HF_TOKEN=hf_xxxxx       # 必须，Llama 是 gated
#   PRE=1 bash repro/run_all.sh    # 只生成初始 CoT/noCoT 结果
#   bash repro/run_all.sh          # 正式 run (~数小时/组合，看 GPU)

set -e
cd "$(dirname "$0")/.."   # cd 到仓库根

EXTRA_ARGS=()
if [[ "${PRE:-0}" == "1" ]]; then
    EXTRA_ARGS=(--n_unlearn 0)
    echo ">>> PRE 模式：只生成初始 CoT/noCoT 结果，跳过 unlearning"
fi

# 4 个组合 (short_model, dataset, best_lr)
# lr 来自源仓库 const.py 的 dataset_model_best_lr
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

echo ">>> 全部完成。初始 CoT/noCoT 缓存在 final_cot/，正式结果在 final_results/"
