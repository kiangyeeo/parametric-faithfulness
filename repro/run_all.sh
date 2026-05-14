#!/usr/bin/env bash
# repro/run_all.sh
#
# 串行跑完 2x2 复现的全部 4 组合。建议第一次先 SMOKE=1 跑一遍，
# pipeline 验通后再 unset SMOKE 跑正式实验。
#
# 用法：
#   export HF_TOKEN=hf_xxxxx       # 必须，Llama 是 gated
#   SMOKE=1 bash repro/run_all.sh  # smoke test (~10 分钟/组合)
#   bash repro/run_all.sh          # 正式 run (~数小时/组合，看 GPU)

set -e
cd "$(dirname "$0")/.."   # cd 到仓库根

SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
    echo ">>> SMOKE 模式：每组合只跑 5 条 × 2 epoch"
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
        $SMOKE_FLAG
done

echo ">>> 全部完成。结果在 final_results/ （或 smoke_results/）"
