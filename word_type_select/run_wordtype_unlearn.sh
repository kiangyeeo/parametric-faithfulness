#!/bin/bash
# ============================================================
# Modifier + LLaMA-3.2-3B Unlearning 训练脚本
# ============================================================
# 功能：对 LLaMA-3.2-3B-Instruct 模型进行 modifier 类型词汇的
#       分步遗忘训练，评估 CoT 推理忠实性
# ============================================================

set -e

# ---------- 环境配置 ----------
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/inspire/hdd/project/fdu-aidake-cfff/public/.huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/.transformers"

# 如需访问 gated model，设置 HF_TOKEN
# export HF_TOKEN="your_token_here"

# 安装 tiktoken（LLaMA 3.x 分词器依赖）
pip install tiktoken --break-system-packages -q 2>/dev/null || pip install tiktoken -q 2>/dev/null || true

# ---------- 训练参数 ----------
MODEL_NAME="/inspire/hdd/project/fdu-aidake-cfff/public/.huggingface/.hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
DATASET="sqa"           # 数据集: openbook / sqa
METHOD="npo_KL"              # 遗忘算法: NPO + KL 正则化
STRATEGY="sentencize"        # 分段策略: sentencize
STEPWISE=True                # 逐步遗忘每个 CoT 步骤
LR=3e-05                     # 学习率 (const.py 中 LLaMA-3-3B 推荐值)
EPOCHS=5                     # 遗忘训练轮数
SEED=1001                    # 随机种子
POS=True                     # 启用词性过滤
FF2=True                     # 仅优化 FF2 层 (mlp.down_proj)
RT_LAMBDA=1.0                # KL 正则化系数
WORD_TYPE_GROUP="modifier"            # 目标词汇类型: modifier

# ---------- 工作目录 ----------
WORK_DIR="/inspire/hdd/project/fdu-aidake-cfff/public/wanyizhou/measuring cot/parametric-faithfulness"
SCRIPT="${WORK_DIR}/unlearn_word_types.py"

cd "${WORK_DIR}"

echo "============================================"
echo "  Random Unlearning - LLaMA-3.2-3B"
echo "============================================"
echo "  Model:          ${MODEL_NAME}"
echo "  Dataset:        ${DATASET}"
echo "  Word Type:      ${WORD_TYPE_GROUP}"
echo "  Method:         ${METHOD}"
echo "  Strategy:       ${STRATEGY}"
echo "  Stepwise:       ${STEPWISE}"
echo "  LR:             ${LR}"
echo "  Epochs:         ${EPOCHS}"
echo "  FF2 only:       ${FF2}"
echo "  RT Lambda:      ${RT_LAMBDA}"
echo "  Seed:           ${SEED}"
echo "============================================"

# ---------- 启动训练 ----------
python "${SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --strategy "${STRATEGY}" \
    --stepwise \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --seed ${SEED} \
    --pos \
    --ff2 \
    --rt_lambda ${RT_LAMBDA} \
    --word_type_group "${WORD_TYPE_GROUP}"

echo "============================================"
echo "  Training completed!"
echo "============================================"
