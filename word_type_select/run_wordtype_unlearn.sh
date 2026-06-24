#!/bin/bash
# ============================================================
# Word-type stepwise unlearning (LLaMA-3.2-3B)
# ============================================================
# Unlearns one vocabulary-type group (e.g. modifier) per CoT step and
# evaluates CoT faithfulness. Run from the repo root.
# ============================================================

set -e
cd "$(dirname "$0")/.."   # cd to repo root

# ---------- environment ----------
export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/path/to/huggingface"           # optional cache location
# export HF_TOKEN="your_token_here"               # required for gated models

# ---------- parameters ----------
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"  # or a local snapshot path
DATASET="sqa"                 # openbook | sqa
METHOD="npo_KL"               # unlearning loss
STRATEGY="sentencize"         # CoT segmentation
LR=3e-05                      # const.py best lr for LLaMA-3-3B
EPOCHS=5
SEED=1001
RT_LAMBDA=1.0                 # retain KL coefficient
WORD_TYPE_GROUP="modifier"    # entity | attribute | action | function | modifier | all_content | random

SCRIPT="word_type_select/unlearn_for_more_kinds_of_words.py"

echo "============================================"
echo "  Word-type unlearning - LLaMA-3.2-3B"
echo "============================================"
echo "  Model:       ${MODEL_NAME}"
echo "  Dataset:     ${DATASET}"
echo "  Word type:   ${WORD_TYPE_GROUP}"
echo "  Method:      ${METHOD}"
echo "  Strategy:    ${STRATEGY}"
echo "  LR:          ${LR}"
echo "  Epochs:      ${EPOCHS}"
echo "  RT lambda:   ${RT_LAMBDA}"
echo "  Seed:        ${SEED}"
echo "============================================"

# --stepwise / --pos / --ff2 are store_true flags (enabled by being present).
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
