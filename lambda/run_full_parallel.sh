#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd/project/fdu-aidake-cfff/public/wangkengyi/lambda-work/parametric-faithfulness
export HUGGINGFACE_HUB_CACHE=/inspire/hdd/project/fdu-aidake-cfff/public/.huggingface/.hub
export TRANSFORMERS_CACHE=/inspire/hdd/project/fdu-aidake-cfff/public/.huggingface/.transformers
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PF=/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/pf/bin/python
mkdir -p lambda/logs
run_one() {
  local gpu=$1
  local model=$2
  local dataset=$3
  local lr=$4
  local lam=$5
  local label=$lam
  local log=lambda/logs/full_${model}_${dataset}_lambda=${label}.log
  echo [start] $(date -Iseconds) gpu=${gpu} model=${model} dataset=${dataset} lambda=${label} >> $log
  CUDA_VISIBLE_DEVICES=$gpu $PF -m repro.run_repro --short_model $model --dataset $dataset --lr $lr --rt_lambda $lam --results_dir lambda/results/lambda=${label} --log_suffix _lambda=${label} >> $log 2>&1
  echo [done] $(date -Iseconds) gpu=${gpu} model=${model} dataset=${dataset} lambda=${label} >> $log
}
run_queue() {
  local gpu=$1
  local model=$2
  local slots=$3
  shift 3
  local active=0
  for spec in $@; do
    IFS=: read -r dataset lr lam <<< $spec
    run_one $gpu $model $dataset $lr $lam &
    active=$((active + 1))
    if (( active >= slots )); then
      wait -n
      active=$((active - 1))
    fi
  done
  wait
}
case ${1:-} in
  phi)
    run_queue 1 Phi-3 3 openbook:0.0001:0.0 openbook:0.0001:0.1 openbook:0.0001:0.3 openbook:0.0001:1.0 openbook:0.0001:3.0 openbook:0.0001:10.0 sqa:0.00005:0.0 sqa:0.00005:0.1 sqa:0.00005:0.3 sqa:0.00005:1.0 sqa:0.00005:3.0 sqa:0.00005:10.0
    ;;
  llama)
    run_queue 0 LLaMA-3-3B 3 openbook:0.00003:0.0 openbook:0.00003:0.1 openbook:0.00003:0.3 openbook:0.00003:1.0 openbook:0.00003:3.0 openbook:0.00003:10.0 sqa:0.00003:0.0 sqa:0.00003:0.1 sqa:0.00003:0.3 sqa:0.00003:1.0 sqa:0.00003:3.0 sqa:0.00003:10.0
    ;;
  *)
    echo usage: $0 phi|llama >&2
    exit 2
    ;;
esac
