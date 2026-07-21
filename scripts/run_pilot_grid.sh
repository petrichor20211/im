#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/120090727/yutong/conda_env/text-omni-flow/bin/python}
DATA=${DATA:-artifacts/data/synthetic_50k.jsonl}
TOKENIZER=${TOKENIZER:-artifacts/tokenizer}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/runs/pilot_grid}
SEED=${SEED:-20260721}

run_one() {
  local group=$1
  local lr=$2
  local device=$3
  local output="$OUTPUT_ROOT/${group}_lr_${lr}"
  mkdir -p "$output"
  echo "[$(date -Is)] starting group=$group lr=$lr gpu=$device output=$output"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$device" "$PYTHON" src/train_qlora.py \
    --data "$DATA" \
    --tokenizer "$TOKENIZER" \
    --output "$output" \
    --group "$group" \
    --max-train-sessions 500 \
    --max-validation-sessions 256 \
    --batch-size 32 \
    --gradient-accumulation 1 \
    --epochs 3 \
    --learning-rate "$lr" \
    --seed "$SEED" \
    --log-every 10 \
    > "$output/launcher.log" 2>&1
  echo "[$(date -Is)] completed group=$group lr=$lr gpu=$device"
}

# C and D use separate GPUs but identical data and hyperparameters. Finish both
# groups at one learning rate before moving to the next candidate.
for lr in 1e-4 2e-4; do
  run_one C "$lr" 0 &
  c_pid=$!
  run_one D "$lr" 1 &
  d_pid=$!
  wait "$c_pid"
  wait "$d_pid"
done
