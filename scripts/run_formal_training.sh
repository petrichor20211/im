#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/120090727/yutong/conda_env/text-omni-flow/bin/python}
DATA=${DATA:-artifacts/data/synthetic_50k.jsonl}
TOKENIZER=${TOKENIZER:-artifacts/tokenizer}
SELECTION=${SELECTION:-artifacts/runs/pilot_eval/selection.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/runs/formal}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-3}

lr=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_learning_rate"])' "$SELECTION")
echo "Selected learning rate: $lr"

run_one() {
  local group=$1
  local seed=$2
  local device=$3
  local output="$OUTPUT_ROOT/seed_${seed}/${group}"
  mkdir -p "$output"
  echo "[$(date -Is)] starting group=$group seed=$seed gpu=$device"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$device" "$PYTHON" src/train_qlora.py \
    --data "$DATA" \
    --tokenizer "$TOKENIZER" \
    --output "$output" \
    --group "$group" \
    --max-validation-sessions 2048 \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation 1 \
    --epochs "$EPOCHS" \
    --learning-rate "$lr" \
    --seed "$seed" \
    --log-every 25 \
    > "$output/launcher.log" 2>&1
  echo "[$(date -Is)] completed group=$group seed=$seed gpu=$device"
}

for seed in 20260721 20260722 20260723; do
  run_one C "$seed" 0 & c_pid=$!
  run_one D "$seed" 1 & d_pid=$!
  wait "$c_pid"
  wait "$d_pid"
done
