#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/120090727/yutong/conda_env/text-omni-flow/bin/python}
DATA=${DATA:-artifacts/data/synthetic_50k.jsonl}
TOKENIZER=${TOKENIZER:-artifacts/tokenizer}
FORMAL_ROOT=${FORMAL_ROOT:-artifacts/runs/formal}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/runs/formal_eval}
SHARDS_PER_GROUP=${SHARDS_PER_GROUP:-8}
SPLITS=(test_in_domain test_template_ood test_timing_ood test_length_ood test_distractor_ood)
SEEDS=(20260721 20260722 20260723)
mkdir -p "$OUTPUT_ROOT"

run_group_split() {
  local seed=$1 group=$2 split=$3 device=$4
  local root="$OUTPUT_ROOT/seed_${seed}/${group}/${split}"
  if [ -s "$root/metrics.json" ]; then
    echo "[$(date -Is)] skipping completed seed=$seed group=$group split=$split"
    return 0
  fi
  rm -rf "$root/shards"
  mkdir -p "$root/shards"
  local pids=()
  for ((shard=0; shard<SHARDS_PER_GROUP; shard++)); do
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$device" "$PYTHON" src/tick_inference.py \
      --data "$DATA" --tokenizer "$TOKENIZER" \
      --adapter "$FORMAL_ROOT/seed_${seed}/${group}/adapter" \
      --group "$group" --cache --split "$split" \
      --num-shards "$SHARDS_PER_GROUP" --shard-index "$shard" \
      --output "$root/shards/${shard}.jsonl" \
      > "$root/shards/${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  [ "$rc" -eq 0 ] || return "$rc"
  cat "$root"/shards/*.jsonl > "$root/predictions.jsonl"
  "$PYTHON" src/evaluate.py "$root/predictions.jsonl" --output "$root/metrics.json" > "$root/evaluate.log"
}

for seed in "${SEEDS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "[$(date -Is)] seed=$seed split=$split"
    run_group_split "$seed" C "$split" 0 & c_pid=$!
    run_group_split "$seed" D "$split" 1 & d_pid=$!
    wait "$c_pid"
    wait "$d_pid"
  done
  # Retention is evaluated for both trained groups with identical examples.
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" src/evaluate_retention.py \
    --adapter "$FORMAL_ROOT/seed_${seed}/C/adapter" \
    --output "$OUTPUT_ROOT/seed_${seed}/C/retention.json" > "$OUTPUT_ROOT/seed_${seed}/C/retention.log" 2>&1 & c_pid=$!
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" src/evaluate_retention.py \
    --adapter "$FORMAL_ROOT/seed_${seed}/D/adapter" \
    --output "$OUTPUT_ROOT/seed_${seed}/D/retention.json" > "$OUTPUT_ROOT/seed_${seed}/D/retention.log" 2>&1 & d_pid=$!
  wait "$c_pid"
  wait "$d_pid"
done

# Untrained chunked baseline B and ordinary multiple-choice baseline are run once.
for split in "${SPLITS[@]}"; do
  root="$OUTPUT_ROOT/baseline_B/$split"
  if [ -s "$root/metrics.json" ]; then
    echo "[$(date -Is)] skipping completed baseline split=$split"
    continue
  fi
  rm -rf "$root/shards"
  mkdir -p "$root/shards"
  pids=()
  total_shards=$((SHARDS_PER_GROUP * 2))
  for ((shard=0; shard<total_shards; shard++)); do
    device=$((shard % 2))
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$device" "$PYTHON" src/tick_inference.py \
      --data "$DATA" --tokenizer "$TOKENIZER" --group D --cache --split "$split" \
      --num-shards "$total_shards" --shard-index "$shard" \
      --output "$root/shards/${shard}.jsonl" > "$root/shards/${shard}.log" 2>&1 &
    pids+=("$!")
  done
  rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done; [ "$rc" -eq 0 ]
  cat "$root"/shards/*.jsonl > "$root/predictions.jsonl"
  "$PYTHON" src/evaluate.py "$root/predictions.jsonl" --output "$root/metrics.json" > "$root/evaluate.log"
done
CUDA_VISIBLE_DEVICES=0 "$PYTHON" src/evaluate_retention.py \
  --output "$OUTPUT_ROOT/baseline_retention.json" > "$OUTPUT_ROOT/baseline_retention.log" 2>&1
