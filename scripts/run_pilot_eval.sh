#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/120090727/yutong/conda_env/text-omni-flow/bin/python}
DATA=${DATA:-artifacts/data/synthetic_50k.jsonl}
TOKENIZER=${TOKENIZER:-artifacts/tokenizer}
GRID_ROOT=${GRID_ROOT:-artifacts/runs/pilot_grid}
EVAL_ROOT=${EVAL_ROOT:-artifacts/runs/pilot_eval}
LIMIT=${LIMIT:-100}
mkdir -p "$EVAL_ROOT"

run_eval() {
  local group=$1
  local lr=$2
  local device=$3
  local output="$EVAL_ROOT/${group}_lr_${lr}"
  mkdir -p "$output"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$device" "$PYTHON" src/tick_inference.py \
    --data "$DATA" \
    --tokenizer "$TOKENIZER" \
    --adapter "$GRID_ROOT/${group}_lr_${lr}/adapter" \
    --group "$group" \
    --cache \
    --split validation \
    --limit "$LIMIT" \
    --output "$output/predictions.jsonl" \
    > "$output/inference.log" 2>&1
  "$PYTHON" src/evaluate.py "$output/predictions.jsonl" --output "$output/metrics.json" \
    > "$output/evaluate.log" 2>&1
}

for lr in 1e-4 2e-4; do
  run_eval C "$lr" 0 & c_pid=$!
  run_eval D "$lr" 1 & d_pid=$!
  wait "$c_pid"
  wait "$d_pid"
done

"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/runs/pilot_eval")
grid = Path("artifacts/runs/pilot_grid")
rows = []
for lr in ("1e-4", "2e-4"):
    metrics = {g: json.loads((root / f"{g}_lr_{lr}" / "metrics.json").read_text()) for g in ("C", "D")}
    summaries = {g: json.loads((grid / f"{g}_lr_{lr}" / "summary.json").read_text()) for g in ("C", "D")}
    rows.append({
        "learning_rate": float(lr),
        "mean_joint_success": sum(metrics[g]["joint_session_success_rate"] for g in ("C", "D")) / 2,
        "mean_control_macro_f1": sum(metrics[g]["control_macro_f1"] for g in ("C", "D")) / 2,
        "mean_validation_loss": sum(summaries[g]["validation_loss"] for g in ("C", "D")) / 2,
        "groups": {g: {"metrics": metrics[g], "training": summaries[g]} for g in ("C", "D")},
    })
selected = max(rows, key=lambda x: (x["mean_joint_success"], x["mean_control_macro_f1"], -x["mean_validation_loss"]))
result = {
    "selection_rule": "max mean C/D joint success, then macro-F1, then minimum mean validation loss",
    "candidates": rows,
    "selected_learning_rate": selected["learning_rate"],
}
(root / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
