#!/usr/bin/env python3
"""Small multiple-choice retention check before and after protocol training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/reasoning/aqua_rat.jsonl"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    model_path = config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    candidates = ["A", "B", "C", "D", "E"]
    candidate_ids = []
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"candidate {candidate} is not one token: {encoded}")
        candidate_ids.append(encoded[0])

    items = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("answer") in candidates:
                items.append(item)
            if len(items) >= args.limit:
                break

    rows = []
    for start in range(0, len(items), args.batch_size):
        batch_items = items[start : start + args.batch_size]
        prompts = [
            "Solve the following multiple-choice problem. "
            "Return only the option letter (A, B, C, D, or E).\n\n"
            f"{item['problem']}\n\nAnswer:"
            for item in batch_items
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        last_positions = inputs["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            batch_logits = model(**inputs, use_cache=False).logits
        for row_index, item in enumerate(batch_items):
            logits = batch_logits[row_index, last_positions[row_index]].float()
            scores = [float(logits[token_id]) for token_id in candidate_ids]
            predicted = candidates[max(range(len(scores)), key=scores.__getitem__)]
            rows.append({"id": item.get("id"), "answer": item["answer"], "predicted": predicted, "correct": predicted == item["answer"]})

    result = {
        "data": str(args.data),
        "adapter": str(args.adapter) if args.adapter else None,
        "count": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / max(len(rows), 1),
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
