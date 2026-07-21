#!/usr/bin/env python3
"""QLoRA training for the text Omni-Flow protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from training_data import OmniFlowDataset, WeightedCollator, load_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("C", "D"), default="D")
    parser.add_argument("--max-train-sessions", type=int)
    parser.add_argument("--max-validation-sessions", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_causal_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # Keep logits in bf16: materializing a full fp32 copy wastes tens of GB at
    # large batches because Qwen3.5 has a 248k-token vocabulary.
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    denominator = shift_weights.sum().clamp_min(1.0)
    return (token_loss * shift_weights).sum() / denominator


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int = 64) -> float:
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        weights = batch.pop("loss_weights")
        labels = batch.pop("labels")
        outputs = model(**batch, use_cache=False)
        losses.append(weighted_causal_loss(outputs.logits, labels, weights).item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA training")
    set_seed(args.seed)
    config = yaml.safe_load(args.config.read_text())
    model_path = config["model"]["base_model"]
    max_length = int(config["sequence"]["context_length"])
    independent_control = args.group == "D"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_manifest = json.loads((args.tokenizer / "protocol_tokens.json").read_text())
    protocol_token_ids = token_manifest["special_token_ids"]
    special_ids = list(protocol_token_ids.values())

    train_sessions = load_sessions(args.data, {"train"})
    validation_sessions = load_sessions(args.data, {"validation"})
    if args.max_train_sessions:
        train_sessions = train_sessions[: args.max_train_sessions]
    validation_sessions = validation_sessions[: args.max_validation_sessions]
    train_dataset = OmniFlowDataset(
        train_sessions,
        tokenizer,
        protocol_token_ids,
        config["loss"]["weights"],
        independent_control=independent_control,
        max_length=max_length,
    )
    validation_dataset = OmniFlowDataset(
        validation_sessions,
        tokenizer,
        protocol_token_ids,
        config["loss"]["weights"],
        independent_control=independent_control,
        max_length=max_length,
    )
    collator = WeightedCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, collate_fn=collator)

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
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    use_gc = not args.no_gradient_checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        exclude_modules=["lm_head"],
        trainable_token_indices=special_ids,
        ensure_weight_tying=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = args.max_steps or updates_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * float(config["training"]["warmup_ratio"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_micro_steps = 0
    started = time.time()
    model.train()

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(args.epochs):
            for batch_index, batch in enumerate(train_loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                weights = batch.pop("loss_weights")
                labels = batch.pop("labels")
                outputs = model(**batch, use_cache=False)
                loss = weighted_causal_loss(outputs.logits, labels, weights)
                (loss / args.gradient_accumulation).backward()
                running_loss += loss.item()
                running_micro_steps += 1
                micro_step += 1

                should_update = micro_step % args.gradient_accumulation == 0 or batch_index + 1 == len(train_loader)
                if not should_update:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clipping"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": running_loss / max(running_micro_steps, 1),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": time.time() - started,
                        "max_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                    }
                    print(json.dumps(record), flush=True)
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    running_loss = 0.0
                    running_micro_steps = 0

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

    validation_loss = evaluate(model, validation_loader, device)
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "tokenizer")
    summary = {
        "group": args.group,
        "independent_control": independent_control,
        "train_sessions": len(train_dataset),
        "validation_sessions": len(validation_dataset),
        "global_steps": global_step,
        "validation_loss": validation_loss,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "wall_seconds": time.time() - started,
        "max_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "environment": os.environ.get("CONDA_PREFIX", ""),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
