#!/usr/bin/env python3
"""Tick-by-tick inference for group D (independent interaction control)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from protocol import CONTROL_TOKEN, Control, Session
from training_data import load_sessions


@dataclass
class TickPrediction:
    control: Control
    output_token_ids: list[int] = field(default_factory=list)
    out_end: bool = False
    violations: list[str] = field(default_factory=list)


class FullContextDecoder:
    """Correctness-first decoder that recomputes the complete context for every token."""

    def __init__(self, model, tokenizer, protocol_ids: dict[str, int]):
        self.model = model
        self.tokenizer = tokenizer
        self.ids = protocol_ids
        self.history: list[int] = []
        self.state = "idle"
        self.special_ids = set(protocol_ids.values())
        self.device = next(model.parameters()).device

    def reset(self, instruction: str) -> None:
        self.history = self.tokenizer.encode(instruction + "\n", add_special_tokens=False)
        self.state = "idle"

    @torch.inference_mode()
    def logits(self) -> torch.Tensor:
        inputs = torch.tensor([self.history], dtype=torch.long, device=self.device)
        return self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0, -1].float()

    def constrained_argmax(self, allowed: list[int]) -> int:
        scores = self.logits()
        allowed_tensor = torch.tensor(allowed, device=scores.device)
        return int(allowed_tensor[scores[allowed_tensor].argmax()].item())

    def raw_argmax(self, *, content: bool = False) -> int:
        scores = self.logits()
        scores[len(self.tokenizer) :] = -torch.inf
        if content:
            for token_id in self.special_ids - {self.ids["</output>"]}:
                scores[token_id] = -torch.inf
        return int(scores.argmax().item())

    def append(self, *token_ids: int) -> None:
        self.history.extend(token_ids)

    def run_tick(self, input_token_ids: list[int], max_output_tokens: int = 4) -> TickPrediction:
        self.append(self.ids["<tick>"], self.ids["<input>"])
        self.history.extend(input_token_ids)
        self.append(self.ids["</input>"])

        allowed_controls = [Control.LISTEN, Control.SPEAK] if self.state == "idle" else [Control.CONTINUE, Control.STOP]
        allowed_ids = [self.ids[CONTROL_TOKEN[control]] for control in allowed_controls]
        control_id = self.constrained_argmax(allowed_ids)
        control = next(control for control in allowed_controls if self.ids[CONTROL_TOKEN[control]] == control_id)
        self.append(control_id)
        prediction = TickPrediction(control)

        if control in {Control.SPEAK, Control.CONTINUE}:
            self.append(self.ids["<output>"])
            for _ in range(max_output_tokens + 1):
                token_id = self.raw_argmax(content=True)
                if token_id == self.ids["</output>"]:
                    self.append(token_id)
                    break
                if len(prediction.output_token_ids) >= max_output_tokens:
                    prediction.violations.append("chunk_overflow")
                    self.append(self.ids["</output>"])
                    break
                prediction.output_token_ids.append(token_id)
                self.append(token_id)
            else:
                prediction.violations.append("missing_output_end")
                self.append(self.ids["</output>"])

            closure = self.raw_argmax()
            if closure == self.ids["<out_end>"]:
                prediction.out_end = True
                self.append(closure)
                self.state = "idle"
                closure = self.raw_argmax()
            else:
                self.state = "speaking"
            if closure != self.ids["<tick_end>"]:
                prediction.violations.append("malformed_tick")
                closure = self.ids["<tick_end>"]
            self.append(closure)
        else:
            closure = self.raw_argmax()
            if closure != self.ids["<tick_end>"]:
                prediction.violations.append("malformed_tick")
                closure = self.ids["<tick_end>"]
            self.append(closure)
            self.state = "idle" if control == Control.STOP else self.state
        return prediction


def load_model(model_path: str, tokenizer_path: Path, adapter: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
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
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    manifest = json.loads((tokenizer_path / "protocol_tokens.json").read_text())
    return model, tokenizer, manifest["special_token_ids"]


def predict_session(decoder: FullContextDecoder, session: Session, max_output_tokens: int) -> dict[str, Any]:
    decoder.reset(session.instruction)
    predictions = []
    for index, tick in enumerate(session.ticks):
        prediction = decoder.run_tick(tick.input_token_ids, max_output_tokens)
        predictions.append(
            {
                "tick": index,
                "expected_control": tick.control.value,
                "predicted_control": prediction.control.value,
                "expected_output_token_ids": tick.output_token_ids,
                "predicted_output_token_ids": prediction.output_token_ids,
                "expected_out_end": tick.out_end,
                "predicted_out_end": prediction.out_end,
                "violations": prediction.violations,
            }
        )
    return {
        "session_id": session.session_id,
        "rule_family": session.rule_family,
        "split": session.split,
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", default="test_in_domain")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    sessions = load_sessions(args.data, {args.split})
    if args.limit:
        sessions = sessions[: args.limit]
    model, tokenizer, protocol_ids = load_model(config["model"]["base_model"], args.tokenizer, args.adapter)
    decoder = FullContextDecoder(model, tokenizer, protocol_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for session in sessions:
            result = predict_session(decoder, session, config["sequence"]["max_output_tokens_per_tick"])
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"completed {session.session_id}", flush=True)


if __name__ == "__main__":
    main()
