"""Token-level training examples and weighted collation for text Omni-Flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from protocol import CONTROL_TOKEN, Control, Session

IGNORE_INDEX = -100


@dataclass
class EncodedSession:
    input_ids: list[int]
    labels: list[int]
    loss_weights: list[float]
    session_id: str


def load_sessions(path: str | Path, splits: set[str] | None = None) -> list[Session]:
    sessions = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            session = Session.from_dict(json.loads(line))
            if splits is None or session.split in splits:
                sessions.append(session)
    return sessions


def encode_session(
    session: Session,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: dict[str, int],
    loss_config: dict[str, Any],
    *,
    independent_control: bool = True,
) -> EncodedSession:
    ids: list[int] = []
    labels: list[int] = []
    weights: list[float] = []

    def append(token_id: int, *, target: bool = False, weight: float = 0.0) -> None:
        ids.append(token_id)
        labels.append(token_id if target else IGNORE_INDEX)
        weights.append(weight if target else 0.0)

    def append_many(token_id_list: Iterable[int], *, target: bool = False, weight: float = 0.0) -> None:
        for token_id in token_id_list:
            append(int(token_id), target=target, weight=weight)

    append_many(tokenizer.encode(session.instruction + "\n", add_special_tokens=False))
    previous_control: Control | None = None
    for tick in session.ticks:
        append(token_ids["<tick>"])
        append(token_ids["<input>"])
        append_many(tick.input_token_ids)
        append(token_ids["</input>"])

        if independent_control:
            control_token = CONTROL_TOKEN[tick.control]
            if tick.control == Control.LISTEN:
                key = "repeated_listen" if previous_control == Control.LISTEN else "first_or_non_repeated_listen"
            else:
                key = tick.control.value
            append(token_ids[control_token], target=True, weight=float(loss_config[key]))

        if not independent_control or tick.control in {Control.SPEAK, Control.CONTINUE}:
            append(token_ids["<output>"])
            append_many(tick.output_token_ids, target=True, weight=float(loss_config["output_text"]))
            append(token_ids["</output>"], target=True, weight=float(loss_config["output_text"]))

        if tick.out_end or (not independent_control and tick.control == Control.STOP):
            append(token_ids["<out_end>"], target=True, weight=float(loss_config["out_end"]))
        append(token_ids["<tick_end>"], target=True, weight=float(loss_config["tick_end"]))
        previous_control = tick.control

    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(f"{session.session_id} has no training targets")
    return EncodedSession(ids, labels, weights, session.session_id)


class OmniFlowDataset(Dataset[EncodedSession]):
    def __init__(
        self,
        sessions: list[Session],
        tokenizer: PreTrainedTokenizerBase,
        token_ids: dict[str, int],
        loss_config: dict[str, Any],
        *,
        independent_control: bool,
        max_length: int,
    ) -> None:
        self.examples = [
            encode_session(
                session,
                tokenizer,
                token_ids,
                loss_config,
                independent_control=independent_control,
            )
            for session in sessions
        ]
        too_long = [example.session_id for example in self.examples if len(example.input_ids) > max_length]
        if too_long:
            raise ValueError(f"{len(too_long)} sessions exceed max_length={max_length}; first={too_long[0]}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedSession:
        return self.examples[index]


class WeightedCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[EncodedSession]) -> dict[str, torch.Tensor]:
        max_length = max(len(example.input_ids) for example in examples)
        batch_size = len(examples)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
        loss_weights = torch.zeros((batch_size, max_length), dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        for row, example in enumerate(examples):
            length = len(example.input_ids)
            input_ids[row, :length] = torch.tensor(example.input_ids)
            labels[row, :length] = torch.tensor(example.labels)
            loss_weights[row, :length] = torch.tensor(example.loss_weights)
            attention_mask[row, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "attention_mask": attention_mask,
        }
