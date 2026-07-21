#!/usr/bin/env python3
"""Create and validate the tokenizer used by the text Omni-Flow experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    model_path = config["model"]["base_model"]
    token_strings = list(config["sequence"]["special_tokens"].values())

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    old_size = len(tokenizer)
    added = tokenizer.add_special_tokens({"additional_special_tokens": token_strings})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in token_strings}
    if len(set(token_ids.values())) != len(token_strings):
        raise ValueError("special tokens do not have unique token IDs")
    for token, token_id in token_ids.items():
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(f"{token} is not encoded as one token: {encoded}")
        if tokenizer.decode([token_id], skip_special_tokens=False) != token:
            raise ValueError(f"{token} does not round-trip")

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output)
    manifest = {
        "base_model": model_path,
        "old_tokenizer_size": old_size,
        "new_tokenizer_size": len(tokenizer),
        "added_tokens": added,
        "pad_token_id": tokenizer.pad_token_id,
        "special_token_ids": token_ids,
    }
    (args.output / "protocol_tokens.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
