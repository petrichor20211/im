#!/usr/bin/env python3
"""Validate text Omni-Flow JSONL sessions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from protocol import Session, validate_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    templates_by_split: dict[str, set[str]] = {}
    total = 0
    with args.path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            session = Session.from_dict(json.loads(line))
            if session.session_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate session_id {session.session_id}")
            seen_ids.add(session.session_id)
            validate_session(
                session,
                max_ticks=128 if session.split == "test_length_ood" else 64,
            )
            if session.rule_family == "periodic":
                start = session.metadata["start_tick"]
                first = start + session.metadata["first_delay"]
                period = session.metadata["period"]
                expected_triggers = list(range(first, len(session.ticks), period))
                actual_triggers = [i for i, tick in enumerate(session.ticks) if tick.control.value == "speak"]
                if actual_triggers != expected_triggers or session.metadata["trigger_ticks"] != expected_triggers:
                    raise ValueError(f"line {line_number}: inconsistent periodic triggers")
            if session.split == "test_timing_ood":
                if session.rule_family != "periodic" or session.metadata.get("period") not in {7, 9}:
                    raise ValueError(f"line {line_number}: invalid timing OOD sample")
            if session.split == "test_length_ood" and len(session.ticks) <= 64:
                raise ValueError(f"line {line_number}: length OOD sample is not longer than training maximum")
            templates_by_split.setdefault(session.split, set()).add(session.template_id)
            counts[session.rule_family] += 1
            total += 1
    leaked = templates_by_split.get("train", set()) & templates_by_split.get("test_template_ood", set())
    if leaked:
        raise ValueError(f"template IDs leaked into test_template_ood: {sorted(leaked)}")
    print(f"validated {total} sessions")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
