#!/usr/bin/env python3
"""Compute control, content, protocol, and joint metrics from tick predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CONTROLS = ("listen", "speak", "continue", "stop")


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confusion: Counter[tuple[str, str]] = Counter()
    total_ticks = exact_outputs = exact_out_end = 0
    false_trigger_numerator = false_trigger_denominator = 0
    malformed = overflow = 0
    session_count = joint_sessions = 0
    exact_controls = 0

    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result = json.loads(line)
            session_count += 1
            session_joint = True
            for tick in result["predictions"]:
                expected = tick["expected_control"]
                predicted = tick["predicted_control"]
                confusion[(expected, predicted)] += 1
                total_ticks += 1
                control_ok = expected == predicted
                output_ok = tick["expected_output_token_ids"] == tick["predicted_output_token_ids"]
                end_ok = tick["expected_out_end"] == tick["predicted_out_end"]
                exact_controls += int(control_ok)
                exact_outputs += int(output_ok)
                exact_out_end += int(end_ok)
                if expected == "listen":
                    false_trigger_denominator += 1
                    false_trigger_numerator += int(predicted == "speak")
                malformed += int("malformed_tick" in tick["violations"] or "missing_output_end" in tick["violations"])
                overflow += int("chunk_overflow" in tick["violations"])
                session_joint &= control_ok and output_ok and end_ok and not tick["violations"]
            joint_sessions += int(session_joint)

    per_control = {}
    f1_values = []
    for control in CONTROLS:
        tp = confusion[(control, control)]
        fp = sum(confusion[(other, control)] for other in CONTROLS if other != control)
        fn = sum(confusion[(control, other)] for other in CONTROLS if other != control)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_control[control] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1_values.append(f1)

    metrics = {
        "sessions": session_count,
        "ticks": total_ticks,
        "control_accuracy": safe_div(exact_controls, total_ticks),
        "control_macro_f1": sum(f1_values) / len(f1_values),
        "per_control": per_control,
        "false_trigger_rate": safe_div(false_trigger_numerator, false_trigger_denominator),
        "tick_output_exact_match": safe_div(exact_outputs, total_ticks),
        "out_end_accuracy": safe_div(exact_out_end, total_ticks),
        "chunk_overflow_rate": safe_div(overflow, total_ticks),
        "malformed_tick_rate": safe_div(malformed, total_ticks),
        "joint_session_success_rate": safe_div(joint_sessions, session_count),
    }
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
