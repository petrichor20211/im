#!/usr/bin/env python3
"""Aggregate formal C/D metrics across seeds and apply pilot gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

SEEDS = (20260721, 20260722, 20260723)
SPLITS = (
    "test_in_domain",
    "test_template_ood",
    "test_timing_ood",
    "test_length_ood",
    "test_distractor_ood",
)
METRICS = (
    "control_macro_f1",
    "false_trigger_rate",
    "onset_exact_0",
    "onset_exact_pm1",
    "onset_mae_ticks",
    "tick_output_exact_match",
    "reconstructed_token_exact_match",
    "chunk_overflow_rate",
    "malformed_tick_rate",
    "joint_session_success_rate",
)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/runs/formal_eval"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/formal_eval/summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate: dict[str, Any] = {"groups": {}, "D_minus_C": {}}
    raw: dict[tuple[str, str, int], dict[str, Any]] = {}
    for group in ("C", "D"):
        aggregate["groups"][group] = {}
        for split in SPLITS:
            rows = []
            for seed in SEEDS:
                path = args.root / f"seed_{seed}" / group / split / "metrics.json"
                row = json.loads(path.read_text())
                raw[(group, split, seed)] = row
                rows.append(row)
            aggregate["groups"][group][split] = {
                metric: stats([float(row[metric]) for row in rows])
                for metric in METRICS
                if all(row.get(metric) is not None for row in rows)
            }

    for split in SPLITS:
        aggregate["D_minus_C"][split] = {}
        for metric in METRICS:
            differences = []
            for seed in SEEDS:
                d_value = raw[("D", split, seed)].get(metric)
                c_value = raw[("C", split, seed)].get(metric)
                if d_value is not None and c_value is not None:
                    differences.append(float(d_value) - float(c_value))
            if differences:
                aggregate["D_minus_C"][split][metric] = stats(differences)

    base_retention = json.loads((args.root / "baseline_retention.json").read_text())["accuracy"]
    aggregate["retention"] = {"baseline_accuracy": base_retention, "groups": {}}
    for group in ("C", "D"):
        values = [
            json.loads((args.root / f"seed_{seed}" / group / "retention.json").read_text())["accuracy"]
            for seed in SEEDS
        ]
        aggregate["retention"]["groups"][group] = {
            **stats(values),
            "mean_ratio_to_baseline": statistics.mean(values) / base_retention if base_retention else None,
        }

    d_in = aggregate["groups"]["D"]["test_in_domain"]
    d_ood_joint = statistics.mean(
        aggregate["groups"]["D"][split]["joint_session_success_rate"]["mean"]
        for split in SPLITS[1:]
    )
    gates = {
        "control_macro_f1": d_in["control_macro_f1"]["mean"] >= 0.90,
        "onset_exact_pm1": d_in["onset_exact_pm1"]["mean"] >= 0.90,
        "reconstructed_token_exact_match": d_in["reconstructed_token_exact_match"]["mean"] >= 0.90,
        "chunk_overflow_rate": d_in["chunk_overflow_rate"]["mean"] <= 0.01,
        "malformed_tick_rate": d_in["malformed_tick_rate"]["mean"] <= 0.01,
        "D_minus_C_joint_success": aggregate["D_minus_C"]["test_in_domain"]["joint_session_success_rate"]["mean"] >= 0.10,
        "ood_joint_retention": d_ood_joint >= 0.80 * d_in["joint_session_success_rate"]["mean"],
    }
    aggregate["pilot_gates"] = {"items": gates, "all_passed": all(gates.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({"pilot_gates": aggregate["pilot_gates"], "retention": aggregate["retention"]}, indent=2))


if __name__ == "__main__":
    main()
