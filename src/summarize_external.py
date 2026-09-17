#!/usr/bin/env python3
"""Aggregate external metrics across the three frozen QLoRA seeds."""
import argparse
import json
import math
from pathlib import Path


METRICS = ["accuracy", "macro_f1", "schema_valid_scene_rate", "family_exact_match"]


def mean(values):
    return sum(values) / len(values)


def sample_sd(values):
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.results)
    summary = {
        "reference_tier": "vistorybench_derived_mit",
        "rule": load(root / "vistory_rule_metrics.json"),
        "base": {
            condition: load(root / f"vistory_base_{condition}_metrics.json")
            for condition in ("direct", "guideline")
        },
        "qlora": {},
    }
    for condition in ("direct", "guideline"):
        rows = [load(root / f"vistory_s{seed}_{condition}_metrics.json") for seed in (17, 29, 43)]
        summary["qlora"][condition] = {
            "seeds": [{key: row[key] for key in METRICS} for row in rows],
            "mean": {key: mean([row[key] for row in rows]) for key in METRICS},
            "sd": {key: sample_sd([row[key] for row in rows]) for key in METRICS},
        }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
