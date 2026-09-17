#!/usr/bin/env python3
"""Aggregate expansion metrics into paper-ready JSON and CSV tables."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from scriptbreak_eval import confusion_and_scores, paired_family_bootstrap, read_jsonl, write_jsonl
from vistory_visibility_eval import bootstrap as external_bootstrap


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def flatten_row(group, name, metrics):
    return {
        "group": group,
        "system": name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "schema_valid_scene_rate": metrics.get("schema_valid_scene_rate"),
        "family_exact_match": metrics.get("family_exact_match"),
        "evidence_exact_match": metrics.get("evidence_exact_match"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/expansion/metrics")
    ap.add_argument("--out-json", default="results/expansion/summary.json")
    ap.add_argument("--out-csv", default="results/expansion/summary.csv")
    ap.add_argument("--raw", default="results/expansion/raw")
    ap.add_argument("--corpus", default="data/synthetic/corpus.jsonl")
    ap.add_argument("--standard-raw", default="results/raw")
    args = ap.parse_args()
    metric_dir = Path(args.metrics)
    paths = sorted(metric_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no metric files in {metric_dir}")
    metrics = {path.stem: load(path) for path in paths}
    rows = []
    for name, item in metrics.items():
        if "template_fold" in name:
            group = "template_holdout"
        elif "curve" in name:
            group = "learning_curve"
        elif "guideline_concise" in name or "guideline_reordered" in name:
            group = "prompt_robustness"
        elif "qwen3_8b" in name:
            group = "model_scale"
        else:
            group = "other"
        rows.append(flatten_row(group, name, item))

    folds = {}
    for prompt in ("direct", "guideline"):
        selected = [
            metrics[f"qwen3_4b_template_fold_{fold}_{prompt}"]
            for fold in range(1, 6)
            if f"qwen3_4b_template_fold_{fold}_{prompt}" in metrics
        ]
        if selected:
            folds[prompt] = {
                key: mean_sd([item[key] for item in selected])
                for key in ("accuracy", "schema_valid_scene_rate", "family_exact_match", "evidence_exact_match")
            }

    pooled = {}
    pooled_coverage = {}
    raw_dir = Path(args.raw)
    corpus_rows = read_jsonl(args.corpus)
    corpus_scene_ids = {row["scene_id"] for row in corpus_rows}
    for prompt in ("direct", "guideline"):
        combined = []
        for fold in range(1, 6):
            pred_path = raw_dir / f"qwen3_4b_template_fold_{fold}_{prompt}_predictions.jsonl"
            if pred_path.exists():
                combined.extend(read_jsonl(pred_path))
        if combined:
            predicted_ids = [row["scene_id"] for row in combined]
            if len(predicted_ids) != len(set(predicted_ids)):
                raise ValueError(f"template holdout {prompt}: duplicate scene predictions")
            unexpected = set(predicted_ids) - corpus_scene_ids
            if unexpected:
                raise ValueError(f"template holdout {prompt}: unexpected scene ids {sorted(unexpected)}")
            pooled_path = raw_dir / f"qwen3_4b_template_holdout_pooled_{prompt}_predictions.jsonl"
            write_jsonl(pooled_path, combined)
            pooled[prompt] = confusion_and_scores(corpus_rows, combined)
            pooled_coverage[prompt] = {
                "expected_scenes": len(corpus_scene_ids),
                "prediction_records": len(predicted_ids),
                "missing_scene_ids": sorted(corpus_scene_ids - set(predicted_ids)),
            }

    comparisons = {"controlled": {}, "external": {}}
    standard_raw = Path(args.standard_raw)
    controlled_gold = read_jsonl("data/synthetic/test.jsonl")
    external_gold = read_jsonl("data/external/vistory_visibility.jsonl")
    for model, standard_name in (("base", "guideline"), ("qlora", "qlora_s17_guideline")):
        standard_controlled = read_jsonl(standard_raw / f"{standard_name}_predictions.jsonl")
        standard_external_name = "vistory_base_guideline" if model == "base" else "vistory_s17_guideline"
        standard_external = read_jsonl(standard_raw / f"{standard_external_name}_predictions.jsonl")
        for rewrite in ("guideline_concise", "guideline_reordered"):
            prefix = f"qwen3_4b_{model}_{rewrite}"
            controlled_path = raw_dir / f"{prefix}_controlled_predictions.jsonl"
            external_path = raw_dir / f"{prefix}_external_predictions.jsonl"
            if controlled_path.exists():
                comparisons["controlled"][f"{model}_{rewrite}_minus_full"] = paired_family_bootstrap(
                    controlled_gold, read_jsonl(controlled_path), standard_controlled
                )
            if external_path.exists():
                comparisons["external"][f"{model}_{rewrite}_minus_full"] = external_bootstrap(
                    external_gold, read_jsonl(external_path), standard_external, 10000, 20260916
                )

    payload = {
        "metric_files": len(metrics),
        "rows": rows,
        "template_holdout_five_fold": folds,
        "template_holdout_pooled": pooled,
        "template_holdout_coverage": pooled_coverage,
        "template_holdout_note": (
            "Per-fold macro-F1 is intentionally not averaged because individual held-out template pairs "
            "omit some ontology labels. Use the pooled 400-scene macro-F1 and fold-level accuracy/family metrics."
        ),
        "prompt_rewrite_vs_full": comparisons,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    out_csv = Path(args.out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metric_files": len(metrics), "rows": len(rows), "out": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
