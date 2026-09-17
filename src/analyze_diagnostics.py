#!/usr/bin/env python3
"""Compute template-level scores and separate label, evidence, and schema errors."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scriptbreak_eval import confusion_and_scores, read_jsonl


def prediction_map(rows):
    return {row.get("scene_id"): row for row in rows}


def subset_predictions(predictions, scene_ids):
    return [row for row in predictions if row.get("scene_id") in scene_ids]


def error_decomposition(gold_rows, prediction_rows):
    predictions = prediction_map(prediction_rows)
    schema_errors = label_errors = evidence_errors = 0
    examples = []
    for row in gold_rows:
        pred_row = predictions.get(row["scene_id"])
        expected = {c["candidate_id"]: c for c in row["candidates"]}
        if not pred_row or not isinstance(pred_row.get("candidates"), list):
            schema_errors += 1
            examples.append({"scene_id": row["scene_id"], "error_type": "schema", "text": row["text"]})
            continue
        got = {c.get("candidate_id"): c for c in pred_row["candidates"] if isinstance(c, dict)}
        if set(got) != set(expected):
            schema_errors += 1
        for candidate_id, target in expected.items():
            guess = got.get(candidate_id, {})
            if guess.get("state") != target["state"]:
                label_errors += 1
                if len(examples) < 30:
                    examples.append({"scene_id": row["scene_id"], "candidate_id": candidate_id,
                                     "error_type": "label", "gold": target["state"],
                                     "prediction": guess.get("state"), "text": row["text"]})
            elif guess.get("evidence") != target["evidence"]:
                evidence_errors += 1
                if len(examples) < 30:
                    examples.append({"scene_id": row["scene_id"], "candidate_id": candidate_id,
                                     "error_type": "evidence", "gold": target["evidence"],
                                     "prediction": guess.get("evidence"), "text": row["text"]})
    return {"schema_error_scenes": schema_errors, "label_error_candidates": label_errors,
            "evidence_only_error_candidates": evidence_errors, "examples": examples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--prediction", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    by_template = defaultdict(list)
    for row in gold:
        by_template[row.get("template_id", row.get("phenomenon", "all"))].append(row)
    output = {}
    for name, path in args.prediction:
        predictions = read_jsonl(path)
        system = {"overall": confusion_and_scores(gold, predictions), "by_template": {},
                  "errors": error_decomposition(gold, predictions)}
        for template_id, rows in sorted(by_template.items()):
            scene_ids = {row["scene_id"] for row in rows}
            metrics = confusion_and_scores(rows, subset_predictions(predictions, scene_ids))
            system["by_template"][template_id] = {
                key: metrics[key] for key in ("n_candidates", "accuracy", "macro_f1",
                                              "evidence_exact_match", "schema_valid_scene_rate",
                                              "family_exact_match")
            }
        output[name] = system
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"accuracy": result["overall"]["accuracy"],
                             "macro_f1": result["overall"]["macro_f1"]}
                      for name, result in output.items()}, indent=2))


if __name__ == "__main__":
    main()
