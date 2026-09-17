#!/usr/bin/env python3
"""Score legal five-state outputs on the binary visibility distinction supported by ViStoryBench.

The derived reference does not adjudicate rationale boundaries. Candidate coverage and
state strings determine output validity; generated evidence remains in raw artifacts.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

STATES = {"visible", "audible_only", "depicted", "referenced_only", "uncertain"}


def read_jsonl(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def index_predictions(gold, predictions):
    expected = {(r["scene_id"], c["candidate_id"]) for r in gold for c in r["candidates"]}
    index = {}
    valid_scenes = set()
    errors = []
    seen = set()
    by_scene = {r["scene_id"]: {c["candidate_id"] for c in r["candidates"]} for r in gold}
    for row in predictions:
        sid = row.get("scene_id")
        if sid not in by_scene or sid in seen or not isinstance(row.get("candidates"), list):
            errors.append({"scene_id": sid, "error": "invalid_or_duplicate_scene"})
            continue
        seen.add(sid)
        local = {}
        ok = True
        for candidate in row["candidates"]:
            cid = candidate.get("candidate_id") if isinstance(candidate, dict) else None
            state = candidate.get("state") if isinstance(candidate, dict) else None
            if (sid, cid) not in expected or cid in local or state not in STATES:
                ok = False
                errors.append({"scene_id": sid, "candidate_id": cid, "error": "invalid_candidate_or_state"})
                continue
            local[cid] = "visible" if state == "visible" else "not_visible"
        if set(local) != by_scene[sid]:
            ok = False
            errors.append({"scene_id": sid, "error": "candidate_set_mismatch"})
        if ok:
            valid_scenes.add(sid)
            index.update({(sid, cid): label for cid, label in local.items()})
    return index, valid_scenes, errors


def metrics(gold, predictions):
    pred, valid_scenes, errors = index_predictions(gold, predictions)
    state_counts = collections.Counter(
        candidate.get("state")
        for row in predictions if isinstance(row.get("candidates"), list)
        for candidate in row["candidates"] if isinstance(candidate, dict)
    )
    labels = ["visible", "not_visible"]
    matrix = {g: {p: 0 for p in labels + ["missing"]} for g in labels}
    families = collections.defaultdict(list)
    for row in gold:
        for candidate in row["candidates"]:
            key = (row["scene_id"], candidate["candidate_id"])
            guess = pred.get(key, "missing")
            target = candidate["visibility"]
            matrix[target][guess] += 1
            families[row["family_id"]].append(guess == target)
    per_label = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in labels if g != label)
        fn = sum(matrix[label][p] for p in labels + ["missing"] if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in labels)
    output_valid_rate = len(valid_scenes) / len(gold)
    return {
        "task": "external_binary_shot_visibility",
        "reference_tier": "vistorybench_derived_mit",
        "n_scenes": len(gold), "n_candidates": total, "n_families": len(families),
        "accuracy": correct / total,
        "macro_f1": sum(v["f1"] for v in per_label.values()) / 2,
        "output_valid_scene_rate": output_valid_rate,
        "schema_valid_scene_rate": output_valid_rate,
        "family_exact_match": sum(all(x) for x in families.values()) / len(families),
        "per_label": per_label, "confusion": matrix,
        "raw_predicted_state_counts": dict(sorted(state_counts.items(), key=lambda item: str(item[0]))),
        "validation_errors": errors,
    }


def bootstrap(gold, a, b, iterations, seed):
    ia, _, _ = index_predictions(gold, a)
    ib, _, _ = index_predictions(gold, b)
    families = collections.defaultdict(list)
    for row in gold:
        for candidate in row["candidates"]:
            key = (row["scene_id"], candidate["candidate_id"])
            families[row["family_id"]].append((key, candidate["visibility"]))
    ids = sorted(families)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        sampled = [rng.choice(ids) for _ in ids]
        ca = cb = total = 0
        for fid in sampled:
            for key, target in families[fid]:
                ca += ia.get(key) == target
                cb += ib.get(key) == target
                total += 1
        diffs.append((ca - cb) / total)
    diffs.sort()
    return {"metric": "binary_visibility_accuracy", "difference_a_minus_b": metrics(gold, a)["accuracy"] - metrics(gold, b)["accuracy"],
            "cluster_bootstrap_95_ci": [diffs[int(.025 * iterations)], diffs[min(iterations - 1, int(.975 * iterations))]],
            "iterations": iterations, "seed": seed, "unit": "source_story", "reference_tier": "vistorybench_derived_mit"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    score = sub.add_parser("evaluate")
    score.add_argument("--gold", required=True); score.add_argument("--predictions", required=True); score.add_argument("--out", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--gold", required=True); compare.add_argument("--predictions-a", required=True); compare.add_argument("--predictions-b", required=True); compare.add_argument("--out", required=True)
    compare.add_argument("--iterations", type=int, default=10000); compare.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()
    gold = read_jsonl(args.gold)
    if args.cmd == "evaluate":
        result = metrics(gold, read_jsonl(args.predictions))
    else:
        result = bootstrap(gold, read_jsonl(args.predictions_a), read_jsonl(args.predictions_b), args.iterations, args.seed)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
