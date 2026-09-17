#!/usr/bin/env python3
"""Validation, scoring, and family-clustered inference for ScriptBreak-ZH."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

STATES = {"visible", "audible_only", "depicted", "referenced_only", "uncertain"}
TYPES = {"person", "animal", "object"}


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def validate(rows, require_reference=False):
    if not rows:
        raise ValueError("dataset is empty")
    seen = set()
    for row in rows:
        sid = row.get("scene_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("scene_id must be nonempty and unique")
        seen.add(sid)
        for key in ("family_id", "text", "phenomenon"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{sid}: nonempty {key} required")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{sid}: nonempty candidates required")
        candidate_ids = set()
        for candidate in candidates:
            cid = candidate.get("candidate_id")
            if not isinstance(cid, str) or not cid or cid in candidate_ids:
                raise ValueError(f"{sid}: candidate_id must be unique within scene")
            candidate_ids.add(cid)
            if candidate.get("entity_type") not in TYPES:
                raise ValueError(f"{sid}/{cid}: invalid entity_type")
            mention = candidate.get("mention")
            if not isinstance(mention, str) or not mention or mention not in row["text"]:
                raise ValueError(f"{sid}/{cid}: mention must be an exact text substring")
            state = candidate.get("state")
            evidence = candidate.get("evidence")
            if require_reference and state not in STATES:
                raise ValueError(f"{sid}/{cid}: realization state required")
            if state is not None and state not in STATES:
                raise ValueError(f"{sid}/{cid}: invalid realization state")
            if state is not None and (
                not isinstance(evidence, list)
                or not evidence
                or any(not isinstance(span, str) or not span or span not in row["text"] for span in evidence)
            ):
                raise ValueError(f"{sid}/{cid}: evidence must contain exact nonempty substrings")
        if require_reference and (
            row.get("status") != "approved_synthetic"
            or row.get("annotation_source") != "programmatic_by_construction"
        ):
            raise ValueError(f"{sid}: controlled reference provenance mismatch")
    return True


def split_families(rows, seed, train, dev, test):
    validate(rows)
    ratios = (train, dev, test)
    if any(value < 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("train/dev/test ratios must be nonnegative and sum to one")
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        grouped[row["phenomenon"]][row["family_id"]].append(row)
    rng = random.Random(seed)
    assignment = {}
    for families in grouped.values():
        ids = sorted(families)
        rng.shuffle(ids)
        n_train = round(len(ids) * train)
        n_dev = min(round(len(ids) * dev), len(ids) - n_train)
        for index, family_id in enumerate(ids):
            assignment[family_id] = "train" if index < n_train else "dev" if index < n_train + n_dev else "test"
    output = []
    for row in rows:
        copy = dict(row)
        copy["split"] = assignment[row["family_id"]]
        output.append(copy)
    memberships = collections.defaultdict(set)
    for row in output:
        memberships[row["family_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in memberships.values()):
        raise AssertionError("family leakage across splits")
    return output


def _prediction_index(reference, predictions):
    scenes = {row["scene_id"]: row for row in reference}
    expected = {(row["scene_id"], candidate["candidate_id"]) for row in reference for candidate in row["candidates"]}
    states = {}
    evidence = {}
    valid_scenes = set()
    seen_scenes = set()
    errors = []
    for row in predictions:
        sid = row.get("scene_id")
        if sid not in scenes or sid in seen_scenes or not isinstance(row.get("candidates"), list):
            errors.append({"scene_id": sid, "error": "invalid_or_duplicate_scene"})
            continue
        seen_scenes.add(sid)
        local_ids = set()
        local_states = {}
        local_evidence = {}
        valid = True
        for candidate in row["candidates"]:
            cid = candidate.get("candidate_id") if isinstance(candidate, dict) else None
            key = (sid, cid)
            state = candidate.get("state") if isinstance(candidate, dict) else None
            spans = candidate.get("evidence") if isinstance(candidate, dict) else None
            if key not in expected or cid in local_ids or state not in STATES:
                errors.append({"scene_id": sid, "candidate_id": cid, "error": "invalid_candidate_or_state"})
                valid = False
                continue
            local_ids.add(cid)
            if not isinstance(spans, list) or not spans or any(
                not isinstance(span, str) or not span or span not in scenes[sid]["text"] for span in spans
            ):
                errors.append({"scene_id": sid, "candidate_id": cid, "error": "invalid_evidence"})
                valid = False
                continue
            local_states[key] = state
            local_evidence[key] = spans
        required = {candidate["candidate_id"] for candidate in scenes[sid]["candidates"]}
        if local_ids != required:
            errors.append({"scene_id": sid, "error": "candidate_set_mismatch"})
            valid = False
        if valid:
            valid_scenes.add(sid)
            states.update(local_states)
            evidence.update(local_evidence)
    return states, evidence, valid_scenes, errors


def confusion_and_scores(reference, predictions):
    validate(reference, require_reference=True)
    gold = {(row["scene_id"], candidate["candidate_id"]): candidate for row in reference for candidate in row["candidates"]}
    pred, pred_evidence, valid_scenes, errors = _prediction_index(reference, predictions)
    labels = sorted(STATES)
    matrix = {gold_label: {pred_label: 0 for pred_label in labels + ["missing"]} for gold_label in labels}
    correct = evidence_exact = evidence_overlap = 0
    for key, candidate in gold.items():
        guess = pred.get(key, "missing")
        target = candidate["state"]
        matrix[target][guess] += 1
        correct += guess == target
        predicted_spans = set(pred_evidence.get(key, []))
        target_spans = set(candidate["evidence"])
        evidence_exact += predicted_spans == target_spans
        evidence_overlap += bool(predicted_spans & target_spans)
    per_label = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][guess] for guess in labels + ["missing"] if guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(matrix[label].values()),
        }
    families = collections.defaultdict(list)
    for row in reference:
        for candidate in row["candidates"]:
            key = (row["scene_id"], candidate["candidate_id"])
            families[row["family_id"]].append(pred.get(key) == candidate["state"])
    total = len(gold)
    return {
        "task": "closed_set_scene_realization_state",
        "reference_tier": "programmatic_synthetic_reference",
        "n_candidates": total,
        "accuracy": correct / total,
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
        "schema_valid_scene_rate": len(valid_scenes) / len(reference),
        "malformed_or_missing_scene_rate": 1.0 - len(valid_scenes) / len(reference),
        "evidence_exact_match": evidence_exact / total,
        "evidence_overlap_recall": evidence_overlap / total,
        "per_label": per_label,
        "confusion": matrix,
        "family_exact_match": sum(all(items) for items in families.values()) / len(families),
        "validation_errors": errors,
    }


def paired_family_bootstrap(reference, predictions_a, predictions_b, iterations=10000, seed=20260916):
    validate(reference, require_reference=True)
    a, _, _, _ = _prediction_index(reference, predictions_a)
    b, _, _, _ = _prediction_index(reference, predictions_b)
    families = collections.defaultdict(list)
    for row in reference:
        for candidate in row["candidates"]:
            key = (row["scene_id"], candidate["candidate_id"])
            families[row["family_id"]].append((key, candidate["state"]))
    ids = sorted(families)
    if len(ids) < 2:
        raise ValueError("at least two families are required")
    rng = random.Random(seed)
    differences = []
    for _ in range(iterations):
        correct_a = correct_b = total = 0
        for family_id in (rng.choice(ids) for _ in ids):
            for key, target in families[family_id]:
                correct_a += a.get(key) == target
                correct_b += b.get(key) == target
                total += 1
        differences.append((correct_a - correct_b) / total)
    differences.sort()
    return {
        "metric": "state_accuracy",
        "difference_a_minus_b": confusion_and_scores(reference, predictions_a)["accuracy"] - confusion_and_scores(reference, predictions_b)["accuracy"],
        "cluster_bootstrap_95_ci": [differences[int(.025 * iterations)], differences[min(iterations - 1, int(.975 * iterations))]],
        "iterations": iterations,
        "seed": seed,
        "unit": "contrast_family",
        "reference_tier": "programmatic_synthetic_reference",
    }


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data", required=True)
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--data", required=True)
    split_parser.add_argument("--out", required=True)
    split_parser.add_argument("--seed", type=int, default=20260916)
    split_parser.add_argument("--train", type=float, default=.6)
    split_parser.add_argument("--dev", type=float, default=.2)
    split_parser.add_argument("--test", type=float, default=.2)
    score_parser = subparsers.add_parser("evaluate")
    score_parser.add_argument("--gold", required=True)
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--out", required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--gold", required=True)
    compare_parser.add_argument("--predictions-a", required=True)
    compare_parser.add_argument("--predictions-b", required=True)
    compare_parser.add_argument("--out", required=True)
    compare_parser.add_argument("--iterations", type=int, default=10000)
    compare_parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    if args.command == "validate":
        rows = read_jsonl(args.data)
        validate(rows, require_reference=True)
        result = {"valid": True, "scenes": len(rows), "sha256": file_sha256(args.data)}
    elif args.command == "split":
        rows = split_families(read_jsonl(args.data), args.seed, args.train, args.dev, args.test)
        write_jsonl(args.out, rows)
        result = dict(collections.Counter(row["split"] for row in rows))
    elif args.command == "evaluate":
        result = confusion_and_scores(read_jsonl(args.gold), read_jsonl(args.predictions))
    else:
        result = paired_family_bootstrap(
            read_jsonl(args.gold), read_jsonl(args.predictions_a), read_jsonl(args.predictions_b), args.iterations, args.seed
        )
    if hasattr(args, "out"):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
