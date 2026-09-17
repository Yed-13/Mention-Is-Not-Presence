#!/usr/bin/env python3
"""Prepare stratified learning curves, template holdouts, and prompt variants."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from prompts import build
from scriptbreak_eval import read_jsonl, write_jsonl


def assistant_record(row):
    answer = {
        "scene_id": row["scene_id"],
        "candidates": [{"candidate_id": c["candidate_id"], "state": c["state"], "evidence": c["evidence"]}
                       for c in row["candidates"]],
    }
    prompt = build([row], "direct")[0]
    prompt["messages"].append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    return prompt


def balanced_family_subset(rows, fraction, seed):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["template_id"]][row["family_id"]].append(row)
    selected = []
    for template_id in sorted(grouped):
        family_ids = sorted(grouped[template_id])
        random.Random(f"{seed}:{template_id}").shuffle(family_ids)
        count = max(1, round(len(family_ids) * fraction))
        for family_id in family_ids[:count]:
            selected.extend(grouped[template_id][family_id])
    return sorted(selected, key=lambda row: row["scene_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = root / "data" / "expansion"
    out.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(root / "data/synthetic/train.jsonl")
    dev = read_jsonl(root / "data/synthetic/dev.jsonl")
    test = read_jsonl(root / "data/synthetic/test.jsonl")
    corpus = read_jsonl(root / "data/synthetic/corpus.jsonl")
    external = read_jsonl(root / "data/external/vistory_visibility.jsonl")

    manifest = {"seed": args.seed, "learning_curves": {}, "template_folds": {}}
    for percent in (25, 50, 75, 100):
        subset = balanced_family_subset(train, percent / 100, args.seed)
        write_jsonl(out / f"train_{percent}.jsonl", subset)
        write_jsonl(out / f"train_{percent}_sft.jsonl", [assistant_record(row) for row in subset])
        manifest["learning_curves"][str(percent)] = {
            "scenes": len(subset),
            "families": len({row["family_id"] for row in subset}),
            "templates": sorted({row["template_id"] for row in subset}),
        }

    template_ids = sorted({row["template_id"] for row in corpus})
    for fold, start in enumerate(range(0, len(template_ids), 2), 1):
        heldout = set(template_ids[start:start + 2])
        fold_train = [row for row in train if row["template_id"] not in heldout]
        fold_dev = [row for row in dev if row["template_id"] not in heldout]
        fold_test = [row for row in corpus if row["template_id"] in heldout]
        prefix = out / f"template_fold_{fold}"
        write_jsonl(f"{prefix}_train.jsonl", fold_train)
        write_jsonl(f"{prefix}_train_sft.jsonl", [assistant_record(row) for row in fold_train])
        write_jsonl(f"{prefix}_dev.jsonl", fold_dev)
        write_jsonl(f"{prefix}_dev_sft.jsonl", [assistant_record(row) for row in fold_dev])
        write_jsonl(f"{prefix}_test.jsonl", fold_test)
        for condition in ("direct", "guideline"):
            write_jsonl(f"{prefix}_{condition}_prompts.jsonl", build(fold_test, condition))
        manifest["template_folds"][str(fold)] = {
            "heldout_templates": sorted(heldout),
            "train_scenes": len(fold_train), "dev_scenes": len(fold_dev), "test_scenes": len(fold_test),
        }

    for condition in ("guideline_concise", "guideline_reordered"):
        write_jsonl(out / f"synthetic_test_{condition}_prompts.jsonl", build(test, condition))
        write_jsonl(out / f"external_{condition}_prompts.jsonl", build(external, condition))

    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
