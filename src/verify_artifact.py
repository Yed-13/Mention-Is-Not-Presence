#!/usr/bin/env python3
"""Verify release structure, data invariants, and reported values."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scriptbreak_eval import read_jsonl, validate

ROOT = Path(__file__).resolve().parents[1]


def require_close(actual, expected, label, tolerance=1e-12):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def verify_controlled_data():
    expected = {"train": (240, 120, 660), "dev": (80, 40, 220), "test": (80, 40, 220)}
    family_splits = defaultdict(set)
    for split, (n_scenes, n_families, n_candidates) in expected.items():
        rows = read_jsonl(ROOT / f"data/synthetic/{split}.jsonl")
        validate(rows, require_reference=True)
        assert len(rows) == n_scenes
        assert len({row["family_id"] for row in rows}) == n_families
        assert sum(len(row["candidates"]) for row in rows) == n_candidates
        for row in rows:
            assert row["split"] == split
            family_splits[row["family_id"]].add(split)
    assert all(len(splits) == 1 for splits in family_splits.values())


def verify_external_data():
    rows = read_jsonl(ROOT / "data/external/vistory_visibility.jsonl")
    assert len(rows) == 77
    assert len({row["source_story_id"] for row in rows}) == 48
    counts = Counter()
    for row in rows:
        assert len(row["candidates"]) == 2
        assert row["source_license"] == "MIT"
        positive = next(candidate for candidate in row["candidates"] if candidate["visibility"] == "visible")
        negative = next(candidate for candidate in row["candidates"] if candidate["visibility"] == "not_visible")
        assert positive["mention"] in row["text"].split("画面描述：", 1)[1]
        assert negative["mention"] not in row["text"].split("画面描述：", 1)[1]
        assert positive["mention"] not in negative["mention"]
        assert negative["mention"] not in positive["mention"]
        counts.update(candidate["visibility"] for candidate in row["candidates"])
    assert counts == Counter({"visible": 77, "not_visible": 77})


def load_metric(name):
    return json.loads((ROOT / "results/metrics" / name).read_text(encoding="utf-8"))


def verify_reported_metrics():
    direct = load_metric("direct_metrics.json")
    guideline = load_metric("guideline_metrics.json")
    controlled = load_metric("qlora_three_seed_summary.json")
    external = load_metric("vistory_three_seed_summary.json")
    comparison = load_metric("guideline_vs_direct.json")
    external_comparison = load_metric("vistory_base_guideline_vs_direct.json")
    majority = load_metric("majority_metrics.json")
    lexical = load_metric("char_ngram_nb_metrics.json")

    require_close(direct["accuracy"], .45, "controlled base direct accuracy")
    require_close(guideline["accuracy"], 150 / 220, "controlled base guideline accuracy")
    require_close(controlled["direct"]["mean"]["accuracy"], 1.0, "controlled QLoRA direct mean")
    require_close(controlled["guideline"]["mean"]["accuracy"], .9484848484848485, "controlled QLoRA guideline mean")
    require_close(comparison["difference_a_minus_b"], .2318181818181818, "controlled guideline gain")
    require_close(external["base"]["guideline"]["accuracy"], .9805194805194806, "external base guideline accuracy")
    require_close(external["qlora"]["guideline"]["mean"]["accuracy"], .8225108225108225, "external QLoRA guideline mean")
    require_close(external_comparison["difference_a_minus_b"], .37662337662337664, "external guideline gain")
    require_close(majority["accuracy"], 172 / 220, "all-visible accuracy")
    require_close(majority["macro_f1"], .17551020408163268, "all-visible macro-F1")
    require_close(lexical["accuracy"], 199 / 220, "character n-gram accuracy")
    require_close(lexical["macro_f1"], .8372701280684837, "character n-gram macro-F1")


def verify_expansion_design():
    manifest = json.loads((ROOT / "data/expansion/experiment_manifest.json").read_text(encoding="utf-8"))
    expected = {"25": (60, 30), "50": (120, 60), "75": (180, 90), "100": (240, 120)}
    for level, (scenes, families) in expected.items():
        item = manifest["learning_curves"][level]
        assert item["scenes"] == scenes and item["families"] == families
        assert len(item["templates"]) == 10
    heldout = []
    for fold in sorted(manifest["template_folds"]):
        item = manifest["template_folds"][fold]
        assert (item["train_scenes"], item["dev_scenes"], item["test_scenes"]) == (192, 64, 80)
        assert len(item["heldout_templates"]) == 2
        heldout.extend(item["heldout_templates"])
    assert sorted(heldout) == [f"F{index:02d}" for index in range(1, 11)]


def verify_expansion_results():
    path = ROOT / "results/expansion/summary.json"
    if not path.exists():
        raise AssertionError("results/expansion/summary.json is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["metric_files"] == 38
    rows = {row["system"]: row for row in summary["rows"]}
    require_close(rows["qwen3_8b_base_guideline_controlled"]["macro_f1"],
                  .9406020299007419, "8B base guideline controlled macro-F1")
    require_close(rows["qwen3_8b_base_guideline_external"]["macro_f1"],
                  .8395104895104895, "8B base guideline external macro-F1")
    require_close(rows["qwen3_8b_qlora_direct_controlled"]["macro_f1"],
                  1.0, "8B QLoRA direct controlled macro-F1")
    require_close(rows["qwen3_8b_qlora_direct_external"]["macro_f1"],
                  .8740789186832353, "8B QLoRA direct external macro-F1")
    pooled = summary["template_holdout_pooled"]
    require_close(pooled["direct"]["macro_f1"], .7994557527383471,
                  "template holdout direct pooled macro-F1")
    require_close(pooled["guideline"]["macro_f1"], .9497435897435897,
                  "template holdout guideline pooled macro-F1")
    assert pooled["direct"]["n_candidates"] == 1100
    assert pooled["guideline"]["n_candidates"] == 1100


def main():
    verify_controlled_data()
    verify_external_data()
    verify_reported_metrics()
    verify_expansion_design()
    verify_expansion_results()
    print("artifact verification passed: data, metrics, and split isolation")


if __name__ == "__main__":
    main()
