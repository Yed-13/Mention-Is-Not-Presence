# Construction and reference audit

Audit date: 2026-09-17. This is a computational provenance and consistency audit,
not independent human annotation.

## Recovered construction chain

`data/construction/seed_scenes.jsonl` contains the original 20 AI-assisted seed
scenes. Its annotation-source fields describe their actual provenance.
`src/generate_synthetic_corpus.py` is the recovered deterministic substitution
program. It expands ten two-scene templates into 20 variants per template.
Labels and evidence strings are inherited through the same substitutions.
The available files do not identify a verifiable generator model/version or
establish separation from the evaluated Qwen family.

Run from the release root:

```bash
PYTHONPATH=src python3 src/generate_synthetic_corpus.py \
  --silver data/construction/seed_scenes.jsonl \
  --out /tmp/scriptbreak-reconstructed.jsonl
```

Reconstruction was executed during this audit. All 400 records match the
released corpus in text, IDs, candidates (including labels and evidence),
template assignment, and split. The only differing key is `annotation_note`,
a descriptive metadata field omitted from the release. This verifies the
construction chain, not the semantic correctness of the labels.

## Exact-text overlap

There are 381 distinct text strings among 400 scene records. One string occurs
20 times, in `SYN_F09_00_1` through `SYN_F09_19_1`: 12 train, four development,
four test. It is the unresolved-person sentence and contains no replaceable
entity name. Hence all uncertainty examples repeat one sentence.

Original family IDs remain disjoint. Standard five-state perfection must not
be described as performance on unseen uncertainty text. In the F09/F10
template-held-out fold, all F09 records are absent from training. Repeated
held-out F09 records still represent one construction, not 20 distinct
linguistic formulations.

Excluding F09 from standard test scoring leaves 72 scenes, 36 families, and
208 candidates in four states. Direct adapters at seeds 17, 29, and 43 remain
perfect. This is a post-hoc scoring sensitivity: training was not repeated.

## Lexical warnings

The generator performs string substitution without classifier agreement
repair. A non-exhaustive seven-pattern screen flags 14 records, including
`一匹黑狗` and `一把U盘`. Some flagged phrases admit alternative readings
(for example, a handful of coins), so these are review flags, not 14 confirmed
errors or a corpus-wide naturalness estimate. Full records and patterns are
in `results/metrics/review_diagnostics.json`.

## External references

The 154 external candidate labels are derived from ViStoryBench fields. The
reported score measures consistency with that derivation. A missing character
in a shot field is not independently adjudicated evidence of physical absence.
The field oracle verifies implementation, not reference validity.

## Outstanding validation

No independent native-speaker naturalness or label audit has been completed.
No new training-seed replication was completed in this revision. Corrected
language, additional templates, or new labels require versioned data and new
model runs; original scores must not be attached to altered inputs.
