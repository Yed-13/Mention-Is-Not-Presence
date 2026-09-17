# Reproducibility guide

## Frozen design

- Model: Qwen3-4B.
- Conditions: base/direct, base/guideline, QLoRA/direct, QLoRA/guideline.
- Decoding: greedy, thinking disabled, maximum 512 new tokens.
- QLoRA: NF4 4-bit base weights, BF16 compute, rank 32, alpha 64, dropout 0.05, learning rate 2e-4, three epochs, batch size 2, gradient accumulation 8, maximum length 1,024.
- Standard full-data seeds: 17, 29, and 43. Original 4090 D template holdout: seed 17. Separate 5090 replication: all five folds at seeds 17, 29, and 43.
- Statistics: 10,000 paired cluster-bootstrap iterations; contrast family is the controlled unit and source story is the external unit.
- Robustness extensions: two semantically equivalent guideline rewrites, 25/50/75/100% template-stratified learning curves, five complete-template holdout folds, and one Qwen3-8B seed replication.

The original runs used two NVIDIA GeForce RTX 4090 D cards with 24 GB memory. Independent inference jobs were parallelized across cards; each adapter training run used one card. The completed revision matrix used one RTX 5090 with 32 GB and batched inference. Its exact recorded environment is in `results/revision/environment.json`; see `docs/REVISION_PROTOCOL.md` and `docs/REVISION_RESULTS.md` for the separate protocol and outcomes.

Recompute the complete 64-condition revision summary without a GPU:

```bash
python3 src/summarize_revision.py
```

This checks completion markers and exact raw-record coverage before recomputing
pooled scores, candidate-state-only diagnostics, evidence scores, and seed summaries.
Do not merge these outputs with the original platform's single-seed predictions.

## CPU-only integrity checks

The post-review fold-matched lexical controls and separate repair candidate are
described in `docs/REVISION_PROTOCOL.md`. Reproduce the CPU steps with:

```bash
PYTHONPATH=src python3 src/holdout_baselines.py
PYTHONPATH=src python3 src/prepare_repaired_corpus.py
```

The repaired version is computationally checked, not independently human
validated. Original metrics apply only to original data. The new-platform GPU
matrix is complete only when its completion marker and all 64 condition metrics
are present; `src/summarize_revision.py` refuses to summarize an incomplete matrix.

Family-disjoint splitting is not a text-deduplication guarantee. See
`docs/CONSTRUCTION_AUDIT.md` for the repeated F09 sentence, lexical warnings,
recovered construction inputs, and the limits of the current validation.

Reproduce the post-hoc evidence, per-template, exclusion, and corpus diagnostics:

```bash
PYTHONPATH=src python3 src/review_diagnostics.py
```

This writes `results/metrics/review_diagnostics.json`. It preserves original
records and outputs. Character evidence F1 is localization agreement, not
faithfulness. Leave-one-template-out scoring reuses saved predictions rather
than retraining. Standard scoring without F09 uses four supported classes.

From the repository root:

```bash
make test
make verify
```

The tests cover strict candidate matching, malformed-output handling, family-safe splitting, evidence validation, binary state collapse, and prompt-contract consistency.

Generate the frozen extension design and CPU baselines:

```bash
make expansion-data
make baselines
make diagnostics
```

`data/expansion/experiment_manifest.json` records the exact family counts and held-out templates.

## Re-score a controlled prediction file

```bash
PYTHONPATH=src python src/scriptbreak_eval.py evaluate \
  --gold data/synthetic/test.jsonl \
  --predictions results/raw/direct_predictions.jsonl \
  --out results/reproduced/direct_metrics.json
```

## Re-score an external prediction file

```bash
PYTHONPATH=src python src/vistory_visibility_eval.py evaluate \
  --gold data/external/vistory_visibility.jsonl \
  --predictions results/raw/vistory_base_guideline_predictions.jsonl \
  --out results/reproduced/vistory_base_guideline_metrics.json
```

## Reproduce a paired comparison

```bash
PYTHONPATH=src python src/scriptbreak_eval.py compare \
  --gold data/synthetic/test.jsonl \
  --predictions-a results/raw/guideline_predictions.jsonl \
  --predictions-b results/raw/direct_predictions.jsonl \
  --out results/reproduced/guideline_vs_direct.json \
  --iterations 10000 --seed 20260916
```

The external scorer exposes the same `compare` interface and clusters by `family_id`, which maps one-to-one to source story in the released set.

## Run the GPU extension matrix

The bounded orchestration script expects cached Qwen3-4B and Qwen3-8B checkpoints and an isolated Python environment. Its path variables can be overridden without editing the script:

```bash
ROOT=/path/to/repository \
EXISTING=/path/to/base/run \
PYTHON=/path/to/venv/bin/python \
MODEL4=/path/to/Qwen3-4B \
MODEL8=/path/to/Qwen3-8B \
ADAPTER4=/path/to/reported/4B/seed17/adapter \
bash scripts/run_expansion_remote.sh
```

The script is restart-safe at the metric-file level. It writes into `results/expansion/` and creates `COMPLETE` only after all stages finish. Afterward:

```bash
make summarize-expansion
```

This creates a flat CSV, a structured JSON summary, and pooled template-held-out predictions. The pooled metric is the valid five-state macro-F1 across all 400 disjoint held-out predictions; individual folds may not contain every state and are therefore summarized primarily by accuracy and family exact match.

## Reproduce the scoring-sensitivity analysis

Primary-metric paired confidence intervals can be reproduced with:

```bash
PYTHONPATH=src python3 src/primary_uncertainty.py
```

This writes `results/metrics/primary_uncertainty.json`. It uses 10,000 paired
cluster-bootstrap replicates, sampling families within each controlled template
or whole external source stories. Macro-F1 is recomputed from aggregated counts
in every replicate; exact match is computed over complete sampled families or
stories. Intervals condition on fixed fitted models, the available clusters,
and the controlled template inventory. They do not estimate retraining or
new-template uncertainty. All candidates in malformed scenes retain the original
scorer's missing-prediction treatment.

```bash
PYTHONPATH=src python3 src/state_sensitivity.py
```

This writes `results/metrics/state_sensitivity.json` from fixed controlled predictions. Strict scoring requires a fully valid scene, including exact-substring evidence. The candidate-state-only diagnostic independently accepts each uniquely identified legal candidate state, regardless of evidence validity. Missing or ambiguous outputs remain incorrect. Both scores use every reference candidate; this is not a valid-output-only analysis. The original predictions and primary metrics are unchanged.

Reproduction starts from the released frozen corpus and cross-source records. The expansion script reconstructs subsets and folds from those records. The recorded training-library versions are PyTorch 2.8.0, Transformers 5.17.0, and PEFT 0.21.0; `requirements-gpu.txt` gives dependency constraints rather than a full environment lock.
