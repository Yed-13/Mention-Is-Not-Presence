# Reproducibility guide

## Frozen design

- Model: Qwen3-4B.
- Conditions: base/direct, base/guideline, QLoRA/direct, QLoRA/guideline.
- Decoding: greedy, thinking disabled, maximum 512 new tokens.
- QLoRA: NF4 4-bit base weights, BF16 compute, rank 32, alpha 64, dropout 0.05, learning rate 2e-4, three epochs, batch size 2, gradient accumulation 8, maximum length 1,024.
- Seeds: 17, 29, and 43.
- Statistics: 10,000 paired cluster-bootstrap iterations; contrast family is the controlled unit and source story is the external unit.
- Robustness extensions: two semantically equivalent guideline rewrites, 25/50/75/100% template-stratified learning curves, five complete-template holdout folds, and one Qwen3-8B seed replication.

The recorded runs used two NVIDIA GeForce RTX 4090 D cards with 24 GB memory. Independent inference jobs were parallelized across cards; each adapter training run used one card.

## CPU-only integrity checks

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

## Compile the paper

```bash
make paper
```

The repository bundles the ACM `acmart` class and reference style from the supplied ACM primary template. The resulting file is `paper/main.pdf`.
