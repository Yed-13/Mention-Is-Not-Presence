# ScriptBreak-ZH

ScriptBreak-ZH is an evidence-grounded scene-realization benchmark for Chinese screenplays. Given a scene and supplied entity candidates, a model predicts one of five states—`visible`, `audible_only`, `depicted`, `referenced_only`, or `uncertain`—and returns exact textual evidence.

The repository contains:

- a 400-record controlled contrast corpus grouped into 200 family-disjoint pairs (381 distinct texts; the repeated F09 uncertainty sentence crosses the standard splits);
- a 77-shot cross-source visibility set derived from ViStoryBench;
- frozen direct and guideline prompts;
- Qwen3-4B inference and QLoRA training entry points;
- majority and candidate-conditioned character n-gram baselines;
- two semantic prompt rewrites, stratified learning curves, five complete-template holdout folds, and Qwen3-8B replication;
- raw generations, parsed predictions, metrics, and paired cluster-bootstrap results.

## Headline result

The controlled and cross-source evaluations measure different capabilities. The all-visible baseline already reaches 78.2% controlled accuracy but only .176 macro-F1, and a character n-gram baseline reaches .837 macro-F1. Three QLoRA runs reach 1.000 macro-F1 with the training-matched direct prompt on the standard controlled split. Under complete-template holdout, the same seed-17 direct condition falls to .799 pooled macro-F1, while full guidelines recover .950. On the cross-source set, the frozen 4B base model with explicit guidelines reaches .981. Scaling that frozen condition to 8B raises controlled macro-F1 to .941 but lowers cross-source macro-F1 to .840; direct-prompt 8B QLoRA reaches 1.000 and .874, respectively. The results distinguish interface learning, structural transfer, and source transfer.

## Repository layout

```text
data/synthetic/        controlled corpus, splits, prompts, and SFT records
data/external/         ViStoryBench-derived visibility set and prompts
data/expansion/        learning-curve subsets, template folds, prompt rewrites
src/                   training, inference, scoring, bootstrap, and verification
tests/                 unit tests for prompts and scorers
results/metrics/       reported metrics and confidence intervals
results/raw/           immutable raw generations and parsed predictions
docs/                  data card and reproducibility guide
```

## Quick verification

The local verification path uses the Python standard library and does not require a GPU:

```bash
make test
make verify
```

`make verify` checks data counts, split isolation, external-set constraints, and key reported values.

The original expansion used two RTX 4090 D GPUs. The separate revision adds 15 RTX 5090 training runs and 64 inference conditions, with matched frozen controls, three seeds, and paired original/repaired tests. See [revision results](docs/REVISION_RESULTS.md), [protocol](docs/REVISION_PROTOCOL.md), and [compute log](docs/COMPUTE_LOG.md). Recompute the complete replication with `python3 src/summarize_revision.py`; the results include prompt-effect reversals, not only favorable comparisons.

The expansion design can be regenerated and audited without a GPU:

```bash
make expansion-data
make baselines
make diagnostics
```

## GPU reproduction

Install the CUDA requirements in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
```

Train the reported QLoRA configuration:

```bash
python src/train_lora.py \
  --model Qwen/Qwen3-4B \
  --train data/synthetic/train_sft.jsonl \
  --dev data/synthetic/dev_sft.jsonl \
  --output checkpoints/seed-17 \
  --seed 17 --qlora
```

Run deterministic inference:

```bash
python src/run_inference.py \
  --model Qwen/Qwen3-4B \
  --adapter checkpoints/seed-17 \
  --prompts data/synthetic/test_direct_prompts.jsonl \
  --predictions results/reproduced/seed-17-direct-predictions.jsonl \
  --raw results/reproduced/seed-17-direct-raw.jsonl
```

The experiment uses greedy decoding with thinking mode disabled and reports seeds 17, 29, and 43. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete protocol.

## Data provenance

Controlled records are generated from disclosed contrast templates and marked `programmatic_by_construction`. The external set is derived from MIT-licensed ViStoryBench fields and retains source-story, source-shot, and license metadata. The two resources have different label spaces and are scored separately. See [docs/DATA_CARD.md](docs/DATA_CARD.md).
