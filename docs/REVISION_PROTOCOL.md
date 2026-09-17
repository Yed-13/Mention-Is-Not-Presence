# Bounded revision experiments

Prepared 2026-09-17 before inspecting the new GPU matrix outputs. This is a
post-review protocol, not retrospective preregistration of the original study.

## Questions and fixed comparisons

1. How do frozen direct/guideline models perform on exactly the records used
   for pooled template holdout? Run both on all 400 original records.
2. Does the within-adapter guideline effect recur across training seeds?
   Train all five original folds at seeds 17, 29, and 43 on the new platform.
   Retain all 15 runs. Do not select checkpoints or prompts using test scores.
3. How sensitive are the same models to computational repairs? Evaluate them
   on the original and `repair_candidate_2` versions of each held-out record.
   This is **test-repair sensitivity**, not training on a corrected corpus.

## Environment and execution

Use the separate RTX 5090 Windows environment, with environment/model revision
recorded by `scripts/prepare_revision_gpu.py`. Original 4090 D results remain
unchanged. The recorded library versions are not a guarantee of identical
training internals across environments; use the new seed-17 run as a bridge,
and analyze the new three-seed matrix internally before comparing with legacy
results. Base-model weights are downloaded from the public Qwen repository.

The original training entry point and explicit hyperparameters are retained:
QLoRA NF4 with double quantization, BF16, rank 32, alpha 64, dropout .05,
learning rate .0002, three epochs, batch 2, accumulation 8, length 1024.
The revision inference runner uses batches of eight, left padding, greedy
decoding, thinking disabled, and maximum 512 new tokens. Batching and platform
differences are recorded; new results must not be represented as the original
single-record inference runs. Raw generations are appended after every batch.

`scripts/run_revision.py` executes four frozen conditions and 60 adapted
inference conditions (15 adapters × two prompt modes × two text versions).
Missing or malformed outputs remain errors. It stops on subprocess failure.
Training completion markers are written only after the training process exits
successfully. Partial adapter directories are not silently reused as complete.

## Analysis plan

Report paired direct/guideline differences separately for each seed, template,
and text version. Pool disjoint folds within each seed for five-state macro-F1;
report the mean, sample SD, and range across seeds. Also report the ten
construction-level effects descriptively. Do not treat overlapping folds as
independent training replications. Retain conditional family bootstrap intervals
only under their stated fixed-inventory interpretation.

Primary metrics remain strict state macro-F1 and family exact match. Secondary
metrics include candidate accuracy, schema validity, exact-span hit, and character
localization F1. Candidate-state-only scoring checks sensitivity to the strict
evidence gate. Report all differences, including reversals or null effects.

## Repair provenance and scope

`src/prepare_repaired_corpus.py` creates a separate version from the original
records. It corrects classifier agreement and F05 surname/title substitution,
uses less restrictive animal actions, and replaces the repeated F09 sentence
with 20 distinct uncertainty formulations. IDs carry the `R2_` prefix and link
back to the original records. Candidate IDs and state labels are preserved;
evidence spans change where necessary to remain verbatim substrings.

There are 157 changed scene texts and 400 distinct texts in this candidate
version. The change log records every affected scene. Exact-text isolation,
reference-substring validity, and state preservation are automatically checked.
These checks do **not** establish independent linguistic validity or turn the
new references into human annotations. The repair remains AI-assisted and
programmatic. No original score is a score on the repaired texts.

## Reproduction

The revision runner's model-loading and generation implementation was assisted
by the Transformers procedure in Scientific Agent Skills. This is software-workflow
provenance, not evidence of experimental validity or retrospective preregistration:
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner
(2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents](https://doi.org/10.48550/arXiv.2609.00065).

```bash
PYTHONPATH=src python3 src/holdout_baselines.py
PYTHONPATH=src python3 src/prepare_repaired_corpus.py
python3 scripts/prepare_revision_gpu.py
python3 scripts/run_revision.py
```

The GPU commands require the documented isolated environment. A completed
matrix is identified by `results/revision/gpu/COMPLETE.json`; absence of that
file means completion must be checked from individual stage results, not assumed.
