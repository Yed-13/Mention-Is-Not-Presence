ScriptBreak-ZH experiment supplementary material

Contents: controlled Chinese scene records and splits; ViStoryBench-derived
visibility records with provenance; frozen prompts; model predictions and raw
generations; metrics and run logs; training, inference, scoring and baseline
code; unit tests; and reproduction documentation.

CPU checks: run `make test` and `make verify` from the extracted directory.
These checks use Python's standard library and do not require a GPU.
Reproduce the additional scoring diagnostic with:
  PYTHONPATH=src python3 src/state_sensitivity.py
Reproduce primary-metric paired bootstrap intervals with:
  PYTHONPATH=src python3 src/primary_uncertainty.py

GPU reproduction requires downloaded Qwen3 model weights, an NVIDIA CUDA
environment, and the dependencies in requirements-gpu.txt. See
docs/REPRODUCIBILITY.md for commands and docs/COMPUTE_LOG.md for recorded runs.
The original runs used RTX 4090 D GPUs with 24 GB memory per card.
The revision adds 15 RTX 5090 adapters and 64 inference conditions, including
matched frozen controls and paired original/repaired-text tests. Recompute:
  python3 src/summarize_revision.py
See docs/REVISION_PROTOCOL.md and docs/REVISION_RESULTS.md. The repaired texts
are computationally constructed, not independently human validated.

The controlled and external resources have distinct reference label spaces
and are scored separately. Reproduction starts from the supplied frozen data.
Source and license metadata for external records remain with those records.
