# Result artifacts

`metrics/` contains the values reported in the manuscript, including per-label scores, confusion matrices, output-validity rates, controlled exact-evidence scores, three-seed summaries, and paired cluster-bootstrap intervals. Controlled schema validity includes exact-substring evidence. Cross-source output validity checks complete candidate coverage and legal five-state strings; the derived binary reference does not independently score rationale boundaries.

`raw/` contains immutable decoded text and the parsed prediction files used by the scorers. Malformed or missing candidate records remain visible in these artifacts and are scored as missing predictions.

The manuscript's two primary summaries are:

- `metrics/qlora_three_seed_summary.json` for the controlled five-state experiment;
- `metrics/vistory_three_seed_summary.json` for cross-source binary visibility.

The robustness expansion is stored under `expansion/`. Its `summary.json` and `summary.csv` aggregate 38 prompt-rewrite, learning-curve, template-holdout, and 8B metric files. The same directory retains parsed predictions and raw generation logs. The five template folds are additionally pooled in `summary.json`; missing or malformed outputs remain scored as failures.

Run `make verify` from the repository root to check the reported headline values against these files.
