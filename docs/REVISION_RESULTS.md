# Completed three-seed replication

All 15 training markers, 64 condition metrics, and the completion marker were
retrieved on 2026-09-17. Each frozen condition covers 400 scenes; each adapted
condition covers 80 held-out scenes. In total, 6,400 raw generations are retained.
Run `python3 src/summarize_revision.py` to recompute the complete analysis.

## Main results

Scores below are strict five-state macro-F1 pooled over all five disjoint folds
within each seed. All runs are retained; training uses the original corpus.

| Test version | Seed | Direct | Guideline | Guideline minus direct |
|---|---:|---:|---:|---:|
| Original | 17 | .835006 | .952705 | +.117699 |
| Original | 29 | .829580 | .710594 | -.118987 |
| Original | 43 | .969218 | .711895 | -.257323 |
| Repaired | 17 | .834086 | .914713 | +.080627 |
| Repaired | 29 | .821795 | .838381 | +.016587 |
| Repaired | 43 | .936697 | .819877 | -.116820 |

Original direct/guideline means (sample SD): .877935 (.079100) and .791731
(.139409). Repaired means: .864193 (.063091) and .857657 (.050271).
Frozen original direct/guideline scores: .486623/.658304; repaired:
.546105/.606968. Every adapter exceeds its prompt-matched frozen control.
The fold-trained lexical baseline reaches .397908 on original held-out records.

## Interpretation and sensitivity

The guideline advantage is not stable across training seeds. Candidate-state-only
scoring preserves the original-text sign pattern (+, -, -), so it is not solely
an artifact of the strict evidence gate. Guideline uncertain-state F1 is 1.0 at
seed 17 and 0.0 at seeds 29 and 43, whereas direct runs score 1.0 at all seeds.
On repaired texts, guideline uncertain-state F1 at seeds 29 and 43 is .709677.
This is sensitivity to the bundled text revision, not an isolated causal effect
of any single repair and not independent label validation.

Original-platform seed-17 macro-F1 was .799456/.949744. New-platform seed 17 is
.835006/.952705. Platform and inference batching changed; the new three-seed
comparison is internally matched and does not silently replace original outputs.

## Timing

Summing one elapsed duration per generation batch gives 5,362.286 seconds.
The 15 reported trainer runtimes sum to 2,133.1 seconds. Their combined timed
execution is 7,495.386 seconds (2.082 hours) on one GPU. This includes execution
overhead within those timers, excludes setup/model loading between stages, and
is not a measured GPU-utilization or energy total. Generation batch time appears
on every row of a batch; divide by its recorded batch size before summing.

## Remaining scope limits

The repaired version has 400 unique texts and 157 changed scenes. Labels and
evidence remain computational references. Original-data training is unchanged.
Three seeds and ten templates cannot establish population-wide robustness,
professional-scene validity, or independent expert agreement. The cross-source
binary task measures consistency with published fields, not adjudicated visibility.
