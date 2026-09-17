# Claim–evidence map

This document links each central manuscript claim to the exact artifact that supports it. It is an audit aid, not an additional source of results. Controlled and cross-source scores are kept separate because their labels and reference construction differ.

| Manuscript claim | Evidence file(s) | Verification logic |
|---|---|---|
| The controlled resource contains 400 scenes in 200 two-scene families, with family-isolated 240/80/80 splits. | `data/synthetic/{train,dev,test}.jsonl`; `src/verify_artifact.py` | `make verify` checks scene, family, and candidate counts and ensures that no family crosses a split. |
| The controlled test set is skewed toward `visible`; accuracy alone is misleading. | `data/synthetic/test.jsonl`; `results/metrics/majority_metrics.json` | The all-visible baseline obtains 172/220 accuracy, .176 macro-F1, and zero family exact match. |
| A surface-form baseline is strong on the standard split. | `src/lexical_baselines.py`; `results/raw/char_ngram_nb_predictions.jsonl`; `results/metrics/char_ngram_nb_metrics.json` | Candidate-conditioned character 1–3 gram Multinomial NB reaches .837 macro-F1. |
| Full guidelines improve the frozen 4B base model on the controlled test. | `results/metrics/direct_metrics.json`; `results/metrics/guideline_metrics.json`; `results/metrics/guideline_vs_direct.json` | The paired family bootstrap compares predictions on identical test families; accuracy rises by .232 with a 95% interval stored in the comparison JSON. |
| Direct-prompt QLoRA fits the standard controlled distribution exactly in three runs. | `results/metrics/qlora_s{17,29,43}_direct_metrics.json`; `results/metrics/qlora_three_seed_summary.json` | Each run has 1.000 state, family, evidence, and schema metrics. |
| The cross-source ranking differs: frozen base + guidelines outperforms guideline-conditioned QLoRA. | `results/metrics/vistory_base_guideline_metrics.json`; `results/metrics/vistory_three_seed_summary.json` | The base condition reaches .981 macro-F1; the three-seed QLoRA guideline mean is .827. |
| Semantically equivalent prompt rewrites are not behaviorally equivalent. | `data/expansion/*concise*`; `data/expansion/*reordered*`; `results/expansion/{metrics,raw}`; `results/expansion/summary.json` | Rewrites preserve the ontology but change controlled and cross-source scores; paired bootstrap results compare each rewrite with the full prompt. |
| The known-template controlled split saturates with little training data, while cross-source transfer is non-monotonic. | `data/expansion/train_{25,50,75}_sft.jsonl`; `results/expansion/metrics/qwen3_4b_curve_*`; `results/expansion/summary.json` | Subsets are template-stratified and family-disjoint; the paper reports all four points rather than selecting a best fraction. |
| Complete-template holdout is a stronger structural test than the standard split. | `data/expansion/template_fold_*`; `data/expansion/experiment_manifest.json`; `results/expansion/metrics/qwen3_4b_template_fold_*`; pooled predictions in `results/expansion/raw/` | Every construction F01–F10 is held out exactly once; pooled scores cover 400 unique scenes. |
| Scale effects are assessed without changing the task contract. | `results/expansion/metrics/qwen3_8b_*`; matching raw prediction files and logs | Qwen3-8B uses the same prompts, test records, LoRA rank, optimizer schedule, and seed-17 adaptation protocol. |
| Controlled evidence scores are strict contract scores, not semantic-overlap scores. | `src/scriptbreak_eval.py`; controlled raw prediction files | Evidence receives credit only when the predicted list matches the programmatic reference list exactly and every span occurs verbatim in the scene. Cross-source rationale boundaries are not scored. |
| The cross-source set is deterministic and binary; it is not presented as five-class professional annotation. | `data/external/vistory_visibility.jsonl`; `src/verify_artifact.py`; `docs/DATA_CARD.md` | Verification checks 77 positive and 77 negative candidates, source IDs, license metadata, and field constraints. |

## Regeneration chain

1. `make expansion-data` regenerates prompt rewrites, learning subsets, and template folds.
2. `make baselines` and `make diagnostics` regenerate the non-neural controls and error counts.
3. GPU inference writes immutable raw generations and parsed predictions before metric calculation.
4. `make summarize-expansion` builds the machine-readable expansion summary and pooled template-holdout scores.
5. `make test`, `make verify`, and `make paper` validate code, artifact invariants, headline values, hashes, and the manuscript build.

No claim in the manuscript relies on a model-generated label represented as independent human annotation.
