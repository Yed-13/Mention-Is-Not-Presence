# Data card

## Controlled contrast corpus

The controlled resource contains 400 short Chinese screenplay scenes in 200 two-scene contrast families. Ten template types vary realization-relevant cues such as current versus past, visible versus offscreen, live versus depicted, and uncertain versus explicitly absent. Entity substitutions create 20 variants per template.

| Split | Scenes | Families | Candidates |
|---|---:|---:|---:|
| Train | 240 | 120 | 660 |
| Development | 80 | 40 | 220 |
| Test | 80 | 40 | 220 |

All members of a contrast family remain in one split. Each candidate contains an entity type, one of five realization states, and exact evidence spans. Records use `status=approved_synthetic` and `annotation_source=programmatic_by_construction`.

Family isolation does not imply text isolation. The F09 uncertainty sentence
is repeated 20 times across standard splits, leaving 381 distinct texts.
`CONSTRUCTION_AUDIT.md` documents the duplication, non-exhaustive lexical
warnings, recovered seed scenes, and deterministic generator. These references
have not undergone independent human validation.

## Generalization partitions

`data/expansion/` adds two predeclared views of the same controlled corpus:

- stratified 25%, 50%, 75%, and 100% training subsets containing 30, 60, 90, and 120 complete families, with equal proportional sampling inside every template;
- five template-held-out folds, each excluding two of the ten complete construction types from training and development and testing on all 80 scenes from those types.

Across the five folds, every template is held out exactly once. These files add no new reference labels; they change only which controlled mechanisms are observable during adaptation.

## Cross-source visibility set

The external resource contains 77 shots from 48 ViStoryBench stories and 154 candidates, balanced between `visible` and `not_visible`. It is deterministically derived from the Chinese plot, setting, static-shot-description, and appearing-character fields. Each record retains dataset, story, shot, and MIT-license metadata.

The external resource evaluates binary consistency with source fields, not independently adjudicated visibility. Five-state system outputs are collapsed to `visible` versus all remaining legal states during scoring. Output validity checks complete candidate coverage and legal state strings. Because the derivation does not supply independently adjudicated rationale boundaries, generated evidence is retained for qualitative audit but is not scored on this resource. External results are not pooled with the controlled five-state results.

Source: [ViStoryBench dataset](https://huggingface.co/datasets/ViStoryBench/ViStoryBench) and [paper](https://arxiv.org/abs/2505.24862).

## Intended use

The resources support research on Chinese narrative information extraction, structured prompting, evidence grounding, controlled contrast evaluation, construction holdout, and source-shift analysis. They are designed for candidate-conditioned classification rather than exhaustive production inventories.

## Fields

Controlled scenes use `scene_id`, `family_id`, `split`, `phenomenon`, `text`, and `candidates`. Candidate records contain `candidate_id`, `mention`, `entity_type`, `state`, and `evidence`.

External scenes additionally retain `source_dataset`, `source_story_id`, `source_shot_id`, and `source_license`; their candidates use `visibility` instead of the five-state `state` field.

## Integrity

`src/verify_artifact.py` checks split isolation, exact counts, candidate balance, and the key reported metrics.
