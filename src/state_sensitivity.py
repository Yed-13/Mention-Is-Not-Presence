"""Recompute strict and evidence-independent candidate-state diagnostics."""
import argparse
import collections
import json
from pathlib import Path

from scriptbreak_eval import STATES, confusion_and_scores, read_jsonl


def candidate_state_scores(reference, predictions):
    gold = {(r['scene_id'], c['candidate_id']): c['state']
            for r in reference for c in r['candidates']}
    scenes = collections.defaultdict(list)
    for row in predictions:
        if isinstance(row, dict):
            scenes[row.get('scene_id')].append(row)
    predicted = {}
    for sid, rows in scenes.items():
        if len(rows) != 1 or not isinstance(rows[0].get('candidates'), list):
            continue
        candidates = collections.defaultdict(list)
        for c in rows[0]['candidates']:
            if isinstance(c, dict) and isinstance(c.get('candidate_id'), str):
                candidates[c['candidate_id']].append(c)
        for cid, items in candidates.items():
            state = items[0].get('state')
            if len(items) == 1 and isinstance(state, str) and state in STATES and (sid, cid) in gold:
                predicted[sid, cid] = state
    per_label = {}
    for label in sorted(STATES):
        tp = sum(v == label and predicted.get(k) == label for k, v in gold.items())
        fp = sum(v != label and predicted.get(k) == label for k, v in gold.items())
        fn = sum(v == label and predicted.get(k) != label for k, v in gold.items())
        per_label[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {'accuracy': sum(predicted.get(k) == v for k, v in gold.items()) / len(gold),
            'macro_f1': sum(per_label.values()) / len(per_label),
            'n_candidates': len(gold), 'per_label_f1': per_label}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--out', default='results/metrics/state_sensitivity.json')
    args = parser.parse_args()
    root = Path(args.root)
    conditions = [(name, 'data/synthetic/test.jsonl', f'results/raw/{name}_predictions.jsonl')
                  for name in ['direct', 'guideline', 'qlora_s17_direct', 'qlora_s17_guideline']]
    conditions += [(f'holdout_{name}', 'data/synthetic/corpus.jsonl',
                    f'results/expansion/raw/qwen3_4b_template_holdout_pooled_{name}_predictions.jsonl')
                   for name in ['direct', 'guideline']]
    result = {'description': 'Post-hoc scoring sensitivity using fixed predictions; original strict metrics are retained.',
              'conditions': {}}
    for name, gold_path, pred_path in conditions:
        gold, pred = read_jsonl(root / gold_path), read_jsonl(root / pred_path)
        strict = confusion_and_scores(gold, pred)
        result['conditions'][name] = {'reference': gold_path, 'predictions': pred_path,
                                     'strict': {k: strict[k] for k in ['accuracy', 'macro_f1', 'n_candidates']},
                                     'candidate_state_only': candidate_state_scores(gold, pred)}
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
