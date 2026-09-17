"""Paired cluster bootstrap for primary metrics, conditional on fixed predictions."""
import collections
import json
import random
from pathlib import Path
from scriptbreak_eval import _prediction_index, read_jsonl, STATES
from vistory_visibility_eval import index_predictions


def scores(v, n_labels):
    f1 = []
    for i in range(n_labels):
        tp, fp, fn = v[3*i:3*i+3]
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(f1)/n_labels, v[-2]/v[-1]


def paired(gold, a, b, external=False, iterations=10000, seed=20260917):
    labels = ['not_visible', 'visible'] if external else sorted(STATES)
    index = index_predictions if external else _prediction_index
    ia, ib = index(gold, a)[0], index(gold, b)[0]
    groups = collections.defaultdict(list)
    strata = collections.defaultdict(list)
    for row in gold:
        groups[row['family_id']].append(row)
    vectors = {}
    for fid, rows in sorted(groups.items()):
        strata['source' if external else rows[0]['template_id']].append(fid)
        pair = []
        for predictions in (ia, ib):
            v = [0] * (3*len(labels)+2)
            exact = True
            for row in rows:
                for candidate in row['candidates']:
                    target = candidate['visibility' if external else 'state']
                    guess = predictions.get((row['scene_id'], candidate['candidate_id']))
                    exact = exact and target == guess
                    for i, label in enumerate(labels):
                        v[3*i] += target == label and guess == label
                        v[3*i+1] += target != label and guess == label
                        v[3*i+2] += target == label and guess != label
            v[-2], v[-1] = int(exact), 1
            pair.append(v)
        vectors[fid] = pair

    def evaluate(ids):
        totals = [[0]*(3*len(labels)+2) for _ in range(2)]
        for fid in ids:
            for side in (0, 1):
                totals[side] = [x+y for x, y in zip(totals[side], vectors[fid][side])]
        return [scores(v, len(labels)) for v in totals]

    point = evaluate(list(vectors))
    samples = [[], []]
    rng = random.Random(seed)
    for _ in range(iterations):
        ids = [rng.choice(group) for group in strata.values() for _ in group]
        sa, sb = evaluate(ids)
        for j in (0, 1):
            samples[j].append(sa[j]-sb[j])
    result = {'iterations': iterations, 'seed': seed,
              'unit': 'source_story' if external else 'contrast_family',
              'stratification': 'none' if external else 'template_id',
              'interpretation': 'Conditional uncertainty over sampled clusters for fixed models, templates and predictions; not training or source-population uncertainty.',
              'n_clusters': len(vectors)}
    for j, metric in enumerate(['macro_f1', 'family_exact_match']):
        ordered = sorted(samples[j])
        result[metric] = {'a': point[0][j], 'b': point[1][j], 'difference_a_minus_b': point[0][j]-point[1][j],
                          'cluster_bootstrap_95_ci': [ordered[int(.025*iterations)], ordered[min(iterations-1, int(.975*iterations))]]}
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    specs = [
        ('standard_base_guideline_vs_direct', 'data/synthetic/test.jsonl', 'results/raw/guideline_predictions.jsonl', 'results/raw/direct_predictions.jsonl', False),
        ('standard_qlora17_vs_base_direct', 'data/synthetic/test.jsonl', 'results/raw/qlora_s17_direct_predictions.jsonl', 'results/raw/direct_predictions.jsonl', False),
        ('holdout_guideline_vs_direct', 'data/synthetic/corpus.jsonl', 'results/expansion/raw/qwen3_4b_template_holdout_pooled_guideline_predictions.jsonl', 'results/expansion/raw/qwen3_4b_template_holdout_pooled_direct_predictions.jsonl', False),
        ('external_base_guideline_vs_direct', 'data/external/vistory_visibility.jsonl', 'results/raw/vistory_base_guideline_predictions.jsonl', 'results/raw/vistory_base_direct_predictions.jsonl', True)]
    out = {}
    for name, ref, a, b, external in specs:
        out[name] = paired(read_jsonl(root/ref), read_jsonl(root/a), read_jsonl(root/b), external)
        out[name]['files'] = {'reference': ref, 'a': a, 'b': b}
        print(name, json.dumps(out[name]), flush=True)
    (root/'results/metrics/primary_uncertainty.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
