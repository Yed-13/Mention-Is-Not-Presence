"""Fit the original lexical controls separately within each held-out fold."""
import json
from pathlib import Path
from lexical_baselines import predict_rows
from scriptbreak_eval import read_jsonl, write_jsonl, confusion_and_scores


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/revision'
    out.mkdir(parents=True, exist_ok=True)
    summary = {'protocol': 'Original frozen folds; no tuning; original baseline implementation',
               'features': 'character 1–4 grams, entity-type prefix, entity-marked scene',
               'folds': {}, 'pooled': {}}
    all_gold = []
    pooled = {method: [] for method in ['majority', 'char_ngram_nb']}
    for fold in range(1, 6):
        prefix = root / f'data/expansion/template_fold_{fold}'
        train = read_jsonl(str(prefix) + '_train.jsonl')
        test = read_jsonl(str(prefix) + '_test.jsonl')
        assert not {r['template_id'] for r in train} & {r['template_id'] for r in test}
        assert not {r['text'] for r in train} & {r['text'] for r in test}
        all_gold.extend(test)
        summary['folds'][str(fold)] = {}
        for method in pooled:
            predictions = predict_rows(train, test, method)
            metrics = confusion_and_scores(test, predictions)
            summary['folds'][str(fold)][method] = metrics
            pooled[method].extend(predictions)
            write_jsonl(out/f'fold_{fold}_{method}_predictions.jsonl', predictions)
    assert len({r['scene_id'] for r in all_gold}) == len(all_gold) == 400
    for method, predictions in pooled.items():
        summary['pooled'][method] = confusion_and_scores(all_gold, predictions)
        write_jsonl(out/f'pooled_{method}_predictions.jsonl', predictions)
    (out/'holdout_baselines.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: {m: v[m] for m in ['accuracy','macro_f1','family_exact_match']}
                      for k,v in summary['pooled'].items()}, indent=2))


if __name__ == '__main__':
    main()
