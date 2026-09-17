"""Score the completed revision matrix without silently dropping missing runs."""
import json
import statistics
from pathlib import Path
from scriptbreak_eval import read_jsonl, confusion_and_scores, write_jsonl
from state_sensitivity import candidate_state_scores
from review_diagnostics import evidence_scores


def mean_sd(values):
    return {'mean':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else None,
            'min':min(values),'max':max(values),'values':values}


def summarize(gold,pred):
    strict=confusion_and_scores(gold,pred)
    return {'strict':strict,'candidate_state_only':candidate_state_scores(gold,pred),
            'evidence_characters':evidence_scores(gold,pred)}


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'results/revision/gpu'
    marker=json.loads((out/'COMPLETE.json').read_text(encoding='utf8'))
    assert marker == {'status':'complete','seeds':[17,29,43],'folds':5}
    for seed in [17,29,43]:
        for fold in range(1,6):
            trained=json.loads((out/f'fold_{fold}_s{seed}_trained.json').read_text(encoding='utf8'))
            assert trained['seed']==seed and trained['fold']==fold
    required=[out/f'base_{v}_{p}_metrics.json' for v in ['original','repair'] for p in ['direct','guideline']]
    required += [out/f'fold_{f}_s{s}_{v}_{p}_metrics.json' for s in [17,29,43] for f in range(1,6)
                 for v in ['original','repair'] for p in ['direct','guideline']]
    missing=[str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f'Matrix incomplete: {len(missing)} of {len(required)} metrics missing. First: {missing[0]}')
    result={'protocol':'RTX 5090 batch-8 separate-platform replication; original training corpus; paired original/repaired tests',
            'base':{},'adapted':{},'seed_summary':{},'paired_prompt_differences':{},'template_results':{}}
    for version,path in [('original','data/synthetic/corpus.jsonl'),('repair','data/repair_candidate_2/corpus.jsonl')]:
        gold=read_jsonl(root/path)
        result['base'][version]={}
        result['adapted'][version]={}
        result['template_results'][version]={}
        for prompt in ['direct','guideline']:
            raw=read_jsonl(out/f'base_{version}_{prompt}_raw.jsonl')
            assert len(raw)==len(gold)==400
            assert {r['scene_id'] for r in raw}=={r['scene_id'] for r in gold}
            result['base'][version][prompt]=summarize(gold,read_jsonl(out/f'base_{version}_{prompt}_predictions.jsonl'))
        for seed in [17,29,43]:
            result['adapted'][version][str(seed)]={}
            result['template_results'][version][str(seed)]={}
            for prompt in ['direct','guideline']:
                predictions=[]
                raw_ids=[]
                for fold in range(1,6):
                    raw=read_jsonl(out/f'fold_{fold}_s{seed}_{version}_{prompt}_raw.jsonl')
                    assert len(raw)==80 and len({r['scene_id'] for r in raw})==80
                    fold_gold=read_jsonl(root/f'data/expansion/template_fold_{fold}_test.jsonl')
                    expected={r['scene_id'] for r in fold_gold}
                    if version=='repair':
                        expected={r['scene_id'] for r in gold if r['source_scene_id'] in expected}
                    assert {r['scene_id'] for r in raw}==expected
                    raw_ids.extend(r['scene_id'] for r in raw)
                    predictions.extend(read_jsonl(out/f'fold_{fold}_s{seed}_{version}_{prompt}_predictions.jsonl'))
                assert len(raw_ids)==len(set(raw_ids))==400
                assert set(raw_ids)=={r['scene_id'] for r in gold}
                result['adapted'][version][str(seed)][prompt]=summarize(gold,predictions)
                write_jsonl(out/f'pooled_s{seed}_{version}_{prompt}_predictions.jsonl',predictions)
                for template in sorted({r['template_id'] for r in gold}):
                    subset=[r for r in gold if r['template_id']==template]
                    ids={r['scene_id'] for r in subset}
                    m=confusion_and_scores(subset,[r for r in predictions if r.get('scene_id') in ids])
                    result['template_results'][version][str(seed)].setdefault(template,{})[prompt]=m
        result['seed_summary'][version]={}
        for prompt in ['direct','guideline']:
            result['seed_summary'][version][prompt]={metric:mean_sd([
                result['adapted'][version][str(s)][prompt]['strict'][metric] for s in [17,29,43]])
                for metric in ['macro_f1','accuracy','family_exact_match']}
        result['paired_prompt_differences'][version]={metric:mean_sd([
            result['adapted'][version][str(s)]['guideline']['strict'][metric]-
            result['adapted'][version][str(s)]['direct']['strict'][metric] for s in [17,29,43]])
            for metric in ['macro_f1','accuracy','family_exact_match']}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'base':{v:{p:r['strict']['macro_f1'] for p,r in conditions.items()}
                                   for v,conditions in result['base'].items()},
                      'seed_summary':result['seed_summary'],
                      'prompt_differences':result['paired_prompt_differences']},indent=2))


if __name__=='__main__': main()
