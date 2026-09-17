"""Construction-level and evidence diagnostics from immutable saved outputs."""
import collections
import json
import random
from pathlib import Path
from scriptbreak_eval import read_jsonl, confusion_and_scores, _prediction_index


def positions(text, spans):
    """Union of character offsets over all exact occurrences of quoted spans."""
    found = set()
    for span in spans:
        start = 0
        while span:
            offset = text.find(span, start)
            if offset < 0:
                break
            found.update(range(offset, offset+len(span)))
            start = offset+1
    return found


def evidence_scores(gold, predictions):
    _, evidence, _, _ = _prediction_index(gold, predictions)
    scores = []
    for row in gold:
        for c in row['candidates']:
            target = positions(row['text'], c['evidence'])
            guess = positions(row['text'], evidence.get((row['scene_id'], c['candidate_id']), []))
            overlap = len(target & guess)
            precision = overlap / len(guess) if guess else 0.
            recall = overlap / len(target) if target else 0.
            f1 = 2*overlap/(len(target)+len(guess)) if target or guess else 0.
            scores.append((precision, recall, f1))
    return dict(zip(['character_precision', 'character_recall', 'character_f1'],
                    [sum(row[j] for row in scores)/len(scores) for j in range(3)]))


def paired_evidence_ci(gold, a, b, iterations=10000):
    groups=collections.defaultdict(list)
    strata=collections.defaultdict(list)
    for row in gold:
        groups[row['family_id']].append(row)
    vectors={}
    for fid, rows in groups.items():
        strata[rows[0]['template_id']].append(fid)
        ids={r['scene_id'] for r in rows}
        n=sum(len(r['candidates']) for r in rows)
        pa=[r for r in a if r['scene_id'] in ids]
        pb=[r for r in b if r['scene_id'] in ids]
        delta=evidence_scores(rows,pa)['character_f1']-evidence_scores(rows,pb)['character_f1']
        ma=confusion_and_scores(rows,pa)['evidence_overlap_recall']
        mb=confusion_and_scores(rows,pb)['evidence_overlap_recall']
        vectors[fid]=(delta*n,(ma-mb)*n,n)
    rng=random.Random(20260917)
    sampled=[[],[]]
    for _ in range(iterations):
        ids=[rng.choice(group) for group in strata.values() for _ in group]
        total=sum(vectors[fid][2] for fid in ids)
        for j in (0,1):
            sampled[j].append(sum(vectors[fid][j] for fid in ids)/total)
    out={'iterations':iterations,'seed':20260917,'unit':'paired families stratified by template','conditioning':'fixed predictions and construction inventory'}
    for j,key in enumerate(['character_f1','exact_span_hit_rate']):
        values=sorted(sampled[j])
        out[key]={'difference':sum(v[j] for v in vectors.values())/sum(v[2] for v in vectors.values()),
                  'ci95':[values[int(.025*iterations)],values[int(.975*iterations)]]}
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    corpus = read_jsonl(root/'data/synthetic/corpus.jsonl')
    test = read_jsonl(root/'data/synthetic/test.jsonl')
    conditions = ['direct', 'guideline'] + [f'qlora_s{s}_{p}' for s in [17,29,43] for p in ['direct','guideline']]
    out = {'evidence': {}, 'templates': {}, 'leave_one_template_out': {}}
    for name in conditions:
        pred = read_jsonl(root/f'results/raw/{name}_predictions.jsonl')
        m = confusion_and_scores(test, pred)
        out['evidence'][name] = {**evidence_scores(test,pred), 'exact_list_match':m['evidence_exact_match'],
                                 'exact_span_hit_rate':m['evidence_overlap_recall']}
    holdout = {p:read_jsonl(root/f'results/expansion/raw/qwen3_4b_template_holdout_pooled_{p}_predictions.jsonl') for p in ['direct','guideline']}
    for p, pred in holdout.items():
        out['evidence']['holdout_'+p] = {**evidence_scores(corpus,pred),
            'exact_span_hit_rate':confusion_and_scores(corpus,pred)['evidence_overlap_recall']}
    for template in sorted({r['template_id'] for r in corpus}):
        for mode in ['templates','leave_one_template_out']:
            gold = [r for r in corpus if (r['template_id']==template)==(mode=='templates')]
            ids = {r['scene_id'] for r in gold}
            out[mode][template] = {}
            for p in holdout:
                pred = [r for r in holdout[p] if r['scene_id'] in ids]
                metrics = confusion_and_scores(gold,pred)
                out[mode][template][p] = {k:metrics[k] for k in ['accuracy','family_exact_match','macro_f1','n_candidates','per_label']}
    # Literal classifier compatibility flags are engineering warnings, not adjudicated language errors.
    patterns = ['一把U盘', '一把工作证', '一把硬币', '一把录音笔', '一匹黑狗', '一匹灰兔', '一匹花猫']
    out['corpus_checks'] = {'literal_naturalness_flags':[
        {'scene_id':r['scene_id'],'template_id':r['template_id'],'pattern':p,'text':r['text']}
        for r in corpus for p in patterns if p in r['text']],
        'scope':'Non-exhaustive literal classifier screening, not human validation or an estimated corpus error rate.'}
    seen=collections.defaultdict(list)
    for r in corpus:
        seen[r['text']].append({'scene_id':r['scene_id'],'split':r['split']})
    out['corpus_checks']['duplicate_text_groups'] = [v for v in seen.values() if len(v)>1]
    out['paired_evidence_ci']={
        'base_guideline_minus_direct':paired_evidence_ci(test,read_jsonl(root/'results/raw/guideline_predictions.jsonl'),read_jsonl(root/'results/raw/direct_predictions.jsonl')),
        'holdout_guideline_minus_direct':paired_evidence_ci(corpus,holdout['guideline'],holdout['direct'])}
    # Exclude the entire affected template to preserve complete families.
    independent_test=[r for r in test if r['template_id']!='F09']
    ids={r['scene_id'] for r in independent_test}
    out['standard_without_f09']={}
    for name in conditions:
        pred=[r for r in read_jsonl(root/f'results/raw/{name}_predictions.jsonl') if r['scene_id'] in ids]
        m=confusion_and_scores(independent_test,pred)
        supported=[v['f1'] for v in m['per_label'].values() if v['support']]
        out['standard_without_f09'][name]={'n_scenes':len(independent_test),'n_candidates':m['n_candidates'],
            'accuracy':m['accuracy'],'family_exact_match':m['family_exact_match'],
            'supported_label_macro_f1':sum(supported)/len(supported),'n_supported_labels':len(supported)}
    out['metadata'] = {'reference':'programmatic controlled labels', 'holdout_seed':17,
        'evidence':'Strict scene gate; candidate-macro character offset overlap, union over all exact occurrences; missing outputs score zero. Post-hoc localization diagnostic, not causal faithfulness.',
        'exact_span_hit_rate':'Original evidence_overlap_recall: at least one identical quoted string in reference and prediction; not partial-span overlap.',
        'leave_one_template_out':'Aggregation sensitivity only; no retraining; retains the original five-label macro-F1 definition.'}
    path=root/'results/metrics/review_diagnostics.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
