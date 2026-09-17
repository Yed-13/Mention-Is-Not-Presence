"""Create a versioned computational repair; never overwrite evaluated v1 data.

References remain AI-assisted/programmatic, not independent human annotations.
No original result is a result on this new version.
"""
import copy
import json
from pathlib import Path
from generate_synthetic_corpus import ITEM_A, ITEM_B, CLOTH, ANIMALS, POSTER_ANIMALS, replace_all
from prepare_expansion_experiments import assistant_record
from prompts import build
from scriptbreak_eval import read_jsonl, write_jsonl, validate

ITEM_A_UNITS = ['把','张','枚','支','张','枚','枚','块','副','张','个','封','张','枚','块','枚','个','把','张','个']
ITEM_B_UNITS = ['把','支','个','封','把','卷','把','本','把','卷','支','把','个','张','台','只','个','个','枚','节']
CLOTH_UNITS = ['条','顶','件','副','条','条','件','条','件','条','顶','条','副','件','条','条','条','件','条','件']

UNCERTAINTY = [
 '门后的轮廓像一个人，但镜头没有交代那里究竟有没有人。',
 '帘子后似乎藏着一个人，画面始终没有确认这个猜测。',
 '窗边的暗影可能是一个人，也可能只是挂起的衣服，镜头没有给出答案。',
 '远处的模糊轮廓疑似一个人，画面清晰度不足以确定其是否真实存在。',
 '柜子旁仿佛有一个人，镜头移开前没有证实那里是否有人。',
 '雾中好像站着一个人，画面没有说明那是人还是树影。',
 '走廊尽头似有一个人，但这个轮廓始终没有得到确认。',
 '床边的黑影让人怀疑有一个人，镜头没有排除它只是衣架的可能。',
 '镜头掠过楼梯口，那里像有一个人，是否有人仍未交代。',
 '墙角隐约出现像一个人的轮廓，画面无法确定它是不是人。',
 '屏风后可能有一个人，但场景在确认之前结束。',
 '阳台上仿佛有一个人，镜头没有澄清那是人影还是晾着的衣物。',
 '门缝里露出疑似一个人的轮廓，后续画面没有确认其真实存在。',
 '光线昏暗，柱子旁像站着一个人，但无法看清那里是否有人。',
 '镜头对准一团模糊的暗影，它可能是一个人，场景没有给出确定答案。',
 '角落里似乎蜷着一个人，也可能是一堆布料，画面未作分辨。',
 '画面中有一道疑似一个人的轮廓，但没有证据确定它不是投影或杂物。',
 '阴影里仿佛坐着一个人，场景结束时仍未确认那里是否有人。',
 '一道像一个人的黑影停在窗帘旁，镜头没有交代它的真实来源。',
 '镜头未能对焦，前方可能有一个人，也可能是其他物体，答案没有揭晓。',
]


def animal_unit(name):
    if name in {'白马','黑马','斑马'}:
        return '匹'
    if name in {'奶牛','水牛'}:
        return '头'
    if name == '骆驼':
        return '峰'
    return '只'


def repair(rows):
    out, changes = [], []
    for original in rows:
        r = copy.deepcopy(original)
        i, t = r['variant_index'], r['template_id']
        reasons = []
        mapping = {}
        if t == 'F01': mapping['一把'+ITEM_A[i]] = '一'+ITEM_A_UNITS[i]+ITEM_A[i]
        if t == 'F03': mapping['一把'+ITEM_B[i]] = '一'+ITEM_B_UNITS[i]+ITEM_B[i]
        if t == 'F04': mapping['有条'+CLOTH[i]] = '有'+CLOTH_UNITS[i]+CLOTH[i]
        if t == 'F05' and r['scene_id'].endswith('_1'):
            name = next(c['mention'] for c in r['candidates'] if c['candidate_id']=='xuning')
            mapping['许老师'] = name[0]+'老师'
        if t == 'F07': mapping['一匹'+ANIMALS[i]] = '一'+animal_unit(ANIMALS[i])+ANIMALS[i]
        r = replace_all(r, mapping)
        if r != original: reasons.append('classifier_or_title_agreement')
        if t == 'F07' and r['scene_id'].endswith('_1'):
            person = next(c for c in r['candidates'] if c['candidate_id']=='linqing')
            phrase = '一'+animal_unit(ANIMALS[i])+ANIMALS[i]+'出现在围栏旁'
            r['text'] = '外景 草地 日。'+person['mention']+'站在围栏旁。'+phrase+'。'
            for c in r['candidates']:
                c['evidence'] = [phrase if c['candidate_id']=='horse' else person['mention']+'站在围栏旁']
            reasons.append('animal_action_compatibility')
        if t == 'F10' and r['scene_id'].endswith('_1'):
            person = next(c for c in r['candidates'] if c['candidate_id']=='chenhe')
            phrase = '一'+animal_unit(POSTER_ANIMALS[i])+POSTER_ANIMALS[i]+'出现在长椅旁'
            r['text'] = '内景 走廊 日。'+phrase+'。'+person['mention']+'站在长椅的另一侧。'
            for c in r['candidates']:
                c['evidence'] = [person['mention']+'站在长椅的另一侧' if c['candidate_id']=='chenhe' else phrase]
            reasons.append('animal_size_and_action_compatibility')
        if t == 'F09' and r['scene_id'].endswith('_1'):
            r['text'] = '内景 储物间 日。'+UNCERTAINTY[i]
            r['candidates'][0]['evidence'] = [UNCERTAINTY[i].rstrip('。')]
            reasons.append('unique_uncertainty_wording')
        r['source_scene_id'] = original['scene_id']
        r['scene_id'] = 'R2_'+original['scene_id']
        r['family_id'] = 'R2_'+original['family_id']
        r['data_version'] = 'repair_candidate_2'
        r['validation_status'] = 'computational_checks_only'
        out.append(r)
        if reasons: changes.append({'source_scene_id':original['scene_id'],'scene_id':r['scene_id'],'reasons':reasons,
                                    'before':original['text'],'after':r['text']})
    validate(out, require_reference=True)
    assert len({r['text'] for r in out}) == len(out)
    for a in ['train','dev','test']:
        for b in ['train','dev','test']:
            if a != b:
                assert not {r['text'] for r in out if r['split']==a} & {r['text'] for r in out if r['split']==b}
    return out, changes


def main():
    root = Path(__file__).resolve().parents[1]
    target = root/'data/repair_candidate_2'
    rows, changes = repair(read_jsonl(root/'data/synthetic/corpus.jsonl'))
    write_jsonl(target/'corpus.jsonl', rows)
    write_jsonl(target/'changes.jsonl', changes)
    for split in ['train','dev','test']:
        selected = [r for r in rows if r['split']==split]
        write_jsonl(target/f'{split}.jsonl', selected)
        write_jsonl(target/f'{split}_sft.jsonl', [assistant_record(r) for r in selected])
    for mode in ['direct','guideline']:
        write_jsonl(target/f'all_{mode}_prompts.jsonl', build(rows,mode))
    print(json.dumps({'records':len(rows),'changed_text_records':len(changes),'unique_texts':len({r['text'] for r in rows}),
                      'status':'repair candidate; not independently human-validated; no inherited model scores'}))


if __name__ == '__main__':
    main()
