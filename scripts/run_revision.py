"""Bounded replication matrix, with separate platform and repair-sensitivity results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scriptbreak_eval import read_jsonl, write_jsonl
from prompts import build


def main():
    os.chdir(ROOT)
    out=ROOT/'results/revision/gpu'; out.mkdir(parents=True,exist_ok=True)
    os.environ['PYTHONIOENCODING']='utf-8'
    def execute(name,args):
        print('START',name,flush=True)
        start=time.monotonic()
        with (out/(name+'.log')).open('a',encoding='utf8') as log:
            subprocess.run([sys.executable,*args],check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        print('FINISH',name,round(time.monotonic()-start,1),flush=True)

    def job(name,gold,mode):
        prompt=out/(name+'_prompts.jsonl')
        write_jsonl(prompt,build(read_jsonl(gold),mode))
        return {'name':name,'gold':str(gold),'prompts':str(prompt),
                **{key:str(out/(name+suffix)) for key,suffix in
                   [('raw','_raw.jsonl'),('predictions','_predictions.jsonl'),('metrics','_metrics.json')]}}

    def infer(name,jobs,adapter=None):
        if all(Path(j['metrics']).exists() for j in jobs): return
        manifest=out/(name+'_jobs.json')
        manifest.write_text(json.dumps(jobs,indent=2),encoding='utf8')
        args=['src/revision_inference.py','--model','model','--jobs',str(manifest),'--batch-size','8']
        if adapter: args+=['--adapter',str(adapter)]
        execute(name,args)

    infer('frozen_controls',[job('base_'+version+'_'+mode,ROOT/path,mode)
          for version,path in [('original','data/synthetic/corpus.jsonl'),('repair','data/repair_candidate_2/corpus.jsonl')]
          for mode in ['direct','guideline']])

    for seed in [17,29,43]:
        for fold in range(1,6):
            name=f'fold_{fold}_s{seed}'
            adapter=out/'adapters'/name
            completion=out/(name+'_trained.json')
            if not completion.exists():
                execute(name+'_train',['src/train_lora.py','--model','model',
                    '--train',f'data/expansion/template_fold_{fold}_train_sft.jsonl',
                    '--dev',f'data/expansion/template_fold_{fold}_dev_sft.jsonl',
                    '--output',str(adapter),'--seed',str(seed),'--rank','32','--alpha','64',
                    '--lr','2e-4','--epochs','3','--qlora'])
                completion.write_text(json.dumps({'seed':seed,'fold':fold,'training_data':'original frozen corpus',
                                                 'platform':'separate 5090 replication'}),encoding='utf8')
            original=ROOT/f'data/expansion/template_fold_{fold}_test.jsonl'
            heldout={r['template_id'] for r in read_jsonl(original)}
            repaired=out/f'fold_{fold}_repair_test.jsonl'
            write_jsonl(repaired,[r for r in read_jsonl(ROOT/'data/repair_candidate_2/corpus.jsonl') if r['template_id'] in heldout])
            # Same adapters on original and repaired held-out texts: paired test-repair sensitivity.
            infer(name,[job(name+'_'+version+'_'+mode,gold,mode)
                        for version,gold in [('original',original),('repair',repaired)]
                        for mode in ['direct','guideline']],adapter)
    (out/'COMPLETE.json').write_text(json.dumps({'status':'complete','seeds':[17,29,43],'folds':5}),encoding='utf8')


if __name__=='__main__': main()
