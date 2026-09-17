"""Batched, append-only inference for the explicitly separate 5090 replication."""
import argparse
import json
import time
from pathlib import Path
from run_inference import extract_json
from scriptbreak_eval import read_jsonl, write_jsonl, confusion_and_scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter')
    p.add_argument('--jobs', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok=AutoTokenizer.from_pretrained(a.model, padding_side='left')
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=q,device_map={'':0},torch_dtype=torch.bfloat16)
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter)
    model.eval()
    for job in json.loads(Path(a.jobs).read_text(encoding='utf8')):
        rows=read_jsonl(job['prompts'])
        raw_path=Path(job['raw']); raw_path.parent.mkdir(parents=True, exist_ok=True)
        existing=read_jsonl(raw_path) if raw_path.exists() else []
        done={r['scene_id'] for r in existing}
        assert len(done)==len(existing), 'duplicate raw records'
        assert done <= {r['scene_id'] for r in rows}
        pending=[r for r in rows if r['scene_id'] not in done]
        with raw_path.open('a',encoding='utf8') as output:
            for start in range(0,len(pending),a.batch_size):
                batch=pending[start:start+a.batch_size]
                strings=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
                inp=tok(strings,return_tensors='pt',padding=True).to(model.device)
                torch.cuda.synchronize(); before=time.monotonic()
                with torch.inference_mode():
                    gen=model.generate(**inp,max_new_tokens=512,do_sample=False,pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize(); elapsed=time.monotonic()-before
                for j,r in enumerate(batch):
                    continuation=gen[j,inp.input_ids.shape[1]:]
                    text=tok.decode(continuation,skip_special_tokens=True)
                    record={'scene_id':r['scene_id'],'family_id':r['family_id'],'raw':text,'parsed':extract_json(text),
                            'batch_seconds':elapsed,'batch_size':len(batch),'input_tokens':int(inp.attention_mask[j].sum()),
                            'output_tokens_including_padding':len(continuation)}
                    output.write(json.dumps(record,ensure_ascii=False)+'\n')
                    existing.append(record)
                output.flush()
                print(job['name'],len(existing),'/',len(rows),f'batch_seconds={elapsed:.2f}',flush=True)
        assert len(existing)==len(rows)
        preds=[r['parsed'] for r in existing if r['parsed'] is not None]
        write_jsonl(job['predictions'],preds)
        metrics=confusion_and_scores(read_jsonl(job['gold']),preds)
        Path(job['metrics']).write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print('COMPLETE',job['name'],metrics['macro_f1'],flush=True)


if __name__=='__main__': main()
