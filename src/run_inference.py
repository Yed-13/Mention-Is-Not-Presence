#!/usr/bin/env python3
"""Deterministic 4-bit inference with immutable raw generations."""
import argparse,json,re,time
from pathlib import Path

def extract_json(text):
    text=re.sub(r"<think>.*?</think>","",text,flags=re.S).strip()
    start=text.find("{"); end=text.rfind("}")
    if start<0 or end<start: return None
    try: return json.loads(text[start:end+1])
    except json.JSONDecodeError: return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--adapter"); p.add_argument("--prompts",required=True); p.add_argument("--predictions",required=True); p.add_argument("--raw",required=True); p.add_argument("--max-new-tokens",type=int,default=512); p.add_argument("--limit",type=int)
    a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=q,device_map={"":0},torch_dtype=torch.bfloat16)
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter)
    model.eval()
    rows=[json.loads(x) for x in Path(a.prompts).read_text(encoding="utf8").splitlines() if x.strip()]
    if a.limit: rows=rows[:a.limit]
    preds=[]; raw=[]
    for i,row in enumerate(rows,1):
        prompt=tok.apply_chat_template(row["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inp=tok(prompt,return_tensors="pt").to(model.device); torch.cuda.synchronize(); t=time.time()
        with torch.inference_mode(): out=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=False)
        torch.cuda.synchronize(); generated=tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True)
        parsed=extract_json(generated)
        raw.append({"scene_id":row["scene_id"],"family_id":row["family_id"],"raw":generated,"parsed":parsed,"seconds":time.time()-t,"input_tokens":int(inp.input_ids.shape[1]),"output_tokens":int(out.shape[1]-inp.input_ids.shape[1])})
        if parsed is not None: preds.append(parsed)
        print(i,row["scene_id"],"ok" if parsed else "malformed",flush=True)
    Path(a.raw).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in raw),encoding="utf8")
    Path(a.predictions).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in preds),encoding="utf8")
if __name__=="__main__": main()
