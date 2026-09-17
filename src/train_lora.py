#!/usr/bin/env python3
"""QLoRA supervised fine-tuning entry point used by ScriptBreak-ZH.

Input JSONL contains chat-template-compatible ``messages`` records. The released
train and development files are separated by complete contrast family.
"""
import argparse, os


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--qlora", action="store_true")
    a=p.parse_args()
    if os.path.exists(a.output) and os.listdir(a.output):
        raise SystemExit("output directory must be absent or empty")
    from datasets import load_dataset
    from peft import LoraConfig
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    set_seed(a.seed)
    ds=load_dataset("json", data_files={"train":a.train,"validation":a.dev})
    tok=AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16) if a.qlora else None
    model=AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, quantization_config=quant, device_map={"":0}, trust_remote_code=a.trust_remote_code)
    if a.qlora: model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    peft=LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=.05, bias="none", target_modules="all-linear", task_type="CAUSAL_LM")
    cfg=SFTConfig(output_dir=a.output, num_train_epochs=a.epochs, learning_rate=a.lr,
                  per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
                  gradient_accumulation_steps=a.grad_accum, max_length=a.max_length,
                  bf16=True, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, logging_steps=5,
                  report_to="none", seed=a.seed, data_seed=a.seed)
    trainer=SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["validation"],
                       processing_class=tok, peft_config=peft)
    trainer.train(); trainer.save_model(a.output); tok.save_pretrained(a.output)

if __name__ == "__main__": main()
