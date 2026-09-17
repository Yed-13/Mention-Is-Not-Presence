#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/scriptbreak_expansion}"
EXISTING="${EXISTING:-/root/autodl-tmp/scriptbreak_zh_v2}"
PYTHON="${PYTHON:-$EXISTING/.venv/bin/python}"
MODEL4="${MODEL4:-$EXISTING/modelscope_cache/models/Qwen--Qwen3-4B/snapshots/master}"
MODEL8="${MODEL8:-$EXISTING/modelscope_cache/models/Qwen--Qwen3-8B/snapshots/master}"
ADAPTER4="${ADAPTER4:-$EXISTING/results/adapters/r32_lr2e4_s17}"

cd "$ROOT"
mkdir -p results/expansion/{raw,metrics,adapters,logs}

run_controlled() {
  local gpu="$1" model="$2" adapter="$3" prompts="$4" prefix="$5" gold="$6"
  local adapter_args=()
  if [[ -n "$adapter" ]]; then adapter_args=(--adapter "$adapter"); fi
  if [[ ! -s "results/expansion/metrics/${prefix}.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" src/run_inference.py \
      --model "$model" "${adapter_args[@]}" --prompts "$prompts" \
      --predictions "results/expansion/raw/${prefix}_predictions.jsonl" \
      --raw "results/expansion/raw/${prefix}_raw.jsonl" \
      >"results/expansion/logs/${prefix}.log" 2>&1
    PYTHONPATH=src "$PYTHON" src/scriptbreak_eval.py evaluate \
      --gold "$gold" --predictions "results/expansion/raw/${prefix}_predictions.jsonl" \
      --out "results/expansion/metrics/${prefix}.json" \
      >>"results/expansion/logs/${prefix}.log" 2>&1
  fi
}

run_external() {
  local gpu="$1" model="$2" adapter="$3" prompts="$4" prefix="$5"
  local adapter_args=()
  if [[ -n "$adapter" ]]; then adapter_args=(--adapter "$adapter"); fi
  if [[ ! -s "results/expansion/metrics/${prefix}.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" src/run_inference.py \
      --model "$model" "${adapter_args[@]}" --prompts "$prompts" \
      --predictions "results/expansion/raw/${prefix}_predictions.jsonl" \
      --raw "results/expansion/raw/${prefix}_raw.jsonl" \
      >"results/expansion/logs/${prefix}.log" 2>&1
    PYTHONPATH=src "$PYTHON" src/vistory_visibility_eval.py evaluate \
      --gold data/external/vistory_visibility.jsonl \
      --predictions "results/expansion/raw/${prefix}_predictions.jsonl" \
      --out "results/expansion/metrics/${prefix}.json" \
      >>"results/expansion/logs/${prefix}.log" 2>&1
  fi
}

train_adapter() {
  local gpu="$1" model="$2" train="$3" dev="$4" output="$5" seed="$6"
  if [[ ! -s "$output/adapter_config.json" ]]; then
    mkdir -p "$(dirname "$output")"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" src/train_lora.py \
      --model "$model" --train "$train" --dev "$dev" --output "$output" \
      --seed "$seed" --rank 32 --alpha 64 --lr 2e-4 --epochs 3 --qlora \
      >"results/expansion/logs/$(basename "$output")_train.log" 2>&1
  fi
}

echo "stage=prompt_robustness start=$(date -Iseconds)"
for condition in guideline_concise guideline_reordered; do
  run_controlled 0 "$MODEL4" "" "data/expansion/synthetic_test_${condition}_prompts.jsonl" \
    "qwen3_4b_base_${condition}_controlled" data/synthetic/test.jsonl &
  run_external 1 "$MODEL4" "" "data/expansion/external_${condition}_prompts.jsonl" \
    "qwen3_4b_base_${condition}_external" &
  wait
  run_controlled 0 "$MODEL4" "$ADAPTER4" "data/expansion/synthetic_test_${condition}_prompts.jsonl" \
    "qwen3_4b_qlora_${condition}_controlled" data/synthetic/test.jsonl &
  run_external 1 "$MODEL4" "$ADAPTER4" "data/expansion/external_${condition}_prompts.jsonl" \
    "qwen3_4b_qlora_${condition}_external" &
  wait
done

echo "stage=learning_curves start=$(date -Iseconds)"
for percent in 25 50 75; do
  adapter="results/expansion/adapters/qwen3_4b_curve_${percent}_s17"
  train_adapter 0 "$MODEL4" "data/expansion/train_${percent}_sft.jsonl" \
    data/synthetic/dev_sft.jsonl "$adapter" 17
  run_controlled 0 "$MODEL4" "$adapter" data/synthetic/test_direct_prompts.jsonl \
    "qwen3_4b_curve_${percent}_direct_controlled" data/synthetic/test.jsonl &
  run_controlled 1 "$MODEL4" "$adapter" data/synthetic/test_guideline_prompts.jsonl \
    "qwen3_4b_curve_${percent}_guideline_controlled" data/synthetic/test.jsonl &
  wait
  run_external 0 "$MODEL4" "$adapter" data/external/direct_prompts.jsonl \
    "qwen3_4b_curve_${percent}_direct_external" &
  run_external 1 "$MODEL4" "$adapter" data/external/guideline_prompts.jsonl \
    "qwen3_4b_curve_${percent}_guideline_external" &
  wait
done

echo "stage=template_holdout start=$(date -Iseconds)"
for fold in 1 2 3 4 5; do
  adapter="results/expansion/adapters/qwen3_4b_template_fold_${fold}_s17"
  train_adapter 0 "$MODEL4" "data/expansion/template_fold_${fold}_train_sft.jsonl" \
    "data/expansion/template_fold_${fold}_dev_sft.jsonl" "$adapter" 17
  run_controlled 0 "$MODEL4" "$adapter" "data/expansion/template_fold_${fold}_direct_prompts.jsonl" \
    "qwen3_4b_template_fold_${fold}_direct" "data/expansion/template_fold_${fold}_test.jsonl" &
  run_controlled 1 "$MODEL4" "$adapter" "data/expansion/template_fold_${fold}_guideline_prompts.jsonl" \
    "qwen3_4b_template_fold_${fold}_guideline" "data/expansion/template_fold_${fold}_test.jsonl" &
  wait
done

echo "stage=qwen3_8b start=$(date -Iseconds)"
run_controlled 0 "$MODEL8" "" data/synthetic/test_direct_prompts.jsonl \
  qwen3_8b_base_direct_controlled data/synthetic/test.jsonl &
run_controlled 1 "$MODEL8" "" data/synthetic/test_guideline_prompts.jsonl \
  qwen3_8b_base_guideline_controlled data/synthetic/test.jsonl &
wait
run_external 0 "$MODEL8" "" data/external/direct_prompts.jsonl qwen3_8b_base_direct_external &
run_external 1 "$MODEL8" "" data/external/guideline_prompts.jsonl qwen3_8b_base_guideline_external &
wait

adapter8="results/expansion/adapters/qwen3_8b_full_s17"
train_adapter 0 "$MODEL8" data/synthetic/train_sft.jsonl data/synthetic/dev_sft.jsonl "$adapter8" 17
run_controlled 0 "$MODEL8" "$adapter8" data/synthetic/test_direct_prompts.jsonl \
  qwen3_8b_qlora_direct_controlled data/synthetic/test.jsonl &
run_controlled 1 "$MODEL8" "$adapter8" data/synthetic/test_guideline_prompts.jsonl \
  qwen3_8b_qlora_guideline_controlled data/synthetic/test.jsonl &
wait
run_external 0 "$MODEL8" "$adapter8" data/external/direct_prompts.jsonl qwen3_8b_qlora_direct_external &
run_external 1 "$MODEL8" "$adapter8" data/external/guideline_prompts.jsonl qwen3_8b_qlora_guideline_external &
wait

date -Iseconds > results/expansion/COMPLETE
echo "stage=complete time=$(cat results/expansion/COMPLETE)"
