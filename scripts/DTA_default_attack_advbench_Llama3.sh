#!/bin/bash

BASE_PATH="../LLMJailbreak" 

TARGET_MODEL_PATH="$BASE_PATH/models/Llama-2-7b-chat-hf"
REFERENCE_MODEL_PATH="$BASE_PATH/models/Mistral-7B-Instruct-v0.3"
JUDGE_MODEL_PATH="$BASE_PATH/models/GPTFuzzer"
DATA_FILE="$BASE_PATH/data/harmful_behaviors.csv"
SAVE_DIR="./results/dta_llama2_mistral_advbench"

# 创建保存目录
mkdir -p "$SAVE_DIR"

python src/main.py \
    --local-llm-name-or-path "$TARGET_MODEL_PATH" \
    --local-llm-device cuda:0 \
    \
    --use-another-reference-llm \
    --reference-model-name-or-path "$REFERENCE_MODEL_PATH" \
    --reference-model-device cuda:1 \
    \
    --judge-llm-name-or-path "$JUDGE_MODEL_PATH" \
    --judge-llm-device cuda:1 \
    \
    --file-name "$DATA_FILE" \
    --reference-num-samples 30 \
    --forward-response-length 30 \
    --num-outer-iterations 15 \
    --num-inner-iterations 150 \
    --learning-rate 1.5 \
    --suffix-length 20 \
    --mask-rej-words \
    --save-dir "$SAVE_DIR" \
    --torch-dtype float16
