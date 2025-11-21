#!/usr/bin/env bash

# Bayes-Adaptive LLM on P4G: SFT + DPO
# Mirrors run_exp_pg_trip.sh style with accelerate.

set -euo pipefail

EXPNAME="P4G_BAYES"

# repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

for i in 1
do
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 8080 --gpu_ids 0 --num_processes 1 run.py \
    --exp_name "${EXPNAME}" \
    --project_name ProactiveLLM \
    --seed "${i}" \
    --scenario persuation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets p4g \
    --models bayes_adaptive_llm \
    --is_so_game \
    --gen_models gpt2 \
    --model_type gpt2 \
    --metrics acc,prf1,sr,total_reward,avg_turn
done
