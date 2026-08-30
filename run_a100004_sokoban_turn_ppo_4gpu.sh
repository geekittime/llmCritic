#!/usr/bin/env bash
# Reproducible four-GPU training profile for the shared A100-004 host.
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export A100004_GPUS_RESERVED="${A100004_GPUS_RESERVED:-1}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export EXPERIMENT_PROFILE=train
export CONFIRM_DEEPSEEK_COST="${CONFIRM_DEEPSEEK_COST:-1}"

# Keep the first measured run large enough for stable PPO mini-batches while
# limiting API cost and contention with the existing processes on GPUs 0-3.
export TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-4}"
export TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
export VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-4}"
export VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export MAX_TURN="${MAX_TURN:-5}"
export MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-1}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-40}"

export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.38}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
export FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"

export DEEPSEEK_MAX_CONCURRENCY="${DEEPSEEK_MAX_CONCURRENCY:-24}"
export DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-15}"
export DEEPSEEK_BATCH_TIMEOUT="${DEEPSEEK_BATCH_TIMEOUT:-120}"
export DEEPSEEK_MAX_RETRIES="${DEEPSEEK_MAX_RETRIES:-1}"
export DEEPSEEK_MAX_PROMPT_CHARS="${DEEPSEEK_MAX_PROMPT_CHARS:-8000}"
export DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-8}"

export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-llm-critic-turn-ppo}"
run_stamp="$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-a100004-sokoban-turnppo-dsflash-4gpu-${run_stamp}}"

exec bash "${ROOT_DIR}/run_a100004_sokoban_turn_ppo.sh"
