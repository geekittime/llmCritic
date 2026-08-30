#!/usr/bin/env bash
# A100-004 launcher for a W&B-tracked Sokoban turn-PPO experiment.
# No credential is stored here.  Put exports in the mode-600 SECRETS_FILE.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CUDA_DEVICES:?Set CUDA_DEVICES to GPUs explicitly reserved for this run}"
if [[ "${A100004_GPUS_RESERVED:-0}" != "1" ]]; then
    echo "Refusing to share an unreserved GPU. Set A100004_GPUS_RESERVED=1 only after confirming ownership." >&2
    exit 2
fi

export PYTHON_BIN="${PYTHON_BIN:-/home/kangshijia/miniconda3/envs/ragen-vanilla/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-3B-Instruct}"
export SECRETS_FILE="${SECRETS_FILE:-${HOME}/.config/llm-critic/secrets.env}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-MuLab-RL}"
export WANDB_PROJECT="${WANDB_PROJECT:-llm-critic-turn-ppo}"

run_stamp="$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-a100004-sokoban-turn-ppo-${EXPERIMENT_PROFILE:-smoke}-${run_stamp}}"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/kangshijia/ray/${RUN_NAME}}"
export WANDB_DIR="${WANDB_DIR:-${RAY_TMPDIR}/wandb}"

# DeepSeek needs the local proxy on A100-004, while W&B works directly.
export DEEPSEEK_PROXY="${DEEPSEEK_PROXY:-http://127.0.0.1:7890}"
export HTTP_PROXY="${HTTP_PROXY:-${DEEPSEEK_PROXY}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${DEEPSEEK_PROXY}}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,api.wandb.ai,wandb.ai"
export no_proxy="${NO_PROXY}"

case "${EXPERIMENT_PROFILE:-smoke}" in
    smoke)
        export TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-1}"
        export TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-1}"
        export VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-1}"
        export VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"
        export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
        export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
        export TOTAL_STEPS="${TOTAL_STEPS:-1}"
        export MAX_TURN="${MAX_TURN:-3}"
        export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
        export TEST_FREQ="${TEST_FREQ:--1}"
        export SAVE_FREQ="${SAVE_FREQ:--1}"
        export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.40}"
        export DEEPSEEK_MAX_CONCURRENCY="${DEEPSEEK_MAX_CONCURRENCY:-4}"
        export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
        ;;
    train)
        export TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-8}"
        export TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
        export VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-8}"
        export VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"
        export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
        export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
        export TOTAL_STEPS="${TOTAL_STEPS:-2000}"
        export MAX_TURN="${MAX_TURN:-5}"
        export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
        export TEST_FREQ="${TEST_FREQ:-25}"
        export SAVE_FREQ="${SAVE_FREQ:-200}"
        export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
        export DEEPSEEK_MAX_CONCURRENCY="${DEEPSEEK_MAX_CONCURRENCY:-16}"
        export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
        ;;
    *)
        echo "EXPERIMENT_PROFILE must be smoke or train" >&2
        exit 2
        ;;
esac

export MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-1}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-40}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
export DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-15}"
export DEEPSEEK_BATCH_TIMEOUT="${DEEPSEEK_BATCH_TIMEOUT:-120}"
export DEEPSEEK_MAX_RETRIES="${DEEPSEEK_MAX_RETRIES:-1}"

exec bash "${ROOT_DIR}/train_sokoban_deepseek_turn_ppo.sh"
