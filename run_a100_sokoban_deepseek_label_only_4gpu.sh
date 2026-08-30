#!/usr/bin/env bash
# Four-GPU diagnostic run: each turn advantage is exactly DeepSeek's -1/0/1 label.
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${LLM_CRITIC_GPUS_RESERVED:-0}" != "1" ]]; then
    echo "Set LLM_CRITIC_GPUS_RESERVED=1 only after reserving CUDA_DEVICES." >&2
    exit 2
fi

: "${CUDA_DEVICES:?Set four explicitly reserved GPU IDs, for example 4,5,6,7}"
export CUDA_DEVICES
export SECRETS_FILE="${SECRETS_FILE-${HOME}/.config/llm-critic/secrets.env}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-MuLab-RL}"
export WANDB_PROJECT="${WANDB_PROJECT:-llm-critic-turn-ppo}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-advantage-ablation}"

# Both A100 hosts expose the working DeepSeek egress proxy on localhost:7890.
# W&B endpoints stay direct so artifact uploads do not contend with judge calls.
export DEEPSEEK_PROXY="${DEEPSEEK_PROXY:-http://127.0.0.1:7890}"
export HTTP_PROXY="${DEEPSEEK_PROXY}"
export HTTPS_PROXY="${DEEPSEEK_PROXY}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,api.wandb.ai,wandb.ai,storage.googleapis.com"
export no_proxy="${NO_PROXY}"

IFS=',' read -r -a gpu_ids <<< "${CUDA_DEVICES}"
if (( ${#gpu_ids[@]} != 4 )); then
    echo "This profile requires exactly four CUDA devices." >&2
    exit 2
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    for gpu_id in "${gpu_ids[@]}"; do
        gpu_state="$(
            nvidia-smi -i "${gpu_id}" \
                --query-gpu=memory.used,utilization.gpu,utilization.memory \
                --format=csv,noheader,nounits
        )"
        IFS=',' read -r memory_used gpu_util memory_util <<< "${gpu_state}"
        memory_used="${memory_used//[[:space:]]/}"
        gpu_util="${gpu_util//[[:space:]]/}"
        memory_util="${memory_util//[[:space:]]/}"
        compute_pids="$(
            nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits \
                | tr -d '[:space:]'
        )"
        if [[ -n "${compute_pids}" ]] || (( memory_used > 500 || gpu_util > 2 || memory_util > 2 )); then
            echo "GPU ${gpu_id} is not idle: memory=${memory_used}MiB gpu=${gpu_util}% memory_util=${memory_util}%" >&2
            exit 2
        fi
    done

    data_available_kib="$(df -Pk /data | awk 'NR==2 {print $4}')"
    if (( data_available_kib < 80 * 1024 * 1024 )); then
        echo "Refusing to train with less than 80 GiB free on /data." >&2
        exit 2
    fi
fi

export TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-4}"
export TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
export VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-16}"
export VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export MAX_TURN="${MAX_TURN:-5}"
export MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-1}"
export RESPONSE_LENGTH="${RESPONSE_LENGTH:-40}"

export TURN_ADVANTAGE_MODE=label_only
export LABEL_WEIGHT=1.0
export OUTCOME_WEIGHT=0.0
# Keep the terminal result as a diagnostic W&B metric; label_only ignores it.
export OUTCOME_BROADCAST=all_turns
export NORMALIZE_TURN_ADVANTAGE=False
export USE_KL_IN_REWARD=False
export ADD_KL_TO_TURN_ADVANTAGE=False
export CRITIC_ENABLE=False
export USE_KL_LOSS=False
export ENTROPY_COEFF=0.0
export PPO_EPOCHS=1

export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
# The run emits exactly 32 rollouts per train/validation batch. Avoid vLLM's
# default 1024-sequence scheduler allocation; shared hosts may additionally set
# VLLM_ENFORCE_EAGER=True to skip CUDA-graph compilation and capture.
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-False}"
export ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-True}"
export ACTOR_FSDP_USE_TORCH_COMPILE="${ACTOR_FSDP_USE_TORCH_COMPILE:-True}"
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
# Evaluate 32 fixed rollouts every ten optimizer steps.
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-50}"

host_tag="$(hostname -s | tr -d '\n' | tr -c 'A-Za-z0-9_.-' '_')"
run_stamp="$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-${host_tag}-sokoban-turnppo-dsflash-labelonly-4gpu-${run_stamp}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/kangshijia/checkpoints/${WANDB_PROJECT}/${RUN_NAME}}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"

exec bash "${ROOT_DIR}/train_sokoban_deepseek_turn_ppo.sh"
