#!/usr/bin/env bash
# A100 launch profile: each turn advantage is exactly DeepSeek's -1/0/1 label.
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${LLM_CRITIC_GPUS_RESERVED:-0}" != "1" ]]; then
    echo "Set LLM_CRITIC_GPUS_RESERVED=1 only after reserving CUDA_DEVICES." >&2
    exit 2
fi

: "${CUDA_DEVICES:?Set two or four explicitly reserved GPU IDs, for example 6,7}"
export CUDA_DEVICES
export SECRETS_FILE="${SECRETS_FILE-${HOME}/.config/llm-critic/secrets.env}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-MuLab-RL}"
export WANDB_PROJECT="${WANDB_PROJECT:-llm-critic-turn-ppo}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-advantage-ablation}"
# The trainer needs only a handful of Ray actors. Prevent Ray from importing
# the full model stack in one idle worker per host CPU at startup.
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
export RAY_WORKER_NICENESS="${RAY_WORKER_NICENESS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Keep Ray's session files off a nearly full shared /data mount. The socket
# root must also remain short enough for Unix-domain socket path limits.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/${USER:-ksj}-lc-rt}"
export TMPDIR="${TMPDIR:-/tmp/${USER:-ksj}-lc-tmp}"

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
if (( ${#gpu_ids[@]} != 2 && ${#gpu_ids[@]} != 4 )); then
    echo "This profile requires exactly two or four CUDA devices." >&2
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

fi

export TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-4}"
export TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
export VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-16}"
export VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export TOTAL_STEPS="${TOTAL_STEPS:-100}"
export MAX_ACTIONS_PER_TRAJ="${MAX_ACTIONS_PER_TRAJ:-10}"
export MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-1}"
# Preserve the original ten primitive-action Sokoban horizon after changing to
# one action per assistant turn. The shared launcher derives MAX_TURN=10 unless
# callers explicitly choose a shorter, consistently advertised ablation.
if [[ -n "${MAX_TURN:-}" ]]; then
    export MAX_TURN
fi
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
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-True}"
export ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
export ACTOR_FSDP_USE_TORCH_COMPILE="${ACTOR_FSDP_USE_TORCH_COMPILE:-False}"
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
export SAVE_FREQ="${SAVE_FREQ:-${TOTAL_STEPS}}"

host_tag="$(hostname -s | tr -d '\n' | tr -c 'A-Za-z0-9_.-' '_')"
run_stamp="$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-${host_tag}-sokoban-turnppo-dsflash-labelonly-${#gpu_ids[@]}gpu-${run_stamp}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HOME}/checkpoints/${WANDB_PROJECT}/${RUN_NAME}}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    min_checkpoint_free_gib="${MIN_CHECKPOINT_FREE_GIB:-50}"
    if [[ ! "${min_checkpoint_free_gib}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MIN_CHECKPOINT_FREE_GIB must be a positive integer." >&2
        exit 2
    fi
    checkpoint_parent="$(dirname "${CHECKPOINT_DIR}")"
    mkdir -p "${checkpoint_parent}"
    checkpoint_available_kib="$(df -Pk "${checkpoint_parent}" | awk 'NR==2 {print $4}')"
    if (( checkpoint_available_kib < min_checkpoint_free_gib * 1024 * 1024 )); then
        echo "Refusing to train with less than ${min_checkpoint_free_gib} GiB free on the checkpoint filesystem." >&2
        exit 2
    fi
fi

exec bash "${ROOT_DIR}/train_sokoban_deepseek_turn_ppo.sh"
