#!/usr/bin/env bash
# Launch the signed, turn-level DeepSeek critic experiment.
#
# Secrets are deliberately read from the environment.  In particular, do not
# add `generative_critic.deepseek_api_key=...` to the Hydra command: Hydra
# stores command-line overrides in `.hydra/overrides.yaml` and they can also be
# visible in process listings.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in the environment before starting the run}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER:-ragen}}"
mkdir -p "${RAY_TMPDIR}"

# W&B uses WANDB_API_KEY when online and does not need a login command.  An
# absent W&B key intentionally falls back to offline logs, which can be synced
# later with `wandb sync`; no credential is embedded in this script.
export WANDB_ENTITY="${WANDB_ENTITY:-MuLab-RL}"
export WANDB_PROJECT="${WANDB_PROJECT:-llm-critic-turn-ppo}"
export WANDB_MODE="${WANDB_MODE:-$([[ -n "${WANDB_API_KEY:-}" ]] && echo online || echo offline)}"
export WANDB_DIR="${WANDB_DIR:-${RAY_TMPDIR}/wandb}"
mkdir -p "${WANDB_DIR}"

if [[ -x /home/kangshijia/venvs/ragen/bin/python ]]; then
    PYTHON_CMD=(/home/kangshijia/venvs/ragen/bin/python)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n ragen python)
else
    PYTHON_CMD=(python)
fi

MODEL_PATH="${MODEL_PATH:-/data/kangshijia/sicheng/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model config not found: ${MODEL_PATH}/config.json" >&2
    echo "Set MODEL_PATH to a local Qwen2/Qwen3 checkpoint." >&2
    exit 2
fi

# Set CUDA_DEVICES only to GPUs reserved for this job.  The default is useful
# for a single-GPU smoke run; on shared machines callers should always pass an
# explicitly reserved device list.
CUDA_DEVICES="${CUDA_DEVICES:-0}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"
N_GPUS="${N_GPUS:-${#GPU_IDS[@]}}"
if [[ "${N_GPUS}" -ne "${#GPU_IDS[@]}" ]]; then
    echo "N_GPUS (${N_GPUS}) must match CUDA_DEVICES (${CUDA_DEVICES})" >&2
    exit 2
fi

RUN_NAME="${RUN_NAME:-sokoban-turn-ppo-deepseek-v4-flash}"
PROJECT_NAME="${WANDB_PROJECT}"

# Keep defaults modest enough for a first validation run.  Increase the group
# and step counts only after parser/API health and turn-mask metrics are sane.
TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-4}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-8}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
MAX_TURN="${MAX_TURN:-5}"
MAX_ACTIONS_PER_TURN="${MAX_ACTIONS_PER_TURN:-1}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-40}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"

# A fresh run is the default; set RESUME_MODE=auto/resume_path explicitly when
# continuing a known checkpoint.  ``DRY_RUN=1`` asks Hydra to render the fully
# resolved job config and exit before Ray/model initialization.

TRAIN_ARGS=(
    --config-name _2_sokoban
    "system.CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
    "trainer.n_gpus_per_node=${N_GPUS}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.logger=['console','wandb']"
    "es_manager.train.env_groups=${TRAIN_ENV_GROUPS}"
    "es_manager.train.group_size=${TRAIN_GROUP_SIZE}"
    "es_manager.train.env_configs.n_groups=[${TRAIN_ENV_GROUPS}]"
    "es_manager.val.env_groups=${VAL_ENV_GROUPS}"
    "es_manager.val.group_size=${VAL_GROUP_SIZE}"
    "es_manager.val.env_configs.n_groups=[${VAL_ENV_GROUPS}]"
    "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "algorithm.adv_estimator=gae"
    "algorithm.use_label_outcome_advantage=True"
    "algorithm.label_weight=1.0"
    "algorithm.outcome_weight=1.0"
    "actor_rollout_ref.actor.use_ref=False"
    "actor_rollout_ref.rollout.rollout_filter_ratio=1.0"
    "actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
    "generative_critic.enable=True"
    "generative_critic.train_enable=False"
    "generative_critic.backend=deepseek_api"
    "generative_critic.response_format=score_only"
    "generative_critic.deepseek_model=${DEEPSEEK_MODEL}"
    "generative_critic.deepseek_api_key_env=DEEPSEEK_API_KEY"
    "generative_critic.deepseek_thinking=${DEEPSEEK_THINKING:-disabled}"
    "generative_critic.deepseek_timeout=${DEEPSEEK_TIMEOUT:-30}"
    "generative_critic.deepseek_max_retries=${DEEPSEEK_MAX_RETRIES:-2}"
    "generative_critic.deepseek_max_concurrency=${DEEPSEEK_MAX_CONCURRENCY:-16}"
    "generative_critic.deepseek_max_tokens=${DEEPSEEK_MAX_TOKENS:-16}"
    "generative_critic.deepseek_max_prompt_chars=${DEEPSEEK_MAX_PROMPT_CHARS:-12000}"
    "generative_critic.parse_fail_score=-1"
    "generative_critic.default_label_if_parse_fail=False"
    "generative_critic.do_sample=False"
    "generative_critic.debug_print_samples=${DEBUG_CRITIC:-False}"
    "agent_proxy.max_turn=${MAX_TURN}"
    "agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN}"
    "agent_proxy.enable_think=${ENABLE_THINK:-False}"
    "agent_proxy.debug_turn_boundary=${DEBUG_TURN_BOUNDARY:-False}"
    "model_path=${MODEL_PATH}"
    "trainer.nnodes=1"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}"
    "trainer.save_freq=${SAVE_FREQ:-500}"
    "trainer.test_freq=${TEST_FREQ:-10}"
    "trainer.total_training_steps=${TOTAL_STEPS}"
    "trainer.resume_mode=${RESUME_MODE:-disable}"
    "algorithm.gamma=${GAMMA:-1.0}"
    "algorithm.lam=${LAMBDA:-0.95}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    MKL_SERVICE_FORCE_INTEL=1 "${PYTHON_CMD[@]}" train.py --cfg job "${TRAIN_ARGS[@]}"
    exit 0
fi

MKL_SERVICE_FORCE_INTEL=1 "${PYTHON_CMD[@]}" train.py \
    "${TRAIN_ARGS[@]}"
