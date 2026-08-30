#!/usr/bin/env bash
# Launch the signed, turn-level DeepSeek critic experiment.
#
# Secrets are deliberately read from the environment.  In particular, do not
# add `generative_critic.deepseek_api_key=...` to the Hydra command: Hydra
# stores command-line overrides in `.hydra/overrides.yaml` and they can also be
# visible in process listings.
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Validate a caller-owned credential file without exporting its contents. The
# Ray TaskRunner reads it after scheduling, keeping values out of Ray job
# runtime_env metadata, process environments, and runtime-env logs.
credential_file_has_deepseek=0
credential_file_has_wandb=0
if [[ -n "${SECRETS_FILE:-}" ]]; then
    if [[ ! -r "${SECRETS_FILE}" ]]; then
        echo "Secrets file is not readable: ${SECRETS_FILE}" >&2
        exit 2
    fi
    secrets_mode="$(stat -c '%a' "${SECRETS_FILE}")"
    if (( (8#${secrets_mode}) & 077 )); then
        echo "Secrets file must not be accessible by group/other: ${SECRETS_FILE}" >&2
        exit 2
    fi
    # Parse only credential assignments. Sourcing arbitrary shell here would
    # let a credentials file override CUDA_DEVICES or other reviewed training
    # settings after the caller reserved a GPU.
    secret_line_number=0
    while IFS= read -r secret_line || [[ -n "${secret_line}" ]]; do
        secret_line_number=$((secret_line_number + 1))
        secret_line="${secret_line%$'\r'}"
        [[ -z "${secret_line}" || "${secret_line}" == \#* ]] && continue
        secret_line="${secret_line#export }"
        secret_name="${secret_line%%=*}"
        secret_value="${secret_line#*=}"
        if [[ "${secret_name}" == "${secret_line}" ]]; then
            echo "Invalid assignment in secrets file at line ${secret_line_number}" >&2
            exit 2
        fi
        case "${secret_name}" in
            DEEPSEEK_API_KEY)
                [[ -n "${secret_value}" ]] || { echo "Empty DEEPSEEK_API_KEY in secrets file" >&2; exit 2; }
                credential_file_has_deepseek=1
                ;;
            WANDB_API_KEY)
                [[ -n "${secret_value}" ]] || { echo "Empty WANDB_API_KEY in secrets file" >&2; exit 2; }
                credential_file_has_wandb=1
                ;;
            *)
                echo "Unsupported variable in secrets file at line ${secret_line_number}" >&2
                exit 2
                ;;
        esac
    done < "${SECRETS_FILE}"
    # Keep rotated or stale inherited values out of raylet/GPU-worker base
    # environments. TaskRunner will install the selected file values locally.
    (( credential_file_has_deepseek == 0 )) || unset DEEPSEEK_API_KEY
    (( credential_file_has_wandb == 0 )) || unset WANDB_API_KEY
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    if [[ -z "${DEEPSEEK_API_KEY:-}" && "${credential_file_has_deepseek}" != "1" ]]; then
        echo "Set DEEPSEEK_API_KEY or provide it in a mode-600 SECRETS_FILE" >&2
        exit 2
    fi
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
# This is a single-node launcher. Never inherit a pointer to another user's
# persistent Ray cluster on a shared host.
unset RAY_ADDRESS
RUN_NAME="${RUN_NAME:-sokoban-turn-ppo-deepseek-v4-flash}"
safe_run_name="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9_.-' '_')"
if [[ -d "/data/${USER:-}" && -w "/data/${USER:-}" ]]; then
    # Ray appends a long session/sockets suffix. Keep this root deliberately
    # short so Unix-domain socket paths stay below the platform limit.
    default_ray_tmp="/data/${USER}/rt"
    default_wandb_dir="/data/${USER}/wb/${safe_run_name}"
    default_tmp_dir="/data/${USER}/tmp/${safe_run_name}"
    default_cache_dir="/data/${USER}/cache"
else
    default_ray_tmp="/tmp/${USER:-ragen}-rt"
    default_wandb_dir="/tmp/${USER:-ragen}-wb/${safe_run_name}"
    default_tmp_dir="/tmp/${USER:-ragen}-tmp/${safe_run_name}"
    default_cache_dir="/tmp/${USER:-ragen}-cache"
fi
export RAY_TMPDIR="${RAY_TMPDIR:-${default_ray_tmp}}"
ray_tmp_bytes="$(LC_ALL=C printf '%s' "${RAY_TMPDIR}" | wc -c)"
if (( ray_tmp_bytes > 32 )); then
    echo "RAY_TMPDIR is too long for Ray Unix sockets (${ray_tmp_bytes} bytes; max 32): ${RAY_TMPDIR}" >&2
    exit 2
fi
mkdir -p "${RAY_TMPDIR}"
chmod 700 "${RAY_TMPDIR}"
export TMPDIR="${TMPDIR:-${default_tmp_dir}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${default_cache_dir}}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# W&B can authenticate through WANDB_API_KEY or a mode-600 ~/.netrc.  Callers
# can always force WANDB_MODE=online/offline; no credential is embedded here.
export WANDB_ENTITY="${WANDB_ENTITY:-MuLab-RL}"
export WANDB_PROJECT="${WANDB_PROJECT:-llm-critic-turn-ppo}"
if [[ -z "${WANDB_MODE:-}" ]]; then
    if [[ -n "${WANDB_API_KEY:-}" || "${credential_file_has_wandb}" == "1" ]] || { [[ -r "${HOME}/.netrc" ]] && grep -q 'api\.wandb\.ai' "${HOME}/.netrc"; }; then
        WANDB_MODE=online
    else
        WANDB_MODE=offline
    fi
fi
export WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-${default_wandb_dir}}"
mkdir -p "${WANDB_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
        exit 2
    fi
    PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -x /home/kangshijia/venvs/ragen/bin/python ]]; then
    PYTHON_CMD=(/home/kangshijia/venvs/ragen/bin/python)
elif [[ -x /home/kangshijia/miniconda3/envs/ragen-vanilla/bin/python ]]; then
    PYTHON_CMD=(/home/kangshijia/miniconda3/envs/ragen-vanilla/bin/python)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n "${RAGEN_CONDA_ENV:-ragen}" python)
else
    PYTHON_CMD=(python)
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
    for candidate in \
        /data/models/Qwen2.5-3B-Instruct \
        /data/kangshijia/models/Qwen2.5-3B-instruct \
        /data/kangshijia/sicheng/AgentGym-RL/models/Qwen2.5-3B-Instruct; do
        if [[ -f "${candidate}/config.json" ]]; then
            MODEL_PATH="${candidate}"
            break
        fi
    done
fi
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-3B-Instruct}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model config not found: ${MODEL_PATH}/config.json" >&2
    echo "Set MODEL_PATH to a local Qwen2/Qwen3 checkpoint." >&2
    exit 2
fi

# Set CUDA_DEVICES only to GPUs reserved for this job.  The default is useful
# for a single-GPU smoke run; on shared machines callers should always pass an
# explicitly reserved device list.
: "${CUDA_DEVICES:?Set CUDA_DEVICES to GPUs explicitly reserved for this run}"
if [[ ! "${CUDA_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "CUDA_DEVICES must be a comma-separated list of integer GPU IDs: ${CUDA_DEVICES}" >&2
    exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"
declare -A seen_gpu_ids=()
available_gpu_ids=""
if command -v nvidia-smi >/dev/null 2>&1; then
    available_gpu_ids=",$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, - | tr -d '[:space:]' || true),"
fi
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
        echo "CUDA_DEVICES contains a duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    seen_gpu_ids["${gpu_id}"]=1
    if [[ "${available_gpu_ids}" != ",," && "${available_gpu_ids}" != *",${gpu_id},"* ]]; then
        echo "CUDA_DEVICES references a GPU not reported by nvidia-smi: ${gpu_id}" >&2
        exit 2
    fi
done
N_GPUS="${N_GPUS:-${#GPU_IDS[@]}}"
if [[ "${N_GPUS}" -ne "${#GPU_IDS[@]}" ]]; then
    echo "N_GPUS (${N_GPUS}) must match CUDA_DEVICES (${CUDA_DEVICES})" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

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

if (( PPO_MINI_BATCH_SIZE % (MICRO_BATCH_SIZE * N_GPUS) != 0 )); then
    echo "PPO_MINI_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * N_GPUS" >&2
    exit 2
fi
if (( TRAIN_ENV_GROUPS * TRAIN_GROUP_SIZE < PPO_MINI_BATCH_SIZE )); then
    echo "TRAIN_ENV_GROUPS * TRAIN_GROUP_SIZE must be >= PPO_MINI_BATCH_SIZE" >&2
    exit 2
fi

# A fresh run is the default; set RESUME_MODE=auto/resume_path explicitly when
# continuing a known checkpoint.  ``DRY_RUN=1`` asks Hydra to render the fully
# resolved job config and exit before Ray/model initialization.

TRAIN_ARGS=(
    --config-name _2_sokoban
    "system.CUDA_VISIBLE_DEVICES='${CUDA_DEVICES}'"
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
    "algorithm.turn_advantage_mode=${TURN_ADVANTAGE_MODE:-weighted}"
    "algorithm.label_weight=${LABEL_WEIGHT:-1.0}"
    "algorithm.outcome_weight=${OUTCOME_WEIGHT:-1.0}"
    "algorithm.outcome_broadcast=${OUTCOME_BROADCAST:-all_turns}"
    "algorithm.normalize_turn_advantage=${NORMALIZE_TURN_ADVANTAGE:-False}"
    "algorithm.use_kl_in_reward=${USE_KL_IN_REWARD:-False}"
    "algorithm.add_kl_to_turn_advantage=${ADD_KL_TO_TURN_ADVANTAGE:-True}"
    "critic.enable=${CRITIC_ENABLE:-False}"
    "actor_rollout_ref.actor.use_ref=False"
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS:-False}"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF:-0.0}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS:-1}"
    "actor_rollout_ref.actor.use_torch_compile=${ACTOR_USE_TORCH_COMPILE:-True}"
    "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=${ACTOR_FSDP_USE_TORCH_COMPILE:-True}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-False}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING:-True}"
    "actor_rollout_ref.rollout.rollout_filter_ratio=1.0"
    "actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
    "actor_rollout_ref.rollout.enforce_eager=${VLLM_ENFORCE_EAGER:-False}"
    "actor_rollout_ref.rollout.max_num_seqs=${VLLM_MAX_NUM_SEQS:-1024}"
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE:-True}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
    "generative_critic.enable=True"
    "generative_critic.train_enable=False"
    "generative_critic.backend=deepseek_api"
    "generative_critic.response_format=score_only"
    "generative_critic.deepseek_model=${DEEPSEEK_MODEL}"
    "generative_critic.deepseek_api_key_env=DEEPSEEK_API_KEY"
    "generative_critic.deepseek_thinking=${DEEPSEEK_THINKING:-disabled}"
    "generative_critic.deepseek_timeout=${DEEPSEEK_TIMEOUT:-15}"
    "generative_critic.deepseek_batch_timeout=${DEEPSEEK_BATCH_TIMEOUT:-120}"
    "generative_critic.deepseek_max_retries=${DEEPSEEK_MAX_RETRIES:-1}"
    "generative_critic.deepseek_max_concurrency=${DEEPSEEK_MAX_CONCURRENCY:-16}"
    "generative_critic.deepseek_max_tokens=${DEEPSEEK_MAX_TOKENS:-8}"
    "generative_critic.deepseek_abort_on_auth_failure=${DEEPSEEK_ABORT_ON_AUTH_FAILURE:-True}"
    "generative_critic.deepseek_max_failure_rate=${DEEPSEEK_MAX_FAILURE_RATE:-0.25}"
    "generative_critic.deepseek_max_parse_fail_rate=${DEEPSEEK_MAX_PARSE_FAIL_RATE:-0.25}"
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
    "trainer.default_local_dir=${CHECKPOINT_DIR:-checkpoints/${PROJECT_NAME}/${RUN_NAME}}"
    "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    "algorithm.gamma=${GAMMA:-1.0}"
    "algorithm.lam=${LAMBDA:-0.95}"
)

if [[ -n "${SECRETS_FILE:-}" ]]; then
    TRAIN_ARGS+=("generative_critic.deepseek_api_key_file=${SECRETS_FILE}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    MKL_SERVICE_FORCE_INTEL=1 "${PYTHON_CMD[@]}" train.py --cfg job "${TRAIN_ARGS[@]}"
    exit 0
fi

MKL_SERVICE_FORCE_INTEL=1 "${PYTHON_CMD[@]}" train.py \
    "${TRAIN_ARGS[@]}"
