#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p /data/kangshijia/ray_tmp
export RAY_TMPDIR="${RAY_TMPDIR:-/data/kangshijia/ray_tmp}"
export PYTHONPATH="$(pwd):$(pwd)/verl:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-ragen}}"
export TOGETHER_NO_BANNER=1

# W&B settings copied from train_sokoban_gen_critic_qwen3_4b.sh.
export WANDB_API_KEY="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"
export WANDB_ENTITY="MuLab-RL"
export WANDB_PROJECT="llm-gen-critic-deepseek"

# DeepSeek critic API key requested by the user.
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-33021c0bec434de4b877c3142cc409c9}"

if [[ -x /home/kangshijia/venvs/ragen/bin/python ]]; then
    PYTHON_CMD=(/home/kangshijia/venvs/ragen/bin/python)
else
    PYTHON_CMD=(conda run -n ragen python)
fi

ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/data/kangshijia/wangbinyu/models/Qwen3-0.6B}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
RUN_NAME="${RUN_NAME:-dev0501-sokoban-qwen3-0.6b-deepseek-api-critic}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
if [[ -z "${N_GPUS:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"
    N_GPUS="${#GPU_IDS[@]}"
else
    N_GPUS="${N_GPUS}"
fi

USE_GEN_CRITIC="algorithm.use_label_outcome_advantage=True critic.enable=False generative_critic.enable=True generative_critic.train_enable=False"
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

CRITIC_DO_SAMPLE="${CRITIC_DO_SAMPLE:-False}"
CRITIC_TEMPERATURE="${CRITIC_TEMPERATURE:-0.0}"
CRITIC_TOP_P="${CRITIC_TOP_P:-1.0}"
CRITIC_MAX_CONCURRENCY="${CRITIC_MAX_CONCURRENCY:-128}"
CRITIC_TIMEOUT="${CRITIC_TIMEOUT:-60}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-500}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
RESUME_MODE="${RESUME_MODE:-auto}"
TRAIN_ENV_GROUPS="${TRAIN_ENV_GROUPS:-4}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
MAX_TURN="${MAX_TURN:-3}"
ROLLOUT_RESPONSE_LENGTH="${ROLLOUT_RESPONSE_LENGTH:-24}"
VAL_ENV_GROUPS="${VAL_ENV_GROUPS:-8}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-4}"

MKL_SERVICE_FORCE_INTEL=1 "${PYTHON_CMD[@]}" train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"${CUDA_DEVICES}\" trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.logger=['console','wandb'] \
    ${USE_BASE} ${USE_GEN_CRITIC} \
    es_manager.train.env_groups="${TRAIN_ENV_GROUPS}" \
    es_manager.train.group_size="${TRAIN_GROUP_SIZE}" \
    es_manager.train.env_configs.n_groups=[${TRAIN_ENV_GROUPS}] \
    es_manager.val.env_groups="${VAL_ENV_GROUPS}" \
    es_manager.val.group_size="${VAL_GROUP_SIZE}" \
    es_manager.val.env_configs.n_groups=[${VAL_ENV_GROUPS}] \
    micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    algorithm.adv_estimator=gae \
    algorithm.critic_score_weight=1.0 \
    algorithm.trajectory_reward_weight=0.1 \
    actor_rollout_ref.actor.use_ref=False \
    generative_critic.backend=deepseek_api \
    generative_critic.eval_enable=False \
    generative_critic.deepseek_model="${DEEPSEEK_MODEL}" \
    generative_critic.deepseek_api_key="${DEEPSEEK_API_KEY}" \
    generative_critic.deepseek_timeout="${CRITIC_TIMEOUT}" \
    generative_critic.deepseek_max_retries=3 \
    generative_critic.deepseek_max_concurrency="${CRITIC_MAX_CONCURRENCY}" \
    generative_critic.response_format=bool_only \
    generative_critic.true_score=1.0 \
    generative_critic.false_score=-1.0 \
    generative_critic.do_sample="${CRITIC_DO_SAMPLE}" \
    generative_critic.temperature="${CRITIC_TEMPERATURE}" \
    generative_critic.top_p="${CRITIC_TOP_P}" \
    generative_critic.max_new_tokens=2 \
    generative_critic.default_label_if_parse_fail=False \
    generative_critic.debug_print_samples=False \
    generative_critic.debug_max_print=0 \
    generative_critic.debug_max_prompt_chars=2048 \
    generative_critic.debug_max_output_chars=256 \
    agent_proxy.debug_turn_boundary=False \
    agent_proxy.max_turn="${MAX_TURN}" \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
    actor_rollout_ref.rollout.response_length="${ROLLOUT_RESPONSE_LENGTH}" \
    model_path="${ACTOR_MODEL_PATH}" \
    agent_proxy.enable_think=False \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_training_steps=2000 \
    trainer.resume_mode="${RESUME_MODE}" \
    algorithm.gamma=1 \
    algorithm.lam=0.95
