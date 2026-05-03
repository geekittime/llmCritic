#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p /data/kangshijia/ray_tmp
export RAY_TMPDIR="${RAY_TMPDIR:-/data/kangshijia/ray_tmp}"
export PYTHONPATH="$(pwd):$(pwd)/verl:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-ragen}}"
export TOGETHER_NO_BANNER=1

# W&B settings copied from /home/kangshijia/wangbinyu/llm-gen-critic/train_sokoban_deepseek_critic.sh.
export WANDB_API_KEY="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"
export WANDB_ENTITY="MuLab-RL"
export WANDB_PROJECT="llm-gen-critic-deepseek"

ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/data/kangshijia/wangbinyu/models/Qwen3-0.6B}"
CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-/data/kangshijia/wangbinyu/models/Qwen3-4B}"
RUN_NAME="${RUN_NAME:-dev0319-sokoban-qwen3-0.6b-critic-qwen3-4b}"
# CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
N_GPUS="${N_GPUS:-4}"

# Trainable generative critic (RLVR) + actor label/outcome advantage.
USE_GEN_CRITIC="algorithm.use_label_outcome_advantage=True critic.enable=True generative_critic.enable=True generative_critic.train_enable=True"
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# Critic generation control:
# - If CRITIC_DO_SAMPLE=False, current vLLM reuse path forces greedy-like decoding (temperature=0).
# - If CRITIC_DO_SAMPLE=True, temperature/top_p/top_k are applied.
CRITIC_DO_SAMPLE="${CRITIC_DO_SAMPLE:-True}"
CRITIC_TEMPERATURE="${CRITIC_TEMPERATURE:-0.3}"
CRITIC_TOP_P="${CRITIC_TOP_P:-1.0}"
CRITIC_TOP_K="${CRITIC_TOP_K:--1}"

# Trainable generative critic RLVR rewards.
RLVR_FORMAT_REWARD="${RLVR_FORMAT_REWARD:-0.2}"
RLVR_LABEL_REWARD="${RLVR_LABEL_REWARD:-1.0}"
RLVR_LABEL_PENALTY="${RLVR_LABEL_PENALTY:--1.0}"
RLVR_PARSE_FAIL_PENALTY="${RLVR_PARSE_FAIL_PENALTY:--1.0}"

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"${CUDA_DEVICES}\" trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.logger=['console','wandb'] \
    ${USE_BASE} ${USE_GEN_CRITIC} \
    es_manager.train.env_groups=8 \
    es_manager.train.group_size=16 \
    es_manager.train.env_configs.n_groups=[8] \
    micro_batch_size_per_gpu=1 \
    ppo_mini_batch_size=16 \
    algorithm.adv_estimator=gae \
    algorithm.label_weight=5.0 \
    algorithm.outcome_weight=1.0 \
    generative_critic.backend=actor_rollout_vllm \
    generative_critic.use_chat_template=True \
    generative_critic.train_strategy=actor_ppo \
    generative_critic.train_temperature=1.0 \
    generative_critic.train_top_p=1.0 \
    generative_critic.train_top_k=-1 \
    generative_critic.train_max_new_tokens=256 \
    generative_critic.train_rollout_gpu_memory_utilization=0.35 \
    generative_critic.train_rollout_max_model_len=4096 \
    generative_critic.train_ppo_mini_batch_size=16 \
    generative_critic.train_ppo_micro_batch_size_per_gpu=1 \
    generative_critic.train_ppo_epochs=1 \
    generative_critic.rlvr_format_reward="${RLVR_FORMAT_REWARD}" \
    generative_critic.rlvr_label_reward="${RLVR_LABEL_REWARD}" \
    generative_critic.rlvr_label_penalty="${RLVR_LABEL_PENALTY}" \
    generative_critic.rlvr_parse_fail_penalty="${RLVR_PARSE_FAIL_PENALTY}" \
    generative_critic.model_path="${CRITIC_MODEL_PATH}" \
    generative_critic.train_model_path="${CRITIC_MODEL_PATH}" \
    generative_critic.do_sample="${CRITIC_DO_SAMPLE}" \
    generative_critic.temperature="${CRITIC_TEMPERATURE}" \
    generative_critic.top_p="${CRITIC_TOP_P}" \
    generative_critic.top_k="${CRITIC_TOP_K}" \
    generative_critic.max_new_tokens=256 \
    generative_critic.inference_batch_size=16 \
    generative_critic.default_label_if_parse_fail=False \
    generative_critic.debug_print_samples=True \
    generative_critic.debug_max_print=8 \
    generative_critic.debug_max_prompt_chars=2048 \
    generative_critic.debug_max_output_chars=700 \
    agent_proxy.debug_turn_boundary=True \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
    model_path="${ACTOR_MODEL_PATH}" \
    agent_proxy.enable_think=False \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    trainer.val_before_train=False \
    trainer.save_freq=2000 \
    trainer.test_freq=10 \
    trainer.total_training_steps=2000 \
    trainer.resume_mode=auto \
    algorithm.gamma=1 \
    algorithm.lam=0.95
