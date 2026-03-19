set -e

mkdir -p /mnt/kangshijia/ray_tmp
export RAY_TMPDIR=/mnt/kangshijia/ray_tmp

# Trainable generative critic (RLVR) + actor label/outcome advantage debug run
# Based on train_sokoban.sh with trainable generative critic enabled.

USE_GEN_CRITIC="algorithm.use_label_outcome_advantage=True critic.enable=True generative_critic.enable=True generative_critic.train_enable=True"
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# Critic generation control:
# - If CRITIC_DO_SAMPLE=False, current vLLM reuse path forces greedy-like decoding (temperature=0).
# - If CRITIC_DO_SAMPLE=True, temperature/top_p/top_k are applied.
CRITIC_DO_SAMPLE=True
CRITIC_TEMPERATURE=0.3
CRITIC_TOP_P=1.0
CRITIC_TOP_K=-1

# Trainable generative critic RLVR rewards
RLVR_FORMAT_REWARD=0.2
RLVR_LABEL_REWARD=1.0
RLVR_LABEL_PENALTY=-1.0
RLVR_PARSE_FAIL_PENALTY=-1.0

# 观测建议（W&B）：
# - gen_critic/train/success_source_field 应尽量接近 1
# - gen_critic/train/success_source_fallback 应尽量接近 0
# - gen_critic/train/label_acc, gen_critic/train/format_rate

wandb login --relogin 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

MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"1,2,3,4\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=sokoban-gen-critic-debug $USE_BASE $USE_GEN_CRITIC \
    es_manager.train.env_groups=8 es_manager.train.group_size=16 es_manager.train.env_configs.n_groups=[8]\
    micro_batch_size_per_gpu=4 \
    ppo_mini_batch_size=32 \
    algorithm.adv_estimator=gae \
    algorithm.label_weight=3.0 \
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
    generative_critic.train_ppo_mini_batch_size=32 \
    generative_critic.train_ppo_micro_batch_size_per_gpu=4 \
    generative_critic.train_ppo_epochs=1 \
    generative_critic.rlvr_format_reward=$RLVR_FORMAT_REWARD \
    generative_critic.rlvr_label_reward=$RLVR_LABEL_REWARD \
    generative_critic.rlvr_label_penalty=$RLVR_LABEL_PENALTY \
    generative_critic.rlvr_parse_fail_penalty=$RLVR_PARSE_FAIL_PENALTY \
    generative_critic.model_path=/mnt/kangshijia/models/Qwen3-1.7B \
    generative_critic.do_sample=$CRITIC_DO_SAMPLE \
    generative_critic.temperature=$CRITIC_TEMPERATURE \
    generative_critic.top_p=$CRITIC_TOP_P \
    generative_critic.top_k=$CRITIC_TOP_K \
    generative_critic.max_new_tokens=256 \
    generative_critic.inference_batch_size=32 \
    generative_critic.default_label_if_parse_fail=False \
    generative_critic.debug_print_samples=True \
    generative_critic.debug_max_print=8 \
    generative_critic.debug_max_prompt_chars=2048 \
    generative_critic.debug_max_output_chars=700 \
    agent_proxy.debug_turn_boundary=True \
    actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
    model_path=/mnt/kangshijia/models/Qwen3-0.6B \
    agent_proxy.enable_think=False \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    trainer.val_before_train=False \
    trainer.save_freq=500 \
    trainer.test_freq=10 \
    trainer.total_training_steps=2000 \
    trainer.resume_mode=auto \
    algorithm.gamma=1 \
    algorithm.lam=0.95 \
    trainer.project_name=gen-ppo-new

