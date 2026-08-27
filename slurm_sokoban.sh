set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo"
# USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# W&B credentials must be supplied out-of-band.  Without a key, keep the
# run usable by writing an offline log that can be synced later.
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    echo "WANDB_API_KEY is unset; using W&B offline mode" >&2
fi

# Section 3.1&3.2 - General Observations


MKL_SERVICE_FORCE_INTEL=1 python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"0,1,2,3\" trainer.n_gpus_per_node=4 \
    trainer.experiment_name=sokoban $USE_PPO $USE_BASE \
    model_path=/ceph/home/muhan01/huggingfacemodels/Qwen3-0.6B \
    micro_batch_size_per_gpu=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.project_name=gen-ppo-new \
