"""
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import os
import uuid
import ray
import torch
import numpy as np
from typing import Optional
from collections import defaultdict
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import time
from pprint import pprint
from copy import deepcopy

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from ragen.trainer.core_algos import compute_grpo_outcome_advantage
from ragen.trainer.core_algos import compute_turn_gae_advantage_return
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


from verl.trainer.ppo.ray_trainer import ResourcePoolManager, compute_response_mask, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

import torch
from verl.utils.torch_functional import masked_mean
from transformers import AutoTokenizer

from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger
from ragen.trainer.rollout_filter import build_rollout_filter
from ragen.trainer.generative_critic import FrozenGenerativeCritic

from tensordict import TensorDict


def _build_left_padded_tensors(tokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids


def _apply_chat_template_batch(tokenizer, texts: list[str], use_chat_template: bool = True) -> list[str]:
    if not use_chat_template:
        return texts

    formatted_texts: list[str] = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except Exception:
            formatted = text
        formatted_texts.append(formatted)
    return formatted_texts


def adjust_batch(batch: DataProto, size_divisor: int, mode: str = "copy") -> DataProto:
    """
    Adjust batch size to be divisible by size_divisor.

    Args:
        batch: The DataProto batch to adjust
        size_divisor: The number that batch size should be divisible by
        mode: "copy" to duplicate samples, "delete" to remove samples

    Returns:
        Adjusted DataProto with batch size divisible by size_divisor
    """
    bs = len(batch.batch) if hasattr(batch.batch, '__len__') else batch.batch.batch_size[0]
    remainder = bs % size_divisor

    if remainder == 0:
        return batch

    if mode == "delete":
        # Remove samples to make it divisible
        remove_indices = np.random.choice(bs, remainder, replace=False)
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool)
        if batch.batch is not None:
            tensor_data = batch.batch[keep_mask_tensor]
        else:
            tensor_data = None

        non_tensor_data = {}
        if batch.non_tensor_batch is not None:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    non_tensor_data[key] = val[keep_mask]
                elif isinstance(val, list):
                    non_tensor_data[key] = [v for v, m in zip(val, keep_mask) if m]
                else:
                    non_tensor_data[key] = val

        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=batch.meta_info)

    elif mode == "copy":
        # Duplicate samples to make it divisible
        to_add = size_divisor - remainder
        if to_add > bs:
            dup_indices = np.random.choice(bs, to_add, replace=True)
        else:
            dup_indices = np.random.choice(bs, to_add, replace=False)

        # Create duplicated batch using TensorDict concat
        dup_indices_tensor = torch.tensor(dup_indices, dtype=torch.long)
        if batch.batch is not None:
            dup_tensor_data = batch.batch[dup_indices_tensor]
            # Use TensorDict's cat method
            tensor_data = TensorDict.cat([batch.batch, dup_tensor_data], dim=0)
        else:
            tensor_data = None

        non_tensor_data = {}
        if batch.non_tensor_batch is not None:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    dup_val = val[dup_indices]
                    non_tensor_data[key] = np.concatenate([val, dup_val], axis=0)
                elif isinstance(val, list):
                    dup_val = [val[i] for i in dup_indices]
                    non_tensor_data[key] = val + dup_val
                else:
                    non_tensor_data[key] = val

        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=batch.meta_info)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'copy' or 'delete'.")

    return adjusted_batch


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, bi_level_gae=False, high_level_gamma=1.0):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        if bi_level_gae:
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                loss_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
                high_level_gamma=high_level_gamma,
            )
        elif "turn_value_mask" in data.batch and "turn_ids" in data.batch and "turn_value_ids" in data.batch:
            advantages, returns = compute_turn_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                turn_ids=data.batch["turn_ids"],
                turn_value_mask=data.batch["turn_value_mask"],
                turn_value_ids=data.batch["turn_value_ids"],
                gamma=gamma,
                lam=lam,
            )
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        # Pass episode_ids for deduplication in single_turn/limited_multi_turn mode
        episode_ids = data.non_tensor_batch.get("episode_ids", None)
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            episode_ids=episode_ids,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


class RayAgentTrainer(VerlRayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        # do not use the original val logger, but use this here
        self.generations_logger = GenerationsLogger()
        self.generative_critic = FrozenGenerativeCritic(config)
        self.use_trainable_generative_critic = bool(
            OmegaConf.select(config, "generative_critic.train_enable", default=False)
        )
        self.generative_critic_train_tokenizer = self.tokenizer

        if self.use_trainable_generative_critic:
            actor_model_path = OmegaConf.select(
                config,
                "actor_rollout_ref.model.path",
                default=OmegaConf.select(config, "model_path", default=None),
            )
            train_model_path = OmegaConf.select(config, "generative_critic.train_model_path", default=actor_model_path)
            if train_model_path is not None and actor_model_path is not None and train_model_path != actor_model_path:
                trust_remote_code = bool(
                    OmegaConf.select(config, "generative_critic.trust_remote_code", default=True)
                )
                self.generative_critic_train_tokenizer = AutoTokenizer.from_pretrained(
                    train_model_path,
                    trust_remote_code=trust_remote_code,
                )
                self.generative_critic_train_tokenizer.padding_side = "left"
                if self.generative_critic_train_tokenizer.pad_token_id is None:
                    self.generative_critic_train_tokenizer.pad_token = self.generative_critic_train_tokenizer.eos_token
                print(
                    "[GEN_CRITIC INIT] loaded dedicated critic tokenizer "
                    f"train_model_path={train_model_path} actor_model_path={actor_model_path}"
                )

    def _generate_with_actor_rollout_vllm(self, prompts: list[str], sampling_overrides: dict) -> list[str]:
        """Reuse actor rollout vLLM engine for generative-critic inference."""
        if len(prompts) == 0:
            return []

        if not hasattr(self, "agent_proxy") or self.agent_proxy is None:
            raise RuntimeError("agent_proxy is not initialized before generative critic inference")

        actor_wg = self.agent_proxy.actor_wg
        if not hasattr(actor_wg, "generate_sequences"):
            raise RuntimeError("actor worker group has no generate_sequences for vLLM reuse")

        use_chat_template = bool(OmegaConf.select(self.config, "generative_critic.use_chat_template", default=True))
        formatted_prompts = _apply_chat_template_batch(self.tokenizer, prompts, use_chat_template=use_chat_template)
        input_ids, attention_mask, position_ids = _build_left_padded_tensors(self.tokenizer, formatted_prompts)

        lm_inputs = DataProto()
        lm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": input_ids[:, 1:],
            },
            batch_size=input_ids.shape[0],
        )
        lm_inputs.non_tensor_batch = {
            "env_ids": np.arange(len(prompts), dtype=int),
            "group_ids": np.zeros(len(prompts), dtype=int),
        }

        val_kwargs = OmegaConf.to_container(self.config.actor_rollout_ref.rollout.val_kwargs, resolve=True)
        do_sample = bool(sampling_overrides.get("do_sample", False))
        temperature = float(sampling_overrides.get("temperature", 0.0))
        top_p = float(sampling_overrides.get("top_p", 1.0))
        top_k = int(sampling_overrides.get("top_k", -1))
        max_tokens = int(sampling_overrides.get("max_tokens", 128))

        # Keep do_sample=True so vLLM rollout will not override kwargs,
        # and emulate greedy decoding with temperature=0 when needed.
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            top_k = -1

        sampling_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n": 1,
        }
        lm_inputs.meta_info = {
            "mode": "singleturn",
            "skip_generation": False,
            "do_sample": True,
            "validate": False,
            "sampling_kwargs": sampling_kwargs,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        lm_outputs = self.agent_proxy.generate_sequences(lm_inputs)

        if lm_outputs.batch is not None and "responses" in lm_outputs.batch.keys():
            return self.tokenizer.batch_decode(lm_outputs.batch["responses"], skip_special_tokens=True)
        if "response_texts" in lm_outputs.non_tensor_batch:
            return lm_outputs.non_tensor_batch["response_texts"].tolist()
        raise RuntimeError("vLLM generative critic path cannot find responses in output")

    @staticmethod
    def _compute_outcome_tensor(token_level_scores: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Broadcast per-trajectory outcome score to all response tokens."""
        outcome = (token_level_scores * response_mask).sum(dim=-1, keepdim=True)
        return outcome * response_mask

    def _build_generative_critic_prompt_batch(self, prompts: list[str]) -> DataProto:
        critic_tokenizer = self.generative_critic_train_tokenizer
        use_chat_template = bool(OmegaConf.select(self.config, "generative_critic.use_chat_template", default=True))
        formatted_prompts = _apply_chat_template_batch(critic_tokenizer, prompts, use_chat_template=use_chat_template)
        input_ids, attention_mask, position_ids = _build_left_padded_tensors(critic_tokenizer, formatted_prompts)

        batch = DataProto()
        batch.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": input_ids[:, 1:],
            },
            batch_size=input_ids.shape[0],
        )
        batch.non_tensor_batch = {
            "env_ids": np.arange(len(prompts), dtype=int),
            "group_ids": np.zeros(len(prompts), dtype=int),
            "uid": np.arange(len(prompts), dtype=int),
        }

        train_temperature = float(OmegaConf.select(self.config, "generative_critic.train_temperature", default=1.0))
        train_max_new_tokens = int(OmegaConf.select(self.config, "generative_critic.train_max_new_tokens", default=256))
        train_top_p = float(
            OmegaConf.select(
                self.config,
                "generative_critic.train_top_p",
                default=OmegaConf.select(self.config, "generative_critic.top_p", default=1.0),
            )
        )
        train_top_k = int(
            OmegaConf.select(
                self.config,
                "generative_critic.train_top_k",
                default=OmegaConf.select(self.config, "generative_critic.top_k", default=-1),
            )
        )

        batch.meta_info = {
            "mode": "singleturn",
            "skip_generation": False,
            "do_sample": True,
            "validate": False,
            "sampling_kwargs": {
                "max_tokens": train_max_new_tokens,
                "temperature": train_temperature,
                "top_p": train_top_p,
                "top_k": train_top_k,
                "n": 1,
            },
            "eos_token_id": critic_tokenizer.eos_token_id,
            "pad_token_id": critic_tokenizer.pad_token_id,
            "temperature": train_temperature,
        }
        return batch

    def _infer_labels_with_trainable_critic(
        self,
        messages_list: list[list[dict]],
        turn_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict, float]:
        """Use trainable critic worker (current parameters) to label actor actions."""
        if turn_ids is None:
            return torch.zeros(0), {
                "gen_critic/enabled": 1.0,
                "gen_critic/used_trainable_critic": 1.0,
                "gen_critic/num_prompts": 0.0,
                "gen_critic/parse_fail_rate": 1.0,
                "gen_critic/true_rate": 0.0,
            }, 0.0

        label_tensor = torch.zeros_like(turn_ids, dtype=torch.float32)

        # Build prompt items from current trajectory content.
        prompt_items = self.generative_critic.build_judge_prompts(messages_list=messages_list, turn_ids=turn_ids)
        if len(prompt_items) == 0:
            return label_tensor, {
                "gen_critic/enabled": 1.0,
                "gen_critic/used_trainable_critic": 1.0,
                "gen_critic/num_prompts": 0.0,
                "gen_critic/parse_fail_rate": 0.0,
                "gen_critic/true_rate": 0.0,
            }, 0.0

        prompts = [item.prompt for item in prompt_items]
        gen_inputs = self._build_generative_critic_prompt_batch(prompts)
        worker_divisor = int(self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes)
        gen_inputs, pad_size = pad_dataproto_to_divisor(gen_inputs, size_divisor=worker_divisor)
        critic_gen_start = time.time()
        critic_rollout = self.critic_wg.generate_critic_sequences(gen_inputs)
        critic_gen_time = time.time() - critic_gen_start
        critic_rollout = unpad_dataproto(critic_rollout, pad_size)

        if critic_rollout.batch is not None and "responses" in critic_rollout.batch.keys():
            outputs = self.generative_critic_train_tokenizer.batch_decode(
                critic_rollout.batch["responses"],
                skip_special_tokens=True,
            )
        elif "response_texts" in critic_rollout.non_tensor_batch:
            outputs = critic_rollout.non_tensor_batch["response_texts"].tolist()
        else:
            raise RuntimeError("Trainable critic inference output has no textual responses")

        parse_fail = 0
        true_count = 0
        for item, text in zip(prompt_items, outputs, strict=True):
            parsed = self.generative_critic.parse_label(text)
            if parsed is None:
                parse_fail += 1
                parsed = self.generative_critic.default_label_if_parse_fail

            value = 1.0 if parsed else 0.0
            if value > 0.5:
                true_count += 1
            mask = turn_ids[item.sample_index] == item.turn_id
            label_tensor[item.sample_index, mask] = value

        n = float(len(prompt_items))
        metrics = {
            "gen_critic/enabled": 1.0,
            "gen_critic/used_trainable_critic": 1.0,
            "gen_critic/num_prompts": n,
            "gen_critic/parse_fail_rate": float(parse_fail) / max(n, 1.0),
            "gen_critic/true_rate": float(true_count) / max(n, 1.0),
        }
        return label_tensor, metrics, critic_gen_time

    def _train_generative_critic_rlvr(self, actor_batch: DataProto) -> tuple[dict, float]:
        if "messages_list" not in actor_batch.non_tensor_batch:
            return {"gen_critic/train/skipped_missing_fields": 1.0}, 0.0

        messages_list = actor_batch.non_tensor_batch["messages_list"].tolist()
        if "trajectory_success" in actor_batch.non_tensor_batch:
            trajectory_success = torch.tensor(
                actor_batch.non_tensor_batch["trajectory_success"],
                dtype=torch.float32,
            ) > 0.5
            success_source = "trajectory_success_field"
        else:
            trajectory_success = self.generative_critic.trajectory_success_from_metrics(messages_list)

        if trajectory_success is None:
            if "token_level_scores" not in actor_batch.batch.keys() and "rm_scores" in actor_batch.batch.keys():
                actor_batch.batch["token_level_scores"] = actor_batch.batch["rm_scores"]
            trajectory_success = self.generative_critic.trajectory_success_from_scores(
                token_level_scores=actor_batch.batch["token_level_scores"],
                response_mask=actor_batch.batch["response_mask"],
            )
            success_source = "score_fallback"
        elif "trajectory_success" not in actor_batch.non_tensor_batch:
            success_source = "metrics"

        train_items = self.generative_critic.build_train_judge_prompts(
            messages_list=messages_list,
            turn_ids=actor_batch.batch["turn_ids"] if "turn_ids" in actor_batch.batch else None,
            trajectory_success=trajectory_success,
        )
        if len(train_items) == 0:
            return {"gen_critic/train/num_samples": 0.0}, 0.0

        prompts = [item.prompt for item in train_items]
        targets = [item.target_label for item in train_items]
        gen_inputs = self._build_generative_critic_prompt_batch(prompts)

        worker_divisor = int(self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes)
        gen_inputs, pad_size = pad_dataproto_to_divisor(gen_inputs, size_divisor=worker_divisor)
        critic_gen_start = time.time()
        critic_rollout = self.critic_wg.generate_critic_sequences(gen_inputs)
        critic_gen_time = time.time() - critic_gen_start
        critic_rollout = unpad_dataproto(critic_rollout, pad_size)
        if critic_rollout.batch is not None and "responses" in critic_rollout.batch.keys():
            outputs = self.generative_critic_train_tokenizer.batch_decode(
                critic_rollout.batch["responses"],
                skip_special_tokens=True,
            )
        elif "response_texts" in critic_rollout.non_tensor_batch:
            outputs = critic_rollout.non_tensor_batch["response_texts"].tolist()
        else:
            raise RuntimeError("Generative critic rollout output has no textual responses")

        scalar_rewards, rlvr_metrics = self.generative_critic.compute_rlvr_scalar_rewards(outputs=outputs, targets=targets)

        # Confusion matrix counts for critic predictions vs ground-truth labels.
        # rows: predict {true,false}, cols: target {true,false}
        pred_true_target_true = 0
        pred_true_target_false = 0
        pred_false_target_true = 0
        pred_false_target_false = 0
        valid_confusion_samples = 0
        for text, target in zip(outputs, targets, strict=True):
            if not self.generative_critic.has_strict_label_format(text):
                continue
            pred = self.generative_critic.parse_label(text)
            if pred is None:
                continue
            valid_confusion_samples += 1
            if pred and target:
                pred_true_target_true += 1
            elif pred and (not target):
                pred_true_target_false += 1
            elif (not pred) and target:
                pred_false_target_true += 1
            else:
                pred_false_target_false += 1

        response_mask = compute_response_mask(critic_rollout)
        critic_rollout.batch["response_mask"] = response_mask
        token_rewards = self.generative_critic.expand_scalar_rewards_to_token(
            scalar_rewards=scalar_rewards,
            response_mask=response_mask,
        )
        critic_rollout.batch["token_level_rewards"] = token_rewards
        critic_rollout.batch["advantages"] = token_rewards
        critic_rollout.batch["returns"] = token_rewards

        critic_rollout, pad_size = pad_dataproto_to_divisor(critic_rollout, size_divisor=worker_divisor)
        old_log_prob = self.critic_wg.compute_critic_log_prob(critic_rollout)
        critic_entropy = None
        if "entropys" in old_log_prob.batch.keys():
            padded_response_mask = critic_rollout.batch["response_mask"]
            entropy_agg = agg_loss(
                loss_mat=old_log_prob.batch["entropys"],
                loss_mask=padded_response_mask,
                loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
            )
            critic_entropy = float(entropy_agg.detach().item())
            old_log_prob.batch.pop("entropys")
        critic_rollout = critic_rollout.union(old_log_prob)

        critic_rollout.meta_info["global_token_num"] = torch.sum(critic_rollout.batch["attention_mask"], dim=-1).tolist()
        critic_rollout.meta_info["temperature"] = float(
            OmegaConf.select(self.config, "generative_critic.train_temperature", default=1.0)
        )

        critic_output = self.critic_wg.update_generative_critic(critic_rollout)
        critic_output = unpad_dataproto(critic_output, pad_size)
        raw_metrics = reduce_metrics(critic_output.meta_info["metrics"])

        prefixed_metrics = {}
        for key, value in raw_metrics.items():
            if key.startswith("actor/"):
                prefixed_metrics[f"gen_critic/{key[len('actor/'):]}"] = value
            else:
                prefixed_metrics[f"gen_critic/{key}"] = value

        prefixed_metrics.update(rlvr_metrics)
        prefixed_metrics["gen_critic/train/num_samples"] = float(valid_confusion_samples)
        prefixed_metrics["gen_critic/train/confusion/num_skipped_non_strict"] = float(len(train_items) - valid_confusion_samples)
        prefixed_metrics["gen_critic/train/success_rate"] = float(trajectory_success.float().mean().item())
        prefixed_metrics["gen_critic/train/confusion/pred_true_target_true"] = float(pred_true_target_true)
        prefixed_metrics["gen_critic/train/confusion/pred_true_target_false"] = float(pred_true_target_false)
        prefixed_metrics["gen_critic/train/confusion/pred_false_target_true"] = float(pred_false_target_true)
        prefixed_metrics["gen_critic/train/confusion/pred_false_target_false"] = float(pred_false_target_false)
        if success_source == "trajectory_success_field":
            prefixed_metrics["gen_critic/train/success_source"] = 2.0
        elif success_source == "metrics":
            prefixed_metrics["gen_critic/train/success_source"] = 1.0
        else:
            prefixed_metrics["gen_critic/train/success_source"] = 0.0
        prefixed_metrics["gen_critic/train/success_source_field"] = 1.0 if success_source == "trajectory_success_field" else 0.0
        prefixed_metrics["gen_critic/train/success_source_metrics"] = 1.0 if success_source == "metrics" else 0.0
        prefixed_metrics["gen_critic/train/success_source_fallback"] = 1.0 if success_source == "score_fallback" else 0.0
        if critic_entropy is not None:
            prefixed_metrics["gen_critic/train/entropy"] = critic_entropy
        return prefixed_metrics, critic_gen_time

    def _compute_critic_confusion_eval_metrics(self, batch: DataProto) -> dict:
        """Compute confusion-matrix-style counts for critic predictions in eval/validation.

        rows: predict {true,false}; cols: target {true,false}
        """
        if "messages_list" not in batch.non_tensor_batch:
            return {"gen_critic/eval/confusion/skipped_missing_messages": 1.0}

        messages_list = batch.non_tensor_batch["messages_list"].tolist()
        if "trajectory_success" in batch.non_tensor_batch:
            trajectory_success = torch.tensor(batch.non_tensor_batch["trajectory_success"], dtype=torch.float32) > 0.5
        else:
            trajectory_success = self.generative_critic.trajectory_success_from_scores(
                token_level_scores=batch.batch.get("token_level_scores", batch.batch.get("rm_scores")),
                response_mask=batch.batch.get("response_mask", batch.batch.get("loss_mask")),
            )

        turn_ids = batch.batch["turn_ids"] if "turn_ids" in batch.batch else None
        eval_items = self.generative_critic.build_train_judge_prompts(
            messages_list=messages_list,
            turn_ids=turn_ids,
            trajectory_success=trajectory_success,
        )
        if len(eval_items) == 0:
            return {"gen_critic/eval/confusion/num_samples": 0.0}

        prompts = [item.prompt for item in eval_items]
        targets = [item.target_label for item in eval_items]

        if self.use_trainable_generative_critic and self.use_critic:
            gen_inputs = self._build_generative_critic_prompt_batch(prompts)
            worker_divisor = int(self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes)
            gen_inputs, pad_size = pad_dataproto_to_divisor(gen_inputs, size_divisor=worker_divisor)
            critic_rollout = self.critic_wg.generate_critic_sequences(gen_inputs)
            critic_rollout = unpad_dataproto(critic_rollout, pad_size)
            if critic_rollout.batch is not None and "responses" in critic_rollout.batch.keys():
                outputs = self.generative_critic_train_tokenizer.batch_decode(
                    critic_rollout.batch["responses"],
                    skip_special_tokens=True,
                )
            elif "response_texts" in critic_rollout.non_tensor_batch:
                outputs = critic_rollout.non_tensor_batch["response_texts"].tolist()
            else:
                return {"gen_critic/eval/confusion/skipped_no_outputs": 1.0}
        else:
            # frozen critic path
            turn_ids_for_infer = turn_ids
            if turn_ids_for_infer is None:
                return {"gen_critic/eval/confusion/skipped_missing_turn_ids": 1.0}
            _, _, outputs = self.generative_critic.infer_turn_labels(
                messages_list=messages_list,
                turn_ids=turn_ids_for_infer,
            )

        pred_true_target_true = 0
        pred_true_target_false = 0
        pred_false_target_true = 0
        pred_false_target_false = 0
        valid_confusion_samples = 0
        for text, target in zip(outputs, targets, strict=True):
            if not self.generative_critic.has_strict_label_format(text):
                continue
            pred = self.generative_critic.parse_label(text)
            if pred is None:
                continue
            valid_confusion_samples += 1
            if pred and target:
                pred_true_target_true += 1
            elif pred and (not target):
                pred_true_target_false += 1
            elif (not pred) and target:
                pred_false_target_true += 1
            else:
                pred_false_target_false += 1

        return {
            "gen_critic/eval/confusion/num_samples": float(valid_confusion_samples),
            "gen_critic/eval/confusion/num_skipped_non_strict": float(len(eval_items) - valid_confusion_samples),
            "gen_critic/eval/confusion/pred_true_target_true": float(pred_true_target_true),
            "gen_critic/eval/confusion/pred_true_target_false": float(pred_true_target_false),
            "gen_critic/eval/confusion/pred_false_target_true": float(pred_false_target_true),
            "gen_critic/eval/confusion/pred_false_target_false": float(pred_false_target_false),
        }

    def _build_trainable_generative_critic_worker_config(self):
        """Build critic worker config by reusing actor config with train overrides."""
        critic_cfg = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))

        train_model_path = OmegaConf.select(self.config, "generative_critic.train_model_path", default=None)
        if train_model_path is not None:
            critic_cfg.model.path = train_model_path

        train_micro_bsz = int(OmegaConf.select(self.config, "generative_critic.train_ppo_micro_batch_size_per_gpu", default=self.config.micro_batch_size_per_gpu))
        train_mini_bsz = int(OmegaConf.select(self.config, "generative_critic.train_ppo_mini_batch_size", default=self.config.ppo_mini_batch_size))
        train_epochs = int(OmegaConf.select(self.config, "generative_critic.train_ppo_epochs", default=self.config.actor_rollout_ref.actor.ppo_epochs))
        train_lr = float(OmegaConf.select(self.config, "generative_critic.train_lr", default=self.config.actor_rollout_ref.actor.optim.lr))
        train_temp = float(OmegaConf.select(self.config, "generative_critic.train_temperature", default=1.0))
        train_max_new_tokens = int(OmegaConf.select(self.config, "generative_critic.train_max_new_tokens", default=256))

        critic_cfg.actor.ppo_micro_batch_size_per_gpu = train_micro_bsz
        critic_cfg.actor.ppo_mini_batch_size = train_mini_bsz
        critic_cfg.actor.ppo_epochs = train_epochs
        critic_cfg.actor.optim.lr = train_lr
        # Generative critic RLVR currently does not provide ref_log_prob; disable KL-loss path
        # to avoid requiring a reference-policy tensor in critic update batches.
        critic_cfg.actor.use_kl_loss = False
        critic_cfg.actor.kl_loss_coef = 0.0
        critic_cfg.actor.use_ref = False
        # Keep critic update independent from rollout-logprob/TIS requirements.
        critic_cfg.actor.tis_imp_ratio_cap = -1
        critic_cfg.rollout.calculate_log_probs = False
        critic_cfg.rollout.temperature = train_temp
        # IMPORTANT: vLLM rollout pads each shard to at least config.response_length.
        # If sampling max_tokens is larger than response_length, different shards may end up
        # with different local response widths and fail at DataProto.concat.
        critic_cfg.rollout.response_length = train_max_new_tokens
        critic_cfg.rollout.log_prob_micro_batch_size_per_gpu = train_micro_bsz
        # Avoid OOM when actor and critic each create their own vLLM rollout engine.
        critic_cfg.rollout.gpu_memory_utilization = float(
            OmegaConf.select(
                self.config,
                "generative_critic.train_rollout_gpu_memory_utilization",
                default=critic_cfg.rollout.gpu_memory_utilization,
            )
        )
        critic_cfg.rollout.max_model_len = int(
            OmegaConf.select(
                self.config,
                "generative_critic.train_rollout_max_model_len",
                default=critic_cfg.rollout.max_model_len,
            )
        )
        # Sleep mode conflicts with multiple vLLM engines in one process.
        critic_cfg.rollout.free_cache_engine = False
        return critic_cfg

        
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
        # val_start = 100000
        # self.train_seeds = [seed for seed in range(0, self.config.trainer.total_training_steps * 1000, 1000)]
        # self.val_seeds = [seed for seed in range(val_start, val_start + self.config.trainer.validation_steps)]

    def init_agent_proxy(self):
        self.agent_proxy = LLMAgentProxy(
            config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer
        )
        if self.generative_critic.enabled and self.generative_critic.backend in {
            "actor_rollout_vllm",
            "vllm_actor_rollout",
            "actor_vllm",
            "vllm",
        }:
            self.generative_critic.set_generate_fn(self._generate_with_actor_rollout_vllm)
            print("[GEN_CRITIC INIT] backend=actor_rollout_vllm (reuse actor rollout engine)")

    def _maybe_log_generations(self, inputs, outputs, scores, _type="val"):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.generations_to_log_to_wandb[_type]

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.generations_logger.log(self.config.trainer.logger, samples, self.global_steps, _type)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            import time
            start_time = time.time()
            test_batch = self.agent_proxy.rollout(test_gen_batch, val=True)
            end_time = time.time()
            print(f"validation generation time: {end_time - start_time} seconds")
            for key, value in test_batch.meta_info["metrics"].items():
                if "val-env/" + key not in env_metric_dict:
                    env_metric_dict["val-env/" + key] = []
                env_metric_dict["val-env/" + key].append(value)

            # Store original inputs and outputs
            batch_size = test_batch.batch["input_ids"].shape[0]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            # Handle single_turn/limited_multi_turn mode: group messages by episode
            context_window_mode = getattr(self.config.agent_proxy, "context_window_mode", "full")
            is_turn_level_mode = context_window_mode in ("single_turn", "limited_multi_turn")
            if is_turn_level_mode and "messages_list" in test_batch.non_tensor_batch:
                # Group samples by episode_id to reconstruct episodes
                episode_ids = test_batch.non_tensor_batch["episode_ids"]
                messages_list = test_batch.non_tensor_batch["messages_list"]

                # Find unique episodes and their samples
                unique_groups = []
                group_to_indices = {}
                for i, eid in enumerate(episode_ids):
                    if eid not in group_to_indices:
                        unique_groups.append(eid)
                        group_to_indices[eid] = []
                    group_to_indices[eid].append(i)

                # Create grouped outputs
                grouped_inputs = []
                grouped_outputs = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Combine all messages from this episode
                    episode_output = ""
                    for idx in indices:
                        msgs = messages_list[idx]
                        for msg in msgs:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            episode_output += f"[{role}]\n{content}\n\n"
                    grouped_inputs.append("")
                    grouped_outputs.append(episode_output.strip())

                sample_inputs.extend(grouped_inputs)
                sample_outputs.extend(grouped_outputs)
            else:
                input_texts = ["" for _ in range(batch_size)]
                sample_inputs.extend(input_texts)
                sample_outputs.extend(output_texts)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            test_batch.batch["token_level_scores"] = reward_tensor
            if "loss_mask" in test_batch.batch:
                test_batch.batch["response_mask"] = test_batch.batch["loss_mask"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            if self.generative_critic.enabled:
                critic_eval_metrics = self._compute_critic_confusion_eval_metrics(test_batch)
                for k, v in critic_eval_metrics.items():
                    if k not in env_metric_dict:
                        env_metric_dict[k] = []
                    env_metric_dict[k].append(v)

            # Group scores by episode if turn-level mode
            if is_turn_level_mode and "messages_list" in test_batch.non_tensor_batch:
                grouped_scores = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Use the first turn's score (all turns have same episode reward)
                    episode_score = scores[indices[0]]
                    grouped_scores.append(episode_score)
                sample_scores.extend(grouped_scores)
                reward_extra_infos_dict["reward"].extend(grouped_scores)
            else:
                sample_scores.extend(scores)
                reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Get data sources and group if needed
            data_sources_batch = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            if is_turn_level_mode and "messages_list" in test_batch.non_tensor_batch:
                # Group data sources by episode
                grouped_data_sources = [data_sources_batch[group_to_indices[gid][0]] for gid in unique_groups]
                data_source_lst.append(grouped_data_sources)
            else:
                data_source_lst.append(data_sources_batch)

        self._maybe_log_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, _type="val")

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = reduce_metrics(env_metric_dict)

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        if self.use_trainable_generative_critic:
            self.resource_pool_manager.create_resource_pool()
            self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

            if self.hybrid_engine:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRollout],
                    config=self.config.actor_rollout_ref,
                    role="actor_rollout",
                )
                self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
            else:
                raise NotImplementedError

            if self.use_critic:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
                critic_cfg = self._build_trainable_generative_critic_worker_config()
                critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
                self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

            if self.use_reference_policy:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
                ref_policy_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RefPolicy],
                    config=self.config.actor_rollout_ref,
                    role="ref",
                )
                self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

            if self.use_rm:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
                self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

            all_wg = {}
            wg_kwargs = {}
            if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
                wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
            if OmegaConf.select(self.config.global_profiler, "steps") is not None:
                wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
                if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                    assert (
                        OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                        is not None
                    ), "worker_nsight_options must be set when using nsys with profile_steps"
                    wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                        OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    )
            wg_kwargs["device_name"] = self.device_name

            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs,
                )
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)

            if self.use_critic:
                self.critic_wg = all_wg["critic"]
                self.critic_wg.init_model()

            if self.use_reference_policy and not self.ref_in_actor:
                self.ref_policy_wg = all_wg["ref"]
                self.ref_policy_wg.init_model()

            self.rm_wg = None
            if self.use_rm:
                self.rm_wg = all_wg["rm"]
                self.rm_wg.init_model()

            self.actor_rollout_wg = all_wg["actor_rollout"]
            self.actor_rollout_wg.init_model()

            self.async_rollout_mode = False
            if self.config.actor_rollout_ref.rollout.mode == "async":
                from verl.experimental.agent_loop import AgentLoopManager

                self.async_rollout_mode = True
                self.async_rollout_manager = AgentLoopManager(
                    config=self.config,
                    worker_group=self.actor_rollout_wg,
                    rm_wg=self.rm_wg,
                )
        else:
            super().init_workers()

        # create rollout filter
        rollout_cfg = self.config.actor_rollout_ref.rollout
        rollout_metric = getattr(rollout_cfg, "rollout_filter_metric", "reward_variance")
        self.rollout_filter = build_rollout_filter(
            ratio=rollout_cfg.rollout_filter_ratio,
            filter_type=rollout_cfg.rollout_filter_type,
            num_groups=self.config.es_manager.train.env_groups,
            group_size=self.config.es_manager.train.group_size,
            metric=rollout_metric,
            compute_log_prob=self.actor_rollout_wg.compute_log_prob,
        )


    def _save_checkpoint(self):
        """ 
        Different from VerlRayPPOTrainer, we have no dataloader so we won"t save it. Other logic is the same.
        """
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
         to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        def _process_batch_for_logging(batch):
            inputs_raw = batch.batch["input_ids"]
            inputs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in inputs_raw]
            outputs = [""] * len(inputs)
            scores = batch.batch["rm_scores"].sum(-1).cpu().tolist()

            # Group by episode if turn-level mode
            context_window_mode = getattr(self.config.agent_proxy, "context_window_mode", "full")
            is_turn_level_mode = context_window_mode in ("single_turn", "limited_multi_turn")
            if is_turn_level_mode and "messages_list" in batch.non_tensor_batch:
                episode_ids = batch.non_tensor_batch["episode_ids"]
                messages_list = batch.non_tensor_batch["messages_list"]

                # Find unique episodes and their samples
                unique_groups = []
                group_to_indices = {}
                for i, eid in enumerate(episode_ids):
                    if eid not in group_to_indices:
                        unique_groups.append(eid)
                        group_to_indices[eid] = []
                    group_to_indices[eid].append(i)

                # Create grouped outputs
                grouped_inputs = []
                grouped_outputs = []
                grouped_scores = []
                for gid in unique_groups:
                    indices = group_to_indices[gid]
                    # Combine all messages from this episode
                    episode_output = ""
                    for idx in indices:
                        msgs = messages_list[idx]
                        for msg in msgs:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            episode_output += f"[{role}]\n{content}\n\n"
                    grouped_inputs.append("")
                    grouped_outputs.append(episode_output.strip())
                    grouped_scores.append(scores[indices[0]])

                return grouped_inputs, grouped_outputs, grouped_scores

            return inputs, outputs, scores

        import time
        self.start_time = time.time()
        for step in range(self.total_training_steps):
            # metrics = {}
            timing_raw = {}
            critic_generation_time = 0.0

            batch: DataProto = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw):
                    batch = self.agent_proxy.rollout(batch, val=False)
                    critic_train_batch = deepcopy(batch) if self.use_trainable_generative_critic else None

                    # Filter first, then adjust batch size
                    batch, metrics = self.rollout_filter.filter(batch)

                    # Adjust batch size to be divisible by num_groups, ppo_mini_batch_size, and n_gpus
                    num_groups = self.config.es_manager.train.env_groups
                    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    n_gpus = self.config.trainer.n_gpus_per_node
                    size_divisor = np.lcm.reduce([num_groups, ppo_mini_batch_size, n_gpus])
                    adjust_mode = getattr(self.config.agent_proxy, "batch_adjust_mode", "copy")
                    batch = adjust_batch(batch, size_divisor, mode=adjust_mode)

                    # Record batch and mini-batch statistics
                    batch_size = batch.batch["input_ids"].shape[0]
                    num_mini_batches = batch_size // ppo_mini_batch_size
                    metrics.update({
                        "train/batch_size": batch_size,
                        "train/num_mini_batches": num_mini_batches,
                    })
                    metrics.update({"train/" + key: value for key, value in batch.meta_info["metrics"].items()})

                    inputs, outputs, scores = _process_batch_for_logging(batch)
                    # self._maybe_log_generations(inputs=inputs, outputs=outputs, scores=scores, _type="train")

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # TODO: check if this is correct. Not tested yer
                    logger.log("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")
                    exit()
                    with marked_timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.agent_proxy.rollout(gen_baseline_batch, val=False)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)

                # NOTE reward normalization already done in ctx_manager, so set group size = 1 here
                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                
                # NOTE: do not do reward normalization in ctx_manager, so we need to do it here
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]

                # batch.batch["response_mask"] = compute_response_mask(batch)
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if self.use_rm:
                    with marked_timer("reward", timing_raw):
                    # compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # recompute old_log_probs

                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(calculate_debug_metrics(batch))

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics.update({"rollout/ref_log_prob": avg_ref_log_prob})

                # compute values
                if self.use_critic and (not self.use_trainable_generative_critic):
                    with marked_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    print(f"{list(reward_extra_infos_dict.keys())=}")
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, multi_turn=True)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    use_label_outcome_advantage = self.config.algorithm.get("use_label_outcome_advantage", False)
                    if use_label_outcome_advantage:
                        print("[GEN_CRITIC FLOW] use_label_outcome_advantage=True, entering label inference path")
                        label_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                        label_metrics = {"gen_critic/enabled": 0.0}

                        if self.generative_critic.enabled and "messages_list" in batch.non_tensor_batch and "turn_ids" in batch.batch:
                            messages_list = batch.non_tensor_batch["messages_list"].tolist()
                            if self.use_trainable_generative_critic and self.use_critic:
                                label_tensor, label_metrics, critic_label_gen_time = self._infer_labels_with_trainable_critic(
                                    messages_list=messages_list,
                                    turn_ids=batch.batch["turn_ids"],
                                )
                                critic_generation_time += critic_label_gen_time
                            else:
                                label_tensor, label_metrics, _ = self.generative_critic.infer_turn_labels(
                                    messages_list=messages_list,
                                    turn_ids=batch.batch["turn_ids"],
                                )
                            label_tensor = label_tensor.to(batch.batch["response_mask"].device)

                        outcome_tensor = self._compute_outcome_tensor(
                            token_level_scores=batch.batch["token_level_scores"],
                            response_mask=batch.batch["response_mask"],
                        )

                        label_weight = float(self.config.algorithm.get("label_weight", 1.0))
                        outcome_weight = float(self.config.algorithm.get("outcome_weight", 1.0))
                        combined_reward = label_weight * label_tensor + outcome_weight * outcome_tensor

                        batch.batch["token_level_rewards"] = combined_reward
                        batch.batch["advantages"] = combined_reward
                        batch.batch["returns"] = combined_reward

                        metrics.update(label_metrics)
                        metrics.update({
                            "train/label_reward_mean": (label_tensor * batch.batch["response_mask"]).sum().item() / max(batch.batch["response_mask"].sum().item(), 1.0),
                            "train/outcome_reward_mean": (outcome_tensor * batch.batch["response_mask"]).sum().item() / max(batch.batch["response_mask"].sum().item(), 1.0),
                            "train/combined_reward_mean": (combined_reward * batch.batch["response_mask"]).sum().item() / max(batch.batch["response_mask"].sum().item(), 1.0),
                        })

                    else:
                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=True,
                            high_level_gamma=self.config.algorithm.high_level_gamma,
                            bi_level_gae=self.config.algorithm.bi_level_gae,
                        )

                ##### A very different setting, just here for testing: Can I normalize the advantages to have a mean of 0?
                if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    response_mask = batch.batch["response_mask"]
                    advantages = batch.batch["advantages"]
                    response_relative_lengths = (torch.sum(response_mask, dim=-1) + 1e-6) / torch.sum(response_mask, dim=-1).float().mean()
                    advantages = advantages / response_relative_lengths.unsqueeze(-1) 
                    batch.batch["advantages"] = advantages

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        if self.use_trainable_generative_critic:
                            critic_source_batch = critic_train_batch if critic_train_batch is not None else batch
                            critic_output_metrics, critic_train_gen_time = self._train_generative_critic_rlvr(critic_source_batch)
                            critic_generation_time += critic_train_gen_time
                        else:
                            critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = True
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw):
                        print(batch.batch.keys())
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            use_value_critic_metrics = self.use_critic and (not self.use_trainable_generative_critic)
            metrics.update(compute_data_metrics(batch=batch, use_critic=use_value_critic_metrics))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            if self.use_trainable_generative_critic:
                metrics["timing_s/critic_generation"] = critic_generation_time
                actor_generation_time = timing_raw.get("gen", 0.0)
                metrics["timing_s/actor_generation"] = actor_generation_time
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # add another timing metric: total time
            metrics.update({"timing_s/total": time.time() - self.start_time})
            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
