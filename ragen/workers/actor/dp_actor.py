# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

from peft import PeftModel


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_turn_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_ids: torch.Tensor,
    cliprange: float,
    cliprange_low: float,
    cliprange_high: float,
    clip_ratio_c: float,
    sample_weights: torch.Tensor = None,
):
    if old_log_prob.shape != log_prob.shape or old_log_prob.shape != response_mask.shape:
        raise ValueError("old_log_prob, log_prob, and response_mask must have identical shapes")
    if turn_ids.shape != response_mask.shape or advantages.shape != response_mask.shape:
        raise ValueError("turn_ids and advantages must be aligned with response_mask")

    batch_size, response_length = turn_ids.shape
    if sample_weights is None:
        row_weights = torch.ones(batch_size, device=log_prob.device, dtype=torch.float32)
    else:
        if sample_weights.numel() != batch_size:
            raise ValueError("sample_weights must contain exactly one value per batch row")
        row_weights = sample_weights.to(device=log_prob.device, dtype=torch.float32).reshape(batch_size)

    token_mask = (turn_ids >= 0) & (response_mask > 0)
    if token_mask.numel() == 0:
        zero = torch.tensor(0.0, device=log_prob.device, dtype=log_prob.dtype)
        return zero, zero, zero, zero

    has_turn = token_mask.any(dim=-1)
    has_any_turn, is_exact_single_turn_trace = torch.stack(
        (has_turn.any(), ((turn_ids == 0) | ~token_mask).all())
    ).tolist()
    if not has_any_turn:
        zero = torch.tensor(0.0, device=log_prob.device, dtype=log_prob.dtype)
        return zero, zero, zero, zero

    if is_exact_single_turn_trace:
        # build_turn_token_metadata assigns id 0 to the sole macro action in
        # every exact-trace row. Keep this path dense so it has no Python loop
        # or per-row device sync.
        turn_old_log_prob = torch.where(token_mask, old_log_prob, 0.0).sum(dim=-1)
        turn_log_prob = torch.where(token_mask, log_prob, 0.0).sum(dim=-1)
        token_positions = torch.arange(response_length, device=turn_ids.device).unsqueeze(0)
        last_token_position = torch.where(token_mask, token_positions, -1).amax(dim=-1).clamp_min(0)
        turn_advantages = advantages.gather(1, last_token_position.unsqueeze(-1)).squeeze(-1)
        turn_weights = row_weights * has_turn.to(row_weights.dtype)
    else:
        # A row may contain several turns. Group by (batch row, turn id), sum
        # token log-probabilities per macro action, and gather its last token's
        # advantage. All grouping remains on device.
        token_indices = torch.nonzero(token_mask, as_tuple=False)
        turn_pairs = torch.stack(
            (token_indices[:, 0], turn_ids[token_mask]),
            dim=-1,
        )
        unique_turn_pairs, inverse = torch.unique(
            turn_pairs,
            sorted=True,
            return_inverse=True,
            dim=0,
        )
        num_turns = unique_turn_pairs.size(0)

        turn_old_log_prob = old_log_prob.new_zeros(num_turns)
        turn_old_log_prob.scatter_add_(0, inverse, old_log_prob[token_mask])
        turn_log_prob = log_prob.new_zeros(num_turns)
        turn_log_prob.scatter_add_(0, inverse, log_prob[token_mask])

        last_token_position = torch.full(
            (num_turns,),
            -1,
            dtype=torch.long,
            device=turn_ids.device,
        )
        last_token_position.scatter_reduce_(
            0,
            inverse,
            token_indices[:, 1],
            reduce="amax",
            include_self=True,
        )
        turn_advantages = advantages[
            unique_turn_pairs[:, 0],
            last_token_position,
        ]
        turn_weights = row_weights[unique_turn_pairs[:, 0]]

    # Reuse the existing PPO implementation with turn-level packed tensors.
    # A numeric mask applies each row's sample weight to all of its turns and
    # normalizes the objective by the weighted number of macro actions.
    turn_old_log_prob = turn_old_log_prob.unsqueeze(-1)
    turn_log_prob = turn_log_prob.unsqueeze(-1)
    turn_advantages = turn_advantages.unsqueeze(-1)
    turn_mask = turn_weights.unsqueeze(-1)

    return compute_policy_loss(
        old_log_prob=turn_old_log_prob,
        log_prob=turn_log_prob,
        advantages=turn_advantages,
        response_mask=turn_mask,
        cliprange=cliprange,
        cliprange_low=cliprange_low,
        cliprange_high=cliprange_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode="token-mean",
    )


def compute_turn_micro_batch_scale(
    micro_turn_weight: float,
    global_turn_weight: float,
    num_mini_batches: int,
    data_parallel_world_size: int,
) -> float:
    """Scale a local weighted-mean loss into the global per-turn objective.

    FSDP averages gradients across data-parallel ranks. Each rank therefore
    contributes its local weighted turn sum multiplied by ``world_size`` and
    divided by one fixed per-step normalizer. Keeping that normalizer fixed
    across mini-batches prevents inverse-weighted padding rows in a short final
    mini-batch from becoming a full-strength optimizer step again.
    """
    if global_turn_weight <= 0:
        raise ValueError("global_turn_weight must be positive")
    if num_mini_batches <= 0:
        raise ValueError("num_mini_batches must be positive")
    if data_parallel_world_size <= 0:
        raise ValueError("data_parallel_world_size must be positive")
    return (
        float(micro_turn_weight)
        * float(data_parallel_world_size)
        * float(num_mini_batches)
        / float(global_turn_weight)
    )


_TURN_POLICY_METRIC_KEYS = (
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/pg_clipfrac_lower",
)


def finalize_turn_policy_metrics(
    metric_sums: dict[str, float],
    turn_weight: float,
    *,
    collective_device: torch.device | None = None,
) -> dict[str, float]:
    """Return exact turn-weighted actor metrics across all data-parallel ranks."""
    if turn_weight <= 0:
        raise ValueError("turn metric weight must be positive")

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if collective_device is None:
        if distributed:
            device_name = get_device_name()
            collective_device = (
                torch.device(device_name, get_device_id())
                if device_name != "cpu"
                else torch.device("cpu")
            )
        else:
            collective_device = torch.device("cpu")

    packed = torch.tensor(
        [*(float(metric_sums[key]) for key in _TURN_POLICY_METRIC_KEYS), float(turn_weight)],
        dtype=torch.float64,
        device=collective_device,
    )
    if distributed:
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)

    global_weight = float(packed[-1].item())
    if global_weight <= 0:
        raise ValueError("global turn metric weight must be positive")
    return {
        key: float(packed[index].item()) / global_weight
        for index, key in enumerate(_TURN_POLICY_METRIC_KEYS)
    }


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @staticmethod
    def _count_turns(
        turn_ids: torch.Tensor,
        sample_weights: torch.Tensor = None,
        response_mask: torch.Tensor = None,
    ) -> float:
        if turn_ids.numel() == 0:
            return 0.0
        if response_mask is not None and response_mask.shape != turn_ids.shape:
            raise ValueError("response_mask must have the same shape as turn_ids")

        batch_size = turn_ids.size(0)
        if sample_weights is None:
            row_weights = torch.ones(batch_size, device=turn_ids.device, dtype=torch.float32)
        else:
            if sample_weights.numel() != batch_size:
                raise ValueError("sample_weights must contain exactly one value per batch row")
            row_weights = sample_weights.to(device=turn_ids.device, dtype=torch.float32).reshape(batch_size)

        token_mask = turn_ids >= 0
        if response_mask is not None:
            token_mask = token_mask & (response_mask > 0)

        has_turn = token_mask.any(dim=-1)
        is_exact_single_turn_trace = ((turn_ids == 0) | ~token_mask).all()

        if bool(is_exact_single_turn_trace):
            total_weight = (row_weights * has_turn.to(row_weights.dtype)).sum()
        else:
            token_indices = torch.nonzero(token_mask, as_tuple=False)
            turn_pairs = torch.stack(
                (token_indices[:, 0], turn_ids[token_mask]),
                dim=-1,
            )
            unique_turn_pairs = torch.unique(turn_pairs, sorted=True, dim=0)
            total_weight = row_weights[unique_turn_pairs[:, 0]].sum()

        # update_policy needs a host scalar for its empty-batch guard and
        # micro-batch accumulation factor. This is one synchronization per
        # batch, rather than one synchronization per row/turn.
        return float(total_weight.item())

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, no_lora=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        is_peft_model = not no_lora and isinstance(self.actor_module._fsdp_wrapped_module, PeftModel)
        if is_peft_model:
            print(f"[INFO] Actor is a PeftModel")
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.merge_adapter()
            print(f"[INFO] Merged adapter actor")

        # Some upstream worker versions request an entropy tensor
        # unconditionally. With a zero entropy coefficient, retain that
        # contract using a zero placeholder and skip the expensive
        # full-vocabulary entropy reduction.
        compute_entropy = calculate_entropy and self.config.entropy_coeff != 0
        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            # Ray dispatch keeps the DataProto payload on CPU. FSDP moves
            # forward inputs internally, but labels and unpadding indices do
            # not pass through that hook. Move the complete micro-batch here
            # so FlashAttention/Triton never receives CPU pointer arguments.
            micro_batch = micro_batch.to(get_device_id())
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=compute_entropy,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy if compute_entropy else torch.zeros_like(log_probs))

        log_probs = torch.concat(log_probs_lst, dim=0)

        if is_peft_model:
            print(f"[INFO] Unmerging adapter actor")
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.unmerge_adapter()
            print(f"[INFO] Unmerged adapter actor")
        

        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        if "turn_ids" in data.batch:
            select_keys.append("turn_ids")
        if "sample_weights" in data.batch:
            select_keys.append("sample_weights")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        turn_global_weight = None
        turn_data_parallel_world_size = 1
        turn_num_mini_batches = len(dataloader)
        if "turn_ids" in batch:
            local_turn_weight = self._count_turns(
                batch["turn_ids"],
                sample_weights=batch.get("sample_weights"),
                response_mask=batch.get("response_mask"),
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if self.ulysses_sequence_parallel_size != 1:
                    raise ValueError(
                        "Turn PPO distributed weighting currently requires "
                        "ulysses_sequence_parallel_size=1"
                    )
                turn_data_parallel_world_size = torch.distributed.get_world_size()
                device_name = get_device_name()
                collective_device = (
                    torch.device(device_name, get_device_id())
                    if device_name != "cpu"
                    else torch.device("cpu")
                )
                turn_weight_tensor = torch.tensor(
                    local_turn_weight,
                    dtype=torch.float64,
                    device=collective_device,
                )
                torch.distributed.all_reduce(turn_weight_tensor, op=torch.distributed.ReduceOp.SUM)
                turn_global_weight = float(turn_weight_tensor.item())
            else:
                turn_global_weight = local_turn_weight
            if turn_global_weight <= 0 or turn_num_mini_batches <= 0:
                raise ValueError("Turn PPO received a batch without any valid turn actions")

        metrics = {}
        turn_metric_sums = {key: 0.0 for key in _TURN_POLICY_METRIC_KEYS}
        turn_metric_weight = 0.0
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                mini_batch_turn_weight = None
                mini_batch_values = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                mini_batch_turn_ids = mini_batch_values.get("turn_ids")
                if mini_batch_turn_ids is not None:
                    mini_batch_turn_weight = self._count_turns(
                        mini_batch_turn_ids,
                        sample_weights=mini_batch_values.get("sample_weights"),
                        response_mask=mini_batch_values.get("response_mask"),
                    )
                    if mini_batch_turn_weight <= 0:
                        raise ValueError("Turn PPO received a mini-batch without any valid turn actions")

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = data["response_mask"]
                    # response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    turn_ids = data.get("turn_ids", None)
                    sample_weights = data.get("sample_weights", None)

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    if turn_ids is not None:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_turn_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            turn_ids=turn_ids,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            sample_weights=sample_weights,
                        )
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    micro_turn_weight = None
                    if turn_ids is not None and turn_global_weight is not None:
                        # compute_turn_policy_loss returns a local weighted mean.
                        # Convert it to a weighted sum, then use one global
                        # normalizer for every optimizer step in this actor
                        # update. This remains correct across dynamic
                        # micro-batches, copied padding rows, and FSDP ranks.
                        micro_turn_weight = self._count_turns(
                            turn_ids,
                            sample_weights=sample_weights,
                            response_mask=response_mask,
                        )
                        loss = policy_loss * compute_turn_micro_batch_scale(
                            micro_turn_weight=micro_turn_weight,
                            global_turn_weight=turn_global_weight,
                            num_mini_batches=turn_num_mini_batches,
                            data_parallel_world_size=turn_data_parallel_world_size,
                        )
                    elif self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    actor_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    if micro_turn_weight is not None:
                        for key in _TURN_POLICY_METRIC_KEYS:
                            turn_metric_sums[key] += actor_metrics[key] * micro_turn_weight
                        turn_metric_weight += micro_turn_weight
                    else:
                        append_to_dict(metrics, actor_metrics)
                    if entropy_coeff != 0:
                        append_to_dict(
                            metrics,
                            {"actor/entropy_loss": entropy_loss.detach().item()},
                        )

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        if turn_global_weight is not None:
            exact_turn_metrics = finalize_turn_policy_metrics(
                turn_metric_sums,
                turn_metric_weight,
            )
            for key, value in exact_turn_metrics.items():
                # Worker-group metric collation expects list-valued samples.
                # Every rank receives the same all-reduced value, so the
                # controller's final mean preserves the exact global result.
                metrics[key] = [value]
        self.actor_optimizer.zero_grad()
        return metrics
