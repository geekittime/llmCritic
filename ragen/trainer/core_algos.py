from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage as _compute_grpo_outcome_advantage,
    compute_reinforce_plus_plus_outcome_advantage,
    compute_reinforce_plus_plus_baseline_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_value_loss,
)


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    episode_ids: Optional[np.ndarray] = None,
):
    """
    Compute advantage for GRPO with episode-level deduplication support.

    When episode_ids is provided (for single_turn/limited_multi_turn mode), each (index, episode_id) pair
    only contributes once to mean/std calculation, avoiding bias from different turn counts.
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Use seen_pairs to deduplicate when episode_ids is provided
        seen_pairs = set()
        for i in range(bsz):
            if episode_ids is not None:
                pair = (index[i], episode_ids[i])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_turn_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    turn_ids: torch.Tensor,
    turn_value_mask: torch.Tensor,
    turn_value_ids: torch.Tensor,
    gamma: float,
    lam: float,
    normalize_advantages: bool = True,
):
    """Compute GAE over macro actions with one scalar reward per turn.

    ``token_level_rewards`` is kept for DataProto compatibility, but every turn
    reward must be written exactly once (normally at ``turn_end_mask``).  The
    resulting scalar advantage is broadcast over the sampled tokens of that
    turn for the actor, while returns are stored only at value boundaries.
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        values = values.float()
        response_mask = response_mask.float()
        turn_value_mask = turn_value_mask > 0
        expected_shape = token_level_rewards.shape
        for name, tensor in {
            "values": values,
            "response_mask": response_mask,
            "turn_ids": turn_ids,
            "turn_value_mask": turn_value_mask,
            "turn_value_ids": turn_value_ids,
        }.items():
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match rewards {tuple(expected_shape)}")

        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        turn_records = []

        for batch_index in range(values.shape[0]):
            boundary_positions = torch.nonzero(turn_value_mask[batch_index], as_tuple=True)[0]
            if boundary_positions.numel() == 0:
                continue

            per_episode = []
            seen_turn_ids = set()
            for boundary_position in boundary_positions:
                turn_id = int(turn_value_ids[batch_index, boundary_position].item())
                if turn_id < 0 or turn_id in seen_turn_ids:
                    raise ValueError("Each valid turn must have exactly one uniquely identified value boundary")
                seen_turn_ids.add(turn_id)
                token_mask = (turn_ids[batch_index] == turn_id) & (response_mask[batch_index] > 0)
                if not torch.any(token_mask):
                    raise ValueError(f"Turn {turn_id} has a value boundary but no sampled action tokens")
                reward = token_level_rewards[batch_index][token_mask].sum()
                per_episode.append(
                    {
                        "batch_index": batch_index,
                        "turn_id": turn_id,
                        "boundary_position": boundary_position,
                        "token_mask": token_mask,
                        "reward": reward,
                        "value": values[batch_index, boundary_position],
                    }
                )

            last_gae = torch.zeros((), dtype=values.dtype, device=values.device)
            for record_index in range(len(per_episode) - 1, -1, -1):
                record = per_episode[record_index]
                next_value = (
                    per_episode[record_index + 1]["value"]
                    if record_index + 1 < len(per_episode)
                    else torch.zeros((), dtype=values.dtype, device=values.device)
                )
                delta = record["reward"] + gamma * next_value - record["value"]
                last_gae = delta + gamma * lam * last_gae
                record["advantage"] = last_gae
                record["return"] = last_gae + record["value"]
            turn_records.extend(per_episode)

        if turn_records:
            scalar_advantages = torch.stack([record["advantage"] for record in turn_records])
            if normalize_advantages and scalar_advantages.numel() > 1:
                scalar_advantages = (scalar_advantages - scalar_advantages.mean()) * torch.rsqrt(
                    scalar_advantages.var(unbiased=False) + 1e-8
                )

            for record, scalar_advantage in zip(turn_records, scalar_advantages):
                batch_index = record["batch_index"]
                advantages[batch_index, record["token_mask"]] = scalar_advantage
                returns[batch_index, record["boundary_position"]] = record["return"]

        advantages *= response_mask

    return advantages, returns

# supported by Kangrui Wang
def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        reward_mask = token_level_rewards.bool()
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]
            
            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


# set up unittest
if __name__ == "__main__":
    token_level_rewards = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    loss_mask = torch.ones(1, 10)
    advantages, returns = compute_bi_level_gae_advantage_return(token_level_rewards, values, loss_mask, 1, 1, 0.95)
    print(advantages)
    print(returns)
