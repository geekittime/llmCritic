"""Frozen generative critic for turn-level action judging.

Step-1 implementation outline (record):
1) Build per-turn judge prompts from rollout `messages_list`.
2) Run a frozen instruct model to generate rationale + `###label: True/False`.
3) Parse labels and map them back to token space with `turn_ids`.
4) Expose lightweight metrics for parser health and label distribution.

This module is intentionally training-agnostic. It only performs inference and
label tensor construction. Trainer integration is handled separately.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


_LABEL_PATTERN = re.compile(r"###\s*label\s*:\s*(true|false)", re.IGNORECASE)
_FALLBACK_BOOL_PATTERN = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_STATE_BLOCK_PATTERN = re.compile(
    r"State:\n(.*?)\n(?:No valid action provided previously\.[^\n]*\n)?You have ",
    re.DOTALL,
)
_TURN_NUM_PATTERN = re.compile(r"Turn\s+(\d+)")
_REWARD_PATTERN = re.compile(r"Reward:\s*\n\s*([^\n]+)")


@dataclass
class JudgePromptItem:
    sample_index: int
    turn_id: int
    prompt: str


@dataclass
class JudgeTrainItem:
    sample_index: int
    turn_id: int
    prompt: str
    target_label: bool


class FrozenGenerativeCritic:
    """Frozen generative action critic.

    The critic reads transition context around one action and emits
    a structured binary judgment in natural language format.
    """

    def __init__(self, config: Any):
        self.config = config
        self.enabled = bool(OmegaConf.select(config, "generative_critic.enable", default=False))
        self.backend = str(OmegaConf.select(config, "generative_critic.backend", default="transformers")).lower()

        self.model_path = OmegaConf.select(config, "generative_critic.model_path", default=None)
        if self.model_path is None:
            self.model_path = OmegaConf.select(config, "model_path", default=None)

        self.max_new_tokens = int(OmegaConf.select(config, "generative_critic.max_new_tokens", default=128))
        self.temperature = float(OmegaConf.select(config, "generative_critic.temperature", default=0.0))
        self.top_p = float(OmegaConf.select(config, "generative_critic.top_p", default=1.0))
        self.top_k = int(OmegaConf.select(config, "generative_critic.top_k", default=-1))
        self.do_sample = bool(OmegaConf.select(config, "generative_critic.do_sample", default=False))
        self.trust_remote_code = bool(OmegaConf.select(config, "generative_critic.trust_remote_code", default=True))
        self.inference_batch_size = int(OmegaConf.select(config, "generative_critic.inference_batch_size", default=8))

        self.default_label_if_parse_fail = bool(
            OmegaConf.select(config, "generative_critic.default_label_if_parse_fail", default=False)
        )
        self.rlvr_format_reward = float(OmegaConf.select(config, "generative_critic.rlvr_format_reward", default=0.2))
        self.rlvr_label_reward = float(OmegaConf.select(config, "generative_critic.rlvr_label_reward", default=1.0))
        self.rlvr_label_penalty = float(OmegaConf.select(config, "generative_critic.rlvr_label_penalty", default=-1.0))
        self.rlvr_parse_fail_penalty = float(
            OmegaConf.select(config, "generative_critic.rlvr_parse_fail_penalty", default=-1.0)
        )
        self.debug_print_samples = bool(
            OmegaConf.select(config, "generative_critic.debug_print_samples", default=False)
        )
        self.debug_max_print = int(OmegaConf.select(config, "generative_critic.debug_max_print", default=2))
        self.debug_max_prompt_chars = int(
            OmegaConf.select(config, "generative_critic.debug_max_prompt_chars", default=2048)
        )
        self.debug_max_output_chars = int(
            OmegaConf.select(config, "generative_critic.debug_max_output_chars", default=600)
        )

        self._tokenizer: Optional[Any] = None
        self._model: Optional[AutoModelForCausalLM] = None
        self._generate_fn: Optional[Callable[[Sequence[str], Dict[str, Any]], List[str]]] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_generate_fn(self, generate_fn: Callable[[Sequence[str], Dict[str, Any]], List[str]]) -> None:
        """Inject generation backend callback (e.g., actor rollout vLLM)."""
        self._generate_fn = generate_fn

    def _load_model(self) -> None:
        if self.backend != "transformers":
            return
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.enabled:
            return
        if self.model_path is None:
            raise ValueError("generative_critic is enabled but model_path is not set")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            dtype=dtype,
        )
        self._model.eval()
        self._model.to(self._device)
        if self.debug_print_samples:
            print(
                "[GEN_CRITIC INIT] "
                f"device={self._device} "
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
                f"model_path={self.model_path}"
            )

    @staticmethod
    def _find_prev_user(messages: Sequence[Dict[str, str]], idx: int) -> str:
        for j in range(idx - 1, -1, -1):
            if messages[j].get("role") == "user":
                return str(messages[j].get("content", ""))
        return ""

    @staticmethod
    def _find_next_user(messages: Sequence[Dict[str, str]], idx: int) -> str:
        for j in range(idx + 1, len(messages)):
            if messages[j].get("role") == "user":
                return str(messages[j].get("content", ""))
        return ""

    @staticmethod
    def _extract_system_instruction(messages: Sequence[Dict[str, str]]) -> str:
        for msg in messages:
            if msg.get("role") == "system":
                return str(msg.get("content", "")).strip()
        return ""

    @staticmethod
    def _extract_last_state(user_content: str) -> str:
        matches = _STATE_BLOCK_PATTERN.findall(user_content)
        if matches:
            return matches[-1].strip()
        if "State:\n" in user_content:
            return user_content.split("State:\n")[-1].strip()
        return user_content.strip()

    @staticmethod
    def _extract_first_state(user_content: str) -> str:
        matches = _STATE_BLOCK_PATTERN.findall(user_content)
        if matches:
            return matches[0].strip()
        if "State:\n" in user_content:
            return user_content.split("State:\n", maxsplit=1)[-1].strip()
        return ""

    @staticmethod
    def _extract_last_turn_number(user_content: str) -> Optional[int]:
        matches = _TURN_NUM_PATTERN.findall(user_content)
        if matches:
            return int(matches[-1])
        return None

    @staticmethod
    def _extract_first_reward(next_user_content: str) -> Optional[str]:
        match = _REWARD_PATTERN.search(next_user_content)
        if match is None:
            return None
        return match.group(1).strip()

    def _extract_transition_context(self, messages: Sequence[Dict[str, str]], assistant_idx: int) -> Dict[str, Any]:
        prev_user = self._find_prev_user(messages, assistant_idx)
        next_user = self._find_next_user(messages, assistant_idx)
        env_instruction = self._extract_system_instruction(messages)

        state_before = self._extract_last_state(prev_user)
        state_after = self._extract_first_state(next_user) if next_user else ""
        observed_reward = self._extract_first_reward(next_user) if next_user else None
        turn_number = self._extract_last_turn_number(prev_user)
        has_after_state = bool(state_after)

        return {
            "state_before": state_before,
            "state_after": state_after,
            "observed_reward": observed_reward,
            "turn_number": turn_number,
            "has_after_state": has_after_state,
            "env_instruction": env_instruction,
        }

    def _get_task_specific_critic_instruction(self, env_instruction: str) -> Optional[str]:
        """Fetch task-specific critic instruction from config/custom_envs.

        Matching strategy: if a custom env's env_instruction appears in the
        system instruction text, use that env's critic_instruction when present.
        """
        custom_envs = OmegaConf.select(self.config, "custom_envs", default=None)
        if custom_envs is None:
            return None

        env_instruction_lower = env_instruction.lower()
        for _, env_cfg in custom_envs.items():
            base_instruction = str(env_cfg.get("env_instruction", "")).strip()
            critic_instruction = env_cfg.get("critic_instruction", None)
            if not base_instruction or critic_instruction is None:
                continue
            if base_instruction.lower() in env_instruction_lower:
                return str(critic_instruction).strip()
        return None

    @staticmethod
    def _build_single_prompt(
        state_before: str,
        action_text: str,
        state_after: str,
        observed_reward: Optional[str],
        turn_number: Optional[int],
        has_after_state: bool,
        env_instruction: str,
        critic_instruction: Optional[str],
    ) -> str:
        turn_text = "unknown"
        if turn_number is not None:
            turn_text = str(turn_number)

        reward_text = observed_reward if observed_reward is not None else "Not provided"
        if has_after_state:
            state_after_text = state_after
        else:
            state_after_text = "Not provided (terminal or truncated context)"

        instruction_block = ""
        if env_instruction:
            instruction_block = (
                "[Environment instruction]\n"
                f"{env_instruction}\n"
                "\n"
            )

        return (
            "You are a strict action critic for step-by-step environment solving.\n"
            "Evaluate one transition only: (s_t, a_t, s_{t+1}).\n"
            "\n"
            f"{instruction_block}"
            f"[Turn]\n{turn_text}\n"
            "\n"
            "[s_t: state before action]\n"
            f"{state_before}\n"
            "\n"
            "[a_t: assistant action]\n"
            f"{action_text}\n"
            "\n"
            "[Observed immediate reward]\n"
            f"{reward_text}\n"
            "\n"
            "[s_{t+1}: state after action]\n"
            f"{state_after_text}\n"
            "\n"
            "Judge whether action a_t moves the agent closer to task completion.\n"
            f"{critic_instruction if critic_instruction else 'Output format (two lines):\\n1) First, give a brief rationale grounded in the transition above. You should check whether the action is helpful/harmful/correct/incorrect.\\n2) A label based on the rationale above: \'###label: True\' or \'###label: False\'.'}"
        )

    def build_judge_prompts(self, messages_list: Sequence[Sequence[Dict[str, str]]], turn_ids: torch.Tensor) -> List[JudgePromptItem]:
        """Build one judge prompt per observed assistant turn."""
        items: List[JudgePromptItem] = []

        for sample_index, messages in enumerate(messages_list):
            max_turn_id = int(turn_ids[sample_index].max().item()) if torch.any(turn_ids[sample_index] >= 0) else -1
            if max_turn_id < 0:
                continue

            assistant_turn_counter = 0
            for msg_idx, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                if assistant_turn_counter > max_turn_id:
                    break

                transition = self._extract_transition_context(messages, msg_idx)
                action_text = str(msg.get("content", ""))
                critic_instruction = self._get_task_specific_critic_instruction(transition["env_instruction"])
                prompt = self._build_single_prompt(
                    state_before=transition["state_before"],
                    action_text=action_text,
                    state_after=transition["state_after"],
                    observed_reward=transition["observed_reward"],
                    turn_number=transition["turn_number"],
                    has_after_state=transition["has_after_state"],
                    env_instruction=transition["env_instruction"],
                    critic_instruction=critic_instruction,
                )
                items.append(
                    JudgePromptItem(
                        sample_index=sample_index,
                        turn_id=assistant_turn_counter,
                        prompt=prompt,
                    )
                )
                assistant_turn_counter += 1

        return items

    @staticmethod
    def parse_label(text: str) -> Optional[bool]:
        """Parse True/False from generated text."""
        match = _LABEL_PATTERN.search(text)
        if match is not None:
            return match.group(1).lower() == "true"

        fallback = _FALLBACK_BOOL_PATTERN.findall(text)
        if fallback:
            return fallback[-1].lower() == "true"
        return None

    @staticmethod
    def has_strict_label_format(text: str) -> bool:
        return _LABEL_PATTERN.search(text) is not None

    @staticmethod
    def trajectory_success_from_metrics(messages_list: Sequence[Sequence[Dict[str, str]]]) -> Optional[torch.Tensor]:
        """Extract trajectory success labels from message-embedded metrics.

        Returns None if metrics cannot be recovered from messages.
        """
        labels: List[bool] = []
        any_found = False
        for messages in messages_list:
            success_val: Optional[bool] = None
            for msg in messages:
                content = str(msg.get("content", ""))
                if not content:
                    continue
                success_matches = re.findall(r"[\"']success[\"']\s*:\s*([0-9\.]+)", content)
                if success_matches:
                    any_found = True
                    success_val = float(success_matches[-1]) > 0.5

            labels.append(False if success_val is None else success_val)

        if not any_found:
            return None
        return torch.tensor(labels, dtype=torch.bool)

    @staticmethod
    def trajectory_success_from_scores(
        token_level_scores: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: convert trajectory-level scores to success labels.

        By convention in current pipeline, total score > 0 indicates success.
        """
        total_score = (token_level_scores * response_mask).sum(dim=-1)
        return total_score > 0

    def build_train_judge_prompts(
        self,
        messages_list: Sequence[Sequence[Dict[str, str]]],
        turn_ids: Optional[torch.Tensor],
        trajectory_success: torch.Tensor,
    ) -> List[JudgeTrainItem]:
        """Build train prompts with trajectory-level True/False targets.

        Target rule:
        - successful trajectory: all turn labels target True
        - failed trajectory: all turn labels target False
        """
        if trajectory_success.ndim != 1:
            raise ValueError(f"trajectory_success must be 1D, got shape {tuple(trajectory_success.shape)}")

        items: List[JudgeTrainItem] = []
        for sample_index, messages in enumerate(messages_list):
            if sample_index >= trajectory_success.shape[0]:
                break

            target_label = bool(trajectory_success[sample_index].item())

            # full mode: use turn_ids to include all assistant turns in trajectory
            if turn_ids is not None:
                max_turn_id = int(turn_ids[sample_index].max().item()) if torch.any(turn_ids[sample_index] >= 0) else -1
                if max_turn_id < 0:
                    continue

                assistant_turn_counter = 0
                for msg_idx, msg in enumerate(messages):
                    if msg.get("role") != "assistant":
                        continue
                    if assistant_turn_counter > max_turn_id:
                        break

                    transition = self._extract_transition_context(messages, msg_idx)
                    action_text = str(msg.get("content", ""))
                    critic_instruction = self._get_task_specific_critic_instruction(transition["env_instruction"])
                    prompt = self._build_single_prompt(
                        state_before=transition["state_before"],
                        action_text=action_text,
                        state_after=transition["state_after"],
                        observed_reward=transition["observed_reward"],
                        turn_number=transition["turn_number"],
                        has_after_state=transition["has_after_state"],
                        env_instruction=transition["env_instruction"],
                        critic_instruction=critic_instruction,
                    )
                    items.append(
                        JudgeTrainItem(
                            sample_index=sample_index,
                            turn_id=assistant_turn_counter,
                            prompt=prompt,
                            target_label=target_label,
                        )
                    )
                    assistant_turn_counter += 1
                continue

            # single_turn / limited_multi_turn fallback:
            # each sample corresponds to one target action, which is the last assistant message.
            assistant_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]
            if len(assistant_indices) == 0:
                continue
            msg_idx = assistant_indices[-1]
            transition = self._extract_transition_context(messages, msg_idx)
            action_text = str(messages[msg_idx].get("content", ""))
            critic_instruction = self._get_task_specific_critic_instruction(transition["env_instruction"])
            prompt = self._build_single_prompt(
                state_before=transition["state_before"],
                action_text=action_text,
                state_after=transition["state_after"],
                observed_reward=transition["observed_reward"],
                turn_number=transition["turn_number"],
                has_after_state=transition["has_after_state"],
                env_instruction=transition["env_instruction"],
                critic_instruction=critic_instruction,
            )
            items.append(
                JudgeTrainItem(
                    sample_index=sample_index,
                    turn_id=0,
                    prompt=prompt,
                    target_label=target_label,
                )
            )
        return items

    def compute_rlvr_scalar_rewards(
        self,
        outputs: Sequence[str],
        targets: Sequence[bool],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute RLVR scalar reward for each generated critic output."""
        if len(outputs) != len(targets):
            raise ValueError(f"outputs/targets length mismatch: {len(outputs)} vs {len(targets)}")

        rewards = torch.zeros(len(outputs), dtype=torch.float32)
        format_ok = 0
        label_correct = 0
        parse_fail = 0

        for i, (text, target) in enumerate(zip(outputs, targets, strict=True)):
            reward = 0.0
            has_format = self.has_strict_label_format(text)
            if has_format:
                format_ok += 1
                reward += self.rlvr_format_reward

            parsed = self.parse_label(text)
            if parsed is None:
                parse_fail += 1
                reward += self.rlvr_parse_fail_penalty
            elif parsed == target:
                label_correct += 1
                reward += self.rlvr_label_reward
            else:
                reward += self.rlvr_label_penalty

            rewards[i] = reward

        n = float(max(len(outputs), 1))
        metrics = {
            "gen_critic/train/format_rate": float(format_ok) / n,
            "gen_critic/train/label_acc": float(label_correct) / n,
            "gen_critic/train/parse_fail_rate": float(parse_fail) / n,
            "gen_critic/train/reward_mean": rewards.mean().item() if len(outputs) > 0 else 0.0,
        }
        return rewards, metrics

    @staticmethod
    def expand_scalar_rewards_to_token(
        scalar_rewards: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast scalar rewards to all valid response tokens."""
        if scalar_rewards.ndim != 1:
            raise ValueError(f"scalar_rewards must be 1D, got shape {tuple(scalar_rewards.shape)}")
        if response_mask.ndim != 2:
            raise ValueError(f"response_mask must be 2D, got shape {tuple(response_mask.shape)}")
        if scalar_rewards.shape[0] != response_mask.shape[0]:
            raise ValueError(
                f"batch mismatch for scalar_rewards/response_mask: {scalar_rewards.shape[0]} vs {response_mask.shape[0]}"
            )

        return scalar_rewards.unsqueeze(-1).to(response_mask.dtype) * response_mask

    def _generate_texts(self, prompts: Sequence[str]) -> List[str]:
        if self.backend in {"actor_rollout_vllm", "vllm_actor_rollout", "actor_vllm", "vllm"}:
            if self._generate_fn is None:
                raise ValueError("generative_critic backend requires generate_fn, but it is not set")

            if self.debug_print_samples:
                print(
                    "[GEN_CRITIC INFER] backend=actor_rollout_vllm "
                    f"num_prompts={len(prompts)} max_tokens={self.max_new_tokens}"
                )

            sampling_overrides: Dict[str, Any] = {
                "max_tokens": self.max_new_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "temperature": self.temperature,
                "do_sample": self.do_sample,
            }
            outputs = self._generate_fn(prompts, sampling_overrides)
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"generative_critic backend returned mismatched output size: {len(outputs)} vs {len(prompts)}"
                )
            return outputs

        self._load_model()
        assert self._model is not None and self._tokenizer is not None

        all_outputs: List[str] = []
        start_time = time.time()
        total_chunks = (len(prompts) + self.inference_batch_size - 1) // self.inference_batch_size
        if self.debug_print_samples:
            print(
                "[GEN_CRITIC INFER] "
                f"num_prompts={len(prompts)} batch_size={self.inference_batch_size} chunks={total_chunks}"
            )
        for start in range(0, len(prompts), self.inference_batch_size):
            chunk = list(prompts[start : start + self.inference_batch_size])
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature if self.do_sample else None,
                    top_p=self.top_p if self.do_sample else None,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            prompt_len = encoded["input_ids"].shape[1]
            completion_ids = generated[:, prompt_len:]
            decoded = self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            all_outputs.extend(decoded)

            if self.debug_print_samples:
                chunk_id = start // self.inference_batch_size + 1
                elapsed = time.time() - start_time
                print(f"[GEN_CRITIC INFER] chunk={chunk_id}/{total_chunks} elapsed={elapsed:.1f}s")

        return all_outputs

    def infer_turn_labels(
        self,
        messages_list: Sequence[Sequence[Dict[str, str]]],
        turn_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], List[str]]:
        """Infer per-token label tensor from per-turn generative judgments.

        Returns:
            label_tensor: float tensor shaped like turn_ids, values in {0,1}
            metrics: parser and label-rate metrics
            raw_outputs: generated critic outputs in prompt order
        """
        label_tensor = torch.zeros_like(turn_ids, dtype=torch.float32)
        if not self.enabled:
            return label_tensor, {"gen_critic/enabled": 0.0}, []

        prompt_items = self.build_judge_prompts(messages_list=messages_list, turn_ids=turn_ids)
        if len(prompt_items) == 0:
            return label_tensor, {
                "gen_critic/enabled": 1.0,
                "gen_critic/num_prompts": 0.0,
                "gen_critic/parse_fail_rate": 0.0,
                "gen_critic/true_rate": 0.0,
            }, []

        outputs = self._generate_texts([item.prompt for item in prompt_items])

        parse_fail = 0
        true_count = 0
        printed = 0
        for item, text in zip(prompt_items, outputs):
            parsed = self.parse_label(text)
            if parsed is None:
                parse_fail += 1
                parsed = self.default_label_if_parse_fail

            value = 1.0 if parsed else 0.0
            if value > 0.5:
                true_count += 1

            mask = turn_ids[item.sample_index] == item.turn_id
            label_tensor[item.sample_index, mask] = value

            if self.debug_print_samples and printed < self.debug_max_print:
                prompt_preview = item.prompt[: self.debug_max_prompt_chars]
                output_preview = text[: self.debug_max_output_chars]
                print("\n[GEN_CRITIC DEBUG]")
                print(f"sample_index={item.sample_index} turn_id={item.turn_id}")
                print("[PROMPT]")
                print(prompt_preview)
                print("[OUTPUT]")
                print(output_preview)
                print(f"[PARSED] label={parsed}")
                printed += 1

        num_prompts = float(len(prompt_items))
        metrics = {
            "gen_critic/enabled": 1.0,
            "gen_critic/num_prompts": num_prompts,
            "gen_critic/parse_fail_rate": float(parse_fail) / max(num_prompts, 1.0),
            "gen_critic/true_rate": float(true_count) / max(num_prompts, 1.0),
        }
        return label_tensor, metrics, outputs
