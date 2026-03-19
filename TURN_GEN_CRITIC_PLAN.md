# Turn Generative Critic Plan (Step 1)

Goal: replace value-based critic usage in actor training with a frozen generative judge signal.

## Phase A: Infrastructure and Config

1. Add a frozen generative critic module.
   - Build per-turn judge prompts from rollout `messages_list` and `turn_ids`.
   - Run frozen instruct model inference.
   - Parse `###label: True/False` and map labels back to token space.

2. Add config flags.
   - `algorithm.use_label_outcome_advantage`
   - `algorithm.label_weight`
   - `algorithm.outcome_weight`
   - `generative_critic.*` inference parameters.

## Phase B: Trainer Integration (Step 1 behavior)

3. In training loop, after reward computation and before actor update:
   - infer turn labels with frozen generative critic,
   - build `label_tensor` on token space,
   - build `outcome_tensor` (trajectory total reward broadcast),
   - combine to `combined_reward = label_weight * label + outcome_weight * outcome`.

4. Bypass GAE when enabled.
   - set `token_level_rewards = combined_reward`
   - set `advantages = combined_reward`
   - set `returns = combined_reward` for compatibility
   - skip value-based computation path for actor training signal.

5. Keep actor optimization code unchanged.
   - still uses existing turn-level policy loss (sum logprob per turn + PPO clip).

## Phase C: Observability and Safety

6. Add metrics.
   - `gen_critic/num_prompts`
   - `gen_critic/parse_fail_rate`
   - `gen_critic/true_rate`
   - `train/label_reward_mean`, `train/outcome_reward_mean`, `train/combined_reward_mean`

7. Add fallback behavior.
   - parse failure -> `default_label_if_parse_fail`.
   - no turn labels -> zero tensor and warning metric.

## Scope Clarification

- This step does NOT train generative critic.
- This step only uses frozen generative critic inference to create actor training signals.
- Critic RL training (GRPO/RLVR for critic itself) is deferred to Step 2.
