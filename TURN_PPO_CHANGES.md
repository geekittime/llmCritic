# Turn-PPO 改造说明

基线 commit（修改前）：`da20dbf10cadb06eb8ea92d7ca050341a5accdfe`

本文档记录从上述基线到当前工作区的全部改动，目标是以**最小代码改动**复现论文 Turn-PPO 的核心训练逻辑（turn-level ratio / turn-level value / turn-level GAE），并保留当前工程中的 value clipping。

## 改动文件总览

- `config/base.yaml`
- `ragen/llm_agent/ctx_manager.py`
- `ragen/trainer/agent_trainer.py`
- `ragen/trainer/core_algos.py`
- `ragen/workers/actor/dp_actor.py`
- `ragen/workers/critic/dp_critic.py`
- `train_sokoban.sh`

---

## 1) `ragen/llm_agent/ctx_manager.py`

### 新增 turn 索引与 turn value 边界标注（full 模式）

在 `_build_samples_full` 中新增并写入 batch：

- `turn_ids`：assistant action token 对应的 turn id（非 action 位置为 `-1`）
- `turn_value_mask`：每个 turn 仅一个 value 监督点（query/env 段最后 token）
- `turn_value_ids`：value 监督点对应的 turn id（其他位置 `-1`）

边界定义：

- 用 `<|im_start|>` 找到每个 assistant turn 的起点 `assistant_start`
- 取 `assistant_start - 1` 作为该 turn 的 value 边界 token（即 query/environment output 的最后 token）

### 新增调试开关（原代码内）

在 `_build_samples_full` 中增加：

- `agent_proxy.debug_turn_boundary=True` 时打印（最多前 2 个样本）：
  - `assistant_starts`
  - `boundary_positions`
  - 边界附近 token（left / boundary / assistant_start）
  - 完整 messages 内容（便于肉眼核对 instruction/grid/reward 文本）

---

## 2) `ragen/trainer/agent_trainer.py`

在 `compute_advantage(...)` 的 `GAE` 分支中接入 turn-GAE：

- 当 batch 含 `turn_value_mask + turn_ids + turn_value_ids` 时，走 `compute_turn_gae_advantage_return(...)`
- 否则保持原 token-level `compute_gae_advantage_return(...)`

这样仅在 turn 标注存在时激活新路径，不影响其他模式。

---

## 3) `ragen/trainer/core_algos.py`

### `compute_turn_gae_advantage_return(...)` 逻辑调整

该函数已用于 turn-level GAE，当前改动点：

- 使用 `boundary_values = values * turn_value_mask`，仅边界 value 参与 turn bootstrap
- 每个 turn 的 reward 采用该 turn action token 上 `token_level_rewards` 的求和
- 按 turn 逆序计算 GAE：
  - `delta = r_turn + gamma * V(next_turn_boundary) - V(curr_turn_boundary)`
- 将 turn advantage / return 广播回对应 turn 的 action tokens
- 额外写 `returns[b, boundary_pos] = turn_ret`，用于 critic 在边界点的 value 回归监督

说明：优势白化仍按 action `response_mask` 做，与 actor 更新口径保持一致。

---

## 4) `ragen/workers/actor/dp_actor.py`

### 新增 `compute_turn_policy_loss(...)`

核心思路：

- 先按 `turn_ids` 聚合 turn 级量：
  - `turn_old_log_prob = sum(old_log_prob over tokens in turn)`
  - `turn_log_prob = sum(log_prob over tokens in turn)`
  - `turn_advantage` 取该 turn 的广播后优势（同 turn token 值相同）
- 再将上述 turn 张量 reshape 为 `(num_turns, 1)`，直接复用原 `compute_policy_loss(...)`

这样实现“先做 token 概率乘积（log 求和）再 clip”的 turn 口径，同时最大化复用原 PPO/clip 逻辑。

### `update_policy(...)` 接入

- 若 batch 含 `turn_ids`，调用 `compute_turn_policy_loss(...)`
- 否则保留原 token-level `compute_policy_loss(...)`

---

## 5) `ragen/workers/critic/dp_critic.py`

### `compute_values(...)`

- turn 模式下（存在 `turn_value_mask`）不再对 `values` 乘 `response_mask`
- 非 turn 模式保持原行为：`values *= response_mask`

原因：turn value 边界位于 query/env token 末端，不在 response mask 中；若强行乘 response mask 会把边界 value 清零。

### `update_critic(...)`

- `select_keys` 增加可选 `turn_value_mask`
- value loss 的 mask 改为：`value_mask = turn_value_mask if exists else response_mask`
- `compute_value_loss(...)` 仍使用原实现（包含 value clipping），仅替换监督 mask

---

## 6) `config/base.yaml`

新增配置项：

- `agent_proxy.debug_turn_boundary: False`

默认关闭，只在需要时显式开启调试打印。

---

## 7) `train_sokoban.sh`

在训练命令中默认开启调试：

- `agent_proxy.debug_turn_boundary=True`

便于直接运行脚本时观察 turn 边界是否正确落在 query/env 最后 token。

---

## 当前实现与论文的对齐情况

已对齐：

- policy clip 粒度：token -> turn
- value 监督粒度：token -> turn 边界（query/env last token）
- GAE 粒度：token -> turn
- value loss 仍使用 clipping（按你的要求保留）

仍保留的工程差异：

- 仍保留 dual-clip（`clip_ratio_c`）
- 当前 `train_sokoban.sh` 仍是 `agent_proxy.use_turn_scores=False`（终局稀疏 reward 形态）
