# DeepSeek Label-Only Turn-PPO 实验记录

日期：2026-08-31（Asia/Singapore）

## 实验问题

本实验检验下面这个消融目标能否直接训练 Sokoban agent：

1. 一个完整 assistant turn 是一个 macro action。
2. macro action 的概率是该 turn 内 sampled token 概率的乘积，即 log 空间中的 token log-prob 求和。
3. DeepSeek 根据 `(s_t, a_t, s_{t+1})` 判断动作让任务更近、无变化或更远，并严格输出 `+1/0/-1`。
4. 每个 turn 的 PPO advantage 直接等于这个标签。
5. 不加入轨迹成功/失败的 outcome reward，不使用 value critic、GAE、reference KL、reward KL 或 entropy bonus。

这不是推荐的最终算法，而是刻意隔离 LLM 进度标签作用的 label-only ablation。

## 可复现信息

- 分支：`experiment/deepseek-only-turn-advantage`
- 训练代码提交：`8276e618a39eb2e0211475b0063659e2a480d70a`
- 训练后记录器修复提交：`66d248d`
- W&B：[run `4mkr3fgx`](https://wandb.ai/MuLab-RL/llm-critic-turn-ppo/runs/4mkr3fgx)
- 主机：A100-006（`lyg0253`），物理 GPU 6、7，2 x A100 80GB
- 模型：`/data/kangshijia/sicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507`
- DeepSeek：`deepseek-v4-flash`，thinking disabled，并发上限 24
- 本地日志：`/data/kangshijia/logs/llm-critic/lyg0253-sokoban-turnppo-dsflash-labelonly-h10-finalcode-2gpu-20260831-122116.log`
- checkpoint：`/home/kangshijia/checkpoints/llm-critic-turn-ppo/lyg0253-sokoban-turnppo-dsflash-labelonly-h10-finalcode-2gpu-20260831-122116`
- train：每 step 32 条轨迹（4 个环境组 x 8），30 steps
- validation：固定 16 个 puzzle seed，每题 2 条 rollout；训练前及每 10 steps 验证
- horizon：10 个 turn，每 turn 最多执行 1 个动作；response 上限 40 token
- actor learning rate：`1e-6`

W&B metadata 已核对为上述 git commit。验证固定的是 puzzle seed；vLLM sampler RNG 没有在每次验证前重置，所以它是固定题集比较，不是逐 token 完全相同的 paired rollout。

## 最终实现审计

训练前后共做了三轮独立代码审计，重点核对：

- rollout 保存原始 sampled token trace，不通过 decoded text 重新分词；每个环境 turn 展开为一个训练 row。
- 因果 LM 的 logit/target 位移正确，left padding、response padding 和未执行 token 不进入 macro action。
- 对每个 turn 计算 `sum(log pi_new) - sum(log pi_old)`，再取指数得到概率乘积的 PPO ratio。
- 为满足 FSDP/microbatch 整除而复制的 row 使用逆 multiplicity 权重，不会改变原始 turn 的目标分布。
- loss 与 `pg_loss/clipfrac/KL` 指标都按有效 turn 权重跨 dynamic microbatch、minibatch、epoch 和 FSDP rank 精确聚合。
- `label_only` 确实不混入 outcome reward、value、KL 或 entropy 项，启动时对冲突配置 fail fast。
- 所有 turn 先构建请求，再用有界 semaphore 并发；batch 内去重和缓存不会重复计权。
- parser 只接受最后一个非空行 `FINAL_SCORE: -1`、`FINAL_SCORE: 0` 或 `FINAL_SCORE: 1`。按本实验需求，API/空响应/格式失败回退为 `-1`；鉴权或失败率异常会在 actor update 前中止。
- success、episode return 与 pass@2 按完整 trajectory/原题聚合，不受 turn 展开或 padding row 数影响。
- checkpoint 只有在 actor shard 保存完成后才更新 latest tracker。

独立审计没有发现会使本次 label-only 实验结论失效的 PPO、turn trace、judge 或指标实现错误。退出审计发现 VERL `Tracking` 仅在 `__del__` 中 finish，导致 step 30 未在解释器退出前 flush。该 P2 已在 `66d248d` 修复为显式、幂等的生命周期管理；同时覆盖成功、训练失败、资源清理失败、backend 半初始化、finish 重试和 W&B/MLflow 状态。原 run 已用 `resume="must"` 只补写从 console 精确解析的 step 30 数值，并以 `recovery/wandb_final_row_reconstructed_from_console=1` 标记来源，云端现为 `state=finished, _step=30`。

### 验证结果

- `/home/kangshijia/venvs/ragen/bin/python -m pytest -q tests`：`131 passed`。
- `compileall -q ragen train.py tests`：通过。
- 8 个相关 shell 脚本 `bash -n`：通过。
- `git diff --check`：通过。
- 当前提交已跟踪文件中的 DeepSeek key 样式匹配数：0。
- 仅有 Gym、Ray 和上游 verl API 的弃用警告；不影响本次数值结果，但需要后续升级兼容。

## 固定题集结果

| step | success | pass@2 | episodic return | action budget exhausted | avg actions | moved player | moved box |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10/32 = 31.25% | 7/16 = 43.75% | 2.6000 | 68.75% | 8.375 | 64.40% | 13.02% |
| 10 | 3/32 = 9.375% | 2/16 = 12.50% | 0.0781 | 90.625% | 9.531 | 76.12% | 3.57% |
| 20 | 2/32 = 6.25% | 1/16 = 6.25% | -0.2625 | 93.75% | 9.500 | 88.85% | 7.92% |
| 30 | 2/32 = 6.25% | 1/16 = 6.25% | -0.2687 | 93.75% | 9.563 | 87.08% | 2.08% |

step 0 到 step 10 的 32 个 validation sample slot 中，7 个由成功变为失败、0 个由失败变为成功。因为 sampler RNG 没有逐次重置，这个 slot 比较只能作为描述性证据；更稳健的结论来自固定题集聚合指标在 step 10 和 step 20 连续下降，以及独立 20-step run 的同方向复现。

step 20 到 step 30 的 success/pass@2 数量没有继续下降，而是停在退化平台。success 的 Wilson 95% 区间从 step 0 的 `[17.95%, 48.57%]` 变为 step 30 的 `[1.73%, 20.15%]`；pass@2 从 `[23.10%, 66.82%]` 变为 `[1.11%, 28.33%]`。按 16 个题独立重采样的描述性 bootstrap，step 30 - step 0 的 success 差值区间约为 `[-46.875, -3.125]` 个百分点。验证集仍然较小，因此这里用于确认退化方向，不用于声称精确总体效应量。

此前的 20-step run [`2wnvb46f`](https://wandb.ai/MuLab-RL/llm-critic-turn-ppo/runs/2wnvb46f) 在同一固定题集上得到 success `10/32 -> 5/32 -> 2/32`、pass@2 `7/16 -> 3/16 -> 1/16`。该 run 在 step 20 后因发现 actor 诊断指标聚合 P2 而主动停止，W&B 只同步到 step 19；step 20 数据来自本地日志，因此不使用其 actor 数值作结论。最终 `8276e61` run 修复了指标聚合并独立复现相同的环境行为退化。

## 学到了什么行为

step 1--10 共 2,842 个原始 turn：

- `+1/0/-1 = 278/1790/774 = 9.78%/62.98%/27.23%`，平均 label 为 `-0.1745`。
- 前 5 step 到后 5 step：正标签 `11.45% -> 8.23%`，中性标签 `56.75% -> 68.80%`，负标签 `31.80% -> 22.98%`。
- 同期训练轨迹：success `21.25% -> 13.125%`，玩家移动 `68.26% -> 75.78%`，推箱 `16.18% -> 9.50%`，动作预算耗尽 `78.75% -> 86.88%`。

因此 judge proxy 的表面改善主要来自把 `-1` 行为变为 `0`，不是增加 `+1` 行为。策略学会了做更多合法、可逆、低风险的走位，却减少真正改变箱子状态的动作，最后耗尽 horizon。固定验证的推箱率在前 10 step 下降约 72.6%，与这一机制完全一致。

DeepSeek 链路在 step 1--20 始终保持 prompt coverage 1.0，API、鉴权、解析、超时失败均为 0；没有 OOM、NaN 或 fatal。退化不能归因于 judge 服务故障。

step 11--20 的 2,957 个原始 turn 中，`+1/0/-1 = 328/2144/485 = 11.09%/72.51%/16.40%`。与前 10 step 相比，平均 label 从 `-0.1745` 改善到 `-0.0531`，但主要仍是负标签下降 10.83 个百分点、中性标签上升 9.52 个百分点；正标签只增加 1.31 个百分点。同期训练 success 从前 10 step 的 17.19% 降到后 10 step 的 12.50%。

step 21--30 的标签进一步变成 `+1/0/-1 = 7.58%/84.04%/8.37%`，非零 advantage 只剩 15.96%，平均 label 接近 0（`-0.0079`）；该段训练 success 只有 5.94%，pass@8 为 12.50%。最后 5 step 的中性标签达到 86.63%。step 24 的 90.32% turn 为中性、0% 为负，梯度范数仅 `0.028`，说明部分 batch 的有效学习信号已经接近消失。

全 30 step 共 8,845 个原始 turn，`+1/0/-1 = 837/6494/1514 = 9.46%/73.42%/17.12%`。DeepSeek 侧共处理 8,992 个 expanded prompt：3,644 个实际请求、5,178 个 batch 去重、123 个 cache hit、47 个规则负分，累计约 4.60M token；API、解析、鉴权、重试和超时失败均为 0。后 10 step 请求避免率显著上升，也与行为/prompt 同质化一致，但这里只作为旁证。

前 10 step 的精确 macro PPO KL 均值为 `0.0643`、最大 `0.1274`；raw clipfrac 均值 `7.75%`、最大 `19.75%`。所有梯度有限，但若只以 non-zero advantage turn 为分母，部分 step 的有效裁剪率偏高。因此更新强度会放大退化速度，但不是退化目标出现的根因。

每 10 step 的 macro KL 均值为 `0.0643 -> 0.0416 -> 0.0171`，raw clipfrac 为 `7.75% -> 3.78% -> 1.19%`，梯度范数均值约为 `39.5 -> 36.1 -> 12.2`。这不是能力收敛，而是中性标签增多后非零 advantage 消失。全程 KL 范围 `[-0.00594, 0.12735]`、clipfrac `[0, 0.19749]`、grad norm `[0.0281, 65.858]`，均 finite。

最终 checkpoint 位于 `global_step_30`，共 36GB：2 个 model shard、2 个 optimizer shard、2 个 extra-state shard及 Hugging Face tokenizer/config 均存在且可读；`latest_checkpointed_iteration.txt=30`。训练计算已结束，相关进程和 tmux 均已退出，GPU 6/7 已完全释放。

## 结论

macro-action PPO 的概率实现是正确且可训练的；失败的是“把硬三分类进度标签直接当 advantage，并移除 outcome anchor”这一学习目标。

`-1/0/+1` 是一个局部代理标签，不是当前策略下的

```text
A^pi(s_t, a_t) = Q^pi(s_t, a_t) - V^pi(s_t)
```

它既不估计动作之后最终成功概率的变化，也不考虑必要绕路、暂时退让、推箱风险、死锁和剩余 horizon。移除 terminal outcome 后，优化器面对的真实任务变成“避免被 judge 判为更远”，而不是“解开 Sokoban”。在这个代理目标下，少推箱、多走可逆空步是合理的局部最优。这是典型的 proxy overoptimization，而不是 PPO wiring 或 DeepSeek 鉴权 bug。

所以 label-only 方案在本配置下不能 work，不应继续靠增加训练 steps 或微调 optimizer 来挽救。必须先修学习信号。

## 下一轮算法建议

优先级从高到低：

1. **恢复终局锚点。** 保留真实终止奖励，并只在最后一个 turn 注入一次。进度信号作为 shaping，而不是替代任务定义。
2. **把状态价值差分当进度。** 使用

   ```text
   r_t = beta * (gamma * Phi(s_{t+1}) - Phi(s_t))
         + 1[t = T] * r_outcome
   ```

   再通过 turn-level return/GAE 计算 advantage。potential difference 是保持原任务最优策略不变的标准形式。
3. **Sokoban 先用 solver teacher。** 用求解器的剩余最短 push、死锁检测或 soft state value 构造 `Phi`，建立便宜、确定且可验证的上限。CAST 在 Sokoban 上正是把 solver value change 转为 turn advantage。
4. **校准 LLM judge，而不是直接相信符号。** 让模型分别输出 `p_success_before`、`p_success_after` 和 confidence，由代码计算 delta；加入 before/after swap consistency。低置信度或服务错误应 mask，不应在正式算法中强制记为负例。
5. **先做 judge golden audit。** 从 replay 中按 push/non-push、成功/失败、死锁、必要 detour 分层抽 1,000--5,000 个 transition，与 solver delta 比较 confusion、排序、校准和 false-negative。现在只能证明优化 judge 标签损害任务，不能证明每个标签具体错在哪里。
6. **加入 value baseline 和 turn GAE。** 直接未中心化的 `-1/0/+1` 方差高且 batch mean 漂移。可以先用 Monte Carlo terminal return 预训练 value，再逐步加入 bootstrap；生成式 critic 可作为后续对照。
7. **限制单次策略漂移。** 在正确信号下再把 actor LR 从 `1e-6` 下调到 `5e-7`，恢复 adaptive reference KL，并监控 active-turn clipfrac。对更长 turn 比较 exact product 与 geometric-mean ratio；当前大多数 turn 只有 5 token，长度问题不是这次主因。
8. **采用同 seed 多臂消融。** 至少比较 outcome-only、label-sign-only、outcome+sign、outcome+LLM-potential-delta、outcome+solver-delta、continuation/value critic；每臂至少 3 个训练 seed。
9. **扩大且固定评测协议。** 每 10--20 step 做 deterministic greedy pass@1 和 stochastic pass@k，至少 128 条 rollout；记录 success/pass@k、push rate、budget exhaustion、label 分布、non-zero advantage、macro KL/clip、entropy、重复行为和 judge failure。checkpoint 只能按环境 success 选择，不能按 judge score 选择。
10. **异常行为做非对称处理。** 成功轨迹中的重复空转可 mask，避免分享成功奖励；失败轨迹中的同类行为可保留负梯度。环境/API 可恢复故障应 mask，实例不可恢复故障应整条丢弃。

## 推荐的下一次最小实验

先不要立刻训练 LLM judge 版本。建议用同一 Qwen3-4B checkpoint、同一固定验证集和 3 个 seed，运行 30--50 steps：

```text
A: terminal outcome only
B: terminal outcome + 0.1 * solver potential delta + turn GAE
C: terminal outcome + 0.1 * calibrated DeepSeek potential delta + turn GAE
```

如果 B 明显优于 A，而 C 落后于 B，问题在 judge accuracy/calibration；如果 B 和 C 都不优于 A，再检查 value/GAE、macro ratio 和训练强度。这个顺序能把“信用分配想法是否有效”和“LLM judge 是否可靠”分开。

## 参考文献

- [Turn-PPO: Turn-Level Advantage Estimation with PPO](https://arxiv.org/abs/2512.17008)
- [ST-PPO: Stabilized Off-Policy PPO for Multi-Turn Agents](https://arxiv.org/abs/2511.20718)
- [Rewarding Progress: Scaling Automated Process Verifiers](https://arxiv.org/abs/2410.08146)
- [Policy invariance under reward transformations](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
- [CAST: Game Solvers as Turn-Level Teachers for LLM Agents](https://arxiv.org/abs/2607.25308)
- [TRACE: Turn-level Reward Assignment via Credit Estimation](https://arxiv.org/abs/2607.13988)
- [Bringing Value Models Back: Generative Critics](https://arxiv.org/abs/2604.10701)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)

## 安全事项

当前分支和 W&B config 未包含 API key。但此前聊天消息和远端旧 `main` 历史已出现过旧 DeepSeek key；应在服务端立即撤销并轮换。不要仅依赖从当前分支删除字符串，因为 git 历史和聊天记录仍然可见。
