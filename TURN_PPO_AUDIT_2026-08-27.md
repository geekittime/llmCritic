# Turn-PPO / LLM-Critic 审计与实验记录

日期：2026-08-27（Asia/Singapore）
仓库：`/home/kangshijia/wangbinyu/llm-critic`
当前 HEAD：`31c44450b889762a37bf44b63c6f1c36df0e2905`
参考备份：`/home/kangshijia/wangbinyu/llm-critic-bak`，HEAD `784e403`

## 结论摘要

把一个完整 assistant turn 当作 macro action，在环境只于 turn 结束后转移的前提下是合法的 SMDP/层次 MDP 建模。若 turn token 为 `y_k`，正确的宏观似然是

```text
log pi(y_k | s_k) = sum_i log pi(y_{k,i} | s_k, y_{k,<i})
rho_k = exp(log pi_new(y_k) - log pi_old(y_k))
```

因此“每个 turn 的 token 概率相乘，再做 PPO clipping”在理论上可以 work；MA-RLHF 和 Turn-PPO 都给出了这一类目标的独立实现/实验依据（见参考文献）。

但当前 HEAD **不能据此称为已经实现并验证了 Turn-PPO**：

1. 训练入口注册的 `ragen.workers.fsdp_workers.ActorRolloutRefWorker` 继承上游 worker，其 `init_model` 实际来自 `verl.workers.fsdp_workers`，加载的是上游 token-PPO actor/critic；`ragen/workers/actor/dp_actor.py` 的 turn loss 没有被接入。
2. 默认 `use_label_outcome_advantage=True` 时，`agent_trainer.py` 直接把 label+outcome 写成 `advantages/returns`，绕过 value、GAE 和 turn-GAE。默认 critic 又是关闭的，实际更接近无 baseline 的 outcome-only REINFORCE/PPO smoke，而不是论文意义上的 Turn-PPO-GAE。
3. 即使强制走自定义路径，action mask 有确定性的 shift 错位，且把下一轮 user marker、固定 chat-template token 纳入 action；重 tokenization 也不保证等于 rollout 的原始 token span。
4. 当前 HEAD 没有 `deepseek_api` backend。参考脚本的 DeepSeek 配置先被 Hydra 拒绝；绕过字段检查也会在 `_generate_texts()` 断言模型未加载。备份工作树的 API 代码还把 API/解析异常转成负标签，风险很高。

所以答案是：**研究方向可行，当前代码原样不可靠；在修复 wiring、mask、reward/GAE 和 judge failure 处理前，不能把训练曲线解释为该算法有效。**

## 今天的运行记录

所有命令均在服务器执行，未杀死或干扰其他用户进程。完整 stdout/stderr 保存在 `/tmp`，没有在此文件记录任何密钥。

| 检查/尝试 | 结果 | 日志 |
|---|---|---|
| 当前 HEAD，参考 DeepSeek 字段、可用模型替代、单卡/极小 batch 的配置 smoke | 在 Ray 启动前失败：`generative_critic.deepseek_model` 不在 structured config；提示只能用 `+` 添加。当前 config 也没有 API 实现，不能简单加 `+` 修好 | `/tmp/llmcritic-current-head-smoke-20260827_195422.log` |
| 当前 HEAD，关闭 critic/gen-critic 的最小 baseline（修正 filter ratio 与 mini batch 约束，Qwen3.5-0.8B 路径） | Hydra 和 Ray head 成功启动，但约 120 秒没有注册训练 worker/rollout，因 8 张 A100 均被其他任务占满而终止；没有产生训练指标或 checkpoint | `/tmp/llmcritic-current-baseline-smoke-20260827_195534.log`（首个配置约束失败），`/tmp/llmcritic-current-baseline-smoke2-20260827_195642.log` |
| 当前 HEAD，venv、短 `RAY_TMPDIR`、Qwen3-4B、`total_training_steps=0` 预启动 | Hydra/Ray 配置成功，排除了长 Unix socket 和 Qwen3.5 架构问题；约 75 秒仍未注册 TaskRunner/worker，因 GPU 全满被终止，没有训练指标 | `/tmp/llmcritic-venv-preflight-short-20260827.log` |
| 备份参考脚本的脱敏 smoke（W&B offline、Qwen3-4B、单卡、1×2 rollout、1 turn） | Hydra/Ray 成功，但约 180 秒未进入 worker/训练指标，因 GPU 全满终止；首次尝试还触发固定 `rollout_filter_ratio=0.5` 与 mini-batch 约束。没有发生 API 请求 | `/tmp/llmcritic-bak-reference-smoke-20260827_205213.log`、`/tmp/llmcritic-bak-reference-smoke-20260827_205251.log` |
| DeepSeek API 健康检查 | 2026-08-27 20:08（+08:00）返回 HTTP 401 `authentication_error`。shell 与 ragen venv 均未设置 `DEEPSEEK_API_KEY`；只验证了参考脚本中旧凭据已不可用 | `/tmp/llmcritic-deepseek-health-20260827.log` |
| venv Sokoban CPU reset/step | 成功。`/home/kangshijia/venvs/ragen` 的 gym 版本支持 `Discrete(start=1)`，reset/step 返回正常 | 命令输出见本次会话；建议固定该 venv |
| conda `ragen` Sokoban CPU 构造 | 失败：conda 中 `gym==0.21` 的 `Discrete` 不接受 `start`；切换 gymnasium 的临时 monkeypatch 可运行。不要混用两套环境 | `/tmp/llmcritic-sokoban-env-smoke-20260827.log` |
| worker wiring 静态/运行时检查 | `ActorRolloutRefWorker.init_model.__module__ == verl.workers.fsdp_workers`；上游 actor 没有 `turn_ids`/turn loss，`update_policy` 源码也没有 `turn_ids`；自定义函数只存在于 `ragen.workers.actor.dp_actor` | `/tmp/llmcritic-worker-wiring-20260827.log` |
| worker 路由复核（venv，无模型加载） | 自定义与上游 `DataParallelPPOActor/Critic` 类 identity 均为 False；继承链和 `init_model` 源码均无 `ragen.workers` 导入 | `/tmp/llmcritic-worker-routing-check-20260827.log` |

### 环境与资源

- ragen venv：Python 3.12.13、torch 2.6.0+cu124、vLLM 0.8.2、transformers 4.57.1、Ray 2.55.1；CUDA 可用。`/home/kangshijia/sicheng/cuda-12.4` 与 torch CUDA 版本匹配，不是当前阻塞。
- 8 张 A100-SXM4-80GB 当时均接近满显存且 100% 利用率，属于其他用户进程；没有强行抢卡。
- 参考脚本默认的 `/data/kangshijia/wangbinyu/models/Qwen3-0.6B` 和当前脚本的 `/mnt/kangshijia/models/Qwen3-0.6B` 均不存在。
- 可见的兼容候选是 `/data/kangshijia/sicheng/AgentGym-RL/models/Qwen3-4B-Instruct-2507` 与 `/data/kangshijia/sicheng/AgentGym-RL/models/Qwen2.5-3B-Instruct`。`Qwen3.5-0.8B` 的 config 在当前 transformers/vLLM 组合中报 `model type qwen3_5 not recognized`，不应作为正式 run 模型。

## 当前 HEAD 的实际数据流

### 两条互斥路径

`config/base.yaml` 默认值是：

```text
critic.enable = False
algorithm.adv_estimator = gae
algorithm.use_label_outcome_advantage = True
generative_critic.enable = False
```

在 `ragen/trainer/agent_trainer.py:1371-1403`，只要 `use_label_outcome_advantage=True`：

```text
label_tensor = 0（critic 未启用时保持全零）
outcome = sum(token_level_scores * response_mask)
combined = label_weight * label_tensor + outcome_weight * outcome
advantages = combined
returns = combined
```

随后不会进入 `compute_advantage()` 的 GAE 分支。因此默认脚本即使把 `adv_estimator=gae` 写出来，也不是 GAE。只有把该 flag 设为 false，且提供正确的 critic `values`，才会调用 `compute_turn_gae_advantage_return()`；若关闭 flag 却仍关闭 critic，通常会缺少 `values`。

版本不要混淆：当前 HEAD 的 label+outcome 组合在上面的分支中；备份 HEAD `784e403` 增加了 DeepSeek API 字段/backend；备份工作树另有未提交改动，把 actor 信号改成 DeepSeek-only 并改变 false 分数。参考脚本与这几个版本不能直接互换。

### 宏概率实现本身

`ragen/workers/actor/dp_actor.py:47-100` 会按 `turn_ids` 对每个 turn 的 old/new token log-prob 求和，再调用普通 PPO loss。这一数学操作等价于 token 概率乘积，方向正确；但它只有在该 actor 类真的被 worker 实例化、且 mask 只含真实 sampled completion 时才成立。

## 需要优先修复的问题

### P0：worker 没有接入自定义 actor/critic

`train.py:209-229` 映射到 `ragen.workers.fsdp_workers`，但该类没有覆写 `init_model`。继承链最终执行 `verl/verl/workers/fsdp_workers.py:732-789`，其中硬编码 `from verl.workers.actor import DataParallelPPOActor`；critic 也在 `1413-1433` 导入上游类。上游 `verl/verl/workers/actor/dp_actor.py` 的 `select_keys/update_policy` 不认识 `turn_ids`，所以当前训练可能完全是 token-PPO。

修复方式：在 ragen worker 覆写 `init_model` 并显式导入 `ragen.workers.actor/critic`，或把自定义实现正式放入 verl 的加载路径；启动时打印 actor/critic 类的 `__module__`，并加一个断言/单测确认 turn loss 被调用。

### P0：action mask 的 off-by-one 与非动作 token

`ctx_manager.py:44-69,996-1003` 在 source 位置计算 assistant mask，然后去掉最后一列；但 actor 的 `logits[j]` 预测的是 `input_ids[j+1]`。assistant 区间 `[p,q)` 实际被训练成目标 `[p+1,q]`：漏掉首 token，却把下一轮 user 的 `<|im_start|>` 纳入上一 turn。rollout 还经过 decode（`skip_special_tokens=True`）再重新 tokenize，不能保证 old log-prob 对应原始行为策略。

修复方式：rollout 保存原始 response token IDs 及每个 turn 的 sampled span；用 destination-aligned mask（至少 `assistant_mask[:,1:]`），排除 role/open/close 标签、next-user marker 和未执行的第二个 primitive action。先将 `max_actions_per_turn=1` 做基准验证，再恢复多动作。

特别是环境在一个 turn 内可能执行不满 `max_actions_per_turn=2`（done 或无效动作会提前停止），但当前 token span 仍覆盖整段 assistant 文本；未执行的动作 token 不应得到 policy gradient。

还应保存 rollout 时的真实行为分布/log-prob；生成后用另一套 actor 重算，或忽略 top-p 截断后的重归一化，会引入额外 off-policy ratio 偏差。

### P0：label/outcome 不是正确的 turn advantage

- `_compute_outcome_tensor()`（`agent_trainer.py:384-387`）把整条轨迹分数复制到每个 response token；若走 turn-GAE，随后按 turn sum 会产生 token-length bias。
- label 是 0/1，false 在当前 HEAD 并不是脚本想象的 `-1`；失败/解析失败可能没有负梯度。Sokoban 失败仍可能有每步 `-0.1`，但量级与成功的 `+10` 不同，不能混为一谈。
- label 分支直接写 `returns`，没有把 turn return 写在 `turn_value_mask` 的 boundary；若同时开 value critic，critic 的有效位置通常拿到 0，后续 GAE 会学错。
- `use_kl_in_reward=True` 时先算出的 KL reward 在 label 分支被重新覆盖，KL 信号被静默丢弃。

推荐目标：每个 turn 只生成一个标量 `R_k`，中间进度奖励和终局奖励在 turn EOS/boundary 放一次；对真实终止给 `R_last += outcome`，对截断给 bootstrap；再按 turn 计算

```text
delta_k = R_k + gamma * V(s_{k+1}) - V(s_k)
A_k = delta_k + gamma * lambda * A_{k+1}
```

若折扣按 token 时间定义，应使用 `gamma ** duration_k`；若目标是 decision-level undiscounted return，明确说明 `gamma=1`。将 `A_k` 广播到该 turn 的真实 sampled tokens，但按 turn（最好先按 trajectory）归一化。

### P1：critic prompt/监督的时间对齐

`_build_samples_full()` 删除 trailing state-only history，并仅在非最后条目追加 reward user 消息；因此最后一个 turn 的 judge 看不到 `s_{k+1}` 和 immediate reward。训练 critic 若按整条轨迹 success 给所有 turn True/False，也不是“这一步是否更接近目标”的 transition label。

应保留终止后的最终状态/成功标志；最后 turn 直接使用环境 verifier。trainable critic 目标应是 transition-level progress，而非把 trajectory outcome 复制到每个 turn。

### P1：宏 ratio 的长度方差与损失加权

raw product 是 exact macro likelihood，但长 turn 的 `sum(log-ratio)` 很快越过 PPO clip：每 token 差 `0.01` 时，长度 20/40/80 的 ratio 约为 1.22/1.49/2.23。当前 flatten 后对所有 turn 全局 mean，turn 多的轨迹权重更大。

建议至少记录 turn ratio 分位数、clipfrac、KL、gradient norm；做 raw-sum 与 geometric mean `exp(mean(log-ratio))` 两个 ablation，并按 trajectory 先平均 turn loss。GSPO/ST-PPO 给出了长度归一化的可复用实现思路。

### P1：LLM judge 的可靠性与失败处理

当前解析器在严格 `###label:` 失败时会回退到输出中最后一个 `True/False`，rationale 中的布尔词可能被误当标签。备份 API 实现把异常变成空字符串，再按默认 False/负分继续更新；401、429、超时因此会变成系统性负反馈。

应使用严格单 token/JSON schema parser；API 失败的 turn mask=0 或跳过该 batch，绝不能等价于 False。judge 使用冻结、独立模型，temperature=0 或多次投票；在 solver/human golden set 上报告准确率、混淆矩阵、swap consistency、置信度/ECE，并缓存 prompt 结果控制成本。

### P1：环境与指标

`ragen/env/sokoban/env.py:49-55` 丢弃了 Gym 原始 `info`，没有 `raw_reward`；`es_manager` 因而记录的 `episodic_return/raw_reward` 常为 0。W&B 中应以 `success`、`pass@k`、有效动作率和环境真实 reward 为主，并修 wrapper 保留 raw reward。还要统一 gym/gymnasium 环境，固定 venv、模型路径、commit 和依赖锁。

另外，rollout filter 如果读取的是已经做过 episode/group normalization 的分数，`reward_variance` 排序可能失去意义；应单独保存 raw episodic reward。参考脚本还关闭了 `actor.use_ref`，因此 `USE_BASE` 中的 KL 系数并不会自动提供 KL 锚定，建议作为明确的有/无 KL ablation。

## 已有 W&B 记录（历史旁证，不是今天的复现）

这些日志来自备份仓库 2026-05 的本地 W&B runs；它们的 commit、模型、batch 和代码工作树并不等于当前 HEAD。尤其 `6km8hkya` 是修过 API parser 的备份工作树，且使用了 DeepSeek-only label 分支，不是本文目标的 label+outcome 组合。备份的 `verl` 子模块仍由上游 `fsdp_workers.init_model` 导入上游 actor；除非当时运行环境另有未记录的 monkeypatch，这些曲线也不能证明自定义 `ragen` turn actor 已执行。

| run | 训练步数 | 观察到的结果 |
|---|---:|---|
| `1igxfoix` | 1 | success .1406，pass@16 .625；critic parse-fail .2708，true rate .0347；首步很慢（约千秒级） |
| `bti26rrm` | 10 | success 均值约 .170，最高 .367；末步 success .0938/pass@16 .375；parse-fail 约 .22-.32，单步常数百至数千秒 |
| `6956b8bx` | 14 | 每步 parse-fail=1、true=0；末步 success .0313/pass@16 .25，action valid .108，manager_invalid .892 |
| `ew2p3u6u` | 19 | parser 失败链持续；末步 success=0/pass@16=0，manager_invalid .992；从初始 success .1895 明显坍塌 |
| `23stigja` | 55 | parser 失败链持续；末步 success=0/pass@8=0、num_actions=0、manager_invalid=1 |
| `6km8hkya` | 28 | parser/API failure=0；末步 success .0625/pass@8 .25，true rate .578，manager_invalid .010；步间 success 高方差、无持续上升趋势；约 5.4 分钟/步 |

这些结果支持两个工程判断：parser/API failure 会直接造成策略坍塌；即使 parser 修好，28 步的短 run 也不足以证明收益，且吞吐/成本很高。当前参考脚本的 critic 并发 128，在每个样本最多 5 turn 时很容易触发限流，建议先用离线缓存和并发 8--16。

## 建议的修复后实验矩阵

固定模型、初始状态、训练/验证 seed、batch 和总 token 预算，至少跑 3 个 seed：

1. **Token-PPO-GAE baseline**：`use_label_outcome_advantage=False`，正确启用 critic，确认上游 token actor 的指标。
2. **Exact Turn-PPO**：接通自定义 worker；真实 turn span mask；terminal-only reward；turn value + GAE。
3. **Oracle progress**：用 Sokoban solver 的 push-distance/deadlock potential，作为 signed `Delta Phi`，验证 credit assignment 上限。
4. **Frozen LLM progress**：只加校准后的 signed progress，限制权重并记录 judge agreement/abstain。
5. **Combined**：`A_k` 来自 progress + terminal outcome；terminal outcome 只出现一次，比较权重和 geometric/raw ratio。
6. **Noisy/random judge 对照**：证明收益来自有效 credit，而不是额外随机噪声。

每步记录：success、pass@k、真实 episodic return、valid/invalid action、turn/token length、label true/parse/API failure、reward 分布、value loss、turn ratio 分位数、clipfrac、KL、entropy、gradient norm，以及 local reward 与 terminal success 的差距。出现 invalid/action collapse 或独立 verifier 下降时提前停止。

## 工程与安全事项

- `train_sokoban.sh` 和备份 DeepSeek 脚本包含硬编码 W&B/DeepSeek 凭据。应立即在服务端撤销/轮换，改为运行时必填的环境变量或 secret manager；不要把脚本原样提交、复制到日志或命令历史。
- 不要仅用 Hydra 的 `+generative_critic.*` 绕过未知字段；先移植/合并真正的 API backend，并加 HTTP 401/429/timeout 的 fail-fast 测试。
- 运行前固定短的 `RAY_TMPDIR`（Unix socket 路径过长会使 Ray 直接失败），固定 `/home/kangshijia/venvs/ragen/bin/python`，并在启动日志打印 commit、模型路径、worker class module、gym/transformers/vLLM 版本。

## 测试状态

- `compileall -q ragen tests`：通过。
- `bash -n`：目标脚本及备份 DeepSeek 脚本语法通过。
- 聚焦测试（`tests/es_manager/test_seed_iteration.py`、`tests/test_rollout_filter.py`、Sokoban render）：4 通过、1 失败；失败测试要求坐标渲染含 `Walls: (...)`，而当前 `format_coordinate_render` 输出没有该字段，属于现有环境渲染契约不一致。原始输出保存在 `/tmp/llmcritic-focused-tests-20260827.log`。
- 全量 pytest 未进入断言阶段，收集时报 49 个错误，主要是缺失可选依赖、重复 `verl`/test 包路径和外部 kimina/prisma 等环境问题；不能把它解释成 49 个算法失败。
- 已完成纯 CPU 的 turn-GAE、宏 ratio、tokenizer turn-boundary 数值检查；宏 log-prob 求和数学上正确，但 boundary/mask 对齐问题被复现。数值输出保存在 `/tmp/llmcritic-turn-numeric-check-20260827.log`。

## 参考文献（联网核验）

- Turn-PPO（RAGEN 基础上的 turn-MDP、宏 ratio、turn-GAE 与 Sokoban 对照）：[arXiv:2512.17008](https://arxiv.org/html/2512.17008)。
- MT-PPO/细粒度多轮奖励；论文同时警告局部 turn reward 可能被过优化：[arXiv:2505.11821v3](https://arxiv.org/html/2505.11821v3)。
- Macro actions 与宏 PPO：[MA-RLHF](https://arxiv.org/html/2410.02743)。
- 长序列 sequence ratio 的几何平均稳定化：[GSPO](https://arxiv.org/html/2507.18071v2)；ST-PPO/SORL 的 turn ratio 与 clipping-bias 讨论：[arXiv:2511.20718](https://arxiv.org/html/2511.20718)。
- 进度奖励应近似 signed advantage，而不是绝对“好/坏”标签：[Rewarding Progress](https://arxiv.org/html/2410.08146)。
- 任意 shaping 只有采用 potential difference `gamma*Phi(s')-Phi(s)` 才保持原最优策略：[Ng et al., ICML 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。
- LLM judge 的位置、冗长和自增强偏差：[Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)。

## 2026-08-28 实现后验证

### 可回溯提交

| 提交 | 内容 |
|---|---|
| `e116389` | 初始代码审计、历史运行记录和风险清单 |
| `ac242d1` | DeepSeek signed turn-PPO 实现、精确 token trace、并行 critic、worker wiring、测试和启动脚本 |
| `3088304` | Ray worker 运行时凭据转发（仅从环境变量读取）以及 legacy critic 默认配置兼容修正 |
| `61f56ca` | 切换到官方当前 `deepseek-v4-flash`，显式关闭 thinking，并将模式纳入请求缓存键 |
| `c36ab18` | 更新 Flash 模型验证记录 |
| `c8477b0` | 统一 console/W&B tracker 配置脱敏，新增凭据不落日志回归测试 |
| `59ce033` | 在系统/turn 提示中区分轨迹动作预算与单 turn 动作上限 |

当前分支为 `feature/turn-ppo-deepseek-progress`，已推送到 `github.com/geekittime/llmCritic`，远端与本地均指向最新实现提交。实现提交没有把 DeepSeek 或 W&B 凭据写入 YAML、shell 参数或日志；Ray 启动时只转发进程环境中已经存在的对应变量。

### 已实现的训练路径

1. 每个 assistant turn 保留 rollout 原始 `prompt_token_ids`/`response_token_ids`，训练时拒绝隐式重新分词（除非显式打开 legacy fallback）。因果 LM 的 response mask 与 logits 做了 `j -> j+1` 对齐。
2. actor 将一个 turn 内所有 token 的 old/new log-prob 在 log 空间求和，再计算 `exp(sum(new)-sum(old))` 的 PPO ratio；loss 以 turn 为单位平均，而不是按 token 数重复加权。
3. 每条轨迹的所有 turn 通过一个 `asyncio.gather` 批次并发请求 DeepSeek，使用有界 semaphore、批内去重、跨 step LRU 缓存、有限重试和 API 健康指标。
4. 提示词明确要求比较 `s_t,a_t,s_{t+1}`，设置/重定位动作可以是正向；最后非空行必须是 `FINAL_SCORE: 1/0/-1`。最终行解析失败、空响应、认证/网络失败都按需求记为 `-1`，padding 不参与训练。
5. 轨迹成功严格映射为 `+1`，失败、截断或缺失 success 字段映射为 `-1`。默认 `outcome_broadcast=all_turns`，每个 turn 的直接分数为 `label_weight * judge + outcome_weight * outcome`，并以 turn 末端 reward 形式接入 PPO。

### 验证结果

- 针对性测试：`33 passed`（`tests/test_generative_critic_api.py`、`tests/test_turn_ppo_core.py`、Sokoban/env、seed iteration、rollout filter），只有 Gym/Ray 的既有弃用警告；其中新增测试确认默认 Flash 请求带 `thinking=disabled`，以及 tracker 配置凭据脱敏。
- `/home/kangshijia/venvs/ragen/bin/python -m compileall -q ragen tests train.py`：通过。
- 目标脚本及旧 launcher `bash -n`：通过；`git diff --check`：通过。
- `DRY_RUN=1` 小 batch Hydra 预检返回码 0，解析了完整配置但没有启动 Ray、模型或 API。日志：`/tmp/llmcritic-dryrun-20260828.log`。
- 在 `61f56ca` 后再次用 `deepseek-v4-flash` 预检返回码 0；解析到 `deepseek_model: deepseek-v4-flash`、`deepseek_thinking: disabled` 和 `parse_fail_score: -1`。日志：`/tmp/llmcritic-dryrun-v4-20260828.log`。
- 无密钥启动预检在模型/Ray 初始化前返回码 1，并明确提示 `DEEPSEEK_API_KEY` 缺失。日志：`/tmp/llmcritic-script-no-key-20260828.log`。
- 最新无密钥启动预检返回码仍为 1；带占位 key 的 `DRY_RUN=1` 返回码为 0，并解析出 `deepseek_model: deepseek-v4-flash`、`deepseek_thinking: disabled`、`parse_fail_score: -1`。日志：`/tmp/llmcritic-script-no-key-final-20260828.log`、`/tmp/llmcritic-dryrun-final-20260828.log`。
- `59ce033` 后重新跑的聚焦测试仍为 `33 passed`；提示词现在同时声明 trajectory budget 与 per-turn action limit，减少被执行列表截断的额外动作。
- 全量 pytest 仍会在收集阶段受服务器缺失可选依赖、重复外部测试包等环境问题阻塞；这不等同于本实现的断言失败，详见上面的历史测试状态。

### 正式训练状态

本次没有伪造训练曲线：当前 shell 没有 `DEEPSEEK_API_KEY`/`WANDB_API_KEY`，且 8 张 GPU 均处于约 94--97% 显存占用、100% 利用率（属于其他任务）；因此没有启动会抢占资源或产生 API 费用的正式 run，也没有新的 checkpoint/W&B 指标。脚本默认使用官方当前的 `deepseek-v4-flash` 非思考模式；取得预留 GPU 和运行时凭据后可用如下方式启动：

```bash
cd /home/kangshijia/wangbinyu/llm-critic
export DEEPSEEK_API_KEY='在 shell 外部注入，勿写入命令历史/配置'
export WANDB_API_KEY='可选；缺失时脚本自动使用 offline'
CUDA_DEVICES=0 N_GPUS=1 RUN_NAME=sokoban-turn-ppo-deepseek-v4-flash \
  bash train_sokoban_deepseek_turn_ppo.sh
```

第一次正式运行建议先使用 `TOTAL_STEPS=1 TRAIN_ENV_GROUPS=1 TRAIN_GROUP_SIZE=1 VAL_ENV_GROUPS=1 VAL_GROUP_SIZE=1`，确认 W&B 中 `gen_critic/api_failure_rate`、`gen_critic/parse_fail_rate`、`train/outcome_success_rate`、`train/turn_count` 和 actor `pg_clipfrac` 正常，再扩大 batch/步数。

### 算法判断与后续建议

把一个完整 turn 视为 SMDP/macro action，在行为策略 tokenization 不变时使用 token 概率乘积（log-prob 求和）是数学上成立的 PPO 目标；相关 turn-level PPO 和 macro-action 文献包括 [Turn-PPO](https://arxiv.org/html/2512.17008) 与 [MA-RLHF](https://arxiv.org/html/2410.02743)。截至本记录日期，DeepSeek 官方文档列出的当前低价模型是 `deepseek-v4-flash`，而 `deepseek-chat`/`deepseek-reasoner` 已被标为兼容旧名称并进入弃用路径；价格和可用模型应以 [官方定价页](https://api-docs.deepseek.com/quick_start/pricing/) 与 [模型列表](https://api-docs.deepseek.com/api/list-models/) 为准。 但 DeepSeek 的 `-1/0/+1` 是冻结 judge 的启发式 reward，不是无偏 advantage：judge 偏差、状态截断、提示注入和 API 失败会直接改变策略梯度。

建议固定总 token 预算跑至少 3 个 seed，并保留以下对照：token-PPO baseline、仅 terminal outcome、oracle Sokoban progress、DeepSeek progress、`label+outcome`、随机/噪声 judge。每个 run 记录 turn 长度、ratio 分位数/clip fraction、KL、有效动作率、judge 失败率和独立 verifier 的 success。长 turn 的 raw product 容易快速越过 PPO clip，可增加 geometric-mean ratio 对照；[GSPO](https://arxiv.org/html/2507.18071v2) 和 [ST-PPO](https://arxiv.org/html/2511.20718) 讨论了这一稳定性问题。若希望 shaping 不改变最优策略，应把进度分数校准为 potential difference `gamma*Phi(s')-Phi(s)`，而不是无条件复制 terminal reward；参见 [Ng et al.](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。

当前 launcher 采用直接 turn advantage、关闭 trainable value critic，因此完全覆盖用户指定的 `judge + outcome` 目标。精确 trace builder 为保持行为策略一致而一 turn 一行；若以后开启 value critic，需要按 `episode_ids` 在行之间恢复跨 turn GAE，不能把每一行当成独立 episode。另一个应做的 ablation 是 `outcome_broadcast=last_turn`，因为 `all_turns` 会重复终局信号并可能放大长轨迹权重。

### 凭据处置

用户消息中出现过 DeepSeek key，且旧仓库历史曾出现 W&B token。即使工作树已清理，也不能消除 Git 历史或服务端日志中的泄露；在正式运行前应立即吊销并轮换两类凭据，改用临时环境变量/secret manager。
