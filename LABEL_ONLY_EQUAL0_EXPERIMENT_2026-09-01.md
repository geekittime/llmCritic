# DeepSeek Label-Only Turn-PPO: Equal=0 实验记录

日期：2026-09-01（Asia/Singapore）

## 实验目标

本实验严格检验以下消融设定：

1. 每个完整 assistant turn 是一个 macro action。
2. turn 概率是 turn 内 sampled token 概率的乘积；实现中在 log 空间求和。
3. DeepSeek 只根据 `(s_t, a_t, s_{t+1})` 输出 `+1/0/-1`：更近为 `+1`，等距为 `0`，更远为 `-1`。
4. DeepSeek 标签直接作为该 turn 的 PPO advantage。
5. outcome weight 为 0，不使用 value critic、GAE、reference KL、reward KL 或 entropy bonus。
6. DeepSeek API、空响应或格式解析失败映射为 0，并单独记录失败指标。

这是 label-only/direct 的诊断性实验，不是推荐的最终长程 RL 算法。

## 关键语义修复

提交 `f64d9d2` 的提示词错误地把 exact-equal、blocked/no-op 和重复状态判成 `-1`。因此诊断 run
[`kv7qtlot`](https://wandb.ai/MuLab-RL/llm-critic-turn-ppo/runs/kv7qtlot) 在 step 1--7 的 neutral rate 始终为 0，
几乎退化成了 `closer=+1, everything else=-1` 的二分类器。该 run 的 W&B 数据同步到 step 7，训练在
step 8 期间被主动停止，不作为最终结果。

提交 `2698a40` 修复了定义：

- `solver_progress_relation=closer -> +1`
- `solver_progress_relation=equal -> 0`，包括状态未变的 blocked/no-op、重复和协议错误
- `solver_progress_relation=farther -> -1`
- 不再把轨迹失败、unsuccessful termination 或 action-budget exhaustion 注入最后一个 turn

真实 DeepSeek 三例网络探针返回 `FINAL_SCORE: 1/0/-1`，解析为 `[1, 0, -1]`；无 API 或鉴权失败。
修复后的 1-step 训练冒烟得到：

- `+1/0/-1 = 22.5%/55.0%/22.5%`
- `closer->+1 = 100%`、`equal->0 = 100%`、`farther->-1 = 100%`
- parse/API/auth failure 全为 0
- `actor/grad_norm=45.95`，确认 actor 确实更新

中止 run 与正式 run 的 step 0 都是 18.75%；step 5 分别为 18.75% 和 28.125%。两者除评分提示词及
run/audit 路径外配置一致，但差值只有 3 条 success，仍只能作为语义修复有效的描述性证据。

## 正式运行配置

- 分支：`experiment/deepseek-only-turn-advantage`
- 训练提交：`2698a40d62e692657f14181ebf80599a4c292109`
- W&B：[`22p8xsjh`](https://wandb.ai/MuLab-RL/llm-critic-turn-ppo/runs/22p8xsjh)，状态 `finished`，最终 step 20
- 主机：A100-006（`lyg0253`），GPU 6、7，2 x A100 80GB
- actor：`Qwen2.5-3B-Instruct`
- DeepSeek：`deepseek-v4-flash`，thinking disabled，并发 16
- train：4 个独立 puzzle x 每题 4 条 rollout，共 16 条 trajectory/step
- validation：32 个独立 puzzle x 每题 1 条 rollout，共 32 条；step 0/5/10/15/20
- seed：train 10000，validation 123
- horizon：10 turns，每 turn 最多执行 1 个动作
- actor LR：`1e-6`；PPO epoch 1；mini-batch 16；micro-batch/GPU 1
- turn advantage：label-only、direct、未归一化；outcome weight 0
- KL/reference/value/entropy：全部关闭
- 本地日志：`/data/kangshijia/logs/llm-critic/lyg0253-sokoban-labelonly-deepseek-equal0-20step-20260901.log`
- judge 审计：`/data/kangshijia/logs/llm-critic/lyg0253-sokoban-labelonly-deepseek-equal0-20step-20260901-critic-audit.jsonl`，权限 600
- `SAVE_FREQ=-1`，本次短实验不保存 checkpoint；W&B 指标和审计记录完整保留

validation 使用固定 puzzle 集合，但采样温度为 0.5，且每次验证前没有重置 vLLM sampler RNG。因此它是同一题集上的分布比较，不是逐 trajectory 配对的确定性评估。

## Validation 结果

| step | success/pass@1 | Wilson 95% CI | episodic return | budget exhausted | avg actions | moved player | moved box | blocked | deadlock after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6/32 = 18.750% | [8.89%, 35.31%] | 1.1875 | 81.250% | 8.750 | 53.594% | 10.990% | 46.406% | 1.875% |
| 5 | 9/32 = 28.125% | [15.56%, 45.37%] | 2.2281 | 71.875% | 8.656 | 60.972% | 12.483% | 39.028% | 5.000% |
| 10 | 5/32 = 15.625% | [6.86%, 31.75%] | 0.8094 | 84.375% | 9.094 | 57.917% | 11.823% | 42.083% | 4.062% |
| 15 | 8/32 = 25.000% | [13.25%, 42.11%] | 1.8750 | 75.000% | 8.750 | 69.635% | 14.167% | 30.365% | 8.438% |
| 20 | 10/32 = 31.250% | [17.95%, 48.57%] | 2.6000 | 68.750% | 8.375 | 66.979% | 17.969% | 33.021% | 4.062% |

step 0 到 step 20 的描述性变化：success `+12.5pp`（多 4 个成功）、budget exhaustion `-12.5pp`、
推箱率 `+6.98pp`、blocked `-13.39pp`。观察到策略变得更主动、推箱行为增加的趋势。

但 success 曲线是 `18.75 -> 28.125 -> 15.625 -> 25.0 -> 31.25%`，中间有明显回落。step 0 与 step 20
的 Wilson 区间大量重叠；若暂按两个独立 32-sample 比例计算，差值的近似 95% 区间为 `[-8.5pp, +33.5pp]`。
这次单 seed、20-step 短跑只支持“有希望的正向趋势”，不能证明稳定提升。

## Judge 与训练信号

20 steps 共审计 2,845 个原始 turn，均经 DeepSeek API backend 获得；重复 prompt 复用同一 API 结果：

| label | count | rate |
|---:|---:|---:|
| -1 | 891 | 31.318% |
| 0 | 1,048 | 36.837% |
| +1 | 906 | 31.845% |

标签总体接近平衡，mean label 为 `(906-891)/2845 = 0.0053`，非零 advantage 率为 63.16%。精确 solver relation 的一致性为：

- closer：905/905 输出 `+1`
- equal：935/935 输出 `0`
- farther：779/779 输出 `-1`
- unknown：226 个，输出 `-1/0/+1 = 112/113/1`

已知 relation 覆盖 `2619/2845 = 92.06%`；在已知 relation 上三类映射为 `2619/2619 = 100%`。
这不是独立的 judge accuracy：提示词已经将 relation 声明为 authoritative，DeepSeek 的任务主要是按给定事实输出标签。

动作审计：1,629 move、961 no-op、252 push、3 invalid。所有 961 个 no-op 都是 0；push 为 `203 positive / 49 negative`；move 为 `703 positive / 86 neutral / 840 negative`。

按 5-step 窗口，no-op 从 `38.38% -> 40.68% -> 28.16% -> 28.47%`，push 从
`8.82% -> 7.97% -> 8.65% -> 9.95%`，最后 10 steps 的行为比前 10 steps 更少 no-op、更常 push。

## API 与效率

- 展开后提交 2,944 个 prompt；1,897 个实际 API 请求，1,047 个批内重复 prompt 被去重，cache hit 0
- 输入 3,703,234 tokens，输出 13,279 tokens，总计 3,716,513 tokens
- 最终失败、鉴权失败、rate limit、server/connection/empty response 和 parse failure 均为 0
- step 13 有 4 次瞬时 timeout、1 次 HTTP retry，重试后全部成功
- DeepSeek API wall time 累计 122.95s；异步 overlap 阶段平均 10.33s/step
- actor 实际等待 judge 累计 5.32s，其余 judge 时间被 rollout/训练流水线覆盖
- API label throughput 的 W&B 逐 step 算术平均为 25.78 labels/s；按总 prompt/累计 API wall time 计算为 23.94 labels/s
- 训练 step 平均 69.90s，总计 1,398.05s（23.30min，不含冷启动）
- W&B 记录的完整 runtime 为 1,445.35s（约 24 分 05 秒）

PPO 稳定性指标：

- `grad_norm` mean/min/max：`77.81 / 37.61 / 108.47`，optimizer 使用 `grad_clip=1`
- macro `pg_clipfrac` mean/max：`8.73% / 18.99%`
- sampled macro `ppo_kl` mean/min/max：`0.0619 / -0.0130 / 0.2606`

所有数值有限，没有 OOM 或训练 fatal。日志末尾的 `BrokenPipeError` 来自 Ray worker 中 W&B atexit socket teardown；主进程返回 0，云端 run 已确认为 `finished` 且最终 step 为 20。

## 代码验证

- 全量测试：`163 passed, 25 warnings`
- critic、turn pipeline、macro PPO targeted tests：通过
- 真实 DeepSeek `closer/equal/farther` 三类探针：通过
- `compileall`、shell `bash -n`、`git diff --check`：通过
- 审计 JSONL 权限：600
- 当前提交与未跟踪文件未包含配置中的 DeepSeek secret

未发现 turn/label 错位、prompt/padding 混入动作概率、old/new log-prob 聚合错误、复制 padding 放大静态梯度或 FSDP micro-batch 缩放错误。

## 算法结论

### 1. 这个具体实验出现了与学习一致的变化，但证据还不够强

在本次单箱、短 horizon Sokoban 中，exact shortest-solution distance 给出了非常强的局部 oracle。沿最短路径的动作使距离减少，反向动作使距离增加，blocked 动作为 equal。因此直接优化三值局部标签近似在做 oracle action classification；最终行为和 success 都观察到描述性的正向变化。

这不能外推为“通用 LLM critic + direct advantage 已经解决长程信用分配”。DeepSeek 提示词直接包含 exact solver relation，905/935/779 个可判定样本被 100% 机械映射；在这个实验里 DeepSeek 主要是在复述 solver 事实，增加了成本和潜在噪声。必须增加 solver-direct baseline 和不提供 solver fact 的 LLM-only baseline。

### 2. 三值 label 不是严格的 advantage

一般情况下，judge 的输出更接近即时 process reward `r_t`，而不是

```text
A^pi(s_t,a_t) = Q^pi(s_t,a_t) - V^pi(s_t)
```

direct 模式只优化当前一步，忽略后续 reward、必要 detour、策略相关 baseline 和终局目标。对一般长程任务，它等价于贪心优化代理目标；去掉 outcome 后可能学会局部看起来正确但无法完成任务的策略。PPO 和 GAE 的标准定义分别见
[PPO](https://arxiv.org/abs/1707.06347) 与 [GAE](https://arxiv.org/abs/1506.02438)。

### 3. 推荐主算法保留 outcome anchor

更稳妥的 Sokoban shaping 是令 `Phi(s)=-d_solver(s)`，使用

```text
r'_t = 1[t=T] * r_outcome + beta * (gamma * Phi(s_{t+1}) - Phi(s_t))
```

再计算 turn-level return/GAE。当 discount 与目标定义一致，并正确处理 terminal/absorbing state potential 时，
potential-based shaping 与原 reward 相加具有策略不变性保证；只保留 sign label 并删除 outcome 不具有该保证，见
[Ng, Harada, Russell 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。

近期 TRACE 也把长程问题建模为工具边界上的 state transition，并用 TD change 分配 turn credit；GenAC 则重新引入 policy-conditioned generative value critic：
[TRACE](https://arxiv.org/abs/2607.13988)、[GenAC](https://arxiv.org/abs/2604.10701)。这些工作支持更细粒度 credit，但不支持把任意一步三值 judge label 直接等同于 advantage。

### 4. 完整 turn 概率乘积数学正确，但有长度风险

若 macro action 定义为完整 token 串，则 `log pi(turn|s)=sum_j log pi(token_j|prefix)` 是精确公式。本实验绝大多数 turn 为 5 tokens，`p95=5`、观测最大值为 13，长度偏差尚不突出。

如果不同 thought/格式字符串映射到同一个环境动作，当前概率是“该 token 串”的概率，不是所有等价字符串概率之和。长 turn 的 summed log-ratio 方差还会随长度增加，固定 PPO clip 会更频繁裁剪长 turn；应记录 active-turn clip by length，并比较 length-aware log clipping。FSPO 对这一固定 clip 的长度不公平给出了分析和 `sqrt(L)` clipping 方案：
[FSPO](https://arxiv.org/abs/2509.09177)。

## 下一步实验优先级

1. 用相同初始 checkpoint 和验证协议跑至少 3 个 train seed、50 steps；使用至少 128 个 validation rollout，同时记录 deterministic greedy pass@1 和 stochastic pass@k。
2. 做五臂等预算对照：outcome-only、solver-sign direct、DeepSeek-sign without solver facts、outcome+solver potential+GAE、outcome+DeepSeek process+GAE。
3. 对本任务先直接使用 solver relation，避免让 DeepSeek 复述已知答案；只有在没有 solver 的任务上评估 LLM critic 的真实增益。
4. 恢复 terminal outcome anchor，并把 process score 当 shaping reward；先用 Monte Carlo return/`lambda=1` 训练 value，再逐步启用 bootstrap GAE。
5. 增加 nonzero-advantage-only clipfrac、macro log-ratio p50/p90/p99、按 turn length 分桶的 clip/KL，以及 deterministic validation RNG reset。
6. API/parse failure 继续映射 0，但必须与真实 neutral 分开统计；超过失败阈值时跳过整批 actor update，不让服务抖动静默改变有效学习率。
7. 若 active clip 持续超过约 20%，优先增大 mini-batch/减少每 rollout optimizer steps或降低 LR，再评估 length-aware clipping；不要先用调参掩盖错误学习信号。

## 结论

修复后的实现完整遵循 `closer=+1, equal=0, farther=-1`，代码链路和 API 均工作。20-step 单次实验从 `18.75%` 到 `31.25%`，并伴随更多有效移动、更多推箱和更少预算耗尽；这些是与该 solver-grounded 局部标签产生有效更新相一致的描述性现象。

但样本区间重叠、曲线波动明显、只有一个 seed，而且 DeepSeek 直接看到了 exact solver relation。因此当前结论应是：**实现可训练，结果有希望，但尚未证明泛化有效；它更像 solver process reward 经 DeepSeek 中转，而不是通用 LLM critic advantage。**
