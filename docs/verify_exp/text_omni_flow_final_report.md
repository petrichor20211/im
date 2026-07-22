# 文本 Omni-Flow 实验报告

## 固定节拍下的持续输入、分块输出与独立控制 token 实验

- **报告日期：2026-07-22**
- **基础模型：Qwen3.5-0.8B**
- **训练方法：QLoRA**
- **主要比较：无独立控制 token 的 C 组 vs. 有独立控制 token 的 D 组**

---

## 摘要

本实验研究一个普通的 decoder-only causal language model 能否学习一种简化的“持续交互”协议：环境按固定 tick 推进，每个 tick 都可能加入新输入；模型必须决定保持沉默、开始回答、继续回答还是停止回答；每个 tick 最多输出 4 个普通 tokenizer token。

实验使用 Qwen3.5-0.8B，不修改 Transformer 架构，只增加 11 个协议特殊 token，并通过 QLoRA 训练。合成数据共 50,000 个 session，覆盖等待完整输入、关键词触发、周期输出、流式分块、用户打断和干扰输入六类任务。

核心对照是：

- **C 组**：不直接预测 `LISTEN/SPEAK/CONTINUE/STOP`；通过输出块是否为空以及历史状态推断行为。
- **D 组**：在每个 tick 显式预测独立的 `LISTEN/SPEAK/CONTINUE/STOP` 控制 token，再决定是否输出内容。

正式训练完成了 C/D 各 3 个随机种子。由于全量逐 tick 评价成本很高，最终行为分析按照用户要求只使用 seed `20260721` 上两个已经完整完成的测试集：in-domain 和 template OOD，每组、每个测试集各 2,000 个 session。

主要结论是：

1. 模型确实学会了固定文本节拍、等待、触发、周期输出和保持沉默；
2. 模型能够完成多 tick 流式输出，但严格要求每个 tick 与唯一参考分块完全相同，会显著压低成功率；
3. D 组没有在总体成功率上优于 C 组，更没有达到预注册的“高 10 个百分点”；
4. D 组的最终内容重建率反而比 C 组低 2.5–3.3 个百分点；
5. 表面上 D 在 interrupt 任务更好，但该结果受到 C 组 STOP 表示与评价标签不一致的结构性偏差影响，不能作为独立控制 token 有效的可靠证据；
6. 两组都没有发生格式崩坏：overflow 和 malformed tick 均为 0。主要困难是内容分块和结束时机，而不是协议语法。

因此，实验支持“普通 causal LLM 可以学习固定 tick 的文本交互协议”，但**不支持“加入独立控制 token 会带来显著总体收益”这一强假设**。

---

## 1. 为什么要做这个实验

常规文本语言模型通常采用一次性接口：用户给出完整 prompt，模型随后连续生成完整答案。真实交互却可能包含以下行为：

- 用户输入尚未结束时，系统应保持沉默；
- 条件满足后，系统才开始输出；
- 一次只能输出很短的一段，然后必须继续接收新输入；
- 用户可能在系统说话时要求停止；
- 系统需要跨多个时间片保存“我现在是否正在说话”的状态。

直接从语音全双工系统开始会同时引入声学模型、实时调度、语音活动检测和延迟等变量。因此本实验先构造一个纯文本、离散时间版本，以隔离最核心的问题：

> 一个未经架构修改的 causal LLM，能否把交互控制也当作 token 序列学习？

这里的 tick 是人工定义的离散时间片，不是真实秒数。实验成功也只能说明模型学会了文本状态机，不能直接证明模型具备真实时间感或语音全双工能力。

---

## 2. 研究问题与假设

### H1：固定时钟学习

模型在看不到绝对 tick 编号的情况下，能否根据历史执行：

- 等待若干 tick；
- 在关键词出现时触发；
- 从 `START` 开始按固定周期输出；
- 条件未满足时持续保持沉默。

### H2：流式输入和输出交错

模型能否遵循以下因果顺序：

```text
读入一部分输入 → 输出最多 4 tokens → 再读新输入 → 继续输出
```

而不是必须先读完整个输入，再一次性给出全部答案。

### H3：控制与内容解耦是否有效

显式预测独立控制 token，再预测文本内容，是否比把“沉默/输出”隐含在同一个输出空间中更稳定？

预注册的强判据是：D 组的 Joint Session Success 至少比 C 组高 **10 个绝对百分点**。

---

## 3. Tick 协议

### 3.1 特殊 token

在 tokenizer 中新增 11 个特殊 token：

```text
<tick> <input> </input> <output> </output>
<listen> <speak> <continue> <stop>
<out_end> <tick_end>
```

其含义如下：

| Token | 含义 |
|---|---|
| `<tick>` | 新 tick 开始 |
| `<input>...</input>` | 当前 tick 新收到的输入 |
| `<output>...</output>` | 当前 tick 的输出块 |
| `<listen>` | 保持沉默，继续等待 |
| `<speak>` | 从 idle 状态开始输出 |
| `<continue>` | 已在输出状态，继续输出下一块 |
| `<stop>` | 立即停止当前输出 |
| `<out_end>` | 当前答案自然结束 |
| `<tick_end>` | 当前 tick 结束 |

每个 tick 最多接收 16 个输入 token，最多产生 4 个普通输出 token。特殊 token 不计入这 4 个 token 的预算。

### 3.2 状态机

模型只有两个抽象状态：

```text
IDLE      尚未输出或已经结束
SPEAKING  正在跨 tick 输出
```

合法转移为：

```text
IDLE + LISTEN      → IDLE
IDLE + SPEAK       → SPEAKING
SPEAKING + CONTINUE → SPEAKING
SPEAKING + STOP     → IDLE
SPEAKING + OUT_END  → IDLE
```

推理时使用 constrained decoding：IDLE 状态只允许 `LISTEN/SPEAK`，SPEAKING 状态只允许 `CONTINUE/STOP`。这保证模型不会产生状态机上非法的控制动作。

### 3.3 一个真实 trigger session

数据中的 `synthetic-040000` 指令是：

> 在 RED 出现前保持沉默，出现后输出 alert。

它共有 52 个 tick。关键片段为：

| Tick | 输入 | 目标动作 | 输出 |
|---:|---|---|---|
| 47 | `GREEN blue` | LISTEN | 空 |
| 48 | `blue green` | LISTEN | 空 |
| 49 | `yellow RED` | SPEAK + OUT_END | `alert` |

D 组中的最后一个 tick 可以抽象写成：

```text
<tick><input>yellow RED</input>
<speak><output>alert</output><out_end><tick_end>
```

这不是普通“一问一答”，因为模型在前 49 个 tick 中一直累积上下文，并在条件满足前保持沉默。

---

## 4. C/D 对照究竟比较什么

### 4.1 C 组：没有四种独立控制 token

C 组每个 tick 都生成一个 output 区域：

```text
空 output                       → LISTEN
非空 output，之前处于 IDLE      → SPEAK
非空 output，之前处于 SPEAKING  → CONTINUE
空 output + <out_end>           → STOP
```

例如沉默 tick 的目标类似：

```text
<output></output><tick_end>
```

### 4.2 D 组：显式控制和内容分离

D 组先预测控制：

```text
<listen><tick_end>
```

或者：

```text
<speak><output>...</output><out_end><tick_end>
```

### 4.3 这个对照的边界

C 组虽然没有 `LISTEN/SPEAK/CONTINUE/STOP`，但仍然拥有：

- `<output>` 与 `</output>`；
- `<out_end>`；
- `<tick_end>`；
- 跨 tick 保存的 IDLE/SPEAKING 状态。

所以本实验检验的不是“有状态控制 vs. 完全无状态控制”，而是：

> 在已经有很强协议脚手架的情况下，再增加四种独立控制 token 是否有额外收益？

这是解读 H3 时最重要的限定。

---

## 5. 六类合成任务及真实例子

### 5.1 Wait：等待问题完整

真实样本 `synthetic-040001`：

> 只有收到完整计算题后才能给答案。

| Tick | 输入 | 目标动作 | 输出 |
|---:|---|---|---|
| 37 | `计算 98` | LISTEN | 空 |
| 38 | ` 加上 4` | LISTEN | 空 |
| 39 | `3` | SPEAK | `1` |
| 40 | 空 | CONTINUE | `4` |
| 41 | 空 | CONTINUE + OUT_END | `1` |

完整问题是“98 加上 43”，最终跨三个 tick 拼成 `141`。该任务检查模型是否会在只看到“98 加上 4”时过早回答。

### 5.2 Trigger：关键词触发

前述 `synthetic-040000` 要求只在 `RED` 出现时输出 `alert`。类似 `GREEN`、`blue` 和大小写变化用于测试误触发。

### 5.3 Periodic：周期输出

真实样本 `synthetic-040004`：

> 看到 START 后先等待 0 个完整 tick，然后输出 ping；此后每隔 2 个 tick 再输出一次 ping。

`START` 在 tick 4 出现，目标在 tick 4、6、8、10……重复输出 `ping`。模型没有看到绝对 tick 编号，必须根据历史维持相位。

### 5.4 Streaming：长答案分块输出

真实样本 `synthetic-040003` 要求原样复述：

> 小猫检查了723颗红色按钮。

目标从 tick 8 开始分成多个小块：

```text
小猫 | 检查 | 了 | 7 | 23颗红色 | 按钮 | 。
```

每块都不超过 4 个 tokenizer token。该任务同时检查内容重建、CONTINUE 状态和结束位置。

### 5.5 Interrupt：用户打断

真实样本 `synthetic-040020`：

| Tick | 输入 | 目标动作 | 输出 |
|---:|---|---|---|
| 21 | `请详细介绍太阳系中的八颗行星` | LISTEN | 空 |
| 22 | `。` | SPEAK | `太阳系的八` |
| 23 | `你说得对` | CONTINUE | `颗行星依次` |
| 24 | 空 | CONTINUE | `包括` |
| 25 | `别说了` | STOP | 空 |

“你说得对”不是打断，而“别说了”是明确打断。模型需要区分两者。

### 5.6 Distractor：相似但不满足条件的输入

真实样本 `synthetic-040019`：

> 只有同一个 tick 的输入同时包含 ALPHA 和 OMEGA 时才输出 done，否则保持沉默。

测试输入包含 `OMEG`、`omega`、`ALPHABET`、单独的 `ALPHA` 和随机数字，但没有任何 tick 同时满足两个精确条件，因此全部目标都是 LISTEN。

---

## 6. 数据集

共生成 50,000 个 session，随机种子为 `20260721`，文件 SHA256 为：

```text
6a102a5544274baa3bd0bb967e39824459161580d754902c64cc41838d14d701
```

### 6.1 任务数量

| 任务 | 数量 |
|---|---:|
| streaming | 13,097 |
| wait | 9,929 |
| periodic | 8,729 |
| trigger | 8,234 |
| interrupt | 5,130 |
| distractor | 4,881 |

### 6.2 Split

| Split | 数量 | 用途 |
|---|---:|---|
| train | 35,000 | 正式训练 |
| validation | 5,000 | 学习率选择与验证 |
| test_in_domain | 2,000 | 相同规则分布的新实例 |
| test_template_ood | 2,000 | 未见指令模板 |
| test_timing_ood | 2,000 | 未见延迟、周期和 offset |
| test_length_ood | 2,000 | 65–128 tick 的更长 session |
| test_distractor_ood | 2,000 | 新型 hard negatives |

本次最终行为报告只使用完整完成的 `test_in_domain` 和 `test_template_ood`。后者实际包含 999 个 trigger 和 1,001 个 wait，因此不能代表所有六类任务的模板泛化。

训练序列平均约 223 tokens，P99 为 411，最大为 480；配置允许的最大上下文长度为 2,048。

---

## 7. 模型与训练设置

### 7.1 基础模型和环境

- 基础模型：Qwen3.5-0.8B；
- 模型类型：decoder-only causal LM；
- 未修改 Transformer 架构；
- Python 3.11；
- PyTorch 2.11.0 + CUDA 12.8；
- Transformers 5.14.1；
- PEFT 0.19.1；
- bitsandbytes 0.49.2；
- GPU：2 × NVIDIA A800 80GB。

### 7.2 QLoRA

- 基座权重：4-bit NF4；
- 计算精度：BF16；
- double quantization：开启；
- LoRA rank：8；
- LoRA alpha：16；
- dropout：0.05；
- target modules：所有 linear 层，排除 `lm_head`；
- 新增协议 token 的 embedding 可训练；
- 可训练参数约 5.42M，占约 757.8M 参数的 0.716%；
- gradient checkpointing：开启。

### 7.3 损失函数

用户输入 token 不计算损失，只监督模型应产生的控制、输出内容和边界 token。为了避免大量 LISTEN 淹没稀有动作，使用加权 token loss：

| 目标 | 权重 |
|---|---:|
| 输出文本 | 1.0 |
| 第一个或非重复 LISTEN | 1.0 |
| 连续重复 LISTEN | 0.3 |
| SPEAK | 2.0 |
| CONTINUE | 1.5 |
| STOP | 2.0 |
| OUT_END | 1.5 |
| TICK_END | 1.0 |

### 7.4 训练流程

1. 先在 20 个 session 上做 100 updates 的过拟合检查；C/D 都能收敛并生成正确小样本。
2. 用 500 个训练 session 比较学习率 `1e-4` 和 `2e-4`。
3. `2e-4` 的 C/D validation loss 和生成指标更好，因此用于正式训练。
4. 正式训练使用 35,000 个训练 session、3 epochs、batch size 32、gradient accumulation 1。
5. 正式运行 3 个随机种子：`20260721`、`20260722`、`20260723`。

每个正式模型完成 3,282 个更新。C 单次约 3.97 小时，D 单次约 3.66 小时，单卡峰值已分配显存约 60.5GB。

### 7.5 正式训练 loss

| 组 | seed 20260721 | seed 20260722 | seed 20260723 |
|---|---:|---:|---:|
| C validation loss | 0.020893 | 0.020898 | 0.020685 |
| D validation loss | 0.025436 | 0.025520 | 0.025183 |

三个 seed 内部非常接近，说明训练过程稳定。但 C 和 D 的监督 token 不同，因此不能根据 C loss 更低就断言 C 行为一定更好。

---

## 8. 推理设置

推理按真实 tick 顺序执行，并保存跨 tick KV cache。每到一个新 tick，只把新输入和新协议 token 追加到缓存后面，不会重新排列历史。

在开发阶段还实现了完整增长上下文版本，并验证它与 KV-cache 版本的逐 tick 输出一致。内容生成使用 greedy argmax；控制 token 使用合法状态集合内的 constrained argmax。

每个输出块达到 4 个普通 token 后强制停止，以保证任何模型都不能突破预算。强制截断、缺失边界或非法结构都会记录为 violation。

---

## 9. 指标及其正确解读方式

### 9.1 Control accuracy

所有 tick 中控制动作完全正确的比例。

但数据中约 90% tick 是 LISTEN，因此一个永远沉默的模型也可能获得很高 accuracy。不能单独使用该指标。

### 9.2 Control Macro-F1

分别计算 LISTEN、SPEAK、CONTINUE、STOP 的 F1，再做等权平均。它比 accuracy 更关注稀有动作。

注意：如果某个 split 根本没有 STOP，当前实现会把 STOP F1 记为 0。因此 template OOD 的 Macro-F1 不能直接与 in-domain 比较。

### 9.3 False-trigger rate

在目标应为 LISTEN 的 tick 中，模型错误预测 SPEAK 的比例。该指标衡量“条件未满足就抢答”。

### 9.4 Onset Exact@0 和 Exact@±1

模型开始 SPEAK 的 tick 是否与参考完全相同，或是否落在前后 1 tick 内。

### 9.5 Tick output exact match

每个 tick 的输出 token ID 是否与参考块完全相同。它非常严格，改变合法分块边界也会失败。

### 9.6 Reconstructed token exact match

把一个 session 中所有输出块拼接后，检查最终 token 序列是否与参考答案完全相同。它不关心答案被切成几块，更接近“最终内容是否完整”。

### 9.7 Joint Session Success

只有一个 session 的每一个 tick 同时满足以下条件才算成功：

```text
控制正确
AND 当前 tick 输出 token 完全正确
AND OUT_END 正确
AND 没有任何 violation
```

这是最严格的指标。一个 50-tick session 即使 49 个 tick 正确，只错一个结束边界，整个 session 仍计为失败。

因此：

- Joint 高，说明协议和内容都非常稳定；
- Joint 低，不一定意味着最终答案错误；
- 必须同时查看 reconstructed EM 和分任务错误。

---

## 10. 正式结果

### 10.1 评价范围

正式训练有 3 seeds，但逐 tick 全量生成很慢。根据用户决定，行为评价在完成以下结果后停止：

- seed `20260721`；
- in-domain：C/D 各 2,000 sessions；
- template OOD：C/D 各 2,000 sessions。

`test_timing_ood` 只有中途停止的部分分片，不进入指标。length OOD、distractor OOD、retention 和未训练基线没有完成。

所以以下结果有大量配对测试 session，但仍然只是**单训练 seed 的行为结果**。

### 10.2 In-domain 总体结果

| 指标 | C | D | D−C |
|---|---:|---:|---:|
| Control accuracy | 98.856% | 98.694% | -0.162 pp |
| Control Macro-F1 | 0.9305 | 0.9235 | -0.0070 |
| SPEAK F1 | 0.9949 | 0.9847 | -0.0102 |
| CONTINUE F1 | 0.7334 | 0.7193 | -0.0141 |
| STOP F1 | 1.0000 | 0.9972 | -0.0028 |
| Onset Exact@±1 | 99.647% | 99.855% | +0.208 pp |
| Tick output EM | 96.517% | 96.425% | -0.092 pp |
| Reconstructed token EM | 87.45% | 84.95% | -2.50 pp |
| OUT_END accuracy | 98.049% | 98.197% | +0.148 pp |
| False-trigger rate | 0.006% | 0.102% | +0.096 pp |
| Joint Session Success | 52.25% | 51.90% | -0.35 pp |
| Overflow rate | 0 | 0 | 0 |
| Malformed rate | 0 | 0 | 0 |

### 10.3 配对差异

C/D 使用完全相同的测试 session，因此可以逐 session 配对比较，而不是只比较两个独立比例。

#### Joint success

- D 成功而 C 失败：86 条；
- C 成功而 D 失败：93 条；
- 两者相同：1,821 条；
- D−C：`-0.35 pp`；
- 近似 95% CI：`[-1.66, +0.96] pp`。

区间跨过 0，不能区分 C/D；但它也远离预注册的 D `+10 pp` 目标。当前数据不支持 D 有大幅总体优势。

#### Reconstructed EM

- D 成功而 C 失败：27 条；
- C 成功而 D 失败：77 条；
- 两者相同：1,896 条；
- D−C：`-2.50 pp`；
- 近似 95% CI：`[-3.49, -1.51] pp`。

这里区间不跨 0，说明在该 seed 和测试集上，D 的最终内容完整率确实更低。

### 10.4 Template OOD

| 指标 | C | D | D−C |
|---|---:|---:|---:|
| Control accuracy | 99.042% | 98.854% | -0.188 pp |
| Onset Exact@±1 | 97.15% | 96.65% | -0.50 pp |
| Tick output EM | 98.158% | 98.057% | -0.101 pp |
| Reconstructed token EM | 96.50% | 93.20% | -3.30 pp |
| False-trigger rate | 0.026% | 0.127% | +0.101 pp |
| Joint Session Success | 68.70% | 67.75% | -0.95 pp |
| Overflow / malformed | 0 / 0 | 0 / 0 | 0 |

配对 Joint 差异为 `-0.95 pp`，近似 95% CI `[-2.50, +0.60] pp`；仍无法区分。Reconstructed EM 差异为 `-3.30 pp`，近似 95% CI `[-4.30, -2.30] pp`，D 仍明显更低。

Template OOD 的 joint 比 in-domain 更高，不代表所有 OOD 都更容易；主要原因是该 split 只有较简单的 trigger 和 wait，没有 streaming 与 interrupt。

---

## 11. 分任务结果

### 11.1 In-domain

| 任务 | C Joint | D Joint | C Recon | D Recon |
|---|---:|---:|---:|---:|
| trigger | 1.000 | 1.000 | 1.000 | 1.000 |
| periodic | 1.000 | 1.000 | 1.000 | 1.000 |
| distractor | 1.000 | 1.000 | 1.000 | 1.000 |
| wait | 0.375 | 0.341 | 0.951 | 0.825 |
| streaming | 0.156 | 0.143 | 0.883 | 0.883 |
| interrupt | 0.000 | 0.085 | 0.102 | 0.108 |

### 11.2 如何理解这个表

- trigger、periodic、distractor 全部为 1，强烈支持模型学会了基本时钟、条件触发和沉默。
- streaming 的 Recon 为 88.3%，但 Joint 只有约 15%。这说明内容经常是对的，严格的逐 tick 分块却不同。
- wait 中 D 的 Recon 比 C 低很多，是 D 总体重建率较低的主要来源之一。
- interrupt 中 D 的 Joint 看似比 C 高，但必须结合后面的评价偏差分析。

---

## 12. 用真实预测解释“指标为什么会这样”

### 12.1 相同最终答案，不同合法分块，却被 Joint 判错

真实 streaming 样本 `synthetic-040024`：

> 计算 629 加 724，只输出结果。

正确答案是 `1353`。参考分块与两组预测如下：

| Tick | 参考 | C | D |
|---:|---|---|---|
| 14 | SPEAK `135`，未结束 | SPEAK `1353`，结束 | SPEAK `1353`，结束 |
| 15 | CONTINUE `3`，结束 | LISTEN | LISTEN |

参考 token ID 分块为：

```text
tick 14: [16, 18, 20]      # “135”
tick 15: [18]              # “3”
```

模型预测为：

```text
tick 14: [16, 18, 20, 18]  # “1353”，正好 4 tokens
tick 15: []
```

模型没有超出每 tick 4-token 预算，最终拼接也完全是 `1353`。从任务语义看，它是合理答案；但由于参考选择了 3+1 的分块，模型选择了 4+0，Tick EM 和 Joint 都判错。

这个例子解释了为什么 streaming Recon 为 88.3%，而 Joint 只有约 15%。**当前 Joint 同时衡量内容正确和模仿唯一参考分块，不是纯粹的任务成功率。**

### 12.2 D 把正确答案多分了一个 tick

真实 wait 样本 `synthetic-040200`：

> 请等待算式完整后再回答。

输入逐步形成“97 加上 85”，答案是 `182`：

| Tick | 参考 | C | D |
|---:|---|---|---|
| 9 | SPEAK `182`，结束 | SPEAK `182`，结束 | SPEAK `18`，未结束 |
| 10 | LISTEN | LISTEN | CONTINUE `2`，结束 |

D 的最终内容仍然是 `182`，协议也没有超预算，但它没有复制参考的结束时机，因此严格 Joint 失败。该例说明一部分 wait 错误同样是“分块策略不同”，不一定是算术错误。

不过总体统计中 D 的 wait Recon 只有 82.5%，低于 C 的 95.1%，说明 D 不只是边界不同，也确实有更多 session 最终内容不完整。

### 12.3 Interrupt 中的结构性评价偏差

真实 interrupt 样本 `synthetic-040800`：

> 回答需要分块输出；用户明确要求停止时立即停止。

| Tick | 输入 | 参考 | C | D |
|---:|---|---|---|---|
| 7 | `。` | SPEAK `太阳系的八颗` | 完全正确 | 完全正确 |
| 8 | `别说了` | STOP，`out_end=false` | STOP，`out_end=true` | STOP，`out_end=false` |

表面上看，D 完全正确，C 因多预测一个 `<out_end>` 而失败。

但 C 没有 `<stop>` token。它的解码器只能通过：

```text
之前处于 SPEAKING
+ 当前 output 为空
+ 当前出现 <out_end>
```

来推断 STOP。也就是说，C 想被识别为 STOP，就必须生成 `<out_end>`；而原始标签又要求 interrupt tick 的 `out_end=false`。因此在当前实现中，C 的 STOP 和 out_end 标签存在不可同时满足的条件。

这会使 C 的 interrupt Joint 被结构性压低到 0。D 的 interrupt 优势不能被直接解释为“独立控制 token 更好”，因为对照评价并不完全公平。更合理的做法应是：

- 对 C 的隐式 STOP 不再比较原始 `out_end`；或
- 为 C 定义与 D 语义等价但表示不同的规范化事件；或
- 只比较“是否在打断 tick 停止旧答案”，而不是比较内部 token 表示。

这一问题不会改变“D 没有总体优于 C”的观察；相反，D 在拥有该评价优势时，总体 Joint 仍与 C 持平。

---

## 13. 对预注册成功标准的判断

| 标准 | seed 20260721 结果 | 判断 |
|---|---:|---|
| D in-domain Control Macro-F1 ≥ 0.90 | 0.9235 | 通过 |
| D SPEAK F1 ≥ 0.85 | 0.9847 | 通过 |
| D STOP F1 ≥ 0.85 | 0.9972 | 通过 |
| Onset Exact@±1 ≥ 0.90 | 0.9985 | 通过 |
| 中断停止延迟中位数 ≤ 1 tick | 0 tick | 通过 |
| Reconstructed EM ≥ 0.90 | 0.8495 | **失败** |
| Overflow ≤ 1% | 0 | 通过 |
| Malformed ≤ 1% | 0 | 通过 |
| D Joint 比 C 高 ≥ 10 pp | -0.35 pp | **失败** |
| OOD Joint 至少保留 in-domain 的 80% | template OOD 达到，但覆盖不完整 | 部分通过 |
| 通用能力无明显下降 | 未完成 retention | 未知 |

最关键的两个失败项正是：D 没有提高 Joint，且内容重建未达到 90%。

---

## 14. 对三个假设的最终判断

### H1：固定时钟学习——支持

证据：

- trigger Joint = 100%；
- periodic Joint = 100%；
- distractor Joint = 100%；
- 两组 false-trigger 都很低；
- onset 基本准确到 ±1 tick。

这说明模型不仅记住输出文本，也学会了何时沉默和何时触发。

### H2：流式输入输出交错——部分支持

证据：

- streaming 最终重建 EM 为 88.3%；
- 模型能在多个 tick 之间维持输出状态；
- 没有 overflow 或协议格式错误。

不足：

- 参考分块下的严格 Joint 只有 14%–16%；
- CONTINUE F1 约 0.72–0.73，明显低于 SPEAK 和 STOP；
- 结束时机和 chunk 边界仍不稳定。

同时，严格 Joint 对合法替代分块过于敏感，因此 14%–16% 低估了语义层面的成功率。

### H3：独立控制 token 更有效——强版本不支持

证据：

- In-domain D−C Joint = -0.35 pp；
- Template OOD D−C Joint = -0.95 pp；
- 两个差异的置信区间均跨 0；
- D 的重建 EM 在两个测试集都显著更低；
- D false-trigger 更高，尽管绝对值仍然很低；
- 预注册的 +10 pp 目标完全没有出现。

Interrupt 的表面局部收益受 C STOP 表示偏差混淆，不能挽救 H3。

---

## 15. 哪些结论可靠，哪些不能下

### 可以较有信心地说

1. Qwen3.5-0.8B 通过 QLoRA 能学习该固定 tick 文本协议。
2. 简单等待、触发、周期和沉默行为可以达到很高准确率。
3. 模型能够生成跨 tick 的分块内容，并严格遵守 4-token 上限。
4. 当前 C/D 设置下，没有观察到独立控制 token 的总体优势。
5. 当前严格 Joint 指标混合了语义正确性和唯一 chunk 调度模仿。

### 不能据此声称

1. 模型拥有真实时间感；tick 只是离散序列位置。
2. 模型已经具备语音全双工能力。
3. D 在所有随机种子都不如 C；行为评价只完整运行了一个 seed。
4. 模型已通过 timing、length 和 distractor OOD；这些全量结果没有完成。
5. QLoRA 没有损害通用能力；retention 未完成。
6. Interrupt 上 D 的优势是真实架构收益；现有评价对 C 不公平。

---

## 16. 实验设计的主要局限

### 16.1 单一参考 chunk 边界

只要最终答案和预算正确，不同的 3+1、4+0 或 2+1 分块都可能合理。当前 Joint 要求复制唯一参考，导致合理输出被判错。

建议增加一个“语义协议成功率”：

```text
触发不早于完整输入
AND 每块 ≤ 4 tokens
AND 最终重建正确
AND 正确停止
```

它应与严格 reference Joint 同时报告。

### 16.2 C 的 STOP 表示与标签不一致

C 必须借助 `<out_end>` 表达 STOP，但评价又要求 interrupt 的 `out_end=false`。应先把 C/D 内部表示映射到统一语义事件，再比较。

### 16.3 Template OOD 覆盖不完整

该 split 只有 trigger 和 wait。它不能支持关于 streaming、periodic 或 interrupt 模板泛化的结论。

### 16.4 行为评价只有一个 seed

训练 loss 在 3 seeds 间稳定，但行为指标仍可能变化。当前置信区间只反映测试 session 的抽样差异，不包含训练 seed 方差。

### 16.5 合成任务较简单

trigger、periodic 和 distractor 的 100% 成功说明协议可学，但不等于自然对话中的复杂控制已经解决。

### 16.6 Constrained decoding 降低了问题难度

模型不能选择状态机上非法的控制 token。因此控制指标衡量的是合法候选之间的判断能力，不是完全自由生成下的协议鲁棒性。

---

## 17. 如果继续实验，最优先的改进

1. **修复语义评价。** 先规范化 C/D 的 STOP、OUT_END 和 chunk 事件，再做任何新训练。
2. **同时报告两种 Joint。** 保留严格 reference Joint，再增加允许合法替代分块的 semantic Joint。
3. **针对 wait/streaming 做错误分类。** 区分内容错误、提前触发、分块不同、结束过早和结束过晚。
4. **重新构造公平 C/D 对照。** 确保两组都能表达同样的语义事件，只改变是否显式预测控制 token。
5. **只在评价修复后补多 seed。** 当前最主要不确定性来自指标定义，而不是缺少更多测试量。
6. **若扩展到语音，保持相同状态机。** 将 tick 输入替换为声学 token 前，先确保文本版 semantic Joint 稳定。

---

## 18. 最终结论

这个实验成功证明了一个较窄但重要的事实：

> 不修改 Transformer 架构，仅通过特殊 token、状态约束和 QLoRA，0.8B causal LLM 就能学习固定 tick 下的持续输入、沉默、触发、周期输出和有限长度分块协议。

但实验没有证明独立的 `LISTEN/SPEAK/CONTINUE/STOP` token 比隐式控制更好。现有完整结果中，C/D 的严格 Joint 基本相同，D 的最终内容重建反而更差。更重要的是，真实案例揭示了两个评价问题：合法替代分块被当作错误，以及 C 的隐式 STOP 与 out_end 标签不兼容。

因此最合理的研究结论不是“Omni-Flow 失败”，也不是“D 组成功”，而是：

1. **固定 tick 文本状态机是可学习的；**
2. **流式内容生成已经出现，但结束和分块仍是主要难点；**
3. **独立控制 token 的额外价值尚未得到证据支持；**
4. **下一步首先应修复语义等价评价，而不是继续扩大同一套全量测试。**

---

## 19. 复现信息与产物

关键配置：

- `configs/protocol.yaml`
- `configs/environment.yaml`

代码：

- `src/generate_synthetic_data.py`
- `src/training_data.py`
- `src/train_qlora.py`
- `src/tick_inference.py`
- `src/evaluate.py`

数据与训练结果：

- `artifacts/data/synthetic_50k.jsonl`
- `artifacts/data/synthetic_50k.manifest.json`
- `artifacts/runs/formal/seed_20260721/`
- `artifacts/runs/formal/seed_20260722/`
- `artifacts/runs/formal/seed_20260723/`

完整行为结果：

- `artifacts/runs/formal_eval/seed_20260721/C/test_in_domain/`
- `artifacts/runs/formal_eval/seed_20260721/D/test_in_domain/`
- `artifacts/runs/formal_eval/seed_20260721/C/test_template_ood/`
- `artifacts/runs/formal_eval/seed_20260721/D/test_template_ood/`

分任务统计：

- `artifacts/runs/formal_eval/partial_analysis/`
