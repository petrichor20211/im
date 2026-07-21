# Turn-based Text LLM → Micro-turn LLM：最小受控验证实验

> 状态：Draft v0.1
>
> 目标：只回答“现有 turn-based Text LLM 是否能在不修改 Transformer 架构的情况下，通过全参数 continued training 学会 micro-turn 执行协议”。
>
> 本实验**不试图同时证明**真实时间感、任意延迟泛化、自然打断、backchannel、完整双工对话或通用能力无损。

---

## 1. 研究问题与结论边界

### 1.1 核心研究问题

给定一个已经能以普通 turn-based 形式完成短答案问答的 Instruct LLM `M0`：

> 在保持模型架构不变的情况下，仅通过全参数 continued training，能否让模型在固定的 micro-turn 协议中，一边接收下一条用户输入的 chunks，一边分 chunks 输出上一条回答？

这里要迁移的是 `M0` **已经具备的语义行为**，而不是在训练中教会模型一个新的知识任务。

### 1.2 本实验中“学会 micro-turn”的操作性定义

模型需要同时做到：

1. 每个 tick 只读取本 tick 新到达的用户 chunk；
2. 每个 tick 决定沉默或输出最多 `K_A` 个 assistant tokens；
3. 在回答上一问题时，仍能接收并记住下一问题的 chunks；
4. 将多个 ticks 的 assistant 输出拼接后，恢复 `M0` 在普通 turn 模式下原本会给出的答案；
5. 不提前回答、不漏答、不重复回答，并遵守每 tick 的输出边界。

### 1.3 允许得出的结论

若实验成功，可以支持：

> 现有 turn-based Text Transformer 无需修改架构，可以通过 continued training，将已有的文本问答能力重新组织为固定的 interleaved chunk-level execution protocol。

### 1.4 不允许外推的结论

本实验不能单独证明：

- 模型理解真实的 200ms 墙上时间；
- 模型能泛化到任意 tick 频率或未见 delay；
- 模型已具备自然 interrupt、barge-in、backchannel；
- 模型能处理任意开放域 duplex conversation；
- continued training 后所有通用能力都无损；
- 该协议是最佳产品协议或最佳训练格式。

---

## 2. 核心设计原则

### 2.1 只改变执行协议，不改变语义任务

训练数据中的目标答案来自 `M0` 自己在普通 turn 模式下的正确输出。这样，continued training 的主要任务是迁移执行方式，而不是学习新的答案分布。

### 2.2 使用成对数据构造两个训练组

对于完全相同的 `(question, answer)` 集合，构造：

- **Blocked-chunk**：先完整接收问题，再分 chunks 输出答案；回答结束后才接收下一问题；
- **Interleaved micro-turn**：输出上一答案时，同时接收下一问题的 chunks。

两组使用：

- 同一个初始化 checkpoint；
- 相同的语义样本；
- 相同的 question/answer chunk 切分；
- 相同的特殊 token；
- 相同的 assistant target tokens；
- 相同训练超参数与优化步数。

核心变量仅是：**是否在 assistant 输出期间继续注入新的 user chunks**。

### 2.3 第一阶段只使用一种任务

默认建议使用“短答案开放域问答”，原因是：

- 能直接验证模型已有知识是否被迁移；
- 可用 normalized exact match / token F1 评价；
- 答案较短，容易构造确定的输出窗口；
- 不需要额外 judge model；
- 比 copy、uppercase、固定符号变换更能代表已有语言能力。

第一阶段不混合翻译、计数、周期输出、interrupt 等任务。

---

## 3. 模型与协议

### 3.1 模型

- 一个 `1.5B–3B` Instruct decoder-only LLM；
- 所有实验组从同一个 checkpoint 初始化；
- 不修改 attention、position encoding、KV cache 或 Transformer block；
- 主实验使用全参数 continued training；
- tokenizer 增加少量特殊 token，并在两个训练组中以相同方式训练新 token embedding。

> LoRA 可以作为工程预实验，但不能替代主实验，否则结论应改成“parameter-efficient adaptation 可行”。

### 3.2 序列协议

建议格式：

```text
<TICK><U>本 tick 新增用户文本</U><A>本 tick assistant 输出</A>
```

用户语义单元结束时，在最后一个 user chunk 中加入 `<EOU>`：

```text
<TICK><U>法国的首都是哪里？<EOU></U><A></A>
```

其中：

- `<TICK>`：一次 scheduler 调用；
- `<U>...</U>`：本 tick 新到达的用户文本，可为空；
- `<EOU>`：当前问题输入结束，可进入待回答队列；
- `<A>...</A>`：本 tick assistant 输出，可为空；
- 每个 `<A>` 最多包含 `K_A=4` 个普通 tokenizer tokens；
- 不提供绝对 tick 编号；
- 不使用 `START / CONT / STOP` 等冗余 ACT 标签。

### 3.3 训练 loss

只在以下位置计算 next-token loss：

- `<A>` 内的 assistant 内容 tokens；
- 每个 assistant slot 的 `</A>`。

不对以下内容计算 loss：

- `<TICK>`；
- `<U>...</U>`；
- `<EOU>`；
- `<A>` 起始标记。

第一版不做 silence reweighting。必须同时报告空 slot 比例和“永远沉默”基线，避免高 silence accuracy 掩盖失败。

### 3.4 推理过程

每条 trajectory 使用同一个持续增长的上下文或等价 KV cache：

1. scheduler append `<TICK><U>...</U><A>`；
2. 模型生成，遇到 `</A>` 停止；
3. 最多允许生成 `K_A` 个内容 tokens，再额外期待一个 `</A>`；
4. 若达到限制仍未生成 `</A>`，强制停止并记为 protocol violation；
5. 将实际生成内容和 `</A>` 一并追加到上下文；
6. 进入下一个 tick。

禁止每个 tick 重置对话上下文，否则实验没有验证跨 tick 状态保持。

---

## 4. 数据构造

### 4.1 原始数据

选择带标准答案或 aliases 的短答案 QA 数据集。候选：

- TriviaQA；
- Natural Questions short-answer subset；
- WebQuestions；
- 一个经过人工检查的中英文短答案 QA 集。

建议约束：

- question 长度：`8–64` 个模型 tokens；
- `M0` 生成答案长度：`1–16` 个模型 tokens；
- 去除需要长解释、多步证明或答案不唯一的样本；
- train / validation / test 按问题去重；
- 尽可能按答案实体去重，减少同一答案跨 split 泄漏。

### 4.2 提取模型已有行为

对每个原始问题，使用固定的普通 turn prompt：

```text
请只给出简短答案，不要解释。
问题：{question}
答案：
```

使用固定 greedy decoding 得到 `M0` 输出 `y0`。

仅保留满足以下条件的样本：

1. `y0` 与数据集任一标准 alias normalized exact match；
2. `y0` 长度不超过上限；
3. `y0` 不包含解释、拒答或额外格式；
4. 问题和答案通过去重检查。

随后将 `y0` 而不是原数据集 reference 作为 micro-turn 训练及测试目标。

这样测试的是：

> 模型能否在新协议下复现自己原来已经能正确生成的行为？

而不是：

> 模型能否在 continued training 中学会新的 QA 数据？

### 4.3 初始数据规模

建议先做低成本 pilot：

| Split | 已过滤语义样本数 | 用途 |
|---|---:|---|
| train | 4,000 | continued training |
| validation | 500 | checkpoint / 超参数选择 |
| test | 1,000 | 最终一次性评价 |

每条 trajectory pack `4` 个 QA item，因此约得到：

- 1,000 条 train trajectories；
- 125 条 validation trajectories；
- 250 条 test trajectories。

如果 pilot 呈现明确学习趋势但未达到成功标准，再扩展到 `16k` 和 `64k` train items，形成 data scaling curve。第一轮不直接预设 10M target tokens。

### 4.4 Chunk 切分

对每个问题和答案使用 tokenizer 后的 token IDs 切分：

- user chunk：每段随机 `1–4` tokens；
- assistant chunk：固定最多 `K_A=4` tokens；
- 最后一段 user chunk 后追加 `<EOU>`；
- chunk 边界由固定随机种子生成并保存，不在运行时临时变化；
- B1 和 B2 必须读取同一份 chunk manifest。

需要保存：

```text
example_id
question_token_ids
answer_token_ids
user_chunk_boundaries
assistant_chunk_boundaries
source_split
M0_turn_output
reference_aliases
```

### 4.5 B1：Blocked-chunk CT

B1 学习 tick/chunk 格式，但训练中没有输入输出重叠：

```text
<TICK><U>法国的首都</U><A></A>
<TICK><U>是哪里？<EOU></U><A></A>
<TICK><U></U><A>巴黎</A>
<TICK><U></U><A></A>

<TICK><U>水的化学式</U><A></A>
<TICK><U>是什么？<EOU></U><A></A>
<TICK><U></U><A>H2O</A>
```

下一问题只能在上一答案完整输出后开始。

### 4.6 B2：Interleaved micro-turn CT

B2 使用完全相同的问题、答案和 chunks，但下一问题在上一答案开始输出时立即进入：

```text
<TICK><U>法国的首都</U><A></A>
<TICK><U>是哪里？<EOU></U><A></A>
<TICK><U>水的化学式</U><A>巴黎</A>
<TICK><U>是什么？<EOU></U><A></A>
<TICK><U>《哈姆雷特》的作者</U><A>H2O</A>
```

确定性调度规则：

1. 一个问题收到 `<EOU>` 后进入 FIFO pending queue；
2. assistant 空闲时，于下一个 tick 开始回答 queue head；
3. 每 tick 输出最多 `K_A` tokens，直到该答案完成；
4. 一旦某答案开始输出，下一问题的第一个 user chunk也在同一 tick 输入；
5. assistant 同一时间只服务一个答案，禁止两个答案交错输出；
6. 若输入速度导致 queue depth 大于预设上限，则暂缓发送下一 user chunk；
7. trajectory 结束时 flush pending answers。

初始实验建议 `max_queue_depth=1`，避免把队列管理变成额外研究问题。

### 4.7 数据一致性检查

生成训练文件后必须自动验证：

- B1/B2 的 `example_id` 顺序完全一致；
- B1/B2 的 user/assistant 普通 token multiset 完全一致；
- B1/B2 的目标答案完全一致；
- 每个 assistant slot 内容不超过 `K_A`；
- 所有问题恰好出现一次 `<EOU>`；
- 所有答案都被完整输出一次；
- B1 不存在非空 U 与非空 A 同 tick；
- B2 中每条 trajectory 至少存在一个非空 U 与非空 A 同 tick；
- train/validation/test 无重复问题；
- chunk manifest 的随机种子和 hash 已保存。

---

## 5. 实验组

| 组别 | 初始化 | 训练 | 主要用途 |
|---|---|---|---|
| B0 | `M0` | 不训练 | 确认目标行为确实已存在于 turn 模型 |
| B1 | `M0` | Blocked-chunk CT | 控制特殊 token、chunk 输出和 continued training 效应 |
| B2 | `M0` | Interleaved micro-turn CT | 主实验 |

### 5.1 公平性约束

B1 与 B2 必须保持一致：

- checkpoint 和 tokenizer；
- 语义样本及顺序；
- target answer tokens；
- batch 中语义 item 数；
- optimizer、learning rate、warmup、weight decay；
- global batch size；
- optimizer updates；
- precision 和 distributed strategy；
- checkpoint selection 规则；
- generation 参数。

由于 interleaving 会改变 trajectory 的 tick 数和上下文 token 数，无法同时严格匹配所有计算量。主实验优先匹配：

1. 语义样本数；
2. assistant target token 数；
3. optimizer updates。

同时报告 B1/B2 实际输入 tokens、总 FLOPs 估计和 wall-clock，不能声称计算量完全相同。

---

## 6. 训练设置

最终数值需根据模型规模确定。建议 pilot 起点：

```yaml
training: full_parameter
objective: masked_next_token_prediction
precision: bf16
optimizer: AdamW
learning_rate_candidates: [1e-6, 3e-6]
weight_decay: 0.1
warmup_ratio: 0.03
epochs: 1-3
gradient_clipping: 1.0
selection_metric: validation_joint_success
```

为避免对 B2 单独调参：

1. 在小规模 shared pilot 上为 B1/B2 使用同一组 LR candidates；
2. 使用预先声明的统一选择规则；
3. 最终比较必须使用相同 LR，或完整报告所有 LR 的结果；
4. 不得只报告对 B2 最有利的 checkpoint。

至少运行 `3` 个训练随机种子。若计算预算不允许，pilot 可先单 seed，但不能据此给出稳定性结论。

---

## 7. 评价设计

### 7.1 Evaluation A：B0 的普通 turn 能力

目的：证明 test item 是 `M0` 原本会做的。

在每个 test question 上重新运行固定 turn prompt，报告：

- normalized EM against stored `M0_turn_output`；
- normalized EM against dataset aliases；
- token F1；
- 平均答案长度。

因为 test 数据经过 `M0` 正确性过滤，理论上该结果应接近 100%。重新运行用于发现环境、prompt 或 checkpoint 不一致。

### 7.2 Evaluation B：Interleaved micro-turn 主测试

B1、B2 都在**同一份 B2 interleaved test trajectories**上运行。不能让 B1 只测 blocked test、B2 只测 interleaved test。

#### 语义指标

按 expected service window 收集每个 item 的 assistant tokens，并重建答案：

```text
reconstructed_answer_i = concat(A slots assigned to item i)
```

报告：

- normalized exact match against `M0_turn_output`；
- token F1；
- item omission rate；
- duplicate answer rate；
- answer order error rate。

主语义指标：`Micro Semantic EM`。

#### 协议指标

报告：

- `Premature Output Rate`：问题 `<EOU>` 前产生非空输出；
- `Onset Exact@0`：应开始回答的 tick 准确开始；
- `Onset Exact@±1`；
- `Onset MAE`；
- `Unexpected Silence Rate`：应输出内容时输出空 slot；
- `Spurious Output Rate`：无待回答 item 时输出内容；
- `Malformed Slot Rate`：未按要求生成 `</A>`；
- `Chunk Overflow Rate`：单 tick 超过 `K_A`；
- `FIFO Violation Rate`：答案顺序错误。

#### 联合指标

定义 item-level `Joint Success`：

```text
Joint Success =
    reconstructed answer normalized EM
    AND no premature output for this item
    AND onset within ±1 tick
    AND no malformed slot
    AND no omission / duplication / order error
```

主实验指标：`Joint Success Rate`。

### 7.3 Evaluation C：Blocked 测试诊断

B1、B2 额外在同一份 blocked test trajectories 上运行，用于区分：

- B2 是否只会 interleaved 格式；
- B1 是否正常学会了基础 tick/chunk 协议；
- 某组失败是否来自特殊 token 或 chunk generation 本身。

该结果是诊断项，不是核心成功判据。

### 7.4 Evaluation D：训练后 turn 能力

用原始 turn prompt 测 B1/B2 的 test QA：

- normalized EM；
- token F1。

这只检查当前任务上的明显遗忘，不在第一阶段运行 MMLU、GSM8K、HumanEval 等完整 benchmark。

若 micro-turn 迁移成立，再在第二阶段系统研究通用能力 retention 和 replay。

### 7.5 统计报告

- 对 test items 做 bootstrap，报告 95% confidence interval；
- 三个训练 seed 分别报告，并给出 mean ± std；
- 同一 test item 上的 B1/B2 使用 paired bootstrap；
- 同时报告绝对差值和相对差值；
- 不只报告最佳 seed。

---

## 8. 预注册成功与失败标准

### 8.1 基础有效性检查

实验进入正式比较前必须满足：

1. B0 在过滤后 test set 上的 turn EM ≥ 95%；
2. B1 在 blocked test 上能够正确解析协议；
3. B1 blocked-test `Malformed Slot Rate ≤ 1%`；
4. B1 blocked-test semantic EM ≥ 90%。

若 B1 连基础 blocked 协议都没学会，应优先视为数据、训练或推理实现失败，不能直接得出“micro-turn 不可迁移”。

### 8.2 主成功标准

建议预注册为同时满足：

1. B2 在 interleaved test 上 `Micro Semantic EM ≥ 90%`；
2. B2 的 `Onset Exact@±1 ≥ 90%`；
3. B2 的 `Premature Output Rate ≤ 5%`；
4. B2 的 `Malformed Slot Rate ≤ 1%`；
5. B2 的 `Joint Success Rate ≥ 80%`；
6. B2 的 Joint Success 比 B1 高至少 `10` 个绝对百分点；
7. B1/B2 paired bootstrap 的 95% CI 不跨 0；
8. B2 训练后的普通 turn EM 不低于 B0 超过 `5` 个绝对百分点。

第 8 项只用于排除“完全覆盖原能力后重新学了一个窄任务”的极端情况，不代表完整通用能力保持证明。

### 8.3 结果解释矩阵

| 观察 | 解释 |
|---|---|
| B1 blocked 成功，B2 interleaved 成功且显著高于 B1 | 支持 micro-turn 协议可通过 CT 迁移 |
| B1 blocked 成功，B2 语义正确但 timing/protocol 差 | 语义能力保留，但 micro-turn 调度未学好 |
| B1 blocked 成功，B2 protocol 正确但答案错误 | 学会了格式，但未成功迁移原语义行为 |
| B1 blocked 失败，B2 也失败 | 实现或训练设置不可诊断，不能否定迁移假设 |
| B1 和 B2 在 interleaved test 都成功 | 可能存在强 zero-shot protocol 泛化；需要更强对照，但仍说明架构能够执行 |
| B2 train 成功、held-out test 失败 | 可能记忆 trajectory，缺乏内容泛化 |
| B2 micro 成功但 turn 能力大幅下降 | 协议迁移可行，但存在严重能力覆盖/遗忘 |

---

## 9. 实验产物与可复现要求

必须保存：

```text
artifacts/
  model_config.json
  tokenizer_config/
  source_dataset_manifest.json
  filtered_examples.jsonl
  split_ids.json
  chunk_manifest.jsonl
  b1_blocked_train.jsonl
  b2_interleaved_train.jsonl
  blocked_test.jsonl
  interleaved_test.jsonl
  training_configs/
  checkpoints/
  generation_outputs/
  metrics/
  environment.txt
  git_commit.txt
```

每次运行记录：

- base checkpoint 精确名称和 revision/hash；
- tokenizer hash；
- 数据集版本；
- 所有随机种子；
- prompt 字符串；
- decoding 参数；
- CUDA/PyTorch/Transformers 版本；
- GPU 类型与数量；
- 实际训练 input/target token 数；
- 总训练时间；
- 最终 checkpoint 选择原因。

---

## 10. 详细 TODO List

### Phase 0：冻结实验问题

- [ ] 确认第一阶段只验证固定 interleaved 协议，不加入 interrupt/backchannel/周期计数。
- [ ] 确认“micro-turn 成功”的操作性定义。
- [ ] 确认主结论措辞和不允许外推的结论。
- [ ] 确认主成功指标为 `Joint Success Rate`。
- [ ] 冻结成功阈值，避免看到 test 结果后修改。

**交付物**：冻结后的本实验设计文档及 decision log。

### Phase 1：模型与环境

- [ ] 选择 base instruct checkpoint 和精确 revision。
- [ ] 验证模型 license 允许 continued training 和结果发布。
- [ ] 确认 tokenizer 是否支持添加 special tokens。
- [ ] 添加 `<TICK> <U> </U> <EOU> <A> </A>`。
- [ ] resize token embeddings，并确认 input/output embedding tying 行为。
- [ ] 实现普通 turn greedy generation。
- [ ] 实现单 trajectory 持续 KV cache 的 tick generation。
- [ ] 实现 `K_A` 限制及 malformed slot 记录。
- [ ] 固定 Python、PyTorch、Transformers、CUDA 环境。
- [ ] 记录单 batch 显存和吞吐，估计全参数训练预算。

**单元测试**：

- [ ] special tokens 编解码后完全可逆；
- [ ] `</A>` stopping criterion 正确；
- [ ] tick 之间 KV cache 不被清空；
- [ ] 空 `<A></A>` 能被正确解析；
- [ ] 超过 `K_A` 时能强制停止并记录 violation。

**交付物**：模型加载脚本、协议 generation demo、环境锁定文件。

### Phase 2：源数据与已有能力过滤

- [ ] 下载并锁定 QA 数据集版本。
- [ ] 清洗 question、aliases 和 split metadata。
- [ ] 使用固定 turn prompt 跑 `M0` greedy generation。
- [ ] 实现 answer normalization。
- [ ] 按 aliases 计算 EM/F1。
- [ ] 只保留 `M0` 已正确回答的样本。
- [ ] 过滤过长问题和过长答案。
- [ ] 过滤包含解释、拒答、列表或异常格式的输出。
- [ ] 做 normalized question 去重。
- [ ] 尽可能做 answer entity 跨 split 去重。
- [ ] 人工抽查至少 100 个保留样本。
- [ ] 生成 train/validation/test split IDs。
- [ ] 冻结 test split，后续不得用于调参。

**数据验收**：

- [ ] test 上 B0 重跑 turn EM ≥ 95%；
- [ ] 没有 train/test 重复问题；
- [ ] 保存每个样本的 `M0_turn_output`；
- [ ] 保存数据过滤各阶段的数量统计。

**交付物**：`filtered_examples.jsonl`、`split_ids.json`、filter report。

### Phase 3：Chunk 与 trajectory 构造

- [ ] 使用目标模型 tokenizer 对 question/answer 编码。
- [ ] 用固定 seed 生成 user chunk boundaries。
- [ ] 按 `K_A=4` 生成 assistant chunks。
- [ ] 实现 B1 blocked scheduler。
- [ ] 实现 B2 interleaved FIFO scheduler。
- [ ] 实现 `max_queue_depth=1`。
- [ ] 每 4 个 QA items pack 为一条 trajectory。
- [ ] 保存统一 chunk manifest。
- [ ] 分别序列化 B1/B2 train、validation、test。
- [ ] 生成 blocked 和 interleaved 两套 test trajectories。
- [ ] 实现 loss mask，并可视化抽查至少 20 条。

**自动一致性测试**：

- [ ] B1/B2 example IDs、答案和 chunk boundaries 一致；
- [ ] assistant slot 不超过 `K_A`；
- [ ] 每个问题恰好一个 `<EOU>`；
- [ ] 每个答案完整出现一次；
- [ ] B1 无输入输出 overlap；
- [ ] B2 存在输入输出 overlap；
- [ ] 所有 trajectory 可 parse → serialize → parse；
- [ ] loss mask 只覆盖 assistant target 和 `</A>`；
- [ ] 数据 hash 固定并写入 manifest。

**交付物**：trajectory builder、manifest、B1/B2 数据文件、数据测试报告。

### Phase 4：训练代码验证

- [ ] 在 20 条 trajectory 上 overfit B1。
- [ ] 在 20 条 trajectory 上 overfit B2。
- [ ] 确认训练 loss 可降至接近 0。
- [ ] 确认 overfit 后生成结果与 target trajectory 一致。
- [ ] 检查新 special token embedding 有梯度。
- [ ] 检查普通 user tokens 没有被计入 loss。
- [ ] 检查 gradient accumulation 后 global batch 正确。
- [ ] 检查 checkpoint save/resume 一致。
- [ ] 实现训练 input tokens、target tokens、FLOPs 估计日志。

**停止条件**：若 20 条样本无法 overfit，不进入正式训练。

**交付物**：overfit report、训练配置模板、可恢复 checkpoint。

### Phase 5：小规模 shared pilot

- [ ] 从 train split 取 500 items，构造 B1/B2 pilot。
- [ ] 使用相同 LR candidates 跑 B1/B2。
- [ ] 只在 validation pilot 上检查是否存在学习信号。
- [ ] 检查 silence collapse、永远输出和格式崩溃。
- [ ] 确认 LR 没有造成明显灾难性发散。
- [ ] 用统一规则选定正式实验 LR。
- [ ] 冻结正式训练配置。

**注意**：pilot 不得查看正式 test 指标。

**交付物**：pilot curves、LR 选择记录、冻结训练配置。

### Phase 6：正式训练

- [ ] 对 B1 运行 3 个 seeds。
- [ ] 对 B2 运行相同 3 个 seeds。
- [ ] 保存所有 validation checkpoints 和日志。
- [ ] 使用同一 validation rule 选择 checkpoint。
- [ ] 记录每组实际 input/target tokens、FLOPs、时间和显存。
- [ ] 检查 B1/B2 optimizer updates 完全一致。
- [ ] 训练结束前不运行正式 test。

**交付物**：6 个正式 checkpoints、训练日志、资源统计。

### Phase 7：正式评价

- [ ] 重跑 B0 普通 turn test。
- [ ] B1/B2 全部 checkpoints 跑 interleaved test。
- [ ] B1/B2 全部 checkpoints 跑 blocked diagnostic test。
- [ ] B1/B2 跑训练后普通 turn test。
- [ ] 计算 semantic metrics。
- [ ] 计算 protocol metrics。
- [ ] 计算 Joint Success。
- [ ] 做 paired bootstrap 95% CI。
- [ ] 汇总三个 seeds 的 mean ± std。
- [ ] 保存逐 tick 原始生成结果，不只保存聚合分数。
- [ ] 人工检查所有主要错误类型各至少 20 例。

**交付物**：metrics JSON/CSV、生成日志、bootstrap report、error taxonomy。

### Phase 8：结论与下一步门控

- [ ] 按预注册阈值判定 success / partial success / failure。
- [ ] 明确区分语义失败、协议失败和实现失败。
- [ ] 不因结果不理想而事后修改主指标。
- [ ] 若成功，运行数据规模曲线 `1k / 4k / 16k / 64k`。
- [ ] 若成功，再设计第二阶段 interrupt/correction 实验。
- [ ] 若 turn retention 明显下降，再引入 general replay 对照。
- [ ] 若 B1 与 B2 都成功，增加更强 scheduling-shift 对照。
- [ ] 若 B2 仅训练内成功，测试新 chunk boundary seed 和更长 trajectory。

**交付物**：最终实验报告和下一阶段 go/no-go 决策。

---

## 11. 暂不进入第一阶段的扩展实验

以下实验只有在最小实验成功后才进入：

1. 不同 user/assistant chunk sizes；
2. 随机或可变 tick interval；
3. unseen delay 和更长 trajectories；
4. 用户在回答中途 correction；
5. interrupt / barge-in；
6. backchannel 与 false-stop；
7. 多任务和开放域对话；
8. general text replay 比例；
9. MMLU、GSM8K、IFEval、HumanEval retention；
10. full parameter 与 LoRA 的比较；
11. 不同模型规模的 scaling；
12. 真实 wall-clock scheduler 和服务吞吐。

---

## 12. 待决策项

在实现前需要冻结以下决策：

| ID | 决策 | 默认建议 | 状态 |
|---|---|---|---|
| D1 | Base model | 选择已有本地训练栈支持的 1.5B–3B Instruct 模型 | 待定 |
| D2 | 主要语言 | 若目标产品以中文为主，使用中文或中英混合 QA | 待定 |
| D3 | QA 数据集 | 优先选择有 aliases、短答案、许可清晰的数据集 | 待定 |
| D4 | 主训练方式 | 全参数 continued training | 待定 |
| D5 | Pilot 数据规模 | 4k train / 500 val / 1k test | 待定 |
| D6 | 每条 trajectory 的 QA 数 | 4 | 待定 |
| D7 | User chunk size | 随机 1–4 model tokens | 待定 |
| D8 | Assistant chunk size `K_A` | 4 model tokens | 待定 |
| D9 | 是否显式使用 `<EOU>` | 使用，避免把 endpoint detection 混入实验 | 待定 |
| D10 | 正式训练 seeds | 3 | 待定 |
| D11 | 成功阈值 | 使用第 8 节建议值 | 待定 |
| D12 | 是否保留原文档 | 保留，将本文作为新的最小实验方案 | 待定 |

所有决策应写入 decision log，包含日期、选择和理由。
