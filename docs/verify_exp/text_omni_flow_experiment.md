# 文本版 Omni-Flow：固定时钟状态机验证实验

> 状态：Draft v1.0
> 目标：用最小成本验证普通 causal LLM 能否学习“持续输入、固定时钟、控制决策、分块输出”的交互机制。
> 范围：只研究文本协议，不引入语音、视觉，不修改 Transformer 架构。

---

## 1. 研究问题

本实验将 Omni-Flow 抽象为固定时钟下的文本状态机。每个 tick 中，模型依次：

1. 接收一小段新增文本；
2. 决定 `LISTEN / SPEAK / CONTINUE / STOP`；
3. 在允许输出时生成不超过固定预算的文本 token；
4. 将本 tick 的输入、控制和输出保留在同一条因果历史中。

实验回答三个问题：

1. 文本 LLM 能否学习固定节拍和跨 tick 状态？
2. 模型能否在持续接收新输入时分块输出完整答案？
3. 模型能否学习何时沉默、开始、继续和停止输出？

若实验成功，只能说明普通 causal LLM 可以学习固定文本 micro-turn 协议，不能据此外推模型已经具备真实时间感、语音全双工能力或自然对话中的完整打断能力。

---

## 2. 核心假设

### H1：固定时钟学习

模型能够在不提供绝对 tick 编号的情况下，根据历史状态执行等待、周期触发和延迟输出。

### H2：流式输入输出交错

模型能够学习如下因果顺序：

```text
input[0] → output[0] → input[1] → output[1] → ...
```

即生成一部分输出后，继续读取新的输入，而不是先收集全部输入再统一回答。

### H3：控制与内容解耦有效

独立预测控制 token，再决定是否生成内容，比将沉默和文本内容放在同一个输出空间中更稳定，尤其能降低错误触发和“永远沉默”。

---

## 3. 序列协议

### 3.1 特殊 token

向 tokenizer 添加以下特殊 token：

```text
<tick>
<input>
</input>
<output>
</output>
<listen>
<speak>
<continue>
<stop>
<out_end>
<tick_end>
```

含义如下：

| Token | 含义 |
|---|---|
| `<tick>` | 新时间片开始 |
| `<input>...</input>` | 本 tick 新到达的文本，可为空 |
| `<listen>` | 当前保持沉默 |
| `<speak>` | 开始一段新输出 |
| `<continue>` | 继续尚未完成的输出 |
| `<stop>` | 用户有效打断，取消当前输出 |
| `<output>...</output>` | 本 tick 的普通文本输出 |
| `<out_end>` | 当前回答自然完成 |
| `<tick_end>` | 当前时间片结束 |

### 3.2 每 tick 的合法格式

沉默：

```text
<tick><input>新增文本</input><listen><tick_end>
```

开始输出：

```text
<tick><input>新增文本</input><speak><output>最多 K 个 token</output><tick_end>
```

继续输出：

```text
<tick><input>新增文本</input><continue><output>最多 K 个 token</output><tick_end>
```

自然结束：

```text
<tick><input></input><continue><output>最后一块</output><out_end><tick_end>
```

用户打断：

```text
<tick><input>停一下，我换个问题</input><stop><tick_end>
```

### 3.3 状态机约束

模型维护两个逻辑状态：

```text
IDLE
SPEAKING
```

合法状态转移：

| 当前状态 | 控制 | 下一状态 |
|---|---|---|
| `IDLE` | `LISTEN` | `IDLE` |
| `IDLE` | `SPEAK` | `SPEAKING`，若同 tick 出现 `<out_end>` 则回到 `IDLE` |
| `SPEAKING` | `CONTINUE` | `SPEAKING`，若出现 `<out_end>` 则回到 `IDLE` |
| `SPEAKING` | `STOP` | `IDLE` |

以下情况均记为协议错误：

- `IDLE` 状态生成 `CONTINUE` 或 `STOP`；
- `SPEAKING` 状态无原因地生成新的 `SPEAK`；
- `LISTEN` 或 `STOP` 后生成非空 `<output>`；
- 单 tick 普通文本超过输出预算；
- 缺失 `<tick_end>`；
- 回答结束后未生成 `<out_end>`。

### 3.4 输出预算

第一版固定：

```text
K = 4 个普通 tokenizer token / tick
```

特殊 token 不计入 K。训练数据中的答案必须预先按同一 tokenizer 切分，推理时也必须强制执行预算，防止模型在第一次 `SPEAK` 时输出完整答案。

协议中不提供绝对 tick 编号。真实时间由外部 scheduler 负责；第一阶段的 tick 只是离散事件，不宣称对应真实毫秒数。

---

## 4. 推理循环

普通 `model.generate()` 会连续生成到终止符，不能在中间插入新输入。因此推理必须实现逐 tick 解码。

```python
state = IDLE
cache = None
history = initial_instruction

for input_chunk in input_stream:
    # 新输入必须接在上一 tick 已生成输出之后。
    tick_prefix = serialize_tick_input(input_chunk)
    history, cache = append_and_prefill(tick_prefix, history, cache)

    # 控制阶段只允许四个控制 token。
    control, cache = decode_control_token(
        allowed={LISTEN, SPEAK, CONTINUE, STOP},
        state=state,
        cache=cache,
    )

    if control in {SPEAK, CONTINUE}:
        output, ended, cache = decode_content(
            cache=cache,
            max_text_tokens=4,
            stop_tokens={OUT_END, TICK_END},
        )
        print_or_stream(output)
        state = IDLE if ended else SPEAKING

    elif control == STOP:
        cancel_current_response()
        state = IDLE

    # 将缺失的结构结束 token 追加到因果历史，再进入下一 tick。
    history, cache = close_tick(history, cache)
```

实现要求：

1. 同一 session 的上下文和 KV cache 不得在 tick 间清空；
2. 新输入必须位于上一 tick 实际生成输出之后；
3. 控制 token 应使用 constrained decoding，仅允许当前状态下的合法控制；
4. 内容生成达到 K 后必须停止，即使模型未生成 `<out_end>`；
5. 强制截断、非法控制和结构缺失都要记录为 protocol violation；
6. 首版以“完整增长上下文”的结果作为正确性基准，再验证 KV cache 优化版本与其逐 token 一致。

---

## 5. 合成任务

第一阶段不训练复杂开放域聊天，优先验证状态机行为。

### 5.1 等待完整信息

```text
tick 1 input: 计算 17 加上
target: LISTEN

tick 2 input: 25
target: SPEAK → 42 → OUT_END
```

要求随机改变：

- 问题切分位置；
- 空 tick 数；
- 数字范围和运算类型；
- 答案长度；
- 指令表述。

### 5.2 关键词触发

```text
instruction: 只有看到 RED 才输出 alert
tick 1: blue green      → LISTEN
tick 2: yellow RED      → SPEAK, alert, OUT_END
```

加入大小写、相似关键词和不满足条件的干扰项，测量 false trigger。

### 5.3 固定周期输出

```text
instruction: 每隔 4 个 tick 输出 ping
tick 1 → LISTEN
tick 2 → LISTEN
tick 3 → LISTEN
tick 4 → SPEAK, ping, OUT_END
```

随机化周期、起始 offset、session 长度和空输入位置。测试集保留训练中未出现的周期与更长序列，以检查相位漂移和长度泛化。

### 5.4 流式连续输出

将一个目标答案严格按 K 个 token 切分：

```text
tick 3: SPEAK    → 北京是
tick 4: CONTINUE → 中国的首都
tick 5: CONTINUE → 。, OUT_END
```

检查最终拼接结果是否完整、是否重复、是否漏 token，以及分块是否超预算。

### 5.5 用户打断

在模型处于 `SPEAKING` 时插入：

```text
停一下
别说了
我换个问题
等等，不是这个意思
```

目标是在当前 tick 输出 `STOP`，并取消未完成答案。加入“继续说”“没错”“嗯”等非打断表达作为 hard negative。

### 5.6 无关输入与干扰

加入噪声、随机句子、无关数字、拼写近似关键词和不完整请求。模型在条件未满足时应保持 `LISTEN`。

---

## 6. 数据构造

### 6.1 初始规模

快速 pilot 配置：

```text
总量：约 50,000 sessions
每条长度：8–64 ticks
每 tick 输入：0–16 tokens
每 tick 输出：最多 4 tokens
LISTEN 比例：70%–90%
上下文长度：2,048–4,096 tokens
```

任务占比：

| 比例 | 数据类型 |
|---:|---|
| 50% | 等待、关键词、周期等确定性任务 |
| 30% | 流式问答、延迟回答、分块生成 |
| 20% | 打断、干扰和长上下文任务 |

在正式生成 50k sessions 前，先生成约 100 条样本进行协议测试，再用 20–100 条样本做过拟合验证。

### 6.2 随机化原则

每类任务必须随机化：

- 起始 offset；
- 触发 tick；
- session 长度；
- 空 tick 数量；
- 输入 chunk 边界；
- 周期和延迟；
- 指令自然语言模板；
- 干扰项的位置和数量。

文本应允许切在单词、数字和句法短语中间，避免模型依赖整齐语义边界。

### 6.3 数据划分

不能只把同一批模板生成的实例随机拆分。建议按规则族划分：

- `train`：训练模板、训练周期和常规长度；
- `validation`：同规则的新实例和部分新模板；
- `test_in_domain`：新实例、新 chunk seed；
- `test_template_ood`：未见指令模板；
- `test_timing_ood`：未见周期、延迟和起始 offset；
- `test_length_ood`：训练最大长度的 2 倍；
- `test_distractor_ood`：新型 hard negatives。

所有 split 按底层任务参数和原始语义样本去重。

### 6.4 每条样本保存的信息

```text
session_id
rule_family
instruction
raw_inputs
input_token_ids
tick_boundaries
expected_controls
expected_output_token_ids
answer_token_ids
interrupt_ticks
trigger_ticks
random_seed
template_id
split
```

生成后保存 tokenizer hash、生成配置和数据 hash。

### 6.5 自动数据校验

必须验证：

- 每个 tick 恰好包含一个控制 token 和一个 `<tick_end>`；
- 控制序列满足状态机转移；
- `LISTEN/STOP` tick 没有普通输出；
- 每个输出块不超过 K；
- 分块拼接后与原始目标答案 token IDs 完全一致；
- 中断后旧答案不再输出；
- 周期任务触发位置与规则一致；
- train/test 不共享模板 ID 或保留规则参数；
- parse → serialize → parse 完全可逆。

---

## 7. 模型与环境

### 7.1 Base model

首版选择 `0.5B–1.5B` decoder-only causal LLM。优先使用当前机器已有的小模型，避免先下载和训练 30B Omni 模型。

候选：

```text
/120090727/yutong/model/Qwen3.5-0.8B
```

运行环境：

```text
/120090727/yutong/conda_env/text-omni-flow
```

环境名只表示使用现有 PyTorch/Transformers 栈，不要求 base checkpoint 必须是 Qwen3-Omni。

### 7.2 Tokenizer 修改

添加特殊 token 后必须：

1. 调用 `resize_token_embeddings`；
2. 检查 input/output embedding 是否 tied；
3. 保存修改后的 tokenizer；
4. 验证所有特殊 token 均编码为单个 token；
5. 验证普通文本编码不因特殊 token 配置发生异常变化。

---

## 8. 训练方案

### 8.1 参数高效训练

第一版使用 QLoRA：

```yaml
training: QLoRA
precision: bf16
quantization: 4bit
context_length: 2048-4096
output_budget_per_tick: 4
optimizer: AdamW
learning_rate_candidates: [1e-4, 2e-4]
warmup_ratio: 0.03
epochs: 1-3
gradient_clipping: 1.0
```

最终超参数应先在 validation pilot 上选择并冻结，不能根据正式 test 调整。

### 8.2 两阶段训练

#### 阶段 A：格式适配

只训练模型正确生成控制和结构 token：

```text
<listen> / <speak> / <continue> / <stop> / <out_end> / <tick_end>
```

目标是快速确认模型理解协议语法和状态转移。

#### 阶段 B：交互行为训练

加入等待、触发、周期、分块输出、打断、干扰和长上下文任务，训练模型根据语义与历史状态做控制决策。

### 8.3 Loss mask 与权重

输入文本和非目标结构前缀不计算 loss。控制 token、输出文本以及必要的结束 token 计算 loss。

建议初始权重：

| Token 类型 | 权重 |
|---|---:|
| 普通输出文本 | 1.0 |
| 首次或非重复 `<listen>` | 1.0 |
| 连续重复 `<listen>` | 0.3 |
| `<speak>` | 2.0 |
| `<stop>` | 2.0 |
| `<continue>` | 1.5 |
| `<out_end>` | 1.5 |
| `<tick_end>` | 1.0 |

“连续重复 LISTEN”定义为当前 tick 与上一 tick 都为 `LISTEN`。必须同时保留一个不重加权配置作为消融实验，避免只把收益归因于协议设计。

### 8.4 训练前门控

正式训练前必须通过：

1. 100 条数据的全部结构校验；
2. 20–100 条 session 过拟合，训练 loss 接近 0；
3. 过拟合样本的控制和输出 token 基本完全复现；
4. 新增特殊 token embedding 有非零梯度；
5. loss mask 不覆盖 input 文本；
6. checkpoint save/resume 输出一致。

若小样本无法过拟合，不进入正式 pilot。

---

## 9. 对照实验

至少运行四组：

| 组别 | 训练形式 | 推理形式 | 目的 |
|---|---|---|---|
| A | 不训练 | 整段输入后回答 | 原始语义能力基线 |
| B | 不训练 | chunked tick | 测量 zero-shot 协议能力 |
| C | QLoRA | chunked，无独立 control token | 控制与内容混合的对照 |
| D | QLoRA | chunked + 独立 control token | 文本版 Omni-Flow 主实验 |

公平性要求：

- C/D 使用相同 base checkpoint、数据语义内容和随机种子；
- 使用相同训练步数、global batch size、优化器和学习率选择规则；
- 输出预算 K 相同；
- 在同一 test trajectories 上评价；
- 报告实际训练 token 数、显存、wall-clock 和 optimizer updates。

附加消融：

- D1：标准平均 loss；
- D2：重复 `LISTEN` 降权；
- D3：去掉显式 `<tick_end>`；
- D4：完整上下文推理与 KV cache 推理一致性。

---

## 10. 评价指标

### 10.1 控制决策

- `Control Macro-F1`：四类控制的 macro-F1；
- 各类 precision / recall / F1；
- `False Trigger Rate`：不应输出时生成 `SPEAK`；
- `Always Listen Baseline`：始终输出 `LISTEN` 的准确率和 macro-F1；
- 非法状态转移率。

不得只报告 control accuracy，因为 `LISTEN` 占比很高。

### 10.2 时序

- `Onset Exact@0`；
- `Onset Exact@±1 tick`；
- onset MAE；
- interrupt 后停止延迟；
- 周期任务漏报率、误报率；
- 长序列 phase drift。

### 10.3 输出内容

- 分块拼接后的 token exact match；
- normalized EM / token F1；
- omission rate；
- duplication rate；
- chunk overflow rate；
- `<out_end>` 缺失或提前率。

### 10.4 协议完整性

- malformed tick rate；
- missing `<tick_end>` rate；
- 非法 control rate；
- `LISTEN/STOP` 后错误输出率；
- session completion rate。

### 10.5 联合成功率

一个任务仅在以下条件全部满足时记为成功：

```text
控制序列正确
AND 触发时间在允许窗口内
AND 最终输出 token 正确
AND 无超预算、重复或遗漏
AND 无协议错误
```

报告 item-level 和 session-level `Joint Success Rate`。

### 10.6 原能力保持

训练前后使用完全相同设置比较：

- 原模型验证文本 perplexity；
- 小型指令遵循集；
- 100–500 道通用问答；
- 固定 prompt 的输出抽样对比。

---

## 11. Pilot 成功标准

首轮 pilot 建议预注册以下门槛：

1. D 在 in-domain test 上 `Control Macro-F1 ≥ 0.90`；
2. `SPEAK` 和 `STOP` 各自 F1 ≥ 0.85；
3. `Onset Exact@±1 tick ≥ 90%`；
4. 中断停止延迟中位数 ≤ 1 tick；
5. 分块重建 token EM ≥ 90%；
6. chunk overflow rate ≤ 1%；
7. malformed tick rate ≤ 1%；
8. D 的 Joint Success 比 C 高至少 10 个绝对百分点；
9. 在 template/timing/length OOD 上达到 in-domain Joint Success 的至少 80%；
10. 通用能力无明显崩溃。

这些阈值用于工程 go/no-go，不代表最终论文结论。正式实验应至少运行 3 个训练随机种子，并报告 mean、std 和 bootstrap 95% CI；低成本 pilot 可以先运行单 seed，但不能据此声称稳定性。

---

## 12. 失败模式与诊断

### 12.1 依赖绝对位置

现象：模型总在固定 tick 输出。
诊断：改变起始 offset、插入空 tick、测试未见周期。
修复：随机化起始位置、触发时刻、周期和 session 长度。

### 12.2 一次输出完整答案

现象：首次 `SPEAK` 超过预算。
修复：训练目标按 tokenizer 预切块；推理强制 K；将 overflow 单独计为错误。

### 12.3 永远沉默

现象：accuracy 高但 `SPEAK/STOP` recall 很低。
修复：降低重复 `LISTEN` 权重、增加 hard positives，并始终报告 macro-F1 和 always-listen baseline。

### 12.4 看到任何输入都触发

现象：false trigger 高。
修复：增加相似关键词、无关数字、不完整问题和非打断表达等 hard negatives。

### 12.5 KV cache 顺序错误

现象：完整上下文正确，cache 版本错误。
修复：逐 token 比较 logits；确保顺序始终为 `input₁ → output₁ → input₂ → output₂`，并正确更新 position IDs 和 attention mask。

### 12.6 只记模板

现象：随机验证集高分，未见模板大幅下降。
修复：按模板族和规则参数划分测试集，而不是随机拆分同模板实例。

---

## 13. 实施阶段与交付物

### Phase 0：冻结协议

- 冻结特殊 token、状态机和 K；
- 冻结任务比例和数据 split 规则；
- 冻结指标与 pilot 门槛。

交付物：协议配置与 decision log。

### Phase 1：数据生成器

- 实现六类合成任务；
- 实现 tokenizer 级随机切分；
- 实现 parser、serializer 和 validator；
- 生成 100 条人工检查样本。

交付物：数据生成脚本、样例、校验报告。

### Phase 2：模型与逐 tick 推理

- 添加特殊 token 并 resize embedding；
- 实现完整上下文 reference loop；
- 实现 KV cache loop；
- 实现 constrained control decoding 和 K 限制。

交付物：模型加载脚本、tick inference demo、单元测试。

### Phase 3：小样本过拟合

- 分别对 C/D 过拟合 20–100 条 session；
- 检查 loss mask、控制预测和输出重建；
- 验证 save/resume。

交付物：overfit report。

### Phase 4：50k pilot

- 训练 C/D 和 loss-weight 消融；
- 只用 validation 选择 checkpoint；
- 记录显存、吞吐和训练时间。

交付物：checkpoint、训练曲线、资源报告。

### Phase 5：正式评价

- 运行 in-domain 和四类 OOD test；
- 计算控制、时序、内容、协议和联合指标；
- 运行原能力保持测试；
- 做错误分类和人工抽查。

交付物：逐 tick generation logs、metrics JSON/CSV、实验报告。

---

## 14. 建议目录结构

```text
src/
  protocol.py
  generate_synthetic_data.py
  validate_data.py
  train_qlora.py
  tick_inference.py
  evaluate.py
configs/
  protocol.yaml
  data.yaml
  train_qlora.yaml
tests/
  test_protocol.py
  test_data_generation.py
  test_tick_inference.py
artifacts/
  data/
  tokenizer/
  checkpoints/
  generation_outputs/
  metrics/
```

`artifacts/` 中的大型生成数据和 checkpoint 不应直接提交 Git；仓库只提交配置、manifest、哈希和小型测试样例。

---

## 15. 第一版冻结配置

```yaml
base_model: /120090727/yutong/model/Qwen3.5-0.8B
environment: /120090727/yutong/conda_env/text-omni-flow
training: qlora
tick_representation: explicit_text_event
max_input_tokens_per_tick: 16
max_output_tokens_per_tick: 4
context_length: 2048
session_ticks: [8, 64]
training_sessions: 50000
listen_ratio: [0.70, 0.90]
tasks:
  deterministic_clock: 0.50
  streaming_qa: 0.30
  interrupt_and_distractor: 0.20
controls:
  - listen
  - speak
  - continue
  - stop
primary_metric: joint_success_rate
```

第一阶段的目标不是获得自然聊天体验，而是先证明模型能否可靠学习：

```text
持续输入 + 固定时钟 + 独立控制决策 + 有预算的分块输出
```

只有该机制在独立规则、未见模板和更长序列上成立后，才进入真实聊天、语音编码和实时 scheduler 实验。
