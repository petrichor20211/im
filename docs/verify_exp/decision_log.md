# 文本版 Omni-Flow 实验 Decision Log

本文件记录实验开始前冻结的关键决定。若后续修改，必须新增记录，不覆盖原记录，并说明修改是否发生在查看正式 test 结果之前。

## 2026-07-21：Phase 0 初始冻结

| ID | 决定 | 选择 | 理由 |
|---|---|---|---|
| D01 | 第一阶段研究范围 | 仅做固定时钟文本状态机 | 最快隔离并验证交互协议，不混入语音、视觉和实时系统变量 |
| D02 | Transformer 架构 | 不修改 decoder-only causal LM | 验证能力是否能通过协议与训练获得 |
| D03 | Base model | `/120090727/yutong/model/Qwen3.5-0.8B` | 本地已有、规模适合快速 QLoRA pilot |
| D04 | Python 环境 | `/120090727/yutong/conda_env/text-omni-flow` | 使用独立环境，避免影响已有 Qwen3-Omni 环境 |
| D05 | 控制形式 | 独立 `LISTEN/SPEAK/CONTINUE/STOP` token | 将“是否说”与“说什么”解耦，并与无独立控制的 C 组比较 |
| D06 | Tick 边界 | 使用显式 `<tick>` 和 `<tick_end>` | 降低模型识别时间块边界的负担 |
| D07 | 绝对 tick 编号 | 不提供 | 防止模型直接依赖绝对位置完成计时任务 |
| D08 | 每 tick 输入上限 | 16 个普通 tokenizer token | 保留随机切分空间，同时控制上下文长度 |
| D09 | 每 tick 输出上限 | 4 个普通 tokenizer token | 强制答案跨 tick 输出；特殊 token 不计入预算 |
| D10 | 初始上下文长度 | 2048 tokens | 降低 pilot 成本；长序列作为独立 OOD 测试 |
| D11 | Session 长度 | 8–64 ticks | 覆盖短触发和中等长度状态保持 |
| D12 | 初始数据规模 | 50,000 sessions | 足以观察学习信号，同时可在机制确认后扩展 |
| D13 | 任务配比 | 50% 确定性时钟、30% 流式问答、20% 打断与干扰 | 优先学习机制，再逐渐增加语义与鲁棒性 |
| D14 | LISTEN 比例 | 70%–90% | 模拟稀疏输出，同时通过加权和 macro-F1 防止沉默投机 |
| D15 | 数据划分 | 按模板族和规则参数划分 OOD test | 避免随机拆分只测到模板记忆 |
| D16 | 训练方式 | QLoRA，4-bit，bf16 compute | 以较低显存和时间成本验证方案 |
| D17 | 重复 LISTEN 权重 | 0.3 | 降低多数类主导；另保留不重加权消融 |
| D18 | 对照组 | A/B/C/D 四组 | 分离训练、chunked inference 和独立 control 的贡献 |
| D19 | 主指标 | Joint Success Rate | 要求控制、时序、内容和协议同时正确 |
| D20 | 推理正确性基准 | 先完整增长上下文，再实现 KV cache | 优先保证语义正确，再优化推理效率，并要求两者一致 |
| D21 | 正式实验随机种子 | 3 | 支持均值、标准差和置信区间；pilot 可先单 seed |

## 2026-07-21：环境修订

最初计划复用 `qwen3omni`，但其中 Transformers 4.57.3 无法识别本地 Qwen3.5 checkpoint。现已创建全新的 Conda 环境 `text-omni-flow`，安装 PyTorch 2.11.0+cu128、Transformers 5.14.1、PEFT 0.19.1 和 bitsandbytes 0.49.2。该修改发生在训练和正式测试之前，不改变模型、数据或评价假设。

## 2026-07-22：停止全量评价

用户要求不再进行全量评价，改为依据当前完整结果分析。已停止 formal evaluation，保留 seed 20260721 的 in-domain 与 template OOD（每个 split、每组各 2,000 sessions）。未完成的 timing OOD 分片不进入结论；length OOD、distractor OOD、retention 和 B baseline 标记为未评价。详细结果见 `partial_results_analysis.md`。

## 2026-07-21：GPU 资源配置修订

实测训练序列平均 223 tokens、P99 411、最大 480。batch size 32 在随机 batch 下稳定运行，两张 A800 的计算利用率均达到 100%；batch size 40 的短样本单步基准可运行，但真实随机 batch 在反向阶段 OOM，batch size 64 必然 OOM。因此 pilot 冻结为每卡 batch size 32、C/D 分别使用 GPU 0/1 并行。正式训练可在实现长度分桶后重新测试动态 batch，但不得降低 C/D 公平性。

## 冻结的状态机

```text
IDLE + LISTEN   -> IDLE
IDLE + SPEAK    -> SPEAKING
SPEAKING + CONTINUE -> SPEAKING
SPEAKING + STOP -> IDLE
SPEAKING + OUT_END -> IDLE
```

非法情况包括：

- `IDLE` 生成 `CONTINUE` 或 `STOP`；
- `SPEAKING` 无原因生成新的 `SPEAK`；
- `LISTEN/STOP` 后生成普通输出；
- 单 tick 普通输出超过 4 tokens；
- 缺少 `<tick_end>`；
- 回答完成但缺少 `<out_end>`。

## Pilot go/no-go 门槛

- Control Macro-F1 ≥ 0.90；
- SPEAK F1、STOP F1 均 ≥ 0.85；
- Onset Exact@±1 tick ≥ 0.90；
- interrupt 停止延迟中位数 ≤ 1 tick；
- 分块重建 token EM ≥ 0.90；
- chunk overflow 与 malformed tick 均 ≤ 1%；
- D 相比 C 的 Joint Success 至少提高 10 个绝对百分点；
- OOD Joint Success 至少保留 in-domain 的 80%。

## 下一步

进入 Phase 1：先实现协议 parser/serializer/validator 和小规模合成数据生成器；在生成大规模数据前，输出约 100 条样本并通过自动校验与人工抽查。
