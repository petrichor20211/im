# Interaction Models: A Scalable Approach to Human-AI Collaboration — Markdown 提取版

- **来源**: Thinking Machines Lab / Connectionism
- **发布日期**: 2026-05-11
- **原文链接**: https://thinkingmachines.ai/blog/interaction-models/
- **整理方式**: 结构化提取与中文概括；保留关键术语、模型设计、benchmark 数字和引用信息，但不做逐字全文转载。

---

## 1. 一句话概括

Thinking Machines Lab 提出 **interaction models**：一种把实时交互能力内化到模型本身的模型范式，而不是依赖外部 harness 去拼接语音检测、多模态输入、打断、工具调用等能力。核心目标是让 AI 像人类协作者一样持续听、看、说、想、行动，并且能在音频、视频、文本之间进行实时双向交互。

---

## 2. 核心问题：collaboration bottleneck

当前很多 AI 系统更强调“自主完成长任务”，但真实工作中，人往往无法一次性把需求完整说清楚。高质量结果通常来自持续协作：人需要随时补充背景、纠正方向、给反馈，而模型也需要边做边解释、询问、响应。

传统 turn-based interface 的限制是：

- 用户输入完成前，模型通常只能等待。
- 模型生成时，它对外界新输入的感知会冻结，除非被打断。
- 这种单线程、轮流说话的交互方式限制了人类知识、意图和判断进入模型的带宽。
- 因此，真正的人机协作不应只是“prompt → response”，而应该支持实时、连续、多模态、可打断的共同工作。

作者认为，如果要让交互能力随着模型智能一起 scale，交互不能只是外部组件拼接，而应成为模型架构与训练目标的一部分。

---

## 3. Interaction model 支持的能力

文章列出的关键能力包括：

### 3.1 Seamless dialog management

模型能隐式判断用户是在思考、让出话轮、自我纠正，还是在邀请模型回答；不需要单独的对话管理模块。

### 3.2 Verbal and visual interjections

模型可以基于上下文主动插话，不必等用户完整说完。例如：当用户说错、写错代码、或者画面中出现新线索时，模型可以及时介入。

### 3.3 Simultaneous speech

用户和模型可以同时说话。典型场景包括实时翻译、同声反馈、口语纠错等。

### 3.4 Time-awareness

模型直接感知时间流逝，因此可以处理“每 4 秒提醒一次”“记录我跑一英里用了多久”这类任务。

### 3.5 Simultaneous tool calls / search / generative UI

模型在听用户说话、继续对话的同时，可以并发进行搜索、浏览网页、调用工具或生成 UI，并把结果自然地织入对话。

---

## 4. 方法总览

### 4.1 Turn-based vs. time-aligned micro-turn

传统模型把输入和输出展平成一个顺序 token 序列：

```text
input 1 → output 1 → input 2 → output 2 → ...
```

interaction model 则把交互放到时间轴上：

```text
每 200ms 作为一个 micro-turn
持续接收 video/audio/text 输入
同时生成 text/audio 输出
```

这种设计让 silence、overlap、interruption 都成为模型上下文的一部分，而不是由外部规则额外处理。

### 4.2 系统结构：interaction model + background model

系统由两部分组成：

| 组件 | 作用 |
|---|---|
| **Interaction model** | 持续与用户实时交互，保持低延迟存在感，处理即时听说看写。 |
| **Background model** | 执行更长链路的推理、工具调用、搜索、规划和 agentic workflow。 |

二者共享上下文。interaction model 可以在需要深入推理时把任务交给 background model，同时继续和用户保持对话，并在合适时机把后台结果融合进交互。

---

## 5. Interaction model 的关键设计

### 5.1 Time-aligned micro-turns

模型以约 **200ms** 为单位，把输入处理和输出生成交错进行。它不是等用户讲完一个完整 turn 再回答，而是把输入/输出都视为连续流。

这种设计使模型可以：

- 在用户讲话时继续听；
- 在自己说话时继续接收新输入；
- 支持打断、backchannel、同声翻译、实时视觉反馈；
- 避免强依赖 VAD 等外部 turn boundary 预测组件。

### 5.2 Encoder-free early fusion

作者没有使用大型独立音频/视频 encoder，而是采用更轻量的早期融合方式：

| 模态 | 处理方式 |
|---|---|
| Audio | 使用 dMel 表示，并通过轻量 embedding layer 转换。 |
| Image / Video | 切成 40×40 patches，再用 hMLP 编码。 |
| Text | 作为 token 输入。 |
| Audio output | 使用 flow head 生成音频。 |

所有组件与 transformer 一起从头联合训练。

### 5.3 Inference optimization

因为 200ms micro-turn 会造成频繁的小 prefill / decode 请求，传统 LLM inference library 的固定开销会变得很明显。作者的优化包括：

- **Streaming sessions**：客户端每 200ms 发一个 chunk，server 在 GPU memory 中维护持久序列，避免频繁重新分配内存和重复计算元数据。
- 将相关能力 upstream 到 SGLang。
- 针对低延迟和双向 serving shape 优化 kernel。
- MoE kernel 使用 gather + gemv 策略，而不是标准 grouped gemm。

### 5.4 Trainer-sampler alignment

作者认为 bitwise trainer-sampler alignment 对训练稳定性和调试很有帮助。他们实现了 batch-invariant kernels，并声称端到端性能开销小于 5%。重点包括：

- 使用 NVLS 实现低延迟、确定性的 all-reduce / reduce-scatter。
- 对 attention 中 Split-KV 带来的 accumulation order 不一致问题进行处理，使 decode 和 prefill 保持一致的累加顺序。

### 5.5 Interaction model 与 background model 的协调

interaction model 不只是给 background model 发一个孤立 query，而是发送完整上下文包。background model 的结果会流式返回，interaction model 再根据当前用户正在做什么，选择合适时机把结果插入对话，避免突兀切换。

### 5.6 Safety

实时语音交互带来不同于文本 turn-based 场景的安全问题。作者的安全训练重点包括：

- 让语音拒答更自然、口语化，但边界仍然明确；
- 使用自动红队 harness 生成多轮 speech-to-speech refusal 数据；
- 保持语音拒答和文本拒答行为尽量一致。

---

## 6. Benchmark 结果摘录

模型名：**TML-Interaction-Small**

文章称该模型在“智能性 + 交互性”组合维度上处于 frontier，尤其强调它既有较强 instruction following / intelligence，又能保持低延迟实时交互。

### 6.1 主要 benchmark 指标

| Benchmark / Metric | Modality | TML-Interaction-Small | 说明 |
|---|---:|---:|---|
| FD-bench V1 turn-taking latency | Audio | 0.40s | 越低越好，衡量简单话轮延迟。 |
| FD-bench V1.5 average quality | Audio | 77.8 | 衡量交互质量。 |
| FD-bench V3 response quality / pass@1 | Audio + Tools | 82.8 / 68.0 | 需要推理或工具调用的任务使用 background agent。 |
| QIVD accuracy | Video + Audio | 54.0 | 流式视频音频 QA。 |
| Audio MultiChallenge APR | Audio | 43.4 | 衡量音频理解、智能性与指令跟随。 |
| BigBench Audio accuracy | Audio | 75.7 / 96.5 | 文章对部分设置作了额外说明。 |
| IFEval VoiceBench accuracy | Audio | 82.1 | 音频指令跟随。 |
| IFEval accuracy | Text | 89.7 | 文本指令跟随。 |
| Harmbench refusal rate | Text | 99.0 | 文本安全拒答率。 |

### 6.2 新交互维度 benchmark

作者认为现有 benchmark 不能充分衡量 interaction model 的新能力，因此提出或改造了一些测试：

| 能力 | Benchmark | TML-Interaction-Small | GPT realtime-2.0 minimal | 任务含义 |
|---|---|---:|---:|---|
| 时间感知 | TimeSpeak macro-acc | 64.7 | 4.3 | 在指定时间点主动说出正确内容。 |
| 语音 cue 触发 | CueSpeak macro-acc | 81.7 | 2.9 | 在用户说话过程中根据 cue 同步反馈。 |
| 视觉计数 | RepCount-A off-by-one | 35.4 | 1.3 | 根据视频动作实时计数。 |
| 视觉 cue 触发 | ProactiveVideoQA PAUC@ω=0.5 | 33.5 | 25.0 | 答案出现时主动回答；25.0 是无响应 baseline。 |
| 视觉 cue 触发 | Charades mIoU | 32.4 | 0 | 判断动作开始/结束时机。 |

作者认为，现有商用实时 API 主要依赖音频 turn detection，通常无法在视觉世界发生变化时主动选择说话。

---

## 7. 局限与未来工作

### 7.1 Long sessions

连续音视频会快速积累上下文。streaming-session 设计能处理短到中等时长交互，但特别长的 session 仍需要更好的上下文管理。

### 7.2 Compute and deployment

低延迟音视频流需要稳定连接。网络不好时体验会显著下降。作者认为可以通过系统可靠性提升和训练模型适应延迟帧来改善。

### 7.3 Alignment and safety

实时交互界面会带来新的 alignment 与 safety 研究问题，作者正在收集反馈并计划相关研究资助。

### 7.4 Scaling model size

当前 **TML-Interaction-Small** 是一个 **276B 参数 MoE，12B active** 的模型。更大的预训练模型目前在这种实时 serving setting 下太慢，作者计划未来发布更大模型。

### 7.5 Better background agents

文章虽然重点是实时交互，但作者认为 agentic intelligence 同样关键。未来方向包括提升 background agent 能力，以及探索它们如何更好地和 interaction model 协作。

---

## 8. 文章提出的研究意义

这篇文章的核心贡献不是单纯提出一个语音模型，而是提出一种新的系统范式：

> 把实时交互、多模态流式感知、打断、同时说话、视觉主动性和工具并发能力作为模型本身的原生能力，而不是外部系统工程的拼接结果。

对 Audio LLM / Multimodal Agent / Computer Use Agent 的启发包括：

- 交互能力本身可以成为 scaling target；
- full-duplex 不只是语音同时输入输出，还包括持续上下文更新与多模态同步；
- 低延迟 serving 和模型架构必须共同设计；
- background reasoning 与 foreground interaction 的分工可能是未来通用 agent 的重要形态；
- 现有 benchmark 对“交互质量”的刻画还不够，需要更多时间感知、主动视觉、重叠语音、实时纠错等任务。

---

## 9. 推荐引用信息

- Thinking Machines Lab. *Interaction Models: A Scalable Approach to Human-AI Collaboration*. Thinking Machines Lab: Connectionism, May 2026.
- DOI: `10.64434/tml.20260511`
- URL: https://thinkingmachines.ai/blog/interaction-models/

---

## 10. 可作为论文阅读笔记的关键词

- Interaction Models
- Full-duplex AI
- Real-time multimodal interaction
- Time-aligned micro-turns
- Streaming sessions
- Encoder-free early fusion
- Background model / Interaction model coordination
- Audio-video-text continuous streams
- Proactive visual interaction
- Trainer-sampler alignment
- Batch-invariant kernels
- Human-AI collaboration
