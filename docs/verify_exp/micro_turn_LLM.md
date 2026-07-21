# 最小受控实验

## 1. 模型与序列格式

使用同一个 `1.5B–3B Instruct LLM`，全参数 continued training，不改 Transformer 架构。增加少量特殊 token：

```text
<TICK>
<U> 用户本时刻新增文本 </U>
<ACT> SIL | START | CONT | STOP </ACT>
<A> 助手本时刻输出，最多 4 tokens </A>
```

示例：

```text
<TICK><U>请每隔三步说一次 ping</U><ACT>SIL</ACT><A></A>
<TICK><U></U><ACT>SIL</ACT><A></A>
<TICK><U></U><ACT>START</ACT><A>ping</A>
<TICK><U>嗯，继续</U><ACT>SIL</ACT><A></A>
```

不提供绝对 tick 编号，避免模型直接读取时间戳。训练时只计算 `ACT` 和 `A` 的 loss；推理时由外部 scheduler 每个 tick 增加一次输入，并强制模型在 `</A>` 处停止生成。

实际 200ms 由 scheduler 保证，模型只学习“一次 tick 对应一次决策”。

## 2. 训练数据

建议先做约 **10M target tokens** 的快速实验，每条序列 32–128 ticks，包含四类任务：

|  占比 | 任务                                      |
| --: | --------------------------------------- |
| 35% | 延迟与周期：第 `k` 个 tick 回答、每 `k` ticks 输出一次  |
| 25% | 流式转换：输入 token 后固定延迟 `d` ticks 输出其变换     |
| 25% | 交互控制：pause、interrupt、backchannel、用户自我修正 |
| 15% | 普通对话，将用户和助手文本按固定速率切成 chunks，并制造重叠       |

训练只使用部分延迟值和序列长度；测试保留未见过的延迟值、提示模板和两倍长度序列，防止只记模板。

重复 `SIL` 极多，建议采用：

[
L=L_{\text{text}}+\lambda L_{\text{act}},\quad \lambda=1
]

连续 `SIL` 权重设为约 `0.3`，`START/STOP` 设为 `1.5`。这与相关工作中单独加强 streaming-control、降低连续 silence 主导效应的思路一致。Audio-Interaction 采用独立 streaming loss；JoyAI 对 silence 与 response onset 进行差异化加权。 

## 3. 对照组

所有组从**同一 checkpoint** 初始化，使用相同学习率、训练步数和原始语义数据。

| 组别 | 训练方式                                 | 目的              |
| -- | ------------------------------------ | --------------- |
| C0 | 原始模型，不训练                             | 原始通用能力基线        |
| C1 | 普通 turn-based CT + 20% 通用数据 replay   | 控制领域数据和继续训练效应   |
| C2 | Micro-turn CT + 20% 通用数据 replay      | 主实验             |
| C3 | Micro-turn CT，不加通用 replay            | 测量灾难性遗忘         |
| C4 | 与 C2 完全相同，但将 ACT/输出时间随机平移 ±2–4 ticks | 排除只学会格式而未学会时间对齐 |

其中：

* **C2 vs. C1**：固定时钟格式是否真正带来交互能力。
* **C2 vs. C4**：模型是否学习了正确的时序关系，而非特殊 token 格式。
* **C2 vs. C3**：混合通用数据能否保护原始能力。

DuplexSLA 在 continued pretraining 中明确加入通用文本以保留知识与推理能力；相关设计因此很适合作为这里的核心对照。

## 4. 评价指标

### 交互能力

1. `ACT macro-F1`：SIL / START / CONT / STOP。
2. `Timing Exact@0`、`Exact@±1 tick` 和 timing MAE。
3. 周期任务的漏报率、误报率和长期 phase drift。
4. Interrupt 后停止延迟。
5. Backchannel false-stop rate。
6. 未见 delay、未见模板和两倍长度序列上的性能。

### 通用能力

训练前后固定测试：

* IFEval：指令遵循
* MMLU 或 C-Eval：知识
* GSM8K：推理
* MBPP/HumanEval：代码
* 通用文本 perplexity

报告每项绝对变化以及平均能力保持率：

[
\text{Retention}=\frac{\text{post-training score}}
{\text{base score}}
]

## 5. 判定标准

认为模型**学会了交互时钟**，至少需要：

* C2 的 `Exact@±1 tick > 90%`；
* 显著优于 C1 和 C4；
* unseen delay 和两倍长度下仍保持训练内性能的 80% 以上；
* Interrupt 延迟不超过 1 tick；
* 长序列周期输出没有持续相位漂移。

认为**通用能力没有明显受损**，可预注册为：

* C2 综合能力保持率不低于 98%；
* 单项 benchmark 下降不超过 2 个绝对点；
* C2 明显优于不带 replay 的 C3。

最有价值的结果不是单纯得到高 timing accuracy，而是得到如下 Pareto 曲线：

> 随着 interaction-data 占比增加，时序能力快速上升；加入约 10%–30% 通用数据 replay 后，通用能力基本保持。

这个实验成本低、变量干净，并能直接回答两个问题：**现有纯文本 Transformer 是否具备学习 micro-turn 协议的归纳能力，以及这种新行为是否必须以牺牲原有通用能力为代价。**
