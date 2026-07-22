# 文本 Omni-Flow 部分正式结果分析

更新时间：2026-07-22

## 1. 分析范围

用户要求停止全量评价。本报告仅使用已经完整生成的结果：

- 模型：Qwen3.5-0.8B，QLoRA；
- 学习率：pilot 选择的 `2e-4`；
- 正式训练：C/D 各 3 seeds、35k sessions、3 epochs，全部完成；
- 行为评价：仅 seed `20260721`；
- 测试集：`test_in_domain` 和 `test_template_ood`，各 2,000 sessions；
- `test_timing_ood` 只有不完整分片，不用于结论；
- length OOD、distractor OOD、retention 和未训练 B 基线未完成。

因此，本报告是有 2,000 条配对测试样本支撑的单 seed 分析，不是多 seed 稳定性结论。

## 2. 训练结果

三个 seed 的 validation loss 很接近：

| 组 | seed 20260721 | seed 20260722 | seed 20260723 |
|---|---:|---:|---:|
| C | 0.020893 | 0.020898 | 0.020685 |
| D | 0.025436 | 0.025520 | 0.025183 |

C/D 的目标空间不同，loss 不能直接横向比较；这里只能说明每组内部训练稳定、没有明显 seed 崩溃。

## 3. 总体结果

### 3.1 In-domain

| 指标 | C | D | D−C |
|---|---:|---:|---:|
| Control accuracy | 98.856% | 98.694% | -0.162 pp |
| Control Macro-F1 | 0.9305 | 0.9235 | -0.0070 |
| Onset Exact@±1 | 99.647% | 99.855% | +0.208 pp |
| Tick output EM | 96.517% | 96.425% | -0.092 pp |
| Reconstructed token EM | 87.45% | 84.95% | -2.50 pp |
| Out-end accuracy | 98.049% | 98.197% | +0.148 pp |
| False-trigger rate | 0.006% | 0.102% | +0.096 pp |
| Joint session success | 52.25% | 51.90% | -0.35 pp |
| Overflow / malformed | 0 / 0 | 0 / 0 | 0 |

对同一批 session 做配对比较：

- Joint success：D 胜 86 条、C 胜 93 条、平 1,821 条；D−C 为 -0.35 pp，近似 95% CI `[-1.66, +0.96] pp`。
- Reconstruction EM：D 胜 27 条、C 胜 77 条、平 1,896 条；D−C 为 -2.50 pp，近似 95% CI `[-3.49, -1.51] pp`。

Joint success 没有可辨别差异；D 的重建完整率则明显低于 C。

### 3.2 Template OOD

该 split 只有 trigger 和 wait 两类，没有 STOP 样本。因此 Macro-F1 会把缺失的 STOP 类按 0 计入，不宜与 in-domain Macro-F1 直接比较。

| 指标 | C | D | D−C |
|---|---:|---:|---:|
| Control accuracy | 99.042% | 98.854% | -0.188 pp |
| Onset Exact@±1 | 97.15% | 96.65% | -0.50 pp |
| Reconstructed token EM | 96.50% | 93.20% | -3.30 pp |
| False-trigger rate | 0.026% | 0.127% | +0.101 pp |
| Joint session success | 68.70% | 67.75% | -0.95 pp |
| Overflow / malformed | 0 / 0 | 0 / 0 | 0 |

配对比较：

- Joint success：D−C 为 -0.95 pp，近似 95% CI `[-2.50, +0.60] pp`，无法区分。
- Reconstruction EM：D−C 为 -3.30 pp，近似 95% CI `[-4.30, -2.30] pp`，D 明显更低。

两组都能迁移到未见模板，但没有观察到 D 的优势。

## 4. 分任务诊断

### In-domain joint / reconstruction

| 任务 | C Joint | D Joint | C Recon | D Recon | 主要观察 |
|---|---:|---:|---:|---:|---|
| trigger | 1.000 | 1.000 | 1.000 | 1.000 | 两组均完全学会触发 |
| periodic | 1.000 | 1.000 | 1.000 | 1.000 | 固定节拍任务成功 |
| distractor | 1.000 | 1.000 | 1.000 | 1.000 | 均能保持沉默 |
| wait | 0.375 | 0.341 | 0.951 | 0.825 | D 更容易误触发/丢失内容 |
| streaming | 0.156 | 0.143 | 0.883 | 0.883 | 内容常可重建，但逐 tick 联合严格成功率低 |
| interrupt | 0.000 | 0.085 | 0.102 | 0.108 | 表面上 D 更高，但受到 C STOP 表示偏差混淆 |

Interrupt 上两组 control accuracy 都是 100%，但：

- C 的 out-end accuracy 为 97.15%，没有一条 session 达到严格 joint success；
- D 的 out-end accuracy 为 100%，joint success 达到 8.52%。

检查真实预测后发现，这不能直接视为 D 的局部收益。C 没有 `<stop>` token，只能用“空 output + `<out_end>`”表达 STOP；原始 interrupt 标签却要求 `out_end=false`。因此 C 若被解码器识别为 STOP，就会必然在 out-end 指标上与标签冲突。当前 interrupt Joint 对 C 存在结构性不公平，必须先把 C/D 的内部表示规范化为同一语义事件再比较。

## 5. 对预注册门槛的判断

基于当前唯一完整 seed：

| 门槛 | 结果 |
|---|---|
| D in-domain Control Macro-F1 ≥ 0.90 | 通过：0.9235 |
| SPEAK / STOP F1 ≥ 0.85 | 通过：0.9847 / 0.9972 |
| Onset Exact@±1 ≥ 0.90 | 通过：0.9985 |
| 中断停止延迟中位数 ≤ 1 tick | 通过：0 tick |
| 重建 token EM ≥ 0.90 | **失败：0.8495** |
| overflow ≤ 1% | 通过：0 |
| malformed ≤ 1% | 通过：0 |
| D Joint 比 C 高至少 10 pp | **失败：-0.35 pp** |
| Template OOD joint 保留 ≥ 80% | 通过，但只覆盖一种 OOD |
| 通用能力无明显崩溃 | 未评价 |

## 6. 结论

1. **H1 基本得到支持。** trigger、periodic 和 distractor 达到完全成功，说明普通 causal LLM 能学习本实验定义下的固定文本节拍和沉默策略。
2. **H2 部分得到支持。** streaming 的重建 EM 达到 88.3%，说明模型能在多 tick 中持续输出；但严格 joint success 仅约 14%–16%，逐 tick continuation/out-end 仍不稳定。
3. **H3 的强版本不受支持。** D 没有在总体 Joint Success 上优于 C，更没有达到预注册的 +10 pp；两个完整测试集上 D 的点估计都略低。
4. **Interrupt 的表面局部信号不可采信。** C 的隐式 STOP 必须伴随 `<out_end>`，但标签要求 `out_end=false`，所以现有比较偏向 D。
5. **D 的主要代价是内容完整性和误触发。** D 的 reconstructed EM 在两个测试集都低 2.5–3.3 pp，false-trigger rate 也更高，尽管绝对值仍低。
6. **协议本身学习成功。** 两组 overflow 和 malformed 都为 0；主要错误来自语义内容、continue/out-end 时序，而不是格式崩坏。

## 7. 解释限制

C 并非完全没有显式边界：它仍使用 `<output>`、`</output>`、`<out_end>` 和 `<tick_end>`，并可通过“输出是否为空 + 当前状态”隐式恢复控制。因此当前 C/D 对比检验的是“额外独立 LISTEN/SPEAK/CONTINUE/STOP token”的增益，而不是“有无任何状态控制”的增益。任务又高度确定、协议脚手架很强，所以 C 已足以解决大部分控制问题。此外，C 的 STOP 表示与原始 out-end 标签不兼容，interrupt 结果不能用于证明 D 的优势。

在没有其余 seeds 行为评价、timing/length/distractor OOD 和 retention 的情况下，不应声称结果具有多 seed 稳定性或完整 OOD 泛化结论。
