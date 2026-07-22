# 文本版 Omni-Flow 实验进度

更新时间：2026-07-22

| 阶段 | 状态 | 产物/说明 |
|---|---|---|
| Phase 0：冻结协议 | 完成 | `configs/protocol.yaml`、`decision_log.md` |
| Phase 1：数据生成 | 完成 | 50k sessions 已生成并通过校验 |
| Phase 2：模型与推理 | 完成 | 完整上下文与 cache 推理逐 tick 一致；C/D 解码通过检查 |
| Phase 3：小样本过拟合 | 完成 | C/D 各 20 sessions、100 updates，生成检查通过 |
| Phase 4：500-session pilot | 完成 | `2e-4` 在 validation loss 和生成指标上优于 `1e-4` |
| Phase 5：正式训练 | 完成 | C/D × 3 seeds；每个模型 35k sessions、3 epochs、3282 steps |
| Phase 6：正式评价 | 按用户要求提前停止 | 完成 seed 20260721 的 in-domain 与 template OOD，各 2,000 sessions；其余不再运行 |
| Phase 7：结果分析 | 完成 | 见 `partial_results_analysis.md` |

## 已完成的正式训练

- 学习率：`2e-4`
- seeds：`20260721`、`20260722`、`20260723`
- C validation loss：0.020893、0.020898、0.020685
- D validation loss：0.025436、0.025520、0.025183
- 单模型训练时间：C 约 3.97 小时，D 约 3.66 小时

## 可用于结论的行为评价

- seed：`20260721`
- `test_in_domain`：C/D 各 2,000 sessions
- `test_template_ood`：C/D 各 2,000 sessions
- 完整结果：`artifacts/runs/formal_eval/seed_20260721/`
- 分任务结果：`artifacts/runs/formal_eval/partial_analysis/`
- 分析报告：`docs/verify_exp/partial_results_analysis.md`

`test_timing_ood` 只有中途停止的分片，不纳入任何指标或结论。length OOD、distractor OOD、retention 和 B baseline 未运行。

## 后台状态

正式评价已按用户要求停止：

- 状态文件：`artifacts/runs/formal_eval_bg/exit_code`（130，表示用户中止）
- 停止说明：`artifacts/runs/formal_eval_bg/cancelled.txt`
- 当前没有实验 GPU 进程。
