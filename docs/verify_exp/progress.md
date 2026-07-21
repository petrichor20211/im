# 文本版 Omni-Flow 实验进度

更新时间：2026-07-21

| 阶段 | 状态 | 产物/说明 |
|---|---|---|
| Phase 0：冻结协议 | 完成 | `configs/protocol.yaml`、`decision_log.md` |
| Phase 1：数据生成 | 进行中 | 100 条预览已通过校验；50k 数据正在后台生成 |
| Phase 2：模型与推理 | 进行中 | tokenizer、完整上下文推理、cache 推理、C/D 解码器已实现；待 GPU 一致性验证 |
| Phase 3：小样本过拟合 | 进行中 | 20 条 D 组、100 updates 正在后台训练 |
| Phase 4：50k pilot | 未开始 | 等待 Phase 3 门控和 50k 数据 |
| Phase 5：正式评价 | 未开始 | 指标与 retention 脚本已实现，等待 checkpoints |

## 当前后台任务

### D 组小样本过拟合

- PID 文件：`artifacts/runs/overfit_D_bg/pid`
- 日志：`artifacts/runs/overfit_D_bg/run.log`
- 退出码：`artifacts/runs/overfit_D_bg/exit_code`

### 50k 数据生成与校验

- PID 文件：`artifacts/data/generate_50k.pid`
- 日志：`artifacts/data/generate_50k.log`
- 退出码：`artifacts/data/generate_50k.exit_code`

只有退出码为 0 且预期产物完整时，阶段才标记为完成。
