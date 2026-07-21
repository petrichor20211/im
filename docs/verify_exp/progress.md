# 文本版 Omni-Flow 实验进度

更新时间：2026-07-21

| 阶段 | 状态 | 产物/说明 |
|---|---|---|
| Phase 0：冻结协议 | 完成 | `configs/protocol.yaml`、`decision_log.md` |
| Phase 1：数据生成 | 完成 | 50k sessions 已生成并通过校验，训练/验证/五类测试划分完整 |
| Phase 2：模型与推理 | 完成 | tokenizer、完整上下文和 cache 推理已实现；D 两种推理逐 tick 一致，C/D 解码均通过 smoke test |
| Phase 3：小样本过拟合 | 完成 | C/D 各 20 sessions、100 updates；最终 loss 分别约 0.025/0.014，生成检查通过 |
| Phase 4：50k pilot | 进行中 | C/D 双 GPU 并行，batch 40，正在比较 1e-4/2e-4 |
| Phase 5：正式评价 | 未开始 | 指标与 retention 脚本已实现，等待 pilot 选定学习率 |

## 当前后台任务

### 500-session C/D pilot

- PID 文件：`artifacts/runs/pilot_grid_bg/pid`
- 总日志：`artifacts/runs/pilot_grid_bg/run.log`
- 单次日志：`artifacts/runs/pilot_grid/<group>_lr_<lr>/launcher.log`
- 退出码：`artifacts/runs/pilot_grid_bg/exit_code`
- 资源：GPU 0 跑 C，GPU 1 跑 D；随机 batch 的安全 batch size 为 32，两卡计算利用率均为 100%

只有退出码为 0 且预期产物完整时，阶段才标记为完成。
