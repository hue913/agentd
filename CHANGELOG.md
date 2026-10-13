# Changelog

## Unreleased

### 安全

- **修复：模型可以自行批准破坏性命令。** `approved` 曾是 `local.exec` / `ssh.exec` 的
  工具参数，模型可借此绕过安全门——包括 `block` 级禁令。现在审批只来自可信的
  `ctx.approved`（即人工回答 `/api/approve` 之后的产物），参数已从两个 schema 中移除，
  传入会被参数校验明确拒绝。MCP 客户端的 `approved` 默认忽略，需 `AGENTD_MCP_PREAPPROVED=1`。
- systemd 单元新增 `UnsetEnvironment=AGENTD_AUTO_APPROVE`；若该变量仍然存在，运行时会打
  `CRITICAL` 日志并把审计记录标记为 `approved_by=auto-env`。

### 修复

- **修复：`agentd demo` 显示记忆无效。** 只有最后一步给奖励，导致每个正确的中间步骤都被
  记为失败，中间状态永远收集不到正回报。改为逐步给奖励后，学习臂从 0/40 变为 27/40。

### 文档

- 用实跑结果更新实测数字表（ops 12 种子学习臂均值 24.8/40，原写 14.4/40；seed 7 为
  21/40，原写 11/40），并补上逐种子明细表；修正 `demo` / `bench` 输出示例。
- 移除两处与代码不符的声明：桌面端 token 并未存入系统钥匙串（`keyring` 声明但从未调用）；
  systemd 此前并未剔除 `AGENTD_AUTO_APPROVE`（现已做实）。
- 测试数量由「约 300」更正为 380（15 个按平台跳过）。

### 新增

- 将 Tauri 桌面端合并进 `desktop/`，与浏览器控制台共用同一份 UI。
- 新增 session、浏览器运行的 SQLite trajectory 审计轨迹。
- 新增任务模板、JEV 配置和浏览器任务发现 API。
- 新增 macOS universal 与 Windows 桌面发布工作流。
- 新增 `tests/test_demo.py`：此前 `demo` 命令没有任何测试覆盖。
