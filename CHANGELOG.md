# Changelog

## Unreleased

### 安全

- **HTTP `/api/ssh/exec` 不再接受客户端 `approved: true`。** 持 token 的调用方不能再自批
  block 级命令；放行只走 `/api/approve`。
- **打开交互式 PTY 必须人工审批。** 请求体里的 `approved` 已删除。终端打开后逐键仍无法过闸，
  控制台会写明这一点。
- **`local.exec` / skill 的审批闭环打通。** 不可证只读的命令会向操作员提问，批准只作用于
  这一次调用，不会写回整集 `CallContext`。
- **`fs.read` / `fs.write` / `fs.ls` 拒绝密钥与控制面路径**（`.ssh`、`agentd.json`、
  `api.token`、`/var/lib/agentd`）。`ssh.upload` 升为 dangerous。
- **出站 MCP 工具默认 dangerous**（除非明确 `readOnlyHint`）。进站 MCP 的 `approved` 参数
  被丢弃，`AGENTD_MCP_PREAPPROVED` 不再放行。
- **校验失败的记忆包拒绝导入。** 不再只警告然后照样写入。
- 插件可按 `plugins_allow` 做 sha256 放行；未列入则拒绝加载。
- `/healthz` 只返回 `{ok, ts}`，不再泄露 kernel 统计。

### 学习信号

- **ops `finish` 不再给 +1 奖励。** 结束 episode 不等于目标达成，避免内核学会「立刻 finish」。

### 其它

- 圆桌复用 primary 的第一次 `choose`，不再为同一席位付两次 token。
- OpenAI 兼容 provider 的 `text()` 可携带原生 `tools` 目录（循环仍用编号选择，接口已铺路）。
- 控制台 HUD 展示 `cache_hit_pct`；桌面端去掉未使用的 `keyring` 与 `shell:allow-open`。
- CI 增加 Python 3.13 与 ruff；`.gitignore` 排除 `desktop/src-tauri/target/`。
- README 图示改为 PTY WebSocket 与 API 同端口 8765。

- **修复：模型可以自行批准破坏性命令。** `approved` 曾是 `local.exec` / `ssh.exec` 的
  工具参数，模型可借此绕过安全门——包括 `block` 级禁令。现在审批只来自可信的
  `ctx.approved`（即人工回答 `/api/approve` 之后的产物），参数已从两个 schema 中移除，
  传入会被参数校验明确拒绝。
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
