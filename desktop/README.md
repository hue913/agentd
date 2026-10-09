# agentd Desktop

这是 `agentd` 的 Tauri 桌面壳，和仓库根目录的 `ui/` 共用同一套控制台。
桌面端只管理系统 SSH 隧道、系统钥匙串和窗口生命周期；圆桌、终端、服务器画面、审批和运维页面都来自 agentd API。

## 本地构建

```bash
cargo install tauri-cli --version '^2'
cd desktop/src-tauri
cargo tauri dev
cargo tauri build --target universal-apple-darwin  # macOS
cargo tauri build                                      # Windows runner 上执行
```

Windows 安装包由 GitHub Actions 的 `windows-latest` 构建，不能在 macOS 上交叉编译验证。
