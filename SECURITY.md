# 安全边界

请不要在公开 Issue 中提交 API Key、SSH 私钥、访问令牌、服务器地址和完整审计日志。

agentd 默认只监听 `127.0.0.1`，远程访问应通过 SSH 隧道。所有受保护 API 都要求
`AGENTD_API_TOKEN`；危险 SSH、文件、浏览器和部署操作经过安全 gate，需要人工审批。
noVNC 也只绑定服务器回环地址。

发现可复现的安全问题，请通过 GitHub Security Advisories 私下报告，并附上版本、最小复现步骤、
影响范围和建议修复方式。不要在未获授权的服务器上运行测试。
