"""Small, editable task templates exposed by the local console."""

from __future__ import annotations

DEFAULT_TASK_TEMPLATES = [
    {"id": "ops-triage", "kind": "ops", "title": "服务器故障排查",
     "goal": "收集健康指标、日志和进程信息，给出可验证的修复步骤。",
     "acceptance": ["结论包含证据", "危险操作明确标注并等待审批"]},
    {"id": "coding-delivery", "kind": "coding", "title": "交付一个可验证的软件功能",
     "goal": "实现一个边界清晰的功能，并运行针对性测试。",
     "acceptance": ["实现覆盖需求", "关键行为有自动化验证"]},
    {"id": "review-security", "kind": "code-review", "title": "审查安全边界",
     "goal": "检查认证、授权、输入处理、密钥和日志边界。",
     "acceptance": ["高风险问题有代码证据", "修复建议遵循最小权限"]},
    {"id": "research-brief", "kind": "research", "title": "形成证据简报",
     "goal": "收集可追溯证据，区分事实、推断和未知项，并保留反证。",
     "acceptance": ["关键结论可追溯", "不确定性和后续动作明确"]},
    {"id": "browser-task", "kind": "browser", "title": "完成浏览器任务",
     "goal": "在受限浏览器环境中完成目标并保存截图和轨迹。",
     "acceptance": ["页面检查器返回成功", "失败步骤和截图可审计"]},
]


def templates_from_config(config: dict) -> list[dict]:
    custom = config.get("task_templates")
    return custom if isinstance(custom, list) else list(DEFAULT_TASK_TEMPLATES)
