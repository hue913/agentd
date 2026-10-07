"""Python plugins: drop a .py file in a directory, get tools.

A plugin module exposes either `TOOLS = [ToolSpec, ...]` or `register(bus)`.
One broken plugin must not take the whole bus down, so import failures are
collected and reported per file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .spec import ToolSpec


def load_plugin_dir(bus, directory: str | Path, namespace: str = "plugin") -> dict:
    directory = Path(directory).expanduser()
    report = {"loaded": [], "errors": {}, "skipped": []}
    if not directory.is_dir():
        report["skipped"].append(f"{directory} does not exist")
        return report

    for file in sorted(directory.glob("*.py")):
        if file.name.startswith("_"):
            continue
        mod_name = f"agentd_plugin_{file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, file)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            report["errors"][file.name] = f"{type(exc).__name__}: {exc}"
            sys.modules.pop(mod_name, None)
            continue

        registered: list[str] = []
        if hasattr(module, "register") and callable(module.register):
            try:
                module.register(bus)
                registered.append("register()")
            except Exception as exc:
                report["errors"][file.name] = f"register() raised {type(exc).__name__}: {exc}"
                continue

        for tool in getattr(module, "TOOLS", []) or []:
            if not isinstance(tool, ToolSpec):
                report["errors"][file.name] = f"TOOLS entry {tool!r} is not a ToolSpec"
                continue
            named = tool
            if "." not in named.name:
                named = ToolSpec(**{**named.__dict__, "name": f"{namespace}.{file.stem}.{named.name}"})
            try:
                bus.register(named, replace=True)
                registered.append(named.name)
            except Exception as exc:
                report["errors"][file.name] = f"register failed: {exc}"

        if registered:
            report["loaded"].append({file.name: registered})

    return report
