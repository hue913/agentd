"""Local inference backends: detect what the hardware can actually run.

This exists because DeepGEMM gets asked about a lot. It is a CUDA kernel library
for **SM90 (Hopper) and SM100 (Blackwell datacentre)** only, needs CUDA 12.9+ and
PyTorch 2.3+, and it lives *inside* an inference engine (SGLang/vLLM) — it is not
something an agent process imports to get faster. On any other GPU the honest
answer is "not applicable, use a GGUF endpoint", and saying so is more useful
than a fake integration that silently no-ops.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

ELIGIBLE_ARCHES = {"9.0": "Hopper (SM90: H100/H200/H800)", "10.0": "Blackwell datacentre (SM100: B200/GB200)"}
MIN_CUDA = (12, 9)
MIN_TORCH = (2, 3)

# No `$(...)` here on purpose: command substitution is (correctly) refused by the
# read-only gate, so each fact is echoed as a label line followed by its value line.
PROBE_SCRIPT = """
echo ARCH
uname -m
echo GPU
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
echo COMPUTE_CAP
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1
echo CUDA
nvcc --version 2>/dev/null | tail -1
echo TORCH
command -v python3 >/dev/null 2>&1 && python3 -c "import torch;print(torch.__version__)" 2>/dev/null
true
""".strip()


@dataclass
class Verdict:
    eligible: bool
    blockers: list[str]
    facts: dict
    plan: list[str]

    def as_dict(self) -> dict:
        return {"eligible": self.eligible, "blockers": self.blockers, "facts": self.facts,
                "plan": self.plan, "backend": "deepgemm" if self.eligible else "llama.cpp/vllm-fallback"}


def _version(text: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", text or "")
    return tuple(int(n) for n in numbers[:3]) if numbers else ()


_CUDA_RELEASE = re.compile(r"release\s+(\d+)\.(\d+)", re.IGNORECASE)


def _cuda_version(text: str) -> tuple[int, ...]:
    """nvcc prints 'Cuda compilation tools, release 12.9, V12.9.41' — read the release."""
    match = _CUDA_RELEASE.search(text or "")
    if match:
        return int(match.group(1)), int(match.group(2))
    return ()


def assess(facts: dict) -> Verdict:
    blockers: list[str] = []
    arch = (facts.get("arch") or "").strip()
    cap = (facts.get("compute_cap") or "").strip()
    gpu = (facts.get("gpu") or "").strip()

    if not gpu:
        blockers.append("no NVIDIA GPU visible (nvidia-smi returned nothing)")
    elif cap not in ELIGIBLE_ARCHES:
        blockers.append(f"compute capability {cap or 'unknown'} is not SM90/SM100; "
                        f"DeepGEMM targets {', '.join(sorted(ELIGIBLE_ARCHES))}")

    cuda = _cuda_version(facts.get("cuda") or "")
    required_cuda = ".".join(str(n) for n in MIN_CUDA)
    if not cuda or cuda[:2] < MIN_CUDA:
        shown = ".".join(str(n) for n in cuda[:2]) if cuda else (facts.get("cuda") or "not found")
        blockers.append(f"CUDA {shown} < required {required_cuda}")

    torch_v = _version(facts.get("torch") or "")
    required_torch = ".".join(str(n) for n in MIN_TORCH)
    if not torch_v or torch_v[:2] < MIN_TORCH:
        shown = ".".join(str(n) for n in torch_v[:2]) if torch_v else (facts.get("torch") or "not found")
        blockers.append(f"PyTorch {shown} < required {required_torch}")

    if arch and arch != "x86_64":
        blockers.append(f"architecture {arch} is not supported by the prebuilt kernels")

    if blockers:
        return Verdict(False, blockers, facts, [
            "keep using an existing OpenAI-compatible endpoint, e.g. llama.cpp llama-server with a "
            "GGUF quant (works on CPU/ARM/consumer GPUs and needs none of the above)",
            "agentd needs no change: AGENTD_BASE_URL=http://127.0.0.1:8080/v1 AGENTD_MODEL=<name>",
        ])

    return Verdict(True, [], facts, [
        "uv pip install 'sglang[all]'  # SGLang calls DeepGEMM for FP8 MoE/linear groups",
        "export SGLANG_ENABLE_JIT_DEEPGEMM=1",
        "python -m sglang.launch_server --model-path <DeepSeek-class FP8 MoE> "
        "--quantization fp8 --mem-fraction-static 0.85 --host 127.0.0.1 --port 8080",
        "agentd probe --base-url http://127.0.0.1:8080/v1 --model <name>   # confirm the decode tier",
    ])


def probe_local(timeout: int = 30) -> Verdict:
    """Ask this machine what it has, using only binaries that exist on it.

    No shell script here: the same command must work on macOS, Linux and Windows,
    and every probe is optional.
    """
    import platform
    import shutil
    import sys

    facts: dict[str, str] = {"arch": platform.machine(), "gpu": "", "compute_cap": "",
                             "cuda": "", "torch": ""}

    def run(argv: list[str]) -> str:
        if not shutil.which(argv[0]):
            return ""
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                  stdin=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    if shutil.which("nvidia-smi"):
        names = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
        facts["gpu"] = names[0].strip() if names else ""
        caps = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).splitlines()
        facts["compute_cap"] = caps[0].strip() if caps else ""
    facts["cuda"] = run(["nvcc", "--version"])
    torch_out = run([sys.executable, "-c", "import torch;print(torch.__version__)"])
    facts["torch"] = torch_out
    return assess(facts)


def probe_remote(hub, host_label: str, timeout: int = 40) -> Verdict:
    from ..envs.ssh_env import _parse_kv

    result = hub.exec(host_label, PROBE_SCRIPT, timeout=timeout, approved=True)
    return assess(_parse_kv(result.stdout))


def _parse(text: str) -> dict:
    """Read the label-line / value-line probe output."""
    facts: dict[str, str] = {}
    lines = [line.strip() for line in (text or "").splitlines()]
    labels = {"ARCH", "GPU", "COMPUTE_CAP", "CUDA", "TORCH"}
    pending: str | None = None
    for line in lines:
        if line in labels:
            pending = line.lower()
            facts[pending] = ""
            continue
        if pending is not None:
            facts[pending] = line
            pending = None
    for key, value in list(facts.items()):
        if value in ("[N/A]", "not found", "N/A"):
            facts[key] = ""
    return facts
