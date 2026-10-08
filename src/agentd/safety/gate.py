"""Destructive-command gate.

An agent that can SSH anywhere will eventually type `rm -rf`. The gate classifies
every outbound command before it leaves the process, so the default answer to an
irreversible action is "a human must confirm", not "the model seemed confident".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LEVEL_ALLOW = "allow"
LEVEL_CONFIRM = "confirm"
LEVEL_BLOCK = "block"

# (pattern, level, reason). Ordered: first match of the highest severity wins.
RULES: list[tuple[str, str, str]] = [
    (r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+", "block", "recursive/forced file deletion"),
    (r"\bmkfs(\.\w+)?\b", "block", "filesystem format"),
    (r"\bdd\b[^\n]*\bof=/dev/", "block", "raw write to a block device"),
    (r">(?:\s*)/dev/(?:sd|nvme|disk|hd)", "block", "redirect onto a disk device"),
    (r":\(\s*\)\s*\{\s*:\|", "block", "fork bomb"),
    (r"\b(?:shutdown|reboot|halt|poweroff)\b", "block", "host power state change"),
    (r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b", "block", "irreversible DDL"),
    (r"\bTRUNCATE\s+(?:TABLE\s+)?", "block", "table truncate"),
    (r"\bkubectl\s+delete\b", "block", "orchestrator resource deletion"),
    (r"\bdocker\s+(?:system\s+prune|\w+\s+rm\b)", "block", "container/image removal"),
    (r"\bgit\s+push\b[^\n]*(?:--force(?:-with-lease)?|\s-f\b)", "block", "force push overwrites shared history"),
    (r"\b(?:curl|wget)\b[^\n]*\|\s*(?:ba)?sh\b", "block", "pipe remote script straight into a shell"),
    (r"\bfind\b[^\n]*-delete\b", "block", "find -delete"),
    (r"\bfind\b[^\n]*-exec(?:dir)?\s+(?:rm|mv|chmod|dd)\b", "block", "find -exec with a mutating binary"),
    (r"\bbase64\s+(-d|--decode)\b[^\n]*\|\s*(?:ba)?sh\b", "block", "obfuscated payload piped to shell"),
    (r"\beval\b[^\n]*\$\(", "block", "eval of a constructed string"),
    # Interpreters and JS runtimes executing inline source can do literally
    # anything (os.system, rmtree, raw syscalls) — the command string itself
    # proves nothing, so a human reads the code before it runs.
    (r"\b(?:python|python\d(?:\.\d+)*|pypy\d*|perl|ruby|php)\b[^|;&\n]*\s-[cC]\b",
     "confirm", "interpreter runs inline code"),
    (r"\b(?:node|nodejs|deno|bun)\b[^|;&\n]*\s-(?:e|eval|p)\b",
     "confirm", "JS runtime evaluates inline code"),
    (r"\bawk\b[^|;&\n]*(?:\bsystem\s*\(|\bgetline\b)",
     "confirm", "awk spawns commands or reads through getline"),
    # partition editors rewrite the disk layout; only `-l` listing is harmless
    # and even that stays behind a confirm because the flag is one typo away
    (r"\b(?:fdisk|sfdisk|cfdisk|gdisk|parted|partprobe)\b",
     "confirm", "partition table tool"),
    (r"\bchmod\s+-[Rr]\s+0?777\b", "confirm", "world-writable permissions"),
    (r"\b(?:apt|apt-get|dnf|yum|pacman|brew)\s+(?:remove|purge|uninstall)\b", "confirm", "package removal"),
    (r"\bsystemctl\s+(?:stop|disable|mask|restart|kill)\b", "confirm", "service state change"),
    (r"\bkill(?:all)?\s+-9\b", "confirm", "forced process kill"),
    (r"\bpkill\b|\bkillall\b", "confirm", "pattern-based process kill"),
    (r"\biptables\b.*-(?:F|X|P)\b", "confirm", "firewall rule flush/policy change"),
    (r"\bDELETE\s+FROM\s+\w+\s*(?:;|$)", "confirm", "unqualified DELETE (no WHERE)"),
    (r"\b(?:crontab|systemctl\s+edit)\b", "confirm", "scheduled-task or unit modification"),
    (r"\bdd\b", "confirm", "raw copy"),
    (r"\bmv\s+.*\s+/(?:dev|proc|sys)\b", "confirm", "move into a virtual filesystem"),
    (r"\b(?:userdel|groupdel|passwd)\b", "confirm", "account change"),
    (r"\bnpm\s+(?:uninstall|publish)\b|\bpip\s+uninstall\b", "confirm", "dependency/publish side effect"),
]

# Interpreters (python*, node, awk, ...) and `env` are deliberately absent:
# each can execute arbitrary code or spawn arbitrary processes, so no command
# starting with one can be *proven* read-only from its first word. fdisk is
# likewise gone — its normal mode rewrites partition tables.
READ_ONLY_FIRST_WORDS = {
    "ls", "cat", "head", "tail", "less", "grep", "egrep", "find", "df", "du", "ps", "top",
    "free", "uptime", "who", "w", "id", "which", "whereis", "stat", "file", "wc", "sort", "uniq",
    "uname", "hostname", "date", "printenv", "ip", "ifconfig", "ss", "netstat", "dig", "nslookup",
    "ping", "traceroute", "curl", "wget", "systemctl", "service", "journalctl", "log", "docker",
    "git", "npm", "nproc", "lscpu", "lsblk", "sensors", "nvidia-smi",
    "sed", "cut", "tr", "tree", "test", "echo", "printf", "md5sum", "sha256sum", "zcat",
}

# Sub-verbs that turn an otherwise read-only binary into a writer.
WRITE_VERBS = {
    "systemctl": {"start", "stop", "restart", "reload", "enable", "disable", "mask", "kill", "edit", "daemon-reload"},
    "docker": {"run", "rm", "rmi", "stop", "kill", "prune", "exec", "build", "compose", "system"},
    "git": {"push", "commit", "reset", "clean", "checkout", "rebase", "merge", "branch", "tag", "stash", "gc"},
    "journalctl": {"--rotate", "--flush"},
    "find": {"-delete", "-exec", "-execdir", "-ok"},
    "sed": {"-i", "--in-place"},
    "curl": {"-o", "--output", "-O", "--remote-name"},
    "npm": {"install", "i", "uninstall", "publish", "ci", "run", "exec", "test", "init"},
}

_COMPILED = [(re.compile(p, re.IGNORECASE), level, why) for p, level, why in RULES]


@dataclass
class Verdict:
    level: str = LEVEL_ALLOW
    reasons: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)

    @property
    def needs_approval(self) -> bool:
        return self.level in (LEVEL_CONFIRM, LEVEL_BLOCK)

    @property
    def hard_blocked(self) -> bool:
        return self.level == LEVEL_BLOCK

    def as_dict(self) -> dict:
        return {"level": self.level, "reasons": self.reasons, "matched": self.matched}


def classify(command: str, *, extra_block: list[str] | None = None,
             extra_confirm: list[str] | None = None) -> Verdict:
    text = command or ""
    verdict = Verdict()

    for pattern, level, reason in _COMPILED:
        match = pattern.search(text)
        if match:
            verdict.matched.append(match.group(0).strip()[:80])
            verdict.reasons.append(reason)
            if level == "block":
                verdict.level = LEVEL_BLOCK
            elif level == "confirm" and verdict.level != LEVEL_BLOCK:
                verdict.level = LEVEL_CONFIRM

    for patterns, level in ((extra_block, "block"), (extra_confirm, "confirm")):
        for custom in patterns or []:
            if re.search(custom, text, re.IGNORECASE):
                verdict.matched.append(custom)
                verdict.reasons.append(f"custom rule ({level})")
                if level == "block":
                    verdict.level = LEVEL_BLOCK
                elif verdict.level != LEVEL_BLOCK:
                    verdict.level = LEVEL_CONFIRM

    return verdict


_SEGMENT_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")


def _segment_readonly(segment: str) -> bool:
    parts = segment.split()
    if not parts:
        return True
    binary = parts[0].rsplit("/", 1)[-1].lower()
    if binary not in READ_ONLY_FIRST_WORDS:
        return False
    rest = " ".join(parts[1:]).lower()
    for verb in WRITE_VERBS.get(binary, set()):
        if re.search(rf"(?:^|[\s='\"-]){re.escape(verb.lower().strip('-'))}(?:$|[\s='\"])", rest):
            return False
        if verb.startswith("-") and verb in rest:
            return False
    return True


def initial_readonly(command: str) -> bool:
    """True only when every statement and pipeline stage is provably read-only.

    Compound commands are checked segment by segment, so `systemctl status nginx
    && rm -rf /` is not waved through just because it starts with an allowed verb.
    """
    text = (command or "").strip()
    if not text:
        return True
    if "`" in text or "$(" in text or "(" in text or "&" in text.replace("&&", ""):
        return False
    stripped = text.replace("2>/dev/null", "").replace("2>&1", "")
    if ">" in stripped or "<" in stripped:
        return False
    for segment in _SEGMENT_SPLIT.split(stripped):
        segment = segment.strip()
        if segment and not _segment_readonly(segment):
            return False
    return True
