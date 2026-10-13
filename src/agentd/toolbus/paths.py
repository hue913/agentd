"""Path confinement for local file tools.

An ops agent has to read logs and configs, so this is not a full sandbox. It
is a stop on the shortest paths to stealing SSH keys or rewriting agentd's
own config/token. Anything that looks like a secret or a control-plane file
is refused before a byte is read or written.
"""

from __future__ import annotations

from pathlib import Path

DENIED_NAMES = {
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_xmss",
    "authorized_keys", "agentd.json", "api.token", "shadow", "gshadow",
    "sudoers",
}


class PathRefused(ValueError):
    pass


def resolve_allowed(path: str, *, write: bool = False) -> Path:
    """Return the expanded path, or raise PathRefused."""
    raw = (path or "").strip()
    if not raw:
        raise PathRefused("empty path")
    target = Path(raw).expanduser()
    try:
        resolved = target.resolve()
    except OSError as exc:
        raise PathRefused(f"cannot resolve {raw}: {exc}") from exc

    parts_l = {p.lower() for p in resolved.parts}
    if parts_l & {p.lower() for p in (".ssh", ".gnupg", ".aws")}:
        raise PathRefused(f"refused path under a secret directory: {resolved}")
    # ~/.config/agentd and /var/lib/agentd hold the token and the live config
    joined = str(resolved).replace("\\", "/").lower()
    if "/.config/agentd" in joined or "/var/lib/agentd" in joined:
        raise PathRefused(f"refused path inside the agentd data directory: {resolved}")

    name = resolved.name.lower()
    if name in {n.lower() for n in DENIED_NAMES} or name.endswith("_ed25519") or name.endswith("_rsa"):
        raise PathRefused(f"refused sensitive file: {resolved.name}")
    if write and name in {"authorized_keys", "known_hosts"}:
        raise PathRefused(f"refused write to {resolved.name}")
    return target
