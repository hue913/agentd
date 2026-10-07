# Deploying agentd on a real server

Worked example, measured on 2026-10-07 against a 2-core / 1.9 GB RAM / 17 GB free Ubuntu 22.04
VPS. Everything here is what actually happened, including the mistake.

## 1. One-command install

```bash
# from the repo root, as root on the target
bash deploy/install_server.sh /path/to/agentd
```

The script installs `uv` + CPython 3.11 (no system Python upgrade needed), creates
`/opt/agentd/.venv`, installs with `--only-binary :all:` (source builds of Rust-backed wheels
break more often than they work), writes `/var/lib/agentd/agentd.json` (`0600`), runs
`agentd doctor`, runs the keyless demo, and installs a systemd unit.

Verified on the target: `agentd doctor` → ready, **151 tests pass**, and
`agentd bench --episodes 40` reproduced the laptop's numbers **exactly**
(control 1/40, learning 11/40) — the measurement is seed-deterministic, not machine-dependent.

```
active  http://127.0.0.1:8765/healthz
{"ok":true,"kernel":{"episodes":0,"steps":0,"enabled":true,"beta":1.0,"gamma":0.95}}
```

## 2. What this size of machine can and cannot run

| | verdict |
|---|---|
| agentd API + kernel + SQLite + memory packs | comfortable; `MemoryMax=512M` in the unit |
| viewer stack (`xvfb x11vnc novnc websockify x11-utils`) | installs and runs; ~40 MB |
| a browser-driven agent (Playwright/Chromium) | **no** — one Chromium page eats more than the free RAM leaves |
| **WebArena** (7 self-hosted sites) | **no.** Officially recommended shape is 4 vCPU / 16 GB / 1000 GB + Docker. `agentd host probe` measures it in one command |
| MiniWoB++ | possible but tight; prefer running the eval on a workstation and pointing it at this server |

Check before you plan:

```bash
agentd host add --label vps --host 203.0.113.10 --user root --key ~/.ssh/deploy
agentd host probe --label vps
# → {"arch":"x86_64","ram_mb":1963,"disk_avail_gb":16.9,"docker":"Docker version 29.1.3",
#    "webarena_viable":false}
```

## 3. Screen viewing without exposing the screen

The stack is `Xvfb :99` → `x11vnc -localhost -rfbport 5900` → `websockify 127.0.0.1:6080`.
The client opens the tunnel; only port 22 is reachable from outside.

```bash
ssh -N -L 8765:127.0.0.1:8765 -L 6080:127.0.0.1:6080 root@your-server
# dashboard   http://127.0.0.1:8765/
# screen      http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale
```

**The mistake, so you do not repeat it.** The first version started websockify as
`websockify 6080 localhost:5900` — which binds `0.0.0.0`. Combined with `auth=none` (loopback was
assumed), the VNC proxy was reachable on the public interface for a few minutes; verified with
`ss -ltnp` and an HTTP request from outside. Fixed by binding `127.0.0.1:6080` explicitly, killing
the listener, and making the status check *fail loudly* instead of staying quiet:

```
novnc=listening-loopback
novnc=EXPOSED-NON-LOOPBACK(FIX: bind 127.0.0.1:6080)
```

A second bug of the same family: the old guards used `pgrep -f 'Xvfb :99'`, which matches the very
shell command carrying that pattern, so the script reported the stack as already running while
nothing had started. The startup logic now uses pid files under `/run/agentd-viewer/` and is
delivered as a script file rather than a one-liner (also because `$(...)` correctly fails the
read-only gate).

Verified after the fix:

```
127.0.0.1:5900  x11vnc      127.0.0.1:6080  websockify     127.0.0.1:8765  agentd
public reachability of 6080 / 8765: timeout (unreachable)
RFB handshake to 5900: greeting "RFB 003.008" + security types
DISPLAY=:99 xdpyinfo: dimensions 1440x900, depth 24
```

## 4. Hardening checklist for this box

Ordered by consequence. Items 1–3 were **not** done without the operator's say-so because they
change how other people reach the machine.

1. **MySQL is listening on `*:3306`.** Pre-existing, unrelated to agentd, and the single biggest
   hole on the box. Bind it to loopback or put it behind a security group.
2. `sshd` still accepts root password authentication. Install a deploy key, then
   `PasswordAuthentication no` + `PermitRootLogin prohibit-password`, and keep `fail2ban`.
3. Nothing else should be on `0.0.0.0:80/888` unless intended.
4. Keep `agentd serve --host 127.0.0.1`. Binding a non-loopback address is refused unless
   `AGENTD_ALLOW_PUBLIC=1`, and there is almost always a better answer (tunnel, WireGuard).
5. Run agentd as a dedicated user with SSH keys scoped to what it may touch.

## 5. Operate

```bash
systemctl status agentd
journalctl -u agentd -f
agentd schedule add --name nightly-ops --cron "17 3 * * *" --task nginx-down   # 巡检
agentd schedule daemon              # or add it to the systemd unit's ExecStart chain
agentd memory export --out ops.agentdmem && cat ops.md   # auditable by a human
sqlite3 /var/lib/agentd/memory.db 'select count(*) from steps;'
```

Back up `/var/lib/agentd/` — the memory database and audit log are the useful state.
