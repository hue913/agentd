#!/usr/bin/env bash
# Install agentd on a Linux server: uv + Python 3.11 + venv + the viewer stack.
# Idempotent: safe to re-run. Never binds a public port.
set -euo pipefail

INSTALL_DIR="${AGENTD_INSTALL_DIR:-/opt/agentd}"
DATA_DIR="${AGENTD_DATA_DIR:-/var/lib/agentd}"
SRC_DIR="${1:-.}"
WITH_VIEWER="${AGENTD_VIEWER:-1}"

say() { printf '\n== %s\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root (this installs packages and a systemd unit)" >&2
  exit 1
fi

say "packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends curl ca-certificates tar \
  python3 python3-venv python3-pip sqlite3 rsync openssh-client
if [ "$WITH_VIEWER" = "1" ]; then
  apt-get install -y --no-install-recommends xvfb x11vnc novnc websockify x11-utils || \
    echo "WARN: viewer packages failed; agentd core still installs"
fi

say "uv + python 3.11"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
uv --version

say "code -> $INSTALL_DIR"
mkdir -p "$INSTALL_DIR" "$DATA_DIR"
rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude 'tests/fixtures/*.log' \
  --exclude '.pytest_cache' --exclude 'data' \
  "$SRC_DIR/" "$INSTALL_DIR/"

say "venv + deps"
cd "$INSTALL_DIR"
uv venv --python 3.11 .venv
# --only-binary: cryptography's Rust wheel has broken source builds many times;
# refusing to compile is the safer failure.
uv pip install --python .venv/bin/python --only-binary :all: -e ".[api]" 2>/dev/null || \
  uv pip install --python .venv/bin/python --only-binary :all: -e .

say "config"
if [ ! -f "$DATA_DIR/agentd.json" ]; then
  cat > "$DATA_DIR/agentd.json" <<'JSON'
{
  "db": "/var/lib/agentd/memory.db",
  "kernel": { "beta": 1.0, "gamma": 0.95, "epsilon": 0.05, "top_k": 8 },
  "providers": {},
  "ssh_hosts": []
}
JSON
  chmod 600 "$DATA_DIR/agentd.json"
fi

say "self-check"
AGENTD_CONFIG="$DATA_DIR/agentd.json" .venv/bin/python -m agentd.cli doctor
AGENTD_CONFIG="$DATA_DIR/agentd.json" .venv/bin/python -m agentd.cli demo --episodes 20 | head -8

say "systemd unit"
cat > /etc/systemd/system/agentd.service <<UNIT
[Unit]
Description=agentd control API
After=network.target

[Service]
Type=simple
Environment=AGENTD_CONFIG=$DATA_DIR/agentd.json
Environment=PYTHONUNBUFFERED=1
ExecStart=$INSTALL_DIR/.venv/bin/python -m agentd.cli serve --host 127.0.0.1 --port 8765
Restart=always
RestartSec=3
MemoryMax=512M
NoNewPrivileges=true
ProtectSystem=full
ReadWritePaths=$DATA_DIR

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable agentd >/dev/null 2>&1 || true

echo
echo "done."
echo "  api        http://127.0.0.1:8765  (loopback only)"
echo "  from mac   ssh -N -L 8765:127.0.0.1:8765 -L 6080:127.0.0.1:6080 root@this-host"
echo "  then       http://127.0.0.1:8765/api/state"
