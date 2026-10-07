"""Viewer stack: token lifetime, tunnel shape, and the startup script's syntax."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from agentd.viewer import (
    START_SCRIPT, TokenVault, install_plan, start_plan, tunnel_command, viewer_status, viewer_url,
)


def test_tokens_expire_and_are_checkable():
    vault = TokenVault(ttl=60)
    token = vault.issue("cli")
    assert vault.check(token.value) is True
    assert vault.check("garbage") is False
    assert vault.active() == 1
    assert token.as_dict()["expires_in_s"] > 0


def test_expired_token_is_rejected():
    vault = TokenVault(ttl=1)
    token = vault.issue("cli")
    vault._issued[token.value] = time.time() - 1
    assert vault.check(token.value) is False
    assert vault.active() == 0


def test_issuing_prunes_old_tokens():
    vault = TokenVault(ttl=5)
    for _ in range(5):
        vault.issue("cli")
    vault._issued["stale"] = time.time() - 100
    vault.issue("cli")
    assert "stale" not in vault._issued


def test_tunnel_command_carries_both_ports_and_quotes_the_target():
    cmd = tunnel_command("ops@app-01.example.com", port=22022)
    assert "-L 8765:127.0.0.1:8765" in cmd
    assert "-L 6080:127.0.0.1:6080" in cmd
    assert "-p 22022" in cmd
    assert cmd.startswith("ssh -N")


def test_viewer_url_only_adds_password_when_given():
    assert "password=" not in viewer_url()
    assert "autoconnect=true" in viewer_url()
    assert "password=abc123" in viewer_url(token="abc123")


def test_install_plan_rejects_unknown_distro():
    assert any("apt-get install" in line for line in install_plan("ubuntu"))
    assert any("unsupported distro" in line for line in install_plan("alpine"))


@pytest.mark.skipif(sys.platform == "win32", reason="sh -n is POSIX")
def test_start_script_is_valid_shell():
    proc = subprocess.run(["sh", "-n"], input=START_SCRIPT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="sh -n is POSIX")
def test_start_script_does_not_self_match_processes():
    """The old guards used `pgrep -f 'Xvfb :99'`, which matches the very command line
    carrying the pattern and reports a dead stack as running. Pidfiles do not."""
    assert "pgrep -f" not in START_SCRIPT
    assert "/run/agentd-viewer" in START_SCRIPT
    assert "kill -0" in START_SCRIPT


def test_start_script_binds_vnc_to_loopback():
    assert "-localhost" in START_SCRIPT
    assert "-nopw" in START_SCRIPT and "AGENTD_VNC_PASS" in START_SCRIPT


def test_start_plan_legacy_is_empty():
    assert start_plan() == []
