"""Memory packs: readable export, idempotent import, provenance warnings."""

from __future__ import annotations

import json

import pytest

from agentd.kernel import JitRLKernel, Store
from agentd.kernel.pack import (
    diff_packs, export_pack, import_pack, load_pack, save_pack, to_markdown,
)

STATE = "ssh session on app-01, nginx reported down by the healthcheck"


@pytest.fixture()
def trained_store():
    store = Store()
    # seeded: epsilon-exploration is genuinely stochastic and must not flake the suite
    kernel = JitRLKernel(store=store, gamma=0.5, seed=2)
    rec = kernel.begin("fix nginx", "make nginx healthy")
    rec.record(STATE, "systemctl restart nginx", z=0.3, reward=0.0, scope="app-01")
    rec.record(STATE, "curl the healthcheck endpoint", z=0.5, reward=1.0, scope="app-01")
    rec.finish(success=True, score=1.0, analysis="restart then verify, do not reboot the host")
    return store, kernel


def test_export_shape_and_checksum(trained_store):
    store, _ = trained_store
    pack = export_pack(store, source={"model": "qwen38-flash", "host": "app-01"})
    assert pack["format"] == "agentd-memory/1"
    assert pack["stats"] == {"episodes": 1, "steps": 2}
    assert pack["source"]["model"] == "qwen38-flash"
    assert len(pack["checksum"]) == 16


def test_markdown_is_human_readable(trained_store):
    store, _ = trained_store
    text = to_markdown(export_pack(store))
    assert "fix nginx — success" in text
    assert "restart then verify" in text
    assert "curl the healthcheck endpoint" in text
    assert "@app-01" in text
    assert "# agentd memory pack" in text
    # credit assignment must not read as causality
    assert "earned the terminal reward" in text
    assert "reward +0.00" in text


def test_roundtrip_through_disk(trained_store, tmp_path):
    store, _ = trained_store
    pack = export_pack(store)
    written = save_pack(pack, tmp_path / "team.agentdmem")
    assert json.loads(open(written["pack"], encoding="utf-8").read())["stats"]["steps"] == 2
    assert (tmp_path / "team.md").exists()

    loaded = load_pack(written["pack"])
    assert loaded["checksum_ok"] is True

    tampered = json.loads((tmp_path / "team.agentdmem").read_text(encoding="utf-8"))
    tampered["episodes"][0]["steps"][0]["ret"] = 99.0
    (tmp_path / "bad.agentdmem").write_text(json.dumps(tampered), encoding="utf-8")
    assert load_pack(tmp_path / "bad.agentdmem")["checksum_ok"] is False

    report = import_pack(Store(), load_pack(tmp_path / "bad.agentdmem"))
    assert any("checksum mismatch" in w for w in report["warnings"])


def test_import_rebuilds_an_equivalent_memory(trained_store):
    store, kernel = trained_store
    pack = export_pack(store, source={"model": "qwen38-flash"})

    fresh = Store()
    report = import_pack(fresh, pack)
    assert report["steps_added"] == 2 and report["episodes_added"] == 1

    before = kernel.decide(STATE, ["systemctl restart nginx", "reboot the whole host"],
                           z={"systemctl restart nginx": 0.3, "reboot the whole host": 0.7})
    after_kernel = JitRLKernel(store=fresh, gamma=0.5, seed=1)
    after = after_kernel.decide(STATE, ["systemctl restart nginx", "reboot the whole host"],
                                z={"systemctl restart nginx": 0.3, "reboot the whole host": 0.7})
    assert before.chosen_action == "systemctl restart nginx"
    assert after.chosen_action == before.chosen_action
    assert fresh.count_steps() == 2


def test_import_is_idempotent(trained_store):
    store, _ = trained_store
    pack = export_pack(store)
    target = Store()
    first = import_pack(target, pack)
    second = import_pack(target, pack)
    assert first["steps_added"] == 2
    assert second["steps_added"] == 0 or second["steps_skipped"] >= 2


def test_provenance_mismatch_is_warned(trained_store):
    store, _ = trained_store
    pack = export_pack(store, source={"model": "qwen38-flash", "host": "app-01"})
    report = import_pack(Store(), pack, current_source={"model": "gpt-5-mini", "host": "app-01"})
    assert any("may not transfer" in w for w in report["warnings"])
    assert not any("host=" in w for w in report["warnings"])


def test_skip_tasks_filter(trained_store):
    store, _ = trained_store
    pack = export_pack(store)
    report = import_pack(Store(), pack, skip_tasks=["fix nginx"])
    assert report["tasks_skipped"] == ["fix nginx"] and report["steps_added"] == 0


def test_diff_reports_task_level_differences(trained_store):
    store, kernel = trained_store
    pack_a = export_pack(store)

    other = Store()
    kernel2 = JitRLKernel(store=other, gamma=0.5)
    rec = kernel2.begin("rotate tokens", "rotate tokens")
    rec.record("gitlab profile page with a access tokens tab", "create project access token", reward=1.0)
    rec.finish(success=True, score=1.0)
    pack_b = export_pack(other)

    delta = diff_packs(pack_a, pack_b)
    assert delta["only_in_a"] == ["fix nginx"]
    assert delta["only_in_b"] == ["rotate tokens"]


def test_forged_checksum_flag_is_ignored(trained_store):
    """A client cannot claim its own pack is signed by setting checksum_ok=True."""
    store, _ = trained_store
    pack = export_pack(store)
    pack["checksum"] = "0" * 16
    pack["checksum_ok"] = True
    report = import_pack(Store(), pack)
    assert report["checksum_ok"] is False
    assert any("checksum mismatch" in w for w in report["warnings"])
