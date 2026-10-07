"""Cron evaluation, scheduler bookkeeping and the DeepGEMM eligibility gate."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from agentd.backends.deepgemm import assess, probe_local
from agentd.scheduler import Cron, CronError, Job, ScheduleStore, Scheduler


# -- cron ------------------------------------------------------------------
def test_cron_aliases_and_fields():
    assert Cron("@daily").matches(datetime(2026, 10, 7, 0, 0))
    assert not Cron("@daily").matches(datetime(2026, 10, 7, 0, 1))
    assert Cron("*/15 9-17 * * 1-5").matches(datetime(2026, 10, 7, 9, 30))
    assert not Cron("*/15 9-17 * * 1-5").matches(datetime(2026, 10, 7, 9, 31))
    assert not Cron("*/15 9-17 * * 1-5").matches(datetime(2026, 10, 10, 9, 30))   # saturday
    assert Cron("30 4 1 * *").matches(datetime(2026, 11, 1, 4, 30))


def test_cron_or_semantics_when_both_day_fields_restricted():
    cron = Cron("0 0 13 * 5")          # the 13th OR any friday
    assert cron.matches(datetime(2026, 8, 13, 0, 0))
    assert cron.matches(datetime(2026, 8, 7, 0, 0))      # a friday, not the 13th
    assert not cron.matches(datetime(2026, 8, 8, 0, 0))   # saturday, not the 13th


def test_cron_rejects_junk():
    for bad in ("* * *", "61 * * * *", "*/0 * * * *", "a b c d e"):
        with pytest.raises(CronError):
            Cron(bad)


def test_next_after_walks_forward():
    cron = Cron("0 3 * * *")
    nxt = cron.next_after(datetime(2026, 10, 7, 4, 0))
    assert nxt == datetime(2026, 10, 8, 3, 0)


# -- scheduler -------------------------------------------------------------
class FakeRuntime:
    def __init__(self):
        self.published = []
        self.calls: list[dict] = []

    def create_session(self, task, provider="", learning=True, max_steps=12, council=False):
        self.calls.append({"task": task, "provider": provider, "learning": learning,
                           "max_steps": max_steps, "council": council})
        return type("S", (), {"id": "s1", "status": "success"})()

    def run_session(self, session):
        return {"success": True, "steps": [{}]}

    def publish(self, event):
        self.published.append(event)


@pytest.fixture()
def scheduler(tmp_path):
    store = ScheduleStore(tmp_path)
    runtime = FakeRuntime()
    return Scheduler(runtime, store), store, runtime


def test_jobs_roundtrip_through_disk(scheduler):
    sched, store, _ = scheduler
    sched.store.upsert(Job(name="nightly", cron="0 3 * * *", task="nginx-down", max_steps=5))
    loaded = store.load()
    assert len(loaded) == 1 and loaded[0].task == "nginx-down"
    assert store.path.exists() and json.loads(store.path.read_text())["jobs"][0]["name"] == "nightly"


def test_due_uses_last_run_stamps(scheduler):
    sched, store, _ = scheduler
    job = Job(name="q15", cron="*/15 * * * *", task="nginx-down")
    now = datetime(2026, 10, 7, 12, 30)
    assert [j.name for j in sched.due([job], now)] == ["q15"]
    job.last_run = now.timestamp()
    assert sched.due([job], now) == []
    assert [j.name for j in sched.due([job], now + timedelta(minutes=15))] == ["q15"]


def test_catchup_fires_once_not_every_missed_minute(scheduler):
    sched, _, _ = scheduler
    job = Job(name="hourly", cron="0 * * * *", task="nginx-down")
    missed = [datetime(2026, 10, 7, h, 0) for h in (5, 6, 7)]
    due = sched.due([job], missed[-1], lookback_minutes=180)
    assert due == [job]
    job.last_run = datetime(2026, 10, 7, 7, 0).timestamp()
    assert sched.due([job], missed[-1], lookback_minutes=180) == []


def test_tick_once_persists_stamps(scheduler):
    sched, store, runtime = scheduler
    store.upsert(Job(name="ops", cron="30 2 * * *", task="nginx-down", max_steps=4))
    now = datetime(2026, 10, 7, 2, 30)
    outputs = sched.tick_once(now)
    assert len(outputs) == 1 and outputs[0]["status"] == "success"
    assert runtime.calls == [{"task": "nginx-down", "provider": "", "learning": True,
                              "max_steps": 4, "council": False}]
    reloaded = store.load()[0]
    assert reloaded.last_run and reloaded.last_status == "success"
    assert sched.tick_once(now) == []                    # same minute: no double fire
    assert store.history.exists()


def test_disabled_jobs_never_fire(scheduler):
    sched, store, runtime = scheduler
    store.upsert(Job(name="off", cron="* * * * *", task="nginx-down", enabled=False))
    assert sched.tick_once(datetime(2026, 10, 7, 2, 30)) == []
    assert runtime.calls == []


def test_broken_cron_is_skipped_not_fatal(scheduler):
    sched, store, _ = scheduler
    store.save([Job(name="bad", cron="nonsense", task="nginx-down")])
    assert sched.tick_once(datetime(2026, 10, 7, 2, 30)) == []


def test_same_job_cannot_overlap(scheduler, monkeypatch):
    sched, store, runtime = scheduler
    store.upsert(Job(name="slow", cron="* * * * *", task="nginx-down"))
    sched._running.add("slow")
    output = sched.run_job(store.load()[0])
    assert output.get("skipped") == "already running"


# -- deepgemm eligibility --------------------------------------------------
def test_h100_is_eligible_and_gets_a_recipe():
    verdict = assess({"arch": "x86_64", "gpu": "NVIDIA H100 80GB HBM3", "compute_cap": "9.0",
                      "cuda": "Cuda compilation tools, release 12.9, V12.9.41", "torch": "2.6.0",
                      "gxx": "12.2.0"})
    assert verdict.eligible and verdict.blockers == []
    assert any("sglang" in line for line in verdict.plan)


def test_consumer_gpu_is_rejected_with_a_reason():
    verdict = assess({"arch": "x86_64", "gpu": "NVIDIA GeForce RTX 4060 Ti", "compute_cap": "8.9",
                      "cuda": "release 12.4", "torch": "2.4.0", "gxx": "11.4.0"})
    assert not verdict.eligible
    assert any("not SM90/SM100" in b for b in verdict.blockers)
    assert any("llama.cpp" in line for line in verdict.plan)


def test_no_gpu_at_all_is_rejected():
    verdict = assess({"arch": "x86_64", "gpu": "", "compute_cap": "", "cuda": "", "torch": ""})
    assert not verdict.eligible
    assert verdict.blockers[0].startswith("no NVIDIA GPU")


def test_arm_box_is_rejected_even_with_a_new_gpu():
    """DGX Spark is aarch64 + SM12x: neither axis qualifies for DeepGEMM."""
    verdict = assess({"arch": "aarch64", "gpu": "NVIDIA GB10", "compute_cap": "12.1",
                      "cuda": "release 13.0", "torch": "2.9.0", "gxx": "13.0.0"})
    assert not verdict.eligible
    assert any("SM90/SM100" in b for b in verdict.blockers)


def test_old_cuda_blocks_eligibility():
    verdict = assess({"arch": "x86_64", "gpu": "NVIDIA H20", "compute_cap": "9.0",
                      "cuda": "release 12.4", "torch": "2.3.0"})
    assert not verdict.eligible
    assert any("CUDA 12.4" in b for b in verdict.blockers)


def test_parse_reads_label_value_pairs():
    from agentd.backends.deepgemm import _parse

    facts = _parse("ARCH\nx86_64\nGPU\nNVIDIA H100\nCOMPUTE_CAP\n9.0\nCUDA\nrelease 12.9\n")
    assert facts["arch"] == "x86_64" and facts["compute_cap"] == "9.0" and facts["gpu"] == "NVIDIA H100"


def test_probe_local_on_this_machine_is_honest():
    """No NVIDIA GPU on this Intel Mac: say so instead of pretending to integrate."""
    verdict = probe_local(timeout=20)
    assert not verdict.eligible
    assert any("no NVIDIA GPU" in b or "compute capability" in b for b in verdict.blockers)
    assert any("llama.cpp" in line for line in verdict.plan)
