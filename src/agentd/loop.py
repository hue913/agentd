"""Agent loop: the spine that ties provider, context, kernel, tools and ledger.

One episode looks like this:

    observe -> enumerate candidates -> model scores (z) -> kernel retrieves
    history and biases (z' = z + beta*A) -> argmax -> execute via the tool bus
    -> record (state, action, return) -> reward at the end

`candidates` are either enumerated by the environment or proposed by the model
first, because the advantage bias needs a closed set to rank.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from .context import ContextBuilder, Ledger
from .council import Council
from .kernel.jitrl import JitRLKernel
from .providers.base import Provider, ProviderError
from .toolbus.bus import CallContext, ToolBus

PROPOSE_INSTRUCTION = (
    "List the {k} most plausible next actions for this state. "
    'Output JSON only: {{"options": ["...", "..."]}}'
)


class Task(Protocol):
    """What an environment owes the loop."""

    name: str
    scope: str

    def reset(self) -> str: ...
    def observe(self) -> str: ...
    def candidates(self, state: str) -> list[str]: ...
    def apply(self, action: str, ctx: CallContext) -> tuple[str, bool, float]: ...
    def goal(self) -> str: ...


@dataclass
class StepTrace:
    t: int
    state: str
    candidates: list[str]
    chosen_index: int
    chosen_action: str
    options: list[dict]
    retrieved: int
    baseline: float
    explored: list[str]
    mode: str
    risk: str
    tool_output: str = ""
    ok: bool = True
    error: str | None = None
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return self.__dict__ | {}


@dataclass
class EpisodeReport:
    task: str
    scope: str
    episode_id: int
    steps: list[StepTrace] = field(default_factory=list)
    success: bool | None = None
    score: float = 0.0
    analysis: str = ""
    stopped_by: str = ""
    tokens: dict = field(default_factory=dict)
    final: str = ""            # the tool task's own answer, when it finished
    council: list = field(default_factory=list)   # one summary per deliberated step
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "task": self.task, "scope": self.scope, "episode_id": self.episode_id,
            "success": self.success, "score": self.score, "analysis": self.analysis,
            "stopped_by": self.stopped_by, "steps": len(self.steps), "tokens": self.tokens,
            "final": self.final, "council": self.council,
            "duration_s": round(time.time() - self.started_at, 2),
            "trace": [s.as_dict() for s in self.steps],
        }


@dataclass
class LoopConfig:
    max_steps: int = 12
    propose_k: int = 4
    risk_hint: str = ""
    deliberate: bool = True
    reflect: bool = True
    # Reward shaping for sparse-reward tasks. WebArena gives 0 on failure, which
    # makes every action in a failed episode look equally worthless (return 0)
    # and the advantage term can only learn from lucky successes. -1 lets the
    # agent learn "do not do that" from a failure alone. Use 0.0 to reproduce the
    # paper's exact reward setting.
    failure_penalty: float = -1.0
    # Time is never free. With step_cost=0 an action that merely avoids terminating
    # scores return 0, which beats "tried something and failed later" — the agent
    # discovers stalling and stays there. See tests/test_loop.py::test_stalling.
    step_cost: float = 0.03


@dataclass
class StepOutcome:
    trace: StepTrace
    decision: object
    observation: object
    done: bool
    reward: float
    council_outcome: object | None = None


class AgentLoop:
    def __init__(self, kernel: JitRLKernel, provider: Provider, bus: ToolBus,
                 context: ContextBuilder | None = None, ledger: Ledger | None = None,
                 council: Council | None = None, config: LoopConfig | None = None,
                 approver=None, catalog_token_budget: int = 900, ssh=None):
        self.kernel = kernel
        self.provider = provider
        self.bus = bus
        self.context = context or ContextBuilder()
        self.ledger = ledger or Ledger()
        self.council = council
        self.config = config or LoopConfig()
        self.approver = approver
        self.catalog_budget = catalog_token_budget
        # The SSH hub rides on the call context so tool handlers (ssh.exec,
        # ssh.ls, ...) behave identically whether a call arrives from the loop
        # or from the HTTP API.
        self.ssh = ssh

    # -- candidate sourcing ----------------------------------------------
    def enumerate_candidates(self, state: str, task: Task) -> tuple[list[str], str]:
        provided = task.candidates(state) if hasattr(task, "candidates") else []
        if provided:
            # An enumerable environment already bounded its own option set; truncating
            # it here would silently delete the action the task requires.
            return list(provided), "environment"
        # Fixed blocks go FIRST, mirroring ContextBuilder.build's ordering: the
        # catalog and mission are step-invariant, so leading with them lets a
        # propose call share its longest prefix with the choose call of the same
        # step and with previous steps' propose calls (provider prefix caches
        # key on exactly that).
        head = []
        catalog = self.bus.catalog(token_budget=self.catalog_budget)
        if catalog:
            head.append(f"# tools\n{catalog}")
        head.append(f"# mission\nTask: {task.goal()}")
        prompt = "\n\n".join(head)
        prompt += f"\n\n# state\n{state}\n\n" + PROPOSE_INSTRUCTION.format(k=self.config.propose_k)
        hint = getattr(task, "propose_hint", "")
        if hint:
            # A tool-driven task proposes real calls: the model needs the accepted
            # action grammar in the same breath as the ask (the tool names are
            # already in the leading catalog).
            prompt += "\n\n" + hint
        if hasattr(self.provider, "text"):
            try:
                text, usage = self.provider.text(prompt)
                self.ledger.record(usage, {"phase": "propose"})
                options = _parse_options(text)
                if options:
                    return options, "model"
            except Exception:
                pass
        return [], "none"

    # -- one step ---------------------------------------------------------
    def step(self, task: Task, state: str, history: list[str], recall: list[str],
             ctx: CallContext) -> StepOutcome:
        candidates, source = self.enumerate_candidates(state, task)
        if not candidates:
            raise ProviderError("no candidate actions available: the task enumerated none and "
                                "the model could not propose any")

        plan = self.context.build(
            instructions=f"Task: {task.goal()}\nPick one of the {len(candidates)} numbered options "
                         "and answer with only its number.",
            catalog=self.bus.catalog(token_budget=self.catalog_budget),
            recall=recall, history=history, state=state, scope=task.scope,
        )
        user = plan.user + "\n\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, 1))
        risk = self._risk_hint(candidates)

        t0 = time.time()
        council_outcome = None
        if self.council is not None and self.config.deliberate:
            council_outcome = self.council.decide(state, candidates, scope=task.scope, task=task.name,
                                                 risk=risk, system=plan.system, user=user,
                                                 episode_id=ctx.episode_id)
            decision = council_outcome.decision
            self.ledger.record(council_outcome.usage,
                               {"phase": "council" if council_outcome.used else "primary"})
        else:
            choice = self.provider.choose(plan.system, user, candidates)
            decision = self.kernel.decide(state, candidates, choice.z, mode=choice.mode.value,
                                          scope=task.scope)
            self.ledger.record(choice.usage, {"phase": "choose"})

        chosen = decision.chosen_action
        raw = task.apply(chosen, ctx)
        observation, done, reward = _unpack_apply(raw)

        trace = StepTrace(
            t=len(history), state=_short(state), candidates=candidates,
            chosen_index=decision.rerank.chosen_index, chosen_action=chosen,
            options=decision.rerank.trace_rows() + [{"index": -1, "note": f"candidates: {source}"}],
            retrieved=len(decision.neighbors), baseline=round(decision.advantage.baseline, 4),
            explored=list(decision.advantage.explored),
            mode=decision.rerank.mode + ("+council" if council_outcome and council_outcome.used else ""),
            risk=risk, tool_output=_short(observation),
            ok=bool(getattr(raw, "ok", True)), error=getattr(raw, "error", None),
            duration_ms=int((time.time() - t0) * 1000),
        )
        return StepOutcome(trace=trace, decision=decision, observation=observation,
                           done=done, reward=reward, council_outcome=council_outcome)

    def _risk_hint(self, candidates: list[str]) -> str:
        from .safety.gate import classify

        level = ""
        for candidate in candidates:
            verdict = classify(candidate)
            if verdict.level == "block":
                return "dangerous"
            if verdict.level == "confirm":
                level = "write"
        return level

    # -- one episode ------------------------------------------------------
    def run(self, task: Task) -> EpisodeReport:
        state = task.reset()
        recorder = self.kernel.begin(task.name, task.goal())
        ctx = CallContext(session=task.name, episode_id=recorder.episode_id,
                          approver=self.approver, kernel=self.kernel, ssh=self.ssh,
                          extra={"bus": self.bus})
        report = EpisodeReport(task=task.name, scope=task.scope, episode_id=recorder.episode_id)
        history: list[str] = []
        council_outcomes: list = []
        # Frozen once per episode: recall sits at the very end of the user block,
        # so re-querying it every step would rewrite the prompt tail each step and
        # defeat the prefix cache the context ordering is built for.
        recall = self.kernel.analyses_for(task.name)

        for _ in range(self.config.max_steps):
            try:
                outcome = self.step(task, state, history, recall, ctx)
            except ProviderError as exc:
                report.stopped_by = f"provider: {exc}"
                break

            decision = outcome.decision
            if outcome.done:
                recorded_reward = outcome.reward if outcome.reward > 0 else self.config.failure_penalty
            else:
                recorded_reward = -self.config.step_cost
            recorder.record(outcome.trace.state, decision.chosen_action,
                            z=decision.rerank.chosen.z, adv=decision.rerank.chosen.advantage,
                            z_prime=decision.rerank.chosen.z_prime,
                            chosen_index=outcome.trace.chosen_index, reward=recorded_reward,
                            scope=task.scope,
                            meta={"mode": outcome.trace.mode, "retrieved": outcome.trace.retrieved})
            report.steps.append(outcome.trace)
            if outcome.council_outcome is not None:
                council_outcomes.append(outcome.council_outcome)
                if getattr(outcome.council_outcome, "used", False):
                    summary = outcome.council_outcome.summary()
                    summary["step"] = outcome.trace.t
                    report.council.append(summary)
            history.append(f"chose {decision.chosen_action} -> {outcome.trace.tool_output[:120]}")

            state = outcome.observation if isinstance(outcome.observation, str) else outcome.trace.state
            if outcome.done:
                report.success, report.score = outcome.reward > 0, float(outcome.reward)
                report.final = str(outcome.observation)[:2000]
                break
        else:
            report.stopped_by = f"step limit ({self.config.max_steps})"
            report.success, report.score = False, 0.0
            if recorder.steps:
                # An unfinished episode is a failure, not a free pass: otherwise
                # "never terminate" carries return 0 and beats "act and lose".
                recorder.steps[-1].reward = self.config.failure_penalty

        if report.success is None:
            report.success, report.score = False, 0.0
        if self.config.reflect:
            report.analysis = self._reflect(task, report)
        recorder.finish(success=report.success, score=report.score, analysis=report.analysis)

        # Council credit is settled once, against the real episode outcome.
        for council_outcome in council_outcomes:
            self.council.credit(council_outcome, success=report.success)

        report.tokens = self.ledger.summary()
        return report

    def _reflect(self, task: Task, report: EpisodeReport) -> str:
        """Reflexion-style verbal self-critique, stored with the episode and recalled later."""
        if not hasattr(self.provider, "text") or not report.steps:
            return ""
        transcript = "\n".join(f"{s.t+1}. {s.chosen_action} -> {s.tool_output[:100]}" for s in report.steps)
        prompt = (
            f"Task: {task.goal()}\nOutcome: {'success' if report.success else 'failure'}\n"
            f"Transcript:\n{transcript}\n\n"
            "In two sentences max, state what to do differently next time. "
            "Be concrete; name the action, not the feeling."
        )
        try:
            text, usage = self.provider.text(prompt)
            self.ledger.record(usage, {"phase": "reflect"})
        except Exception as exc:
            return f"reflection failed: {type(exc).__name__}"
        return (text or "").strip()[:600]


class _Noop:
    used = False


def _unpack_apply(result) -> tuple[object, bool, float]:
    if isinstance(result, tuple) and len(result) == 3:
        return result
    ok = bool(getattr(result, "ok", True))
    state = getattr(result, "output", None) or str(result)
    done = bool(getattr(result, "done", False))
    reward = float(getattr(result, "reward", 0.0) or 0.0)
    return state, done, reward if ok else 0.0


def _short(text: object) -> str:
    out = str(text or "")
    return out if len(out) <= 600 else out[:300] + " … " + out[-200:]


def _parse_options(text: str) -> list[str]:
    import json

    from .providers.openai_compat import _balanced_objects

    for obj in _balanced_objects(text or ""):
        rows = obj.get("options")
        if isinstance(rows, list):
            return [str(r).strip() for r in rows if str(r).strip()][:8]
    return [line.split(".", 1)[-1].strip() for line in (text or "").splitlines()
            if line.strip() and line.strip()[0].isdigit()][:8]
