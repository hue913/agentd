"""Council: three-phase multi-model deliberation, fused with the JitRL kernel.

Borrowed from the round-table pattern (independent drafting -> critique under
permission boundaries -> adjudication, with disagreements *kept* instead of
averaged away), and made useful to an agent that acts rather than writes:

* every phase costs tokens, so deliberation is triggered by conditions, not by
  default: a thin margin between the top two candidate scores, a dangerous
  action, or a task the operator pinned;
* the adjudication weight of each member comes from the memory store, so the
  council learns *which model wins in which kind of state* instead of trusting
  the same vendor forever;
* dissent is written into a risk table, never folded into a fake consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .kernel.jitrl import Decision, JitRLKernel
from .log import get_logger
from .providers.base import Choice, Provider, Usage

log = get_logger("agentd.council")


@dataclass
class Proposal:
    member: str
    action: str
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: str | None = None
    mode: str = "token"


@dataclass
class Objection:
    member: str
    target_member: str
    target_action: str
    stance: str            # support | object | unsure
    note: str = ""
    severity: str = "medium"


@dataclass
class CouncilOutcome:
    used: bool
    reason: str
    decision: Decision | None = None
    proposals: list[Proposal] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    consensus: bool = True
    scope: str = ""
    task: str = ""

    def summary(self) -> dict:
        return {
            "used": self.used, "reason": self.reason, "consensus": self.consensus,
            "chosen": self.decision.chosen_action if self.decision else None,
            "weights": self.weights,
            "proposals": {p.member: p.action for p in self.proposals},
            "objections": [f"{o.member}->{o.target_member}:{o.stance}" for o in self.objections],
            "risks": self.risks,
            "tokens": self.usage.total_tokens,
        }


@dataclass
class TriggerPolicy:
    """When is deliberation worth the tokens?"""

    margin_threshold: float = 0.12      # probability gap between 1st and 2nd choice
    deliberate_on_risk: tuple[str, ...] = ("dangerous",)
    pin_tasks: tuple[str, ...] = ()     # substring match on task name
    max_members: int = 3

    def evaluate(self, primary: Choice, risk: str = "", task: str = "") -> tuple[bool, str]:
        for needle in self.pin_tasks:
            if needle and needle in (task or ""):
                return True, f"task pinned ({needle})"
        if risk in self.deliberate_on_risk:
            return True, f"action risk={risk}"
        top = sorted((v for v in primary.z.values()), reverse=True)
        if len(top) >= 2:
            margin = top[0] - top[1]
            if margin < self.margin_threshold:
                return True, f"margin {margin:.3f} < {self.margin_threshold}"
        return False, "confident single model"


class Council:
    def __init__(self, kernel: JitRLKernel, members: list[Provider],
                 trigger: TriggerPolicy | None = None, critique_enabled: bool = True):
        self.kernel = kernel
        self.members = members
        self.trigger = trigger or TriggerPolicy()
        self.critique_enabled = critique_enabled

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _probabilities(z: dict[str, float]) -> dict[str, float]:
        finite = {k: (0.0 if v == float("-inf") else max(float(v), 0.0)) for k, v in z.items()}
        total = sum(finite.values())
        if total <= 0:
            n = max(len(finite), 1)
            return {k: 1.0 / n for k in finite}
        return {k: v / total for k, v in finite.items()}

    # -- phase 1: independent drafting ------------------------------------
    def _draft(self, system: str, user: str, candidates: list[str]) -> tuple[list[Proposal], Usage]:
        proposals: list[Proposal] = []
        usage = Usage(calls=0)
        for member in self.members[: max(self.trigger.max_members, 1)]:
            try:
                choice = member.choose(system, user, candidates)
            except Exception as exc:
                # A member without a key, or with a dead endpoint, abstains --
                # it must not take the whole deliberation down. "Not configured
                # yet" is the normal state during rollout, and the abstention is
                # recorded so the outcome is visibly thinner, not quietly wrong.
                proposals.append(Proposal(
                    member=member.label, action="", scores={},
                    reasoning=f"unavailable: {type(exc).__name__}: {exc}"[:200],
                    mode="unavailable"))
                continue
            usage = usage + choice.usage
            probs = self._probabilities(choice.z)
            top = max(probs, key=probs.get) if probs else candidates[0]
            proposals.append(Proposal(member=member.label, action=top, scores=probs,
                                      reasoning=choice.reasoning, mode=choice.mode.value))
        return proposals, usage

    # -- phase 2: critique under permission boundaries --------------------
    def _critique(self, state: str, proposals: list[Proposal]) -> tuple[list[Objection], Usage]:
        if not self.critique_enabled or len({p.action for p in proposals}) <= 1:
            return [], Usage(calls=0)
        total = Usage(calls=0)
        objections: list[Objection] = []
        listing = "\n".join(f"- {p.member}: {p.action}" for p in proposals)
        prompt = (
            f"State:\n{state}\n\nThese are independent proposals:\n{listing}\n\n"
            "You may only flag problems with someone else's proposal; you may not rewrite the plan. "
            'Reply with JSON: {"objections": [{"target_member": "...", "stance": '
            '"support"|"object"|"unsure", "note": "...", "severity": "low|medium|high"}]}'
        )
        for member in self.members:
            if not hasattr(member, "text"):
                continue
            try:
                text, used = member.text(prompt)
            except Exception as exc:
                # An abstaining member thins the deliberation but must not end
                # it; the abstention is still observable in the logs.
                log.debug("critique from %s failed: %s: %s",
                          member.label, type(exc).__name__, exc)
                continue
            total = total + used
            for item in _objection_rows(text):
                if item["target_member"] == member.label:
                    continue
                objections.append(Objection(
                    member=member.label, target_member=item["target_member"],
                    target_action=_action_of(proposals, item["target_member"]),
                    stance=item["stance"], note=str(item.get("note", ""))[:400],
                    severity=str(item.get("severity", "medium")),
                ))
        return objections, total

    # -- phase 3: adjudicate, keeping dissent ------------------------------
    def decide(self, state: str, candidates: list[str], scope: str = "", task: str = "",
               risk: str = "", system: str = "", user: str | None = None,
               episode_id: int | None = None) -> CouncilOutcome:
        primary = self.members[0]
        first = primary.choose(system or state, user or state, candidates)
        decision = self.kernel.decide(state, candidates, first.z, mode=first.mode.value, scope=scope)

        should, reason = self.trigger.evaluate(first, risk=risk, task=task)
        if not should:
            return CouncilOutcome(used=False, reason=reason, decision=decision,
                                  usage=first.usage, weights={primary.label: 1.0},
                                  scope=scope, task=task)

        proposals, usage = self._draft(system or state, user or state, candidates)
        usable = [p for p in proposals if p.scores]
        objections, usage2 = self._critique(state, usable)
        usage = usage + usage2

        weights = self.kernel.store.member_reliability(scope)
        for member in self.members:
            weights.setdefault(member.label, 0.5)      # unknown members start neutral
        weight_total = sum(max(weights.get(p.member, 0.5), 1e-6) for p in usable) or 1.0

        merged: dict[str, float] = {}
        for action in candidates:
            merged[action] = sum(weights.get(p.member, 0.5) / weight_total * p.scores.get(action, 0.0)
                                 for p in usable)
        penalised = _apply_objections(merged, objections, usable, weights, weight_total)

        adjudicated = self.kernel.decide(state, candidates, penalised, mode="council", scope=scope)
        distinct = {p.action for p in usable}
        risks = [
            {"member": o.member, "against": o.target_member, "action": o.target_action,
             "stance": o.stance, "note": o.note, "severity": o.severity}
            for o in objections if o.stance == "object"
        ]
        for p in proposals:
            if not p.scores:
                risks.append({"member": p.member, "against": "", "action": "",
                              "stance": "unavailable", "note": p.reasoning or "no scores returned",
                              "severity": "low"})
        if len(distinct) > 1:
            risks.append({"member": "council", "against": "", "action": "", "stance": "split",
                          "note": f"members disagreed: {sorted(distinct)}", "severity": "medium"})
            adjudicated.rerank.reasoning = ((adjudicated.rerank.reasoning or "") +
                                            f" | split vote among {sorted(distinct)}").strip(" |")

        for row in risks:
            self.kernel.store.add_risk(episode_id, row["member"], row["note"],
                                       row["severity"], detail=task or scope)

        return CouncilOutcome(
            used=True, reason=reason, decision=adjudicated, proposals=proposals,
            objections=objections, risks=risks, weights=weights, usage=usage,
            consensus=len(distinct) <= 1, scope=scope, task=task,
        )

    def credit(self, outcome: CouncilOutcome, success: bool) -> None:
        """Learn which member is right in this kind of state.

        Only proposals that were actually executed earn credit or blame: a member
        whose advice was ignored has no evidence either way, and recording it as a
        loss would teach the router to distrust whoever dissented.
        """
        if not outcome.used or not outcome.decision:
            return
        chosen = outcome.decision.chosen_action
        for proposal in outcome.proposals:
            if proposal.action != chosen:
                continue
            self.kernel.store.record_member_outcome(
                proposal.member, scope=outcome.scope, won=success, ret=1.0 if success else 0.0,
            )


def _action_of(proposals: list[Proposal], member: str) -> str:
    return next((p.action for p in proposals if p.member == member), "")


def _apply_objections(merged: dict[str, float], objections: list[Objection],
                      proposals: list[Proposal], weights: dict[str, float],
                      weight_total: float) -> dict[str, float]:
    """An objection lowers the score of the action it targets, and raises the
    credibility-weighted share of the alternatives. Nothing is deleted: the model
    can still choose the flagged action if the memory says it is the right call."""
    out = dict(merged)
    by_member_weight = {p.member: weights.get(p.member, 0.5) / weight_total for p in proposals}
    penalties: dict[str, float] = {}
    for objection in objections:
        if objection.stance != "object":
            continue
        severity = {"low": 0.05, "medium": 0.15, "high": 0.35}.get(objection.severity, 0.15)
        penalties[objection.target_action] = penalties.get(objection.target_action, 0.0) + \
            severity * by_member_weight.get(objection.member, 1.0 / max(len(proposals), 1))
    for action, penalty in penalties.items():
        out[action] = max(out.get(action, 0.0) - penalty, 0.0)
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


def _objection_rows(text: str) -> list[dict]:
    import json
    import re

    from .providers.jsonutil import balanced_objects

    for obj in balanced_objects(text or ""):
        rows = obj.get("objections")
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict) and r.get("target_member") and r.get("stance")]
    match = re.search(r"\[\s*\{.*\}\s*\]", text or "", re.DOTALL)
    if match:
        try:
            rows = json.loads(match.group(0))
        except ValueError:
            return []
        return [r for r in rows if isinstance(r, dict) and r.get("target_member")]
    return []
