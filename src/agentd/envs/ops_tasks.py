"""Small deterministic ops tasks the agent can be measured on without a browser.

These exist for two reasons: the loop needs an environment whose action space is
enumerable (the advantage bias requires a closed set), and a contributor must be
able to run `agentd bench --learning on|off` on a laptop with no API key, no
Docker and no network, and still see the memory effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _State:
    text: str
    options: list[str]
    next: dict[str, str] = field(default_factory=dict)
    terminal: dict[str, float] = field(default_factory=dict)


@dataclass
class OpsTask:
    """A scripted chain: each state offers options, some of which end the run."""

    name: str
    goal_text: str
    scope: str
    chain: list[_State]
    steps_taken: int = 0
    _index: int = 0

    # -- Task protocol ----------------------------------------------------
    def reset(self) -> str:
        self._index = 0
        self.steps_taken = 0
        return self.chain[0].text

    def observe(self) -> str:
        return self.chain[self._index].text

    def goal(self) -> str:
        return self.goal_text

    def candidates(self, state: str) -> list[str]:
        for item in self.chain:
            if item.text == state:
                return list(item.options)
        return list(self.chain[self._index].options)

    def apply(self, action: str, ctx=None) -> tuple[str, bool, float]:
        current = self.chain[self._index]
        self.steps_taken += 1

        moved = self._move(current.next.get(action))
        if action in current.terminal:
            reward = float(current.terminal[action])
            return self.chain[self._index].text, True, reward
        if not moved:
            return current.text, False, 0.0            # no progress, stay put
        return self.chain[self._index].text, False, 0.0

    def _move(self, nxt: str | None) -> bool:
        if nxt is None:
            return False
        for i, item in enumerate(self.chain):
            if item.text == nxt:
                self._index = i
                return True
        return False


def nginx_down() -> OpsTask:
    return OpsTask(
        name="nginx-down",
        goal_text="Make nginx healthy again on app-01 without disrupting anything else.",
        scope="app-01",
        chain=[
            _State(
                text="ssh session on app-01; nginx is reported down by the healthcheck",
                options=["systemctl restart nginx", "reboot the whole host", "tail the nginx error log"],
                next={"systemctl restart nginx": "nginx restarted on app-01; healthcheck still pending",
                      "tail the nginx error log": "ssh session on app-01; nginx is reported down by the healthcheck"},
                terminal={"reboot the whole host": 0.0},
            ),
            _State(
                text="nginx restarted on app-01; healthcheck still pending",
                options=["curl the healthcheck endpoint", "run apt upgrade", "edit the nginx config file"],
                next={"curl the healthcheck endpoint": "healthcheck returns 200 on app-01"},
                terminal={"curl the healthcheck endpoint": 1.0, "run apt upgrade": 0.0,
                          "edit the nginx config file": 0.0},
            ),
        ],
    )


def approve_merge_request() -> OpsTask:
    return OpsTask(
        name="approve-mr",
        goal_text="Approve the ready merge request on gitlab and nothing else.",
        scope="gitlab",
        chain=[
            _State(
                text="gitlab merge request list; one MR is green and ready to approve",
                options=["click approve on the green merge request", "force push the source branch",
                         "open the pipeline tab"],
                next={"click approve on the green merge request": "merge request shows approved badge",
                      "open the pipeline tab": "gitlab merge request list; one MR is green and ready to approve"},
                terminal={"force push the source branch": 0.0},
            ),
            _State(
                text="merge request shows approved badge",
                options=["click merge", "revert the approval", "close the merge request"],
                next={"click merge": "merge request merged"},
                terminal={"click merge": 1.0, "revert the approval": 0.0,
                          "close the merge request": 0.0},
            ),
        ],
    )


def zzz_never_ends() -> OpsTask:
    """A task with no terminal action, to prove the step limit works."""
    return OpsTask(
        name="never-ends",
        goal_text="Loop forever unless the harness stops it.",
        scope="test",
        chain=[_State(text="a state that never resolves",
                      options=["look around", "look around again"],
                      next={"look around": "a state that never resolves",
                            "look around again": "a state that never resolves"})],
    )


BUILTIN_TASKS = {
    "nginx-down": nginx_down,
    "approve-mr": approve_merge_request,
    "never-ends": zzz_never_ends,
}


def suite() -> list[OpsTask]:
    return [nginx_down(), approve_merge_request()]
