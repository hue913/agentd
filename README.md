# agentd

**English** · [中文](README.zh-CN.md)

**An agent that gets better without being trained.**

It stores every action it takes together with the return that followed, retrieves the
situations it has already lived through, estimates the advantage of each option, and biasing
the model's own next-token probabilities: `z'(s,a) = z(s,a) + β·Â(s,a)`. No gradients, no
fine-tuning, no GPU rental, no API key required to see the mechanism work.

Then you point it at your servers: a real interactive SSH terminal, a live view of the
machine's screen, multi-model councils that argue before they decide, and a safety gate
that will not let `rm -rf` through without a human.

It runs on a $5 VPS. It runs keyless. It runs today.

[![ci](https://github.com/hue913/agentd/actions/workflows/ci.yml/badge.svg)](https://github.com/hue913/agentd/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-brightgreen)
![gpu](https://img.shields.io/badge/GPU-not%20required-success)

![The console: a council run with the seats picker](docs/images/console-council.png)

> Desktop client (Tauri, macOS Intel + Apple Silicon, Windows) and the same console served
> by the agent itself: **[hue913/council-agent](https://github.com/hue913/council-agent)**.

---

## 60 seconds, no API key, no GPU, no network

```bash
git clone https://github.com/hue913/agentd && cd agentd
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[api]"

agentd demo                     # watch memory flip decisions, keyless
agentd bench --episodes 40      # the two curves below
```

```
learning OFF (control)   1/40 (  2.5%)    1st half  5.0%   2nd half  0.0%
learning ON  (JitRL)    11/40 ( 27.5%)    1st half  0.0%   2nd half 55.0%
```

Not a lucky seed. Across **12 seeds × 40 episodes** on the same suite: control averages
**1.1/40**, learning averages **14.4/40**, learning is strictly better in **11 of 12**
seeds — the twelfth is a tie where neither arm ever succeeds. Seed-to-seed spread is wide
(0 to 30), which is why the command prints both halves of the curve: the mechanism shows up
as a trend, not as a single number.

```bash
for s in $(seq 1 12); do agentd bench --episodes 40 --seed $s | sed -n 3,4p; done
```

### And on real web pages

`--suite web` drives actual Chromium through Playwright over six MiniWoB-shaped tasks
(numbered elements, one goal, a DOM checker deciding reward — the page decides, not a model
grading itself). It runs keyless with a lexical baseline policy:

| seed | memory OFF | memory ON | delta |
|---|---|---|---|
| 7  | 20/60 | 19/60 | −0.02 |
| 21 | 18/60 | 23/60 (2nd half .27 → **.50**) | +0.08 |
| 33 | 18/60 | 29/60 | +0.18 |
| | mean 18.7/60 | mean 23.7/60 | **mean +0.08** |

Read that honestly: on real pages the gain is **modest and seed-dependent** — advantage can
only amplify successes the base policy occasionally reaches, and three of the six tasks are
beyond the lexical baseline entirely. `--policy model` with a real endpoint is where the
comparison gets sharp, and the command prints which policy produced the numbers so nobody
can mistake the baseline for a language model.

---

## Why "without being trained" is the point

Deployed LLM agents have frozen weights: they repeat mistakes forever, and conventional RL
fixes that with the two costs you cannot pay on a small box — compute and catastrophic
forgetting.

[**JitRL** (Just-In-Time Reinforcement Learning, arXiv:2601.18510)](https://arxiv.org/abs/2601.18510)
moves the RL to *inference time*: a dynamic, non-parametric memory of `<state, action,
reward>` triplets, retrieval of similar past trajectories, an on-the-fly advantage estimate,
and a direct additive modulation of the output logits — for which the paper proves the exact
closed-form solution to the KL-constrained policy objective. It beats fine-tuning baselines
(WebRL) on WebArena *without any training*, at a fraction of the cost.

**agentd is an independent, productised implementation of that idea** — built for operators
rather than for a benchmark table. Where it deviates from the reference implementation, it
says so: retrieval here is lexical (Jaccard n-grams over an inverted index, zero embedding
calls), a deliberate cheap subset of the paper's BM25 + embedding + LLM-scored stack; the
advantage and exploration terms follow the published equations (including the
"recompute-baseline" detail and the fixed ε=0.05 the published path actually uses).

The rest of the loop is Reflexion-shaped: at the end of an episode the agent writes itself a
two-sentence critique, stores it, and recalls it the next time a similar situation appears.

## What it does

| | |
|---|---|
| **Learns while running** | `(state, action, discounted_return)` triplets in SQLite; n-gram retrieval with an inverted index; advantage re-ranking; verbal self-critique stored and recalled. Per-memory credit accounting shows which memories actually get adopted |
| **Any model you want** | one `base_url + api_key + model` — an OpenAI-compatible endpoint (your own relay included) or a native Anthropic / Gemini seat. Seats are managed live from the console or `POST /api/providers`; a capability probe picks the decode tier per endpoint and a missing key makes that seat *abstain out loud*, never fail silently |
| **Operates your servers** | SSH through the system `ssh`/`scp` (inherits `~/.ssh/config`, agent, `ProxyJump`), an interactive PTY terminal served over WebSocket, resumable uploads with md5 reconciliation |
| **An agent loop that acts** | open goals (`kind:"ops"`) run through the tool bus: the model proposes real calls — `ssh.exec {"host":"self","command":"df -h"}`, `ssh.exec df -h`, `finish {...}` — and every one passes the safety gate before execution |
| **You can see it work** | `Xvfb + x11vnc + noVNC` reached only through an SSH tunnel; the screen panel also states plainly when no vision model is attached instead of implying it saw something |
| **Refuses to destroy things** | every command is classified before execution; `rm -rf`, `mkfs`, `DROP TABLE`, force-push, `curl \| sh` etc. need a human approval, with the pending request showing command, reasons and host |
| **Multi-model councils** | independent proposals → cross-examination under a permission boundary → adjudication. Dissent is rendered as its own block and never averaged into a fake consensus |
| **Extensible four ways** | built-in tools, drop-in Python plugins, `SKILL.md` skill packs, and any MCP server — plus agentd itself **is** an MCP server |
| **Costs you less** | stable prompt prefix for provider-side caching, observation compaction, a per-step token ledger you can reconcile against your bill, cheap/strong tier routing |
| **Recurring work** | dependency-free cron evaluator (`agentd schedule`), with one-shot catch-up after downtime |
| **Shareable experience** | `agentd memory export` writes a signed `.agentdmem` pack **and** human-readable Markdown; import is idempotent and warns when provenance differs |

## Run it

### 1. Point it at any model — three decode tiers

JitRL needs a score per candidate action. Not every endpoint gives one, so `agentd` probes
and degrades on purpose:

| tier | endpoint gives | how the score is obtained |
|---|---|---|
| `token` | `logprobs` + `top_logprobs` | exactly the paper: `z = exp(logprob)` at the decision token |
| `n_sample` | logprob of the sampled token only | k independent samples; frequency becomes the score |
| `verbalized` | nothing | the model grades each option 0–100 |

```bash
agentd probe --base-url http://127.0.0.1:8080/v1 --model qwen3-8b
# → { "reachable": true, "decode_mode": "token", "notes": "top_logprobs at decision position …" }
```

The reference implementation reads `logprobs.content[-2]`, which breaks the moment a chat
template emits a different number of trailing tokens (reasoning models do). Here the
decision position is *found*, not assumed — that is what makes "swap the model" safe.

### 2. Put it to work on a server

```bash
agentd host add --label prod-1 --host 10.0.0.5 --user ops --key ~/.ssh/id_ed25519
agentd host probe --label prod-1        # arch / RAM / disk / docker / WebArena-viability, in one call
agentd viewer install                   # Xvfb + x11vnc + noVNC, loopback-only
agentd serve                            # HTTP API + SSE on 127.0.0.1:8765
```

Everything — API, PTY WebSocket, screen — travels through one SSH tunnel; the server exposes
no public ports besides 22:

```bash
ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8765 -L 6080:127.0.0.1:6080 root@your-server
# console   http://127.0.0.1:8765/
# screen    http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale
```

### 3. The council: who sits at the table is yours

Seats are not pinned to any vendor. Add an OpenAI-compatible endpoint, or a native
Anthropic / Gemini seat, from the console's seats page or:

```bash
curl -X POST http://127.0.0.1:8765/api/providers -H "Authorization: Bearer $TOKEN" \
  -d '{"name":"lab","kind":"openai_compat","model":"qwen3-8b",
       "base_url":"https://your-relay.example/v1","api_key_env":"AGENTD_LAB_KEY","tier":"cheap"}'
```

Then run an open goal with the members you choose:

```json
POST /api/session
{ "task": "nginx 挂了该怎么排查", "kind": "ops", "council": true,
  "members": ["claude", "lab"], "max_steps": 12 }
```

![Seats management](docs/images/console-seats.png)

What the council does, concretely:

* **Independent proposals first.** Every member scores the candidate actions on its own.
* **Cross-examination inside a boundary.** Members may only *flag problems* with someone
  else's proposal — never rewrite the plan. The critique phase sees prior output as
  untrusted input and cannot change tool permissions.
* **Learned trust.** Each member's weight is its historical win rate *in states like this
  one*, so the council discovers that model A is right about nginx and model B is right
  about gitlab, instead of trusting the same vendor forever.
* **Dissent is kept.** Objections become risk rows (`object` / `split` / `unavailable`) and
  render as their own block. A seat whose key is missing abstains with a recorded reason;
  the outcome is visibly thinner, never quietly wrong.
* **Conditional by default.** Deliberation fires on a thin margin between the top two
  options, a dangerous action, or a task you pin — confident steps cost one call.

### 4. Desktop client & console

![Onboarding: tunnel, token, pick where to start](docs/images/console-onboarding.png)

One bundle, two hosts: the console is served by agentd itself *and* loaded by the Tauri
desktop shell — there is no second UI to keep in sync. The client opens the SSH tunnel,
supervises it, and adds tray presence. Builds: macOS universal (Intel + Apple Silicon) and
Windows (via GitHub Actions — Tauri cannot cross-compile a Windows bundle from a Mac).

→ **[hue913/council-agent](https://github.com/hue913/council-agent)**

### 5. Recurring work

```bash
agentd schedule add --name nightly-ops --cron "17 3 * * *" --task nginx-down
agentd schedule daemon
agentd run --suite web --policy model          # measure with your real endpoint
```

## How it works

```mermaid
flowchart LR
    subgraph clients [Clients]
        B[Browser console]
        D[Tauri desktop<br/>Intel · Apple Silicon · Windows]
    end
    subgraph tunnel [One SSH tunnel — the whole security model]
        T1[8765 API + SSE]:::p
        T2[8766 PTY WebSocket]:::p
        T3[6080 noVNC]:::p
    end
    subgraph agentd [agentd · 127.0.0.1 only]
        API[FastAPI + bearer token]
        LOOP[Agent loop]
        KERNEL[JitRL kernel<br/>memory · retrieval · advantage · credit]
        COUNCIL[Council<br/>draft → critique → adjudicate]
        BUS[Tool bus<br/>builtin · plugins · skills · MCP]
        GATE[Safety gate + approvals + audit]
        PROV[Providers<br/>token / n_sample / verbalized]
    end
    subgraph targets [What it operates]
        SSH1[Your servers<br/>via system ssh]
        SCR[Xvfb screen<br/>x11vnc]
    end
    B & D --> T1 & T2 & T3 --> API --> LOOP
    LOOP --> KERNEL & COUNCIL & PROV
    LOOP --> BUS --> GATE --> SSH1
    API --> SCR
    classDef p fill:#eef,stroke:#88a;
```

The loop: `observe → enumerate/propose candidates → models score → kernel retrieves history
and biases → argmax → execute via the tool bus → record (state, action, return) → reflect`.

Repo map:

```
src/agentd/
  kernel/      JitRL: store, retrieval, advantage, credit, memory packs
  providers/   openai_compat · anthropic (native) · gemini (native) · capability probe
  council.py   three-phase deliberation with learned member weights
  loop.py      the agent loop, Reflexion-style reflection
  envs/        scripted bench tasks · real tool tasks · web (MiniWoB) · SSH · PTY
  toolbus/     builtin tools · plugins · skills · MCP client
  safety/      command classification · audit log
  api/         HTTP + SSE + PTY WebSocket + console serving
  screen/      capture · change detection · perception · gated synthetic input
  sysinfo/     ops panel metrics (no request data ever reaches a shell)
ui/            the console, served same-origin by agentd
tests/         ~300 tests; platform-specific ones skip with a stated reason
```

## Measured, honestly

Everything below is reproducible on a laptop; numbers are printed by the commands that
produce them, not typed into a README by hand.

| suite | policy | scale | memory OFF | memory ON | delta |
|---|---|---|---|---|---|
| ops (scripted chains) | synthetic noisy model | 40 eps × 12 seeds | mean 1.1/40 | mean 14.4/40 | better in 11/12 seeds |
| ops, default seed | same | 40 eps | 1/40 (2.5%) | 11/40 (27.5%) | 2nd half 55% |
| web (real Chromium) | lexical baseline (not an LLM) | 60 eps × 3 seeds | mean 18.7/60 | mean 23.7/60 | **+0.08, one seed negative** |

Cross-machine reproducibility: the deployment server (2 vCPU Xeon, Ubuntu) reproduces the
laptop's seed-7 numbers exactly (1/40 vs 11/40) — the measurement is seed-deterministic, not
machine-dependent.

**What is deliberately not measured here:** full WebArena. The official shape is 4 vCPU /
16 GB / 1000 GB + 7 self-hosted sites; this project's reference box is a 2 GB VPS, so the
web suite above is the MiniWoB-shaped compromise and `agentd host probe` tells you in one
command whether your box could host the real thing. Benchmarks you cannot run are worth
less than numbers you can.

Also from the live box (the exact deploy walkthrough is in [`docs/DEPLOY.md`](docs/DEPLOY.md)):

![The agent's view of the server screen](docs/images/server-screen.jpg)

*The screen panel's capture path, unedited: Xvfb → x11vnc → agentd → JPEG, fetched through
the tunnel. What the model sees is what you see here.*

## Deploy to a small VPS

```bash
# on the target, as root — installs uv + CPython 3.11, creates /opt/agentd/.venv,
# writes a 0600 config, runs doctor + the keyless demo, installs a hardened systemd unit
bash deploy/install_server.sh /path/to/agentd
```

The unit runs `agentd serve --host 127.0.0.1`, drops `AGENTD_AUTO_APPROVE` from the
environment on purpose (the lab-only auto-approve switch can never reach a deployed
service), and caps memory. The full worked example — including the mistake (a briefly
public noVNC port, and the status check that now fails loudly instead of staying quiet) —
is in [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Extending

```bash
# a tool in one file
cp examples/plugins/cert_days.py ~/.config/agentd/plugins/

# a skill pack: SKILL.md + scripts, loaded as tools
cp -r examples/skills/nginx-triage ~/.config/agentd/skills/

# agentd is also an MCP server for other agents
agentd mcp
```

Config lives in one file (`agentd.json`, 0600) + environment variables; keys are referenced
as `api_key_env` so they live in your systemd environment, not in the repo or the UI.

## Security model, stated plainly

* **Loopback + tunnel.** agentd binds `127.0.0.1` and refuses non-loopback without an
  explicit `AGENTD_ALLOW_PUBLIC=1`. Remote reach is one `ssh -N -L`.
* **Fail-closed auth.** Protected routes require a bearer token; with no token configured
  they answer 503, not "open".
* **The gate decides, not the model.** Commands are classified *before* execution;
  destructive intents need a human click that a disconnected socket can never auto-answer.
* **Audit without secrets.** The PTY records a rolling digest of what was typed — never the
  raw bytes — because an append-only log is exactly where secrets would end up.
* **Honest failure everywhere.** Unreachable endpoint, missing key, refused command,
  blocked approval: each is rendered as itself. Nothing in this project fabricates a
  success.

## Honest limits

* The council's trust weights need a few hundred episodes before they mean anything; early
  runs are deliberately neutral (0.5).
* The lexical retrieval is weaker than the paper's embedded retriever on large memories —
  it is fast, free and dependency-zero, and it is labelled as a subset.
* The web-suite gains are small (see the table). If you need benchmark-grade numbers, run
  `--policy model` with your own endpoint on real WebArena hardware.
* One agent, one box: there is no cluster mode, and `agentd` does not pretend to be a
  general-purpose coding agent. It operates servers, deliberately.

## Acknowledgments

**The paper this project exists because of.** agentd is an independent implementation of
[JitRL — *Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without
Gradient Updates*](https://arxiv.org/abs/2601.18510) (arXiv:2601.18510, ICML 2026
Spotlight). Thank you to its authors — **Yibo Li, Zijie Lin, Ailin Deng, Xuan Zhang, Yufei
He, Shuo Ji, Tri Cao, and Bryan Hooi** — for the method, the theorems, and for publishing a
reference implementation ([liushiliushi/JitRL](https://github.com/liushiliushi/JitRL)) that
made it possible to check this codebase against the source of truth. No code from that
repository is included here; the attribution is for the method and the ideas.

**Borrowed ideas, credited:**

* **[open-project-council](https://github.com/hue913/open-project-council)** — the round-table
  protocol this project's council descends from: independent proposals, bounded critique,
  adjudication, and the insistence that unresolved dissent is shown, not averaged away.
* **[Reflexion](https://arxiv.org/abs/2303.11366)** (Shinn et al.) — verbal reinforcement via
  stored self-reflection, the training-free critique the loop performs at episode end.
* **[WebArena](https://arxiv.org/abs/2307.13854)** (Zhou et al.) — the evaluation philosophy
  of real environments and functional correctness; the web suite here is the small-box
  descendant of it.
* **[WebRL](https://arxiv.org/abs/2411.02337)** (Qi et al.) — the strongest fine-tuning
  baseline JitRL is measured against; its existence is why "training-free" is worth
  claiming.

**Standing on:** [xterm.js](https://xtermjs.org/) · [FastAPI](https://fastapi.tiangolo.com/)
· [uvicorn](https://www.uvicorn.org/) · [Playwright](https://playwright.dev/) ·
[Xvfb / x11vnc / noVNC](https://novnc.com/) · [Tauri](https://tauri.app/) ·
[SQLite](https://sqlite.org/).

## License

[Apache-2.0](LICENSE). The artefacts you build with it are yours; credentials never enter
the repo, and the audit trail is designed not to accumulate secrets.

---

*If this project is useful to you, the best thanks are a star, an issue with your setup, and
a bench run against your own model — those numbers are the ones worth having.*
