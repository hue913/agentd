/* Council Agent console.
 *
 * A no-build single bundle served directly by agentd at "/" and opened in a
 * browser over the SSH tunnel. There is no second host loading this file: the
 * desktop shell is an independent implementation in another repo.
 *
 * The onboarding layer is not decoration. Someone opening this for the first
 * time needs to know three things in order: open the tunnel, prove who you are,
 * pick somewhere to start. Each step reports its own state, so "is it working"
 * is never a guess.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    token: sessionStorage.getItem("agentd_token") || "",
    session: null,
    pty: null, ws: null,
    term: null, fit: null,
    auto: null,
    onboardSeen: sessionStorage.getItem("agentd_onboarded") === "1",
  };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ───────────────────────── transport ───────────────────────── */

  const API_TIMEOUT_MS = 15000;

  async function api(path, opts = {}, attempt = 0) {
    const headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
    if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
    // Every call gets a hard deadline; a hung tunnel must surface as an error,
    // not as a spinner that never resolves.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), API_TIMEOUT_MS);
    let res;
    try {
      res = await fetch(path, Object.assign({}, opts, { headers, signal: ctrl.signal }));
    } catch (e) {
      // Network failure or timeout. Idempotent GETs retry once after a short
      // backoff; mutating calls must never be replayed blindly.
      if (attempt === 0 && (!opts.method || opts.method === "GET")) {
        await new Promise((r) => setTimeout(r, 500));
        return api(path, opts, attempt + 1);
      }
      throw new Error(e.name === "AbortError" ? "请求超时（15s）" : "网络错误");
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      let detail = `${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      const err = new Error(detail);
      err.status = res.status;   // callers distinguish 404 (quiet) from real faults
      throw err;
    }
    return res.status === 204 ? null : res.json();
  }

  /* At most three toasts on screen at once; each stacks above the last and
   * removes itself, so a burst of failures cannot hide earlier messages. */
  const toastStack = [];
  function toast(msg, kind) {
    const el = document.createElement("div");
    el.className = "toast" + (kind ? " " + kind : "");
    el.textContent = msg;
    document.body.appendChild(el);
    el.style.bottom = `${20 + toastStack.length * 48}px`;
    toastStack.push(el);
    while (toastStack.length > 3) toastStack.shift().remove();
    setTimeout(() => {
      const i = toastStack.indexOf(el);
      if (i >= 0) toastStack.splice(i, 1);
      // Re-stack the survivors so no gap is left behind.
      toastStack.forEach((t, idx) => { t.style.bottom = `${20 + idx * 48}px`; });
      el.remove();
    }, 4200);
  }

  function conn(ok, text) {
    $("conn-dot").className = "dot " + (ok ? "ok" : "bad");
    $("conn-text").textContent = text;
  }

  /* ───────────────────────── onboarding ───────────────────────── */

  async function probeHealth() {
    try {
      const h = await fetch("healthz").then((r) => r.json());
      conn(true, `已连接 · ${h.kernel?.steps ?? 0} 条记忆`);
      markStep(1, true, "隧道已通，可以继续下一步");
      return true;
    } catch (e) {
      conn(false, "隧道未建立");
      markStep(1, false, "还没连上 —— 先在终端里跑上面那条 ssh 命令");
      return false;
    }
  }

  function markStep(n, done, hint) {
    const el = document.querySelector(`.step[data-step="${n}"]`);
    if (!el) return;
    el.classList.toggle("done", !!done);
    const target = el.querySelector(".step-hint");
    if (target && hint) {
      target.textContent = hint;
      target.classList.toggle("ok", !!done);
    }
  }

  function wireCopy() {
    document.querySelectorAll("[data-copy]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const text = $(btn.dataset.copy)?.textContent ?? "";
        if (await copy(text)) flash(btn, "已复制");
        else toast("复制失败 —— 请手动选中复制", "bad");
      });
    });
    document.querySelectorAll("[data-copy2]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (await copy(btn.dataset.copy2)) flash(btn, "已复制");
        else toast("复制失败 —— 请手动选中复制", "bad");
      });
    });
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      // Clipboard API needs a secure context; the SSH tunnel is plain http, so
      // fall back rather than silently doing nothing. The caller is told
      // honestly whether either path worked.
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (__) { ok = false; }
      ta.remove();
      return ok;
    }
  }

  function flash(btn, msg) {
    const old = btn.textContent;
    btn.textContent = msg;
    btn.classList.add("ok");
    setTimeout(() => { btn.textContent = old; btn.classList.remove("ok"); }, 1400);
  }

  async function verifyToken() {
    const token = $("token-input").value.trim();
    if (!token) return toast("先粘贴令牌", "bad");
    state.token = token;
    try {
      await api("api/state");
      sessionStorage.setItem("agentd_token", token);
      markStep(2, true, "验证通过，令牌已存在本标签页");
      $("onboard-status").textContent = "一切就绪 —— 挑一个开始吧。";
      toast("验证通过", "good");
      await loadState();
      return true;
    } catch (e) {
      state.token = sessionStorage.getItem("agentd_token") || "";
      $("token-input").value = "";
      markStep(2, false, `验证失败：${e.message}`);
      return false;
    }
  }

  function closeOnboard() {
    $("onboard").hidden = true;
    sessionStorage.setItem("agentd_onboarded", "1");
    state.onboardSeen = true;
  }

  async function initOnboarding() {
    wireCopy();
    $("btn-verify").addEventListener("click", verifyToken);
    $("btn-skip").addEventListener("click", closeOnboard);
    $("btn-help").addEventListener("click", () => { $("onboard").hidden = false; });
    $("token-input").addEventListener("keydown", (e) => { if (e.key === "Enter") verifyToken(); });

    document.querySelectorAll(".pick-card").forEach((card) => {
      card.addEventListener("click", () => {
        closeOnboard();
        goto(card.dataset.goto);
      });
    });

    if (state.onboardSeen) $("onboard").hidden = true;

    // Step 1 is a real check, not a claim.
    const up = await probeHealth();
    const status = $("onboard-status");
    if (up && state.token) {
      try {
        await api("api/state");
        markStep(2, true, "令牌有效");
        status.textContent = "隧道已通、令牌有效 —— 一切就绪。";
      } catch (_) {
        status.textContent = "隧道已通，还需要填一次令牌。";
      }
    } else if (up) {
      status.textContent = "隧道已通，填一下令牌就能用。";
      $("token-input").focus();
    } else {
      status.textContent = "先建立隧道，再回来。";
    }
  }

  /* ───────────────────────── navigation ───────────────────────── */

  function goto(view) {
    document.querySelectorAll(".tabs button").forEach((b) =>
      b.classList.toggle("on", b.dataset.view === view));
    document.querySelectorAll(".view").forEach((v) =>
      v.classList.toggle("on", v.id === "view-" + view));

    if (view === "ops") { refreshOps(); loadLogs(); loadCredit(); }
    if (view === "screen") { checkVision(); if (state.auto === null) grabFrame(); }
    if (view === "models") loadSeats();
    if (view === "council") loadSeatPicker();
    if (view === "terminal" && state.term) setTimeout(() => { try { state.fit.fit(); } catch (_) {} }, 40);
  }

  /* ───────────────────────── state ───────────────────────── */

  async function loadState() {
    try {
      const st = await api("api/state");
      const nModels = Object.keys(st.providers || {}).length;
      const nHosts = (st.hosts || []).length;
      conn(true, `${nModels} 个模型 · ${nHosts} 台主机`);

      const hostSel = $("pty-host");
      if (hostSel && !hostSel.options.length) {
        (st.hosts || []).forEach((h) => hostSel.add(new Option(h, h)));
        if (!hostSel.options.length) {
          hostSel.add(new Option("（没有配置主机）", ""));
          hostSel.disabled = true;
        }
      }
      const logSel = $("log-unit");
      if (logSel && !logSel.options.length) {
        (st.services || []).forEach((s) => logSel.add(new Option(s.unit, s.unit)));
        if (!logSel.options.length) {
          ["agentd.service", "ssh.service", "nginx.service"].forEach((u) => logSel.add(new Option(u, u)));
        }
      }
      return st;
    } catch (e) {
      conn(false, e.message);
      return null;
    }
  }

  /* ───────────────────────── council ───────────────────────── */

  /* Generation token for the two polling chains. Starting a new session (or
   * reaching a terminal state) bumps it; any in-flight timer whose captured
   * generation no longer matches simply exits instead of rescheduling. This
   * is what keeps repeated "开始" clicks from leaving permanent 1.2s/4s
   * polling loops behind. */
  let pollGeneration = 0;
  let approvalsTimer = null;

  function startApprovalsPolling() {
    stopApprovalsPolling();
    approvalsTimer = setInterval(refreshApprovals, 4000);
  }

  function stopApprovalsPolling() {
    if (approvalsTimer !== null) { clearInterval(approvalsTimer); approvalsTimer = null; }
  }

  /* Terminal state: stop both chains and drop the session reference so the
   * approvals view stops claiming there is something to watch. */
  function finishSession() {
    stopApprovalsPolling();
    pollGeneration += 1;
    state.session = null;
    $("appr-badge").hidden = true;
    $("appr-badge").textContent = "0";
  }

  async function runCouncil() {
    const task = $("task-input").value.trim();
    if (!task) return toast("先写下要问的问题", "bad");
    const btn = $("btn-run");
    btn.disabled = true;
    btn.textContent = "进行中…";
    // Any polling chain left over from a previous run is orphaned here: bump
    // the generation so its timers exit on their next tick.
    stopApprovalsPolling();
    pollGeneration += 1;
    $("council-cols").innerHTML = "";
    $("council-empty").hidden = true;
    $("dissent").hidden = true;
    $("council-answer").hidden = true;
    $("council-trace-card").hidden = true;

    try {
      // kind=ops: the goal is open text, and actions are real tool calls
      // (ssh.exec / local.exec / fs.*) executed through the same safety gate.
      const r = await api("api/session", {
        method: "POST",
        body: JSON.stringify({
          task,
          kind: "ops",
          council: $("use-council").checked,
          members: chosenMembers(),
          max_steps: 12,
        }),
      });
      state.session = r.session;
      toast(`已启动 · ${r.session}`);
      const gen = ++pollGeneration;   // this run owns the new generation
      pollSession(gen);
      startApprovalsPolling();
    } catch (e) {
      toast("启动失败：" + e.message, "bad");
      $("council-empty").hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "开始";
    }
  }

  async function pollSession(gen) {
    if (gen !== pollGeneration || !state.session) return;
    try {
      const s = await api(`api/session/${state.session}`);
      if (gen !== pollGeneration) return;   // superseded by a newer run
      if (s.status === "running" || s.status === "pending") {
        setTimeout(() => pollSession(gen), 1200);
        return;
      }
      renderReport(s);
      finishSession();
    } catch (e) {
      if (gen !== pollGeneration) return;
      toast(e.message, "bad");
      finishSession();
    }
  }

  function renderReport(s) {
    const rep = s.report || {};
    const tok = rep.tokens || {};

    $("cost-box").hidden = false;
    $("cost-total").textContent = tok.total_tokens
      ? `${tok.total_tokens.toLocaleString()} tok${tok.est_usd ? ` · $${(+tok.est_usd).toFixed(4)}` : ""}`
      : "—";

    // 最终答案（工具任务收尾时给的那段话）
    if (rep.final) {
      $("council-answer").hidden = false;
      $("answer-text").textContent = rep.final;
      const okRun = s.status === "success";
      $("answer-tag").textContent = okRun ? "正常结束" : s.status;
      $("answer-tag").className = "tag " + (okRun ? "ok" : "warn");
    } else {
      $("council-answer").hidden = true;
    }

    // 圆桌的每一步：谁提了什么、谁质疑了谁
    const cols = $("council-cols");
    cols.innerHTML = "";
    const council = rep.council || [];
    council.forEach((c) => {
      const card = document.createElement("div");
      card.className = "speech" + (c.consensus ? " verdict" : "");
      const rows = Object.entries(c.proposals || {}).map(([m, action]) => {
        const w = c.weights && c.weights[m] != null ? ` · 权重 ${(+c.weights[m]).toFixed(2)}` : "";
        const act = action ? esc(action) : '<span class="muted">（弃权 / 不可用）</span>';
        return `<div class="proposal"><b>${esc(m)}</b><span class="muted sm">${w}</span><div class="act">${act}</div></div>`;
      }).join("");
      const objs = (c.objections || [])
        .map((o) => `<li>${esc(o)}</li>`).join("");
      card.innerHTML = `
        <h4>第 ${(Number(c.step) || 0) + 1} 步 · ${c.consensus ? "达成一致" : "有分歧"}</h4>
        ${rows || '<p class="muted">（这一步没有提案记录）</p>'}
        ${objs ? `<div class="objs"><b class="muted sm">质疑</b><ul class="muted sm">${objs}</ul></div>` : ""}
        <div class="meta">裁决：${esc(c.chosen || "—")}${c.tokens ? ` · ${esc(c.tokens)} tok` : ""}</div>`;
      cols.appendChild(card);
    });

    // 单模型运行（或没有圆桌记录）时给一个朴素摘要
    if (!council.length) {
      const picks = (rep.trace || []).map((t) => t.chosen_action).filter(Boolean);
      const card = document.createElement("div");
      card.className = "speech";
      card.innerHTML =
        `<h4>裁决</h4><p>${esc([...new Set(picks)].slice(0, 8).join("\n") || rep.analysis || "（没有轨迹）")}</p>` +
        `<div class="meta">${picks.length} 步${rep.stopped_by ? ` · ${esc(rep.stopped_by)}` : ""}</div>`;
      cols.appendChild(card);
    }

    // 分歧永远独立成块：反对、分裂、席位不可用
    const risks = [];
    council.forEach((c) => (c.risks || []).forEach((r) => risks.push(r)));
    if (risks.length) {
      $("dissent").hidden = false;
      const ul = $("dissent-list");
      ul.innerHTML = "";
      risks.forEach((r) => {
        const who = r.member ? `${r.member}${r.against ? " → " + r.against : ""}` : "";
        const stance = { object: "反对", split: "分歧", unavailable: "席位不可用" }[r.stance] || r.stance || "";
        const head = [who, stance].filter(Boolean).join(" · ");
        const li = document.createElement("li");
        li.textContent = head + (r.note ? `：${r.note}` : "");
        ul.appendChild(li);
      });
    } else {
      $("dissent").hidden = true;
    }

    // 执行轨迹：一个可展开的回放
    const trace = rep.trace || [];
    $("council-trace-card").hidden = trace.length === 0;
    const box = $("council-trace");
    box.innerHTML = "";
    trace.forEach((t) => {
      const det = document.createElement("details");
      det.className = "trace-step" + (t.ok === false ? " failed" : "");
      const viaCouncil = (t.mode || "").includes("council");
      const failed = t.ok === false ? '<span class="tag warn">失败</span>' : "";
      det.innerHTML =
        `<summary><span class="n">${(Number(t.t) || 0) + 1}</span>` +
        `<span class="act">${esc(t.chosen_action || "—")}</span>` +
        `<span class="meta">${failed} ${esc(t.retrieved || 0)} 条召回 · ${esc(t.duration_ms || 0)}ms ` +
        `${viaCouncil ? '<span class="tag ok">圆桌</span>' : '<span class="tag">单模型</span>'}</span></summary>` +
        `<div class="trace-body">` +
        `<div class="muted sm">当时的状态</div><pre>${esc(String(t.state || "").slice(0, 700))}</pre>` +
        (t.tool_output ? `<div class="muted sm">执行结果</div><pre>${esc(t.tool_output)}</pre>` : "") +
        (t.error ? `<div class="muted sm">错误</div><pre class="err">${esc(t.error)}</pre>` : "") +
        `<div class="meta sm">优势基线 ${(+t.baseline || 0).toFixed(3)} · 探索 ${(t.explored || []).length} 项 · 风险档 ${esc(t.risk || "—")}</div>` +
        `</div>`;
      box.appendChild(det);
    });

    $("council-empty").hidden = cols.children.length > 0;
    toast(`运行结束 · ${s.status}`);
  }

  /* ───────────────────────── 模型席位 ─────────────────────────
   * 圆桌会不绑定任何厂商：这里管理的就是"谁能上桌"。席位保存在服务端
   * agentd.json（0600），保存后热生效；缺密钥的席位会被明确标出，
   * 圆桌会自动跳过它并记一条"席位不可用"的风险，而不是假装参与。
   */

  let seats = [];

  /* api/providers is fetched from two views (seat manager and council picker);
   * share one response with a short TTL so entering the council view does not
   * double-hit the endpoint. Saving or deleting a seat invalidates the cache. */
  let seatsCache = null;
  let seatsCacheAt = 0;
  const SEATS_TTL_MS = 30000;

  async function fetchSeats(force = false) {
    if (!force && seatsCache !== null && Date.now() - seatsCacheAt < SEATS_TTL_MS) return seatsCache;
    seatsCache = await api("api/providers");
    seatsCacheAt = Date.now();
    return seatsCache;
  }

  function invalidateSeats() { seatsCache = null; seatsCacheAt = 0; }

  async function loadSeats() {
    const box = $("seat-list");
    if (!box) return;
    try {
      const data = await fetchSeats();
      seats = data.providers || [];
      if (!seats.length) {
        box.innerHTML =
          `<div class="empty" style="grid-column:1/-1">` +
          `<p><b>还没有配置任何模型。</b></p>` +
          `<p class="muted">在下面填一个席位就能加入圆桌。OpenAI 兼容端点（含自建中转）填 Base URL；` +
          `Anthropic / Gemini 原生协议留空即可。密钥优先用环境变量，不方便时也可直接填。</p></div>`;
        return;
      }
      box.innerHTML = seats.map((s) => {
        const key = !s.key_required
          ? '<span class="tag">无需密钥</span>'
          : s.key_present
            ? '<span class="tag ok">密钥就绪</span>'
            : '<span class="tag warn">缺密钥</span>';
        return `<div class="speech seat" data-seat="${esc(s.name)}">
          <h4>${esc(s.name)}${s.default ? ' <span class="tag ok">默认</span>' : ""}</h4>
          <p class="seat-line">${esc(s.model)}</p>
          <div class="meta">
            <div>协议 ${esc(s.kind)} · 档位 ${esc(s.tier)}</div>
            ${s.base_url ? `<div class="ellip">${esc(s.base_url)}</div>` : ""}
            ${s.key_env ? `<div>密钥变量 <code>${esc(s.key_env)}</code></div>` : ""}
            <div class="seat-tags">${key}</div>
            <div class="seat-btns">
              <button class="ghost sm" data-act="probe">探活</button>
              <button class="ghost sm" data-act="edit">编辑</button>
              <button class="ghost sm" data-act="del">删除</button>
            </div>
          </div>
        </div>`;
      }).join("");
      box.querySelectorAll(".seat").forEach((el) => {
        const name = el.dataset.seat || "";
        el.querySelector('[data-act="probe"]').onclick = () => probeSeat(name);
        el.querySelector('[data-act="edit"]').onclick = () => fillSeatForm(name);
        el.querySelector('[data-act="del"]').onclick = () => deleteSeat(name);
      });
    } catch (e) {
      box.innerHTML = `<div class="muted" style="grid-column:1/-1">读取失败：${esc(e.message)}</div>`;
    }
  }

  async function probeSeat(name) {
    toast(`正在探活 ${name}…`);
    try {
      const r = await api(`api/providers/${encodeURIComponent(name)}/probe`, { method: "POST" });
      if (r.reachable) {
        toast(`${name} 可达 · 解码档位 ${r.decode_mode || "?"}${r.notes ? " · " + r.notes.slice(0, 80) : ""}`, "good");
      } else {
        toast(`${name} 不可达：${(r.notes || "无响应").slice(0, 120)}`, "bad");
      }
    } catch (e) {
      toast(`探活失败：${e.message}`, "bad");
    }
  }

  function fillSeatForm(name) {
    const s = seats.find((x) => x.name === name);
    if (!s) return;
    $("seat-name").value = s.name;
    $("seat-kind").value = s.kind || "openai_compat";
    $("seat-model").value = s.model || "";
    $("seat-base").value = s.base_url || "";
    $("seat-keyenv").value = s.key_env || "";
    $("seat-key").value = "";
    $("seat-tier").value = s.tier || "strong";
    $("seat-model").focus();
    toast(`已载入 ${name}，改完点「保存席位」`);
  }

  function clearSeatForm() {
    ["seat-name", "seat-model", "seat-base", "seat-keyenv", "seat-key"].forEach((id) => {
      const el = $(id);
      if (el) el.value = "";
    });
  }

  async function saveSeat() {
    const body = {
      name: $("seat-name").value.trim(),
      kind: $("seat-kind").value,
      model: $("seat-model").value.trim(),
      base_url: $("seat-base").value.trim(),
      api_key_env: $("seat-keyenv").value.trim(),
      api_key: $("seat-key").value.trim(),
      tier: $("seat-tier").value,
    };
    if (!body.name || !body.model) return toast("席位名和模型名必填", "bad");
    const btn = $("btn-seat-save");
    btn.disabled = true;
    try {
      const r = await api("api/providers", { method: "POST", body: JSON.stringify(body) });
      toast(`已保存席位 ${r.name}${r.note ? " · " + r.note : ""}`, "good");
      clearSeatForm();
      invalidateSeats();
      await loadSeats();
      await loadSeatPicker();
    } catch (e) {
      toast("保存失败：" + e.message, "bad");
    } finally {
      btn.disabled = false;
    }
  }

  async function deleteSeat(name) {
    if (!name) return;
    if (!confirm(`删除席位 ${name}？服务端配置文件会同步更新，圆桌将不再使用它。`)) return;
    try {
      await api(`api/providers/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast(`已删除 ${name}`);
      invalidateSeats();
      await loadSeats();
      await loadSeatPicker();
    } catch (e) {
      toast("删除失败：" + e.message, "bad");
    }
  }

  async function loadSeatPicker() {
    const box = $("council-seats");
    if (!box) return;
    try {
      const data = await fetchSeats();
      const list = data.providers || [];
      if (!list.length) {
        box.innerHTML = '<span class="muted sm">还没有模型 —— 去「模型」页添加任意端点（缺密钥会如实标出）。</span>';
        return;
      }
      // 保持用户已勾选的状态；首次进入默认全选 —— 缺密钥的席位也在列，
      // 只是标注出来，圆桌会跳过它并把「席位不可用」记进分歧块。
      const prev = new Set([...box.querySelectorAll("input:checked")].map((i) => i.value));
      const first = !box.querySelectorAll("input").length;
      box.innerHTML = list.map((s) => {
        const missing = s.key_required && !s.key_present;
        const on = first ? true : prev.has(s.name);
        return `<label class="chk sm${missing ? " dim" : ""}" title="${missing ? "缺密钥，圆桌会跳过它并记录一条风险" : ""}">` +
          `<input type="checkbox" value="${esc(s.name)}"${on ? " checked" : ""}> ${esc(s.name)}` +
          `${missing ? '<span class="muted">（缺密钥）</span>' : ""}</label>`;
      }).join("");
    } catch (_) {
      box.innerHTML = '<span class="muted sm">读取席位失败（检查连接与令牌）</span>';
    }
  }

  function chosenMembers() {
    const boxes = [...document.querySelectorAll("#council-seats input")];
    const checked = boxes.filter((i) => i.checked).map((i) => i.value);
    // 全选（或没有席位）时交给服务端默认（主模型 + 按上限补齐）；
    // 手动取消过才发送显式名单。空名单不发送 —— 圆的还是那张桌子。
    return checked.length && checked.length < boxes.length ? checked : null;
  }

  /* ───────────────────────── terminal ───────────────────────── */

  async function openPty() {
    const host = $("pty-host").value;
    if (!host) return toast("没有可用主机", "bad");
    try {
      const s = await api("api/pty/open", {
        method: "POST",
        body: JSON.stringify({ host, purpose: $("pty-purpose").value.trim(), rows: 30, cols: 100 }),
      });
      state.pty = s.session;
      $("pty-state").textContent = `${s.session} · ${s.host}`;
      $("term-empty").hidden = true;
      connectPty();
      toast("已打开终端");
    } catch (e) {
      toast("打开失败：" + e.message, "bad");
    }
  }

  async function closePty() {
    if (!state.pty) return;
    try { await api(`api/pty/${state.pty}/close`, { method: "POST" }); } catch (_) {}
    teardownPtyConnection();
    state.pty = null;
    $("pty-state").textContent = "未打开";
    $("term-empty").hidden = false;
    toast("终端已关闭");
  }

  /* Tear down everything a previous connection owned: the socket, the resize
   * observer's push path, and the state references. Without this, reopening a
   * PTY would leave the old WebSocket alive and its output interleaved with
   * the new session's. Handlers are detached first so the deliberate close
   * does not fire the "已断开" UI path. */
  function teardownPtyConnection() {
    if (state.ws) {
      const ws = state.ws;
      ws.onopen = ws.onmessage = ws.onclose = null;
      try { ws.close(); } catch (_) {}
      state.ws = null;
    }
    if (ptyResizeObserver) {
      try { ptyResizeObserver.disconnect(); } catch (_) {}
    }
  }

  /* One observer for the lifetime of the page, created lazily and reused
   * across reconnects; a fresh ResizeObserver per onopen would leak one per
   * session. It only pushes when a live socket exists. */
  let ptyResizeObserver = null;

  function ensurePtyResizeObserver() {
    if (ptyResizeObserver) { ptyResizeObserver.observe($("term-wrap")); return; }
    ptyResizeObserver = new ResizeObserver(() => {
      if (!state.ws || state.ws.readyState !== 1) return;
      try { state.fit.fit(); } catch (_) {}
      try { state.ws.send(JSON.stringify({ type: "resize", rows: state.term.rows, cols: state.term.cols })); } catch (_) {}
    });
    ptyResizeObserver.observe($("term-wrap"));
  }

  function connectPty() {
    if (!state.pty) return;
    // A previous connection may still be open (re-open without explicit
    // close); fully tear it down before opening a new one.
    teardownPtyConnection();
    if (!state.term) {
      state.term = new Terminal({
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: 12, cursorBlink: true, scrollback: 6000,
        theme: { background: "#08090d", foreground: "#e8ebf1", cursor: "#6aa9ff" },
      });
      state.fit = new FitAddon.FitAddon();
      state.term.loadAddon(state.fit);
      state.term.open($("term"));
      state.fit.fit();
      state.term.onData((d) => {
        if (state.ws && state.ws.readyState === 1) state.ws.send(d);
      });
    }
    // A browser WebSocket cannot set an Authorization header, so the token
    // rides in the query string -- the path the server documents.
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/api/pty/${state.pty}/stream?token=${encodeURIComponent(state.token)}`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    state.ws = ws;

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") return;   // ping / exit control frame
      state.term.write(new Uint8Array(ev.data));
    };
    ws.onclose = (ev) => {
      if (state.ws === ws) state.ws = null;
      $("pty-state").textContent = ev.code === 1000 ? "已关闭" : `已断开 (${ev.code})`;
      if (ev.code === 4401) toast("令牌不对，PTY 被拒绝", "bad");
    };
    ws.onopen = () => {
      ensurePtyResizeObserver();
      ws.send(JSON.stringify({ type: "resize", rows: state.term.rows, cols: state.term.cols }));
    };
  }

  /* ───────────────────────── screen ───────────────────────── */

  async function checkVision() {
    try {
      const v = await api("api/screen/vision");
      const el = $("vision-state");
      if (v.has_vision) {
        el.textContent = `视觉模型：${v.model}`;
        el.className = "tag ok";
      } else {
        el.textContent = "未接视觉模型（只有结构性描述）";
        el.className = "tag warn";
      }
    } catch (e) {
      // Not in the 5s grab loop (this runs once per view switch), so a plain
      // toast is enough — but 401/network failures must not be silent.
      toast("视觉状态检查失败：" + e.message, "bad");
    }
  }

  /* Only one grab in flight at a time: each round is three serial requests,
   * and an overlapping round would reorder frames and thrash the backend. */
  let grabInFlight = false;

  async function grabFrame() {
    if (grabInFlight) return;
    grabInFlight = true;
    try {
      const f = await api("api/screen/frame");
      $("screen-img").src = `api/screen/frame.jpg?t=${f.ts}`;
      $("screen-meta").textContent =
        `${f.width}×${f.height} · 有内容 ${(f.ink_ratio * 100).toFixed(0)}% · ` +
        `${f.changed ? "画面有变化" : "画面未变"} · ${(f.jpeg_bytes / 1024).toFixed(0)} KB`;
      $("screen-blank").textContent = f.blank
        ? "画面是空的 —— 服务器上还没跑任何窗口程序"
        : "";

      const obs = await api("api/screen/observe");
      $("observe").textContent = obs.text;

      const w = await api("api/screen/windows");
      $("windows").innerHTML = (w.windows || [])
        .map((x) => `<div>${esc(x.name || "(无名窗口)")} · ${esc(x.w)}×${esc(x.h)} @ (${esc(x.x)},${esc(x.y)})</div>`)
        .join("") || '<span>画面上没有窗口</span>';
    } catch (e) {
      toast("抓帧失败：" + e.message, "bad");
    } finally {
      grabInFlight = false;
    }
  }

  /* ───────────────────────── ops ───────────────────────── */

  function stat(k, v, extra) {
    return `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>${extra || ""}</div>`;
  }
  function meter(pct) {
    const cls = pct > 88 ? "bad" : pct > 70 ? "warn" : "";
    return `<div class="meter"><i class="${cls}" style="width:${Math.min(100, pct)}%"></i></div>`;
  }
  function uptime(s) {
    if (!s) return "—";
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    return d ? `${d} 天 ${h} 小时` : h ? `${h} 小时 ${m} 分` : `${m} 分`;
  }
  function pctOf(a, b) { return b ? Math.round((a / b) * 100) : 0; }

  async function refreshOps() {
    try {
      const snap = await api("api/sys/snapshot");
      const o = snap.overview || {};
      const root = (o.disk || []).find((d) => d.mount === "/") || {};
      $("ops-grid").innerHTML = [
        stat("已运行", uptime(o.uptime_s)),
        stat("负载", `${(o.load1 ?? 0).toFixed(2)} · ${(o.load5 ?? 0).toFixed(2)} · ${(o.load15 ?? 0).toFixed(2)}`),
        stat("内存", pctOf((o.mem_total_kb || 0) - (o.mem_avail_kb || 0), o.mem_total_kb) + "%", meter(o.mem_used_pct || 0)),
        stat("Swap", pctOf(o.swap_used_kb, o.swap_total_kb) + "%",
          meter(pctOf(o.swap_used_kb, o.swap_total_kb))),
        stat("根分区", (root.use_pct ?? "?") + "%", meter(parseFloat(root.use_pct || 0))),
        stat("磁盘可用", (((root.avail || 0) / 2 ** 30).toFixed(1)) + " GB"),
      ].join("");

      $("proc-table").querySelector("tbody").innerHTML = (snap.processes || [])
        .map((p) => `<tr><td>${esc(p.comm)}</td><td>${(p.rss_kb / 1024).toFixed(1)} MB</td>` +
                    `<td>${esc(p.cpu_pct)}%</td><td>${uptime(p.etime_s)}</td></tr>`)
        .join("") || '<tr><td colspan="4" class="muted">—</td></tr>';
    } catch (e) {
      toast("运维数据拉取失败：" + e.message, "bad");
    }
  }

  async function loadCredit() {
    const box = $("credit-box");
    try {
      const d = await api("api/credit?limit=12");
      const t = d.totals || {};
      if (!t.decisions) {
        box.innerHTML = `还没有可统计的记忆。<br><span class="muted">共 ${t.steps || 0} 条记忆，` +
                        `等 Agent 真的检索过并做出选择后，这里会出现「被采纳 / 被否决」的计数。</span>`;
        return;
      }
      const head = `<div class="muted sm" style="margin-bottom:8px">` +
        `召回 ${esc(t.recalls)} 次 · 采纳 ${esc(t.adopted)} · 否决 ${esc(t.rejected)} · ` +
        `采纳率 <b>${((t.adoption_rate ?? 0) * 100).toFixed(1)}%</b></div>`;
      box.innerHTML = head + (d.entries || []).map((e) => {
        const cls = e.credit >= 1 ? "up" : "down";
        return `<div class="credit-row"><span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">` +
          `${esc((e.action || "").slice(0, 40))}</span>` +
          `<span class="credit-val ${cls}">×${(+e.credit).toFixed(2)}</span>` +
          `<span class="muted">${esc(e.adopted)}/${esc(e.rejected)}</span></div>`;
      }).join("");
    } catch (e) {
      box.textContent = "拉取失败：" + e.message;
    }
  }

  async function loadLogs() {
    const unit = $("log-unit").value;
    if (!unit) return;
    try {
      const r = await api(`api/sys/logs/${encodeURIComponent(unit)}?lines=120`);
      $("logs").textContent = (r.entries || []).join("\n") || "（这个服务还没有日志）";
    } catch (e) {
      $("logs").textContent = "加载失败：" + e.message;
    }
  }

  /* ───────────────────────── approvals ───────────────────────── */

  /* Throttle for the "审批列表不可达" toast: the 4s poll would otherwise
   * spam one toast per tick while the tunnel is down. */
  let apprErrorShown = false;

  async function refreshApprovals() {
    if (!state.session) return;
    try {
      const s = await api(`api/session/${state.session}`);
      const pending = s.pending_approvals || [];
      const list = $("appr-list");
      list.innerHTML = "";
      $("appr-empty").hidden = pending.length > 0;
      $("appr-badge").hidden = pending.length === 0;
      $("appr-badge").textContent = pending.length;

      pending.forEach((p) => {
        // Entries are objects now ({token, command|tool+args, reasons, level,
        // host}); tolerate a bare token string in case of an older server.
        if (typeof p === "string") p = { token: p };
        const what = p.command
          || (p.tool ? `${p.tool} ${JSON.stringify(p.args || {})}` : "(未知请求)");
        const why = (p.reasons || []).join("；")
          || (p.level === "block" ? "被安全门拦下，需要人工确认" : "需要人工确认");
        const el = document.createElement("div");
        el.className = "appr " + (p.level === "block" ? "block" : "");
        el.innerHTML =
          `<div class="why">${esc(why)}</div>` +
          `<div class="cmd">${p.host ? esc(`[${p.host}] `) : ""}${esc(what)}</div>` +
          `<div class="acts"><button class="approve">允许</button><button class="deny">拒绝</button></div>`;
        const [ok, no] = el.querySelectorAll("button");
        ok.onclick = () => decide(p.token, true, el);
        no.onclick = () => decide(p.token, false, el);
        list.appendChild(el);
      });
    } catch (e) {
      // A 404 just means the session is not queryable (not started, already
      // reaped) -- quiet is correct there. Anything else (401, network,
      // timeout) is a real fault and gets told to the user, but throttled so
      // the 4s poll cannot flood the screen.
      if (e.status === 404) return;
      if (apprErrorShown) return;
      apprErrorShown = true;
      toast("审批列表不可达：" + e.message, "bad");
      setTimeout(() => { apprErrorShown = false; }, 15000);
    }
  }

  async function decide(token, approved, el) {
    el.remove();
    try {
      await api("api/approve", { method: "POST", body: JSON.stringify({ token, approved }) });
      toast(approved ? "已允许执行" : "已拒绝", approved ? "good" : "");
    } catch (e) {
      toast("审批失败：" + e.message, "bad");
      // The request is still pending on the server; putting the entry back at
      // the head of the list keeps the UI honest instead of pretending the
      // decision went through.
      const list = $("appr-list");
      if (list) list.insertBefore(el, list.firstChild);
      else refreshApprovals();
    }
  }

  /* ───────────────────────── boot ───────────────────────── */

  function init() {
    $("tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-view]");
      if (b) goto(b.dataset.view);
    });
    $("btn-run").onclick = runCouncil;
    $("btn-pty-open").onclick = openPty;
    $("btn-pty-close").onclick = closePty;
    $("btn-grab").onclick = grabFrame;
    $("btn-refresh").onclick = async () => { await loadState(); refreshOps(); loadSeatPicker(); };
    $("btn-seat-save").onclick = saveSeat;
    $("btn-seat-clear").onclick = clearSeatForm;
    // The explicit 刷新 button must bypass the 30s seats cache.
    $("btn-seat-reload").onclick = () => { invalidateSeats(); loadSeats(); };
    $("btn-seat-manage").onclick = () => goto("models");
    $("log-unit").onchange = loadLogs;
    $("btn-auto").onclick = () => {
      if (state.auto) {
        clearInterval(state.auto);
        state.auto = null;
        $("btn-auto").textContent = "自动刷新";
      } else {
        // Each tick is three serial requests; skip the tick entirely unless
        // the screen view is visible, and grabFrame skips overlapping rounds.
        state.auto = setInterval(() => {
          if ($("view-screen").classList.contains("on")) grabFrame();
        }, 5000);
        $("btn-auto").textContent = "停止自动";
      }
    };

    // The noVNC link only makes sense once a tunnel exists; keep it honest.
    $("novnc-link").href = "http://127.0.0.1:6080/vnc.html";

    initOnboarding().then(async () => {
      if (state.token) {
        try {
          await api("api/state");
          await loadState();
          loadSeatPicker();
          refreshOps();
        } catch (_) {
          state.token = "";
          sessionStorage.removeItem("agentd_token");
        }
      }
    });

    // Approvals polling is bound to a live session (started in runCouncil,
    // stopped in finishSession) -- a permanent interval here would keep
    // hitting a dead session forever.

    // Leaving the page: close the PTY socket and stop every timer we own so
    // the server sees a clean disconnect.
    window.addEventListener("beforeunload", () => {
      teardownPtyConnection();
      if (state.auto !== null) { clearInterval(state.auto); state.auto = null; }
      stopApprovalsPolling();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();