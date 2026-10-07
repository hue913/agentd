/* Council Agent console.
 *
 * One bundle serves both surfaces: the browser console and the Tauri shell
 * load this same file. Only the transport differs -- see api() below.
 *
 * Two conventions worth knowing:
 *  - The bearer token lives in sessionStorage, never localStorage, so it does
 *    not outlive the tab and is not readable by another page on the host.
 *  - Approval is always an explicit click. A disconnected socket means the
 *    pending request is abandoned, never auto-approved.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    token: sessionStorage.getItem("agentd_token") || "",
    session: null,
    pty: null,
    term: null,
    fit: null,
    auto: false,
  };

  // ---------- transport ----------
  async function api(path, opts = {}) {
    const headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
    if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    if (!res.ok) {
      let detail = `${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    return res.status === 204 ? null : res.json();
  }

  function toast(msg, bad) {
    const el = $("toast");
    el.textContent = msg;
    el.className = "toast" + (bad ? " bad" : "");
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, 4200);
  }

  function setConn(ok, text) {
    $("conn-dot").className = "dot " + (ok ? "ok" : "bad");
    $("conn-text").textContent = text;
  }

  // ---------- token gate ----------
  // /healthz is deliberately public, so it is used to discover whether a token
  // is required rather than assuming one.
  async function bootstrap() {
    try {
      const health = await fetch("healthz").then((r) => r.json());
      setConn(true, `kernel β=${health.kernel?.beta ?? "?"} γ=${health.kernel?.gamma ?? "?"}`);
    } catch (e) {
      setConn(false, "cannot reach agentd");
      return;
    }
    try {
      await api("api/state");
    } catch (e) {
      const token = prompt("agentd 需要 Bearer token（服务器上 /var/lib/agentd/api.token）");
      if (!token) { setConn(false, "no token"); return; }
      state.token = token.trim();
      sessionStorage.setItem("agentd_token", state.token);
      await refreshState();
    }
  }

  // ---------- council ----------
  async function refreshState() {
    const st = await api("api/state");
    setConn(true, `${Object.keys(st.providers || {}).length} 模型 · ${(st.hosts || []).length} 主机`);

    const sel = $("pty-host");
    if (sel && sel.options.length === 0) {
      (st.hosts || []).forEach((h) => sel.add(new Option(h, h)));
    }
    const lu = $("log-unit");
    if (lu && lu.options.length === 0) {
      (st.services || []).forEach((s) => lu.add(new Option(s.unit, s.unit)));
      if (!lu.options.length) {
        ["agentd.service", "ssh.service", "nginx.service"].forEach((u) => lu.add(new Option(u, u)));
      }
    }
  }

  async function runCouncil() {
    const task = $("task-input").value.trim();
    if (!task) return toast("先写下要问圆桌会的问题", true);
    $("btn-run").disabled = true;
    $("council-cols").innerHTML = "";
    $("dissent").hidden = true;
    try {
      const r = await api("api/session", {
        method: "POST",
        body: JSON.stringify({ task, council: $("use-council").checked, max_steps: 12 }),
      });
      state.session = r.session;
      toast(`session ${r.session}`);
      pollSession();
    } catch (e) {
      toast("启动失败: " + e.message, true);
    } finally {
      $("btn-run").disabled = false;
    }
  }

  async function pollSession() {
    if (!state.session) return;
    try {
      const s = await api(`api/session/${state.session}`);
      if (s.status === "running" || s.status === "pending") {
        return setTimeout(pollSession, 1200);
      }
      renderReport(s);
    } catch (e) {
      toast(e.message, true);
    }
  }

  function renderReport(s) {
    const rep = s.report || {};
    const cols = $("council-cols");
    cols.innerHTML = "";

    // Costs: the panel most products do not show at all.
    const tok = rep.tokens || {};
    $("cost-total").textContent = tok.total_tokens
      ? `${tok.total_tokens.toLocaleString()} tok${tok.est_usd ? ` / $${tok.est_usd.toFixed(4)}` : ""}`
      : "—";

    const trace = rep.trace || [];
    const byModel = new Map();
    for (const step of trace) {
      const key = step.model || "model";
      if (!byModel.has(key)) byModel.set(key, { picks: [], score: 0 });
      const g = byModel.get(key);
      if (step.chosen_action) g.picks.push(step.chosen_action);
      g.score += (step.options || []).length;
    }
    if (!byModel.size) byModel.set("model", { picks: [rep.analysis || "(无轨迹)"], score: 0 });

    for (const [name, g] of byModel) {
      const card = document.createElement("div");
      card.className = "card";
      const picks = [...new Set(g.picks)].slice(0, 6);
      card.innerHTML = `<h4>${escapeHtml(name)}</h4><p>${escapeHtml(picks.join("\n"))}</p>
        <div class="meta">${g.picks.length} 步 · ${g.score} 个候选打分</div>`;
      cols.appendChild(card);
    }

    // Dissent is rendered as its own block, never folded into a verdict.
    const risks = rep.risks || s.dissent || [];
    if (risks.length) {
      $("dissent").hidden = false;
      const ul = $("dissent-list");
      ul.innerHTML = "";
      risks.forEach((r) => {
        const li = document.createElement("li");
        li.textContent = typeof r === "string" ? r : (r.text || JSON.stringify(r));
        ul.appendChild(li);
      });
    }
    toast(`运行结束：${s.status}`);
  }

  // ---------- terminal ----------
  async function openPty() {
    const host = $("pty-host").value;
    if (!host) return toast("没有可用主机", true);
    const purpose = $("pty-purpose").value.trim();
    try {
      const s = await api("api/pty/open", {
        method: "POST",
        body: JSON.stringify({ host, purpose, rows: 30, cols: 100 }),
      });
      state.pty = s.session;
      $("pty-state").textContent = `${s.session} · ${s.host}`;
      connectPty();
    } catch (e) {
      toast("打开失败: " + e.message, true);
    }
  }

  function connectPty() {
    if (!state.pty) return;
    if (!state.term) {
      state.term = new Terminal({
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: 12, cursorBlink: true, scrollback: 5000,
        theme: { background: "#0b0d11", foreground: "#e6e9ef" },
      });
      state.fit = new FitAddon.FitAddon();
      state.term.loadAddon(state.fit);
      state.term.open($("term"));
      state.fit.fit();
      state.term.onData((d) => {
        if (state.ws && state.ws.readyState === 1) state.ws.send(d);
      });
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    // A browser WebSocket cannot set an Authorization header, so the token
    // rides in the query string -- the same path the server documents.
    const url = `${proto}//${location.host}/api/pty/${state.pty}/stream?token=${encodeURIComponent(state.token)}`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    state.ws = ws;
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") return;      // control frame (ping/exit)
      state.term.write(new Uint8Array(ev.data));
    };
    ws.onclose = (ev) => {
      $("pty-state").textContent = `已断开 (${ev.code})`;
      if (ev.code === 4401) toast("PTY 鉴权失败：token 不对", true);
    };
    ws.onopen = () => {
      const send = (rows, cols) => ws.send(JSON.stringify({ type: "resize", rows, cols }));
      send(state.term.rows, state.term.cols);
      new ResizeObserver(() => {
        try { state.fit.fit(); send(state.term.rows, state.term.cols); } catch (_) {}
      }).observe($("term"));
    };
  }

  // ---------- screen ----------
  async function grabFrame() {
    try {
      const f = await api("api/screen/frame");
      $("screen-img").src = `api/screen/frame.jpg?t=${f.ts}&_=${state.token}`;
      $("screen-meta").textContent =
        `${f.width}x${f.height} · ink ${(f.ink_ratio * 100).toFixed(1)}% · ` +
        `${f.changed ? "有变化" : "无变化"} · ${f.blank ? "空白" : "有内容"} · ${(f.jpeg_bytes / 1024).toFixed(1)} KB`;
      const obs = await api("api/screen/observe");
      $("observe").textContent = obs.text;
      const w = await api("api/screen/windows");
      $("windows").innerHTML = (w.windows || [])
        .map((x) => `<div>${escapeHtml(x.name || "(unnamed)")} · ${x.w}x${x.h} @ (${x.x},${x.y})</div>`)
        .join("") || "<div class='sub'>none</div>";
    } catch (e) {
      toast("抓帧失败: " + e.message, true);
    }
  }

  async function checkVision() {
    try {
      const v = await api("api/screen/vision");
      const el = $("vision-state");
      el.textContent = v.has_vision ? `视觉模型: ${v.model}` : "未接视觉模型（只有结构性描述）";
      el.className = "tag " + (v.has_vision ? "ok" : "warn");
    } catch (_) {}
  }

  // ---------- ops ----------
  async function refreshOps() {
    try {
      const snap = await api("api/sys/snapshot");
      const o = snap.overview || {};
      const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);
      const root = (o.disk || []).find((d) => d.mount === "/") || {};
      $("ops-grid").innerHTML = [
        stat("运行时长", fmtUptime(o.uptime_s), ""),
        stat("负载", `${(o.load1 ?? 0).toFixed(2)} / ${(o.load5 ?? 0).toFixed(2)} / ${(o.load15 ?? 0).toFixed(2)}`, ""),
        stat("内存", `${pct(o.mem_total_kb - o.mem_avail_kb, o.mem_total_kb)}%`,
          meter(o.mem_used_pct)),
        stat("Swap", `${pct(o.swap_used_kb, o.swap_total_kb)}%`, meter(o.swap_total_kb ? (o.swap_used_kb / o.swap_total_kb) * 100 : 0)),
        stat("根分区", `${root.use_pct ?? "?"}%`, meter(parseFloat(root.use_pct || 0))),
        stat("磁盘可用", `${((root.avail || 0) / 2 ** 30).toFixed(1)} GB`, ""),
      ].join("");

      $("proc-table").querySelector("tbody").innerHTML = (snap.processes || [])
        .map((p) => `<tr><td>${escapeHtml(p.comm)}</td><td>${(p.rss_kb / 1024).toFixed(1)} MB</td>
                     <td>${p.cpu_pct}%</td><td>${fmtUptime(p.etime_s)}</td></tr>`).join("");
    } catch (e) {
      toast("运维数据失败: " + e.message, true);
    }
  }

  async function loadLogs() {
    const unit = $("log-unit").value;
    if (!unit) return;
    try {
      const r = await api(`api/sys/logs/${encodeURIComponent(unit)}?lines=120`);
      $("logs").textContent = (r.entries || []).join("\n") || "(空)";
    } catch (e) {
      $("logs").textContent = "加载失败: " + e.message;
    }
  }

  function stat(k, v, extra) {
    return `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div>${extra || ""}</div>`;
  }
  function meter(pct) {
    const cls = pct > 88 ? "bad" : pct > 70 ? "warn" : "";
    return `<div class="bar-meter"><i class="${cls}" style="width:${Math.min(100, pct)}%"></i></div>`;
  }
  function fmtUptime(s) {
    if (!s) return "—";
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------- approvals ----------
  async function refreshApprovals() {
    try {
      const s = state.session ? await api(`api/session/${state.session}`) : null;
      const pending = (s && s.pending_approvals) || [];
      const list = $("appr-list");
      list.innerHTML = "";
      $("appr-empty").hidden = pending.length > 0;
      $("appr-badge").hidden = pending.length === 0;
      $("appr-badge").textContent = pending.length;
      pending.forEach((p) => {
        const kind = (p.command || "").startsWith("[pty") ? "info" : (p.level || "confirm");
        const el = document.createElement("div");
        el.className = "appr " + (kind === "block" ? "block" : "");
        el.innerHTML = `<div class="why">${escapeHtml((p.reasons || []).join("; ") || p.level || "")}</div>
          <div class="cmd">${escapeHtml(p.command || "")}</div>
          <div class="acts"><button class="approve">允许</button><button class="deny">拒绝</button></div>`;
        const [ok, no] = el.querySelectorAll("button");
        ok.onclick = () => decide(p.token, true, el);
        no.onclick = () => decide(p.token, false, el);
        list.appendChild(el);
      });
    } catch (e) {
      /* no session yet is not an error */
    }
  }

  async function decide(token, approved, el) {
    el.remove();
    try {
      await api("api/approve", { method: "POST", body: JSON.stringify({ token, approved }) });
      toast(approved ? "已允许" : "已拒绝");
    } catch (e) {
      toast("审批失败: " + e.message, true);
    }
  }

  // ---------- wiring ----------
  function init() {
    $("tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-view]");
      if (!b) return;
      document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
      document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === "view-" + b.dataset.view));
      if (b.dataset.view === "ops") { refreshOps(); loadLogs(); }
      if (b.dataset.view === "screen") checkVision();
      if (b.dataset.view === "terminal" && state.term) setTimeout(() => state.fit.fit(), 30);
    });
    $("btn-run").onclick = runCouncil;
    $("btn-pty-open").onclick = openPty;
    $("btn-grab").onclick = grabFrame;
    $("btn-refresh").onclick = () => { refreshState(); refreshOps(); };
    $("btn-auto").onclick = () => {
      state.auto = !state.auto;
      $("btn-auto").textContent = state.auto ? "停止自动" : "自动刷新";
      clearInterval(state._t);
      if (state.auto) state._t = setInterval(grabFrame, 5000);
    };
    $("log-unit").onchange = loadLogs;

    bootstrap().then(() => { refreshState(); refreshOps(); });
    setInterval(refreshApprovals, 4000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
