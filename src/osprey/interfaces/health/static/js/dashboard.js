// @ts-check
/*
 * System Health dashboard — render + polling entry module.
 *
 * Consumes the P1 health envelope served by GET /checks (CheckReport.to_dict()
 * plus the P2 envelope keys stale / warming / interval_s / title) and paints the
 * static skeleton in index.html: the SVG ring hero, per-status summary badges,
 * per-category cards with a stacked status bar and LED check rows, latency
 * badges, expandable details, and single-toggle status filters.
 *
 * Contract with dashboard.css (inline color literals are banned here): every
 * status maps to a CSS modifier class the stylesheet colors — ok→s-ok,
 * warning→s-wn, error→s-er, skip→s-sk (plus led-<s> for LEDs and b-<s> for the
 * stacked bar). The JS only toggles classes; it never sets a color.
 *
 * Theme is initialized in index.html's <head> (initTheme follower +
 * applyEmbedded) — deliberately not duplicated here.
 */

import { el, fmtName, fmtMs, msCls, worst, byCategory } from "./helpers.js";

/**
 * @typedef {Object} CheckResult
 * @property {string} name
 * @property {string} category
 * @property {string} status
 * @property {string} message
 * @property {string} [value]
 * @property {number} [latency_ms]
 * @property {string} [details]
 */

/**
 * @typedef {Object} Envelope
 * @property {string} summary
 * @property {number} ok
 * @property {number} warnings
 * @property {number} errors
 * @property {number} skips
 * @property {number} total
 * @property {number} elapsed_ms
 * @property {boolean} deadline_hit
 * @property {CheckResult[]} results
 * @property {boolean} stale
 * @property {boolean} warming
 * @property {number} interval_s
 * @property {string} title
 */

const RING_R = 80;
const RING_C = 2 * Math.PI * RING_R;
const RING_GAP = 4;
const DEFAULT_INTERVAL_S = 60;
const WARMING_REPOLL_S = 3;

// An on_demand category the web suite did not run surfaces as skip rows whose
// message begins with this prefix (see runner._on_demand_hint). It is the only
// wire signal that a category is on_demand — the CheckResult shape carries no
// cost field — so the informational-card rendering keys off it.
const ON_DEMAND_HINT = "on_demand category";

/** @type {Record<string, string>} */
const STATUS_CLS = { ok: "s-ok", warning: "s-wn", error: "s-er", skip: "s-sk" };
/** @type {Record<string, string>} */
const LED_CLS = { ok: "led-ok", warning: "led-wn", error: "led-er", skip: "led-sk" };

/** @type {Envelope | null} */
let data = null;
let busy = false;
let cd = DEFAULT_INTERVAL_S;
/** @type {ReturnType<typeof setInterval> | undefined} */
let tmr;
/** @type {string | null} */
let filt = null;

/**
 * Resolve a required skeleton element, throwing if the bundle drifted out of
 * sync with index.html (a loud failure beats silently rendering nothing).
 *
 * @param {string} id
 * @returns {HTMLElement}
 */
function must(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error("missing element: " + id);
  return node;
}

/** @param {string} status @returns {string} */
function statusCls(status) {
  return STATUS_CLS[status] || "s-sk";
}

/**
 * Replace a node's class with its base plus the status modifier.
 *
 * @param {HTMLElement} node
 * @param {string} base
 * @param {string} status
 */
function setStatusClass(node, base, status) {
  node.setAttribute("class", base + " " + statusCls(status));
}

/** @param {string} text */
function copyText(text) {
  try {
    if (navigator.clipboard) navigator.clipboard.writeText(text);
  } catch {
    /* clipboard unavailable (insecure context) — silently ignore */
  }
}

// -- row / card builders -----------------------------------------------------

/**
 * @param {CheckResult} ck
 * @returns {{ row: HTMLElement, detail: HTMLElement | null }}
 */
function buildCheckRow(ck) {
  const row = el("div", { class: "ck" });
  row.appendChild(el("span", { class: "led " + (LED_CLS[ck.status] || "led-sk") }));
  row.appendChild(
    el("span", { class: "ck-nm" + (ck.status === "skip" ? " sk" : ""), text: fmtName(ck.name) }),
  );
  if (ck.value) row.appendChild(el("span", { class: "ck-val", text: ck.value }));

  const cls = msCls(ck.latency_ms);
  row.appendChild(el("span", { class: "ck-ms" + (cls ? " " + cls : ""), text: fmtMs(ck.latency_ms) }));

  // Expandable text: prefer explicit details; otherwise, for a row with no
  // value, fall back to its message so an informational row (e.g. the
  // restart-notice: name=control_system, no value/details, message IS the
  // payload) carries its text instead of dropping it. Value-bearing rows are
  // untouched (ALS fidelity).
  const extra = ck.details || (ck.value ? "" : ck.message);
  /** @type {HTMLElement | null} */
  let detail = null;
  if (extra) {
    const btn = el("button", { class: "dt-btn", text: "▸" });
    // Accent the box by the row's status (see dashboard.css .dt-box.s-*):
    // error→red, warning→amber, ok/skip→neutral.
    const box = el("div", { class: "dt-box " + statusCls(ck.status), text: extra });
    btn.addEventListener("click", () => {
      box.classList.toggle("show");
      btn.classList.toggle("open");
    });
    row.appendChild(btn);
    detail = box;
  }
  return { row, detail };
}

/**
 * @param {HTMLElement} bar
 * @param {number} n
 * @param {number} total
 * @param {string} cls
 */
function appendBar(bar, n, total, cls) {
  if (!n) return;
  const seg = el("div", { class: cls });
  seg.style.width = (n / total) * 100 + "%";
  bar.appendChild(seg);
}

/**
 * A poll-class category card: status-accented header, stacked ok/warn/err/skip
 * bar, and one LED row per (filter-passing) check.
 *
 * @param {string} cat
 * @param {CheckResult[]} checks
 * @param {number} delay
 * @returns {HTMLElement}
 */
function buildCard(cat, checks, delay) {
  const w = worst(checks);
  let ok = 0;
  let wn = 0;
  let er = 0;
  let sk = 0;
  for (const c of checks) {
    if (c.status === "ok") ok++;
    else if (c.status === "warning") wn++;
    else if (c.status === "error") er++;
    else sk++;
  }
  const total = checks.length;

  const card = el("div", {
    class: "card" + (w === "error" ? " c-err" : w === "warning" ? " c-wrn" : ""),
  });
  card.style.animationDelay = delay + "ms";

  const hd = el("div", { class: "c-hd " + statusCls(w) });
  hd.appendChild(el("span", { class: "c-nm", text: cat.replace(/_/g, " ") }));
  hd.appendChild(el("span", { class: "c-ct", text: ok + "/" + total }));
  card.appendChild(hd);

  const bar = el("div", { class: "c-bar" });
  appendBar(bar, ok, total, "b-ok");
  appendBar(bar, wn, total, "b-wn");
  appendBar(bar, er, total, "b-er");
  appendBar(bar, sk, total, "b-sk");
  card.appendChild(bar);

  const body = el("div", { class: "c-bd" });
  for (const ck of checks) {
    if (filt && ck.status !== filt) continue;
    const { row, detail } = buildCheckRow(ck);
    body.appendChild(row);
    if (detail) body.appendChild(detail);
  }
  card.appendChild(body);
  return card;
}

/**
 * An on_demand category renders as an informational card (no run buttons): the
 * web never executes these, so it offers the copyable CLI command instead.
 *
 * @param {string} cat
 * @returns {HTMLElement}
 */
function buildOnDemandCard(cat) {
  const card = el("div", { class: "card" });

  const hd = el("div", { class: "c-hd s-sk" });
  hd.appendChild(el("span", { class: "c-nm", text: cat.replace(/_/g, " ") }));
  hd.appendChild(el("span", { class: "c-ct", text: "on demand" }));
  card.appendChild(hd);

  const body = el("div", { class: "c-bd" });
  const info = el("div", { class: "ck" });
  info.appendChild(el("span", { class: "led led-sk" }));
  info.appendChild(el("span", { class: "ck-nm sk", text: "Runs on demand — not polled here." }));
  body.appendChild(info);

  const cmd = "osprey health --full --category " + cat;
  const cmdRow = el("div", { class: "ck", title: "Click to copy" });
  cmdRow.style.cursor = "pointer";
  cmdRow.appendChild(el("span", { class: "ck-nm", text: "Copy CLI command" }));
  cmdRow.appendChild(el("span", { class: "ck-val", text: cmd }));
  cmdRow.addEventListener("click", () => copyText(cmd));
  body.appendChild(cmdRow);

  card.appendChild(body);
  return card;
}

/**
 * @param {CheckResult[]} checks
 * @returns {boolean}
 */
function isOnDemand(checks) {
  return (
    checks.length > 0 &&
    checks.every((c) => c.status === "skip" && (c.message || "").indexOf(ON_DEMAND_HINT) === 0)
  );
}

// -- summary / ring / grid ---------------------------------------------------

/**
 * @param {string} label
 * @param {number} count
 * @param {string} status
 * @returns {HTMLElement}
 */
function buildBadge(label, count, status) {
  const badge = el("div", { class: "badge" + (filt === status ? " on" : "") });
  badge.setAttribute("data-s", status);
  badge.appendChild(el("span", { class: "dot " + statusCls(status) }));
  badge.appendChild(el("span", { text: count + " " + label }));
  badge.addEventListener("click", () => {
    filt = filt === status ? null : status;
    if (data) {
      renderSumm(data);
      renderGrid(data);
    }
  });
  return badge;
}

/** @param {Envelope} d */
function renderSumm(d) {
  const stats = [
    { l: "OK", c: d.ok, s: "ok" },
    { l: "WARN", c: d.warnings, s: "warning" },
    { l: "ERR", c: d.errors, s: "error" },
  ];
  if (d.skips > 0) stats.push({ l: "SKIP", c: d.skips, s: "skip" });

  const container = must("summ");
  container.textContent = "";
  for (const s of stats) container.appendChild(buildBadge(s.l, s.c, s.s));
}

/** @param {Envelope} d */
function renderRing(d) {
  const svg = must("ring");
  for (const seg of svg.querySelectorAll(".ring-seg")) seg.remove();

  const beam = must("beam");
  const rsc = must("rsc");
  must("rok").textContent = String(d.ok);
  must("rtot").textContent = "/" + d.total;

  const tot = d.results.length;
  if (tot === 0) {
    setStatusClass(rsc, "ring-sc", "skip");
    setStatusClass(beam, "beam", "ok");
    return;
  }

  let off = 0;
  byCategory(d.results).forEach((checks) => {
    const arc = (checks.length / tot) * RING_C;
    const vis = Math.max(arc - RING_GAP, 2);
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("class", "ring-seg " + statusCls(worst(checks)));
    c.setAttribute("cx", "100");
    c.setAttribute("cy", "100");
    c.setAttribute("r", String(RING_R));
    c.setAttribute("stroke-dasharray", vis + " " + (RING_C - vis));
    c.setAttribute("stroke-dashoffset", String(-(off + RING_GAP / 2)));
    c.setAttribute("transform", "rotate(-90 100 100)");
    c.setAttribute("filter", "url(#glow)");
    svg.insertBefore(c, beam);
    off += arc;
  });

  const ov = worst(d.results);
  setStatusClass(rsc, "ring-sc", ov);
  setStatusClass(beam, "beam", ov === "error" ? "error" : ov === "warning" ? "warning" : "ok");
}

/** @param {Envelope} d */
function renderGrid(d) {
  const grid = must("grid");
  grid.textContent = "";
  let delay = 0;
  let count = 0;
  byCategory(d.results).forEach((checks, cat) => {
    if (filt && !checks.some((c) => c.status === filt)) return;
    grid.appendChild(isOnDemand(checks) ? buildOnDemandCard(cat) : buildCard(cat, checks, delay));
    delay += 40;
    count++;
  });
  if (count === 0) {
    const msg = el("div", { class: "ld" });
    msg.appendChild(el("span", { text: "No checks match filter" }));
    grid.appendChild(msg);
  }
}

/** The cold, no-cache state: a first scan is running; suppress stale chrome. */
function renderWarming() {
  const svg = must("ring");
  for (const seg of svg.querySelectorAll(".ring-seg")) seg.remove();
  must("rok").textContent = "--";
  must("rtot").textContent = "";
  setStatusClass(must("rsc"), "ring-sc", "skip");
  must("summ").textContent = "";

  const grid = must("grid");
  grid.textContent = "";
  const msg = el("div", { class: "ld" });
  msg.appendChild(el("div", { class: "ld-spin" }));
  msg.appendChild(el("span", { text: "First scan in progress…" }));
  grid.appendChild(msg);
}

/** @param {string} message */
function renderError(message) {
  const grid = must("grid");
  grid.textContent = "";
  const wrap = el("div", { class: "err-s" });
  wrap.appendChild(el("div", { class: "em", text: "Cannot reach checks endpoint" }));
  wrap.appendChild(el("div", { text: message }));
  wrap.appendChild(document.createElement("br"));
  const btn = el("button", { text: "Retry" });
  btn.addEventListener("click", doRefresh);
  wrap.appendChild(btn);
  grid.appendChild(wrap);
}

// -- envelope application + polling ------------------------------------------

/** @param {Envelope} d */
function applyEnvelope(d) {
  data = d;
  const title = d.title || "System Health";
  must("ttl").textContent = title;
  document.title = title;

  if (d.warming) {
    renderWarming();
  } else {
    renderRing(d);
    renderSumm(d);
    renderGrid(d);
  }

  const ts = must("ts");
  ts.textContent = new Date().toLocaleTimeString();
  ts.setAttribute("title", d.elapsed_ms > 0 ? "scan " + fmtMs(d.elapsed_ms) : "");

  // Footer status line: warming suppresses the staleness note (rendering rule);
  // a truthful stale flag surfaces otherwise, and a hit suite deadline is noted.
  let status = d.summary;
  if (d.warming) status = "First scan in progress…";
  else if (d.stale) status = "Data may be stale · " + d.summary;
  if (!d.warming && d.deadline_hit) status += " · deadline hit";
  must("stxt").textContent = status;
}

function doRefresh() {
  if (busy) return;
  busy = true;
  must("sw").classList.add("on");
  must("refBtn").classList.add("spin");

  fetchData()
    .then((d) => {
      applyEnvelope(d);
      resetCd(d.warming ? WARMING_REPOLL_S : Math.max(1, Math.round(d.interval_s || DEFAULT_INTERVAL_S)));
    })
    .catch((e) => {
      // Keep the last good render on a transient error; only the cold path
      // (no prior data) shows the error panel.
      if (!data) renderError(e instanceof Error ? e.message : String(e));
      resetCd(DEFAULT_INTERVAL_S);
    })
    .finally(() => {
      busy = false;
      must("sw").classList.remove("on");
      must("refBtn").classList.remove("spin");
    });
}

/** @returns {Promise<Envelope>} */
function fetchData() {
  const params = new URLSearchParams(window.location.search);
  if (params.has("demo")) return Promise.resolve(demoData(params.get("demo") || ""));
  return fetch("/checks").then((r) => {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  });
}

/** @param {number} seconds */
function resetCd(seconds) {
  cd = seconds;
  must("cd").textContent = String(cd);
  if (tmr !== undefined) clearInterval(tmr);
  tmr = setInterval(() => {
    cd--;
    must("cd").textContent = String(cd);
    if (cd <= 0) doRefresh();
  }, 1000);
}

// -- demo fixture ------------------------------------------------------------

/** @type {CheckResult[]} */
const DEMO_RESULTS = [
  { name: "configuration.required_settings", category: "configuration", status: "ok", message: "All required settings present" },
  { name: "configuration.connector_selected", category: "configuration", status: "ok", message: "Connector selected", value: "epics" },
  { name: "control_system", category: "configuration", status: "warning", message: "control_system config changed; restart the web terminal to apply" },
  { name: "file_system.workspace_writable", category: "file_system", status: "ok", message: "Workspace writable", value: "12 GB free", latency_ms: 4 },
  { name: "file_system.scratch_writable", category: "file_system", status: "ok", message: "Scratch dir writable", latency_ms: 3 },
  { name: "providers.anthropic", category: "providers", status: "ok", message: "Anthropic reachable", latency_ms: 88 },
  { name: "providers.cborg", category: "providers", status: "warning", message: "CBORG slow response", latency_ms: 2140 },
  { name: "control_system.beam_current", category: "control_system", status: "ok", message: "Beam current", value: "401.2 mA", latency_ms: 12 },
  { name: "control_system.rf_frequency", category: "control_system", status: "ok", message: "RF frequency", value: "499.654 MHz", latency_ms: 9 },
  { name: "control_system.corrector_write", category: "control_system", status: "error", message: "Corrector setpoint unreachable", details: "caput SR:C01:HCM:SP timed out after 5s\nIOC may be offline" },
  { name: "web.dashboard_http", category: "web", status: "ok", message: "Dashboard endpoint reachable", value: "200", latency_ms: 15 },
  { name: "claude_cli_pinned", category: "claude_cli_pinned", status: "skip", message: "on_demand category; run with --full (e.g. `osprey health --full --category claude_cli_pinned`)" },
  { name: "model_chat", category: "model_chat", status: "skip", message: "on_demand category; run with --full (e.g. `osprey health --full --category model_chat`)" },
];

/**
 * @param {CheckResult[]} results
 * @param {Partial<Envelope>} [extra]
 * @returns {Envelope}
 */
function demoEnvelope(results, extra) {
  const count = (/** @type {string} */ s) => results.filter((r) => r.status === s).length;
  const ok = count("ok");
  const warnings = count("warning");
  const errors = count("error");
  const skips = count("skip");
  const total = results.length;

  /** @type {string[]} */
  const parts = [];
  if (warnings) parts.push(warnings + " warning" + (warnings !== 1 ? "s" : ""));
  if (errors) parts.push(errors + " error" + (errors !== 1 ? "s" : ""));
  if (skips) parts.push(skips + " skipped");
  const summary =
    ok + "/" + total + " checks passed" + (parts.length ? " (" + parts.join(", ") + ")" : "");

  return {
    summary,
    ok,
    warnings,
    errors,
    skips,
    total,
    elapsed_ms: 1240,
    deadline_hit: false,
    results,
    stale: false,
    warming: false,
    interval_s: DEFAULT_INTERVAL_S,
    title: "System Health",
    ...extra,
  };
}

/**
 * Fixtures for local review without a backend. `?demo` is the healthy-ish
 * snapshot; `?demo=stale` and `?demo=warming` exercise the new envelope states.
 *
 * @param {string} variant
 * @returns {Envelope}
 */
function demoData(variant) {
  if (variant === "warming") {
    return demoEnvelope([], { warming: true, stale: true });
  }
  if (variant === "stale") {
    return demoEnvelope(DEMO_RESULTS, { stale: true, deadline_hit: true });
  }
  return demoEnvelope(DEMO_RESULTS);
}

// -- boot (module is deferred; the skeleton is already parsed) ---------------

must("refBtn").addEventListener("click", doRefresh);
doRefresh();
