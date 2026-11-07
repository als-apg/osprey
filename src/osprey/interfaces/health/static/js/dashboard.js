// @ts-check
/*
 * System Health dashboard — render + polling entry module.
 *
 * Consumes the P1 health envelope served by GET /checks (CheckReport.to_dict()
 * plus the P2 envelope keys stale / warming / interval_s / title) and paints the
 * static skeleton in index.html: the summary card (a static SVG gauge and the
 * passed count), per-category cards with a stacked status bar and LED check
 * rows, latency badges and expandable details. The panel chrome (the toolbar's
 * status filter and Scan now icon, the categories sidebar, the tile ⋯ entries)
 * is built by chrome.js; this module owns the data and the actions.
 *
 * Contract with dashboard.css (inline color literals are banned here): every
 * status maps to a CSS modifier class the stylesheet colors — ok→s-ok,
 * warning→s-wn, error→s-er, skip→s-sk (plus led-<s> for LEDs and b-<s> for the
 * stacked bar). The JS only toggles classes; it never sets a color.
 *
 * Theme is initialized in index.html's <head> (initTheme follower +
 * applyEmbedded) — deliberately not duplicated here.
 */

import { copyText } from '/design-system/js/clipboard.js';
import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';
import { clearState, renderState } from '/design-system/js/panel-state.js';
import {
  el,
  fmtName,
  fmtMs,
  msCls,
  worstStatus,
  byCategory,
  detailText,
  detailKey,
} from './helpers.js';
import {
  MENU_ENTRY,
  cliCommand,
  cliCommands,
  contributeMenu,
  countsText,
  initFilter,
  initScanButton,
  onMenuAction,
  renderCategories,
  summaryText,
} from './chrome.js';
import { demoData } from './demo.js';

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

/** @typedef {{cat: string, checks: CheckResult[], onDemand: boolean}} Group */

const GAUGE_R = 80;
const GAUGE_C = 2 * Math.PI * GAUGE_R;
const GAUGE_GAP = 4;
const DEFAULT_INTERVAL_S = 60;
const WARMING_REPOLL_S = 3;

// An on_demand category the web suite did not run surfaces as skip rows whose
// message begins with this prefix (see runner._on_demand_hint). It is the only
// wire signal that a category is on_demand — the CheckResult shape carries no
// cost field — so the informational-card rendering keys off it.
const ON_DEMAND_HINT = 'on_demand category';

/** @type {Record<string, string>} */
const STATUS_CLS = { ok: 's-ok', warning: 's-wn', error: 's-er', skip: 's-sk' };
/** @type {Record<string, string>} */
const LED_CLS = { ok: 'led-ok', warning: 'led-wn', error: 'led-er', skip: 'led-sk' };

/** @type {Envelope | null} */
let data = null;
let busy = false;
let cd = DEFAULT_INTERVAL_S;
/** @type {ReturnType<typeof setInterval> | undefined} */
let tmr;
/** @type {string | null} */
let filt = null;
/** @type {string | null} */
let selectedCat = null;
/** The detail keys (helpers.detailKey) of the open detail boxes. */
const openDetails = new Set();

/**
 * Resolve a required skeleton element, throwing if the bundle drifted out of
 * sync with index.html (a loud failure beats silently rendering nothing).
 *
 * @param {string} id
 * @returns {HTMLElement}
 */
function must(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error('missing element: ' + id);
  return node;
}

const grid = must('grid');
const hero = must('hero');
const stateHost = must('state');
const categories = must('categories');

/** @param {string} status @returns {string} */
function statusCls(status) {
  return STATUS_CLS[status] || 's-sk';
}

/**
 * Replace a node's class with its base plus the status modifier.
 *
 * @param {HTMLElement} node
 * @param {string} base
 * @param {string} status
 */
function setStatusClass(node, base, status) {
  node.setAttribute('class', base + ' ' + statusCls(status));
}

/**
 * @param {CheckResult[]} checks
 * @returns {boolean}
 */
function isOnDemand(checks) {
  return (
    checks.length > 0 &&
    checks.every((c) => c.status === 'skip' && (c.message || '').indexOf(ON_DEMAND_HINT) === 0)
  );
}

/**
 * @param {Envelope} d
 * @returns {Group[]}
 */
function groupsOf(d) {
  /** @type {Group[]} */
  const groups = [];
  byCategory(d.results).forEach((checks, cat) =>
    groups.push({ cat, checks, onDemand: isOnDemand(checks) })
  );
  return groups;
}

// -- row / card builders -----------------------------------------------------

/**
 * @param {CheckResult} ck
 * @returns {{ row: HTMLElement, detail: HTMLElement | null }}
 */
function buildCheckRow(ck) {
  const row = el('div', { class: 'ck' });
  row.appendChild(el('span', { class: 'led ' + (LED_CLS[ck.status] || 'led-sk') }));
  row.appendChild(
    el('span', {
      class: 'ck-nm' + (ck.status === 'skip' ? ' sk' : ''),
      text: fmtName(ck.name, ck.category),
    })
  );
  if (ck.value) row.appendChild(el('span', { class: 'ck-val', text: ck.value }));

  const cls = msCls(ck.latency_ms);
  row.appendChild(
    el('span', { class: 'ck-ms' + (cls ? ' ' + cls : ''), text: fmtMs(ck.latency_ms) })
  );

  const extra = detailText(ck);
  /** @type {HTMLElement | null} */
  let detail = null;
  if (extra) {
    const key = detailKey(ck);
    const open = openDetails.has(key);
    const btn = el('button', {
      type: 'button',
      class: 'dt-btn' + (open ? ' open' : ''),
      'data-detail': key,
      text: '▸',
    });
    // Accent the box by the row's status (see dashboard.css .dt-box.s-*):
    // error→red, warning→amber, ok/skip→neutral.
    const box = el('div', {
      class: 'dt-box ' + statusCls(ck.status) + (open ? ' show' : ''),
      'data-detail': key,
      text: extra,
    });
    btn.addEventListener('click', () => {
      if (openDetails.has(key)) openDetails.delete(key);
      else openDetails.add(key);
      syncDetails();
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
  const seg = el('div', { class: cls });
  seg.style.width = (n / total) * 100 + '%';
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
  const w = worstStatus(checks);
  let ok = 0;
  let wn = 0;
  let er = 0;
  let sk = 0;
  for (const c of checks) {
    if (c.status === 'ok') ok++;
    else if (c.status === 'warning') wn++;
    else if (c.status === 'error') er++;
    else sk++;
  }
  const total = checks.length;

  const card = el('div', {
    class: 'card' + (w === 'error' ? ' c-err' : w === 'warning' ? ' c-wrn' : ''),
    'data-category': cat,
  });
  card.style.animationDelay = delay + 'ms';

  const hd = el('div', { class: 'c-hd ' + statusCls(w) });
  hd.appendChild(el('span', { class: 'c-nm', text: cat.replace(/_/g, ' ') }));
  hd.appendChild(el('span', { class: 'c-ct', text: ok + '/' + total }));
  card.appendChild(hd);

  const bar = el('div', { class: 'c-bar' });
  appendBar(bar, ok, total, 'b-ok');
  appendBar(bar, wn, total, 'b-wn');
  appendBar(bar, er, total, 'b-er');
  appendBar(bar, sk, total, 'b-sk');
  card.appendChild(bar);

  const body = el('div', { class: 'c-bd' });
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
 * web never executes these, so it shows the CLI command that runs it. The tile
 * ⋯ entry "Copy on-demand CLI commands" copies every such command.
 *
 * @param {string} cat
 * @returns {HTMLElement}
 */
function buildOnDemandCard(cat) {
  const card = el('div', { class: 'card', 'data-category': cat });

  const hd = el('div', { class: 'c-hd s-sk' });
  hd.appendChild(el('span', { class: 'c-nm', text: cat.replace(/_/g, ' ') }));
  hd.appendChild(el('span', { class: 'c-ct', text: 'on demand' }));
  card.appendChild(hd);

  const body = el('div', { class: 'c-bd' });
  const info = el('div', { class: 'ck' });
  info.appendChild(el('span', { class: 'led led-sk' }));
  info.appendChild(el('span', { class: 'ck-nm sk', text: 'Runs on demand — not polled here.' }));
  body.appendChild(info);

  const cmdRow = el('div', { class: 'ck' });
  cmdRow.appendChild(el('span', { class: 'ck-nm ck-cmd', text: cliCommand(cat) }));
  body.appendChild(cmdRow);

  card.appendChild(body);
  return card;
}

// -- summary card / grid / sidebar ---------------------------------------------

/** @param {Envelope} d */
function renderGauge(d) {
  const svg = must('gauge');
  for (const seg of svg.querySelectorAll('.gauge-seg')) seg.remove();

  const rsc = must('rsc');
  rsc.textContent = String(d.ok);
  must('rtot').textContent = '/' + d.total;

  const tot = d.results.length;
  if (tot === 0) {
    setStatusClass(rsc, 'gauge-sc', 'skip');
    return;
  }

  let off = 0;
  byCategory(d.results).forEach((checks) => {
    const arc = (checks.length / tot) * GAUGE_C;
    const vis = Math.max(arc - GAUGE_GAP, 2);
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('class', 'gauge-seg ' + statusCls(worstStatus(checks)));
    c.setAttribute('cx', '100');
    c.setAttribute('cy', '100');
    c.setAttribute('r', String(GAUGE_R));
    c.setAttribute('stroke-dasharray', vis + ' ' + (GAUGE_C - vis));
    c.setAttribute('stroke-dashoffset', String(-(off + GAUGE_GAP / 2)));
    c.setAttribute('transform', 'rotate(-90 100 100)');
    svg.appendChild(c);
    off += arc;
  });

  setStatusClass(rsc, 'gauge-sc', worstStatus(d.results));
}

/**
 * The summary card's two lines. The second carries what the footer status
 * line used to: the stale flag and a hit suite deadline.
 *
 * @param {Envelope} d
 */
function renderSummary(d) {
  must('hero-title').textContent = d.ok + '/' + d.total + ' checks passed';
  const notes = [countsText(d)];
  if (d.stale) notes.push('data may be stale');
  if (d.deadline_hit) notes.push('deadline hit');
  must('hero-detail').textContent = notes.filter(Boolean).join(' · ');
}

/** @param {Envelope} d */
function renderGrid(d) {
  grid.textContent = '';
  let delay = 0;
  let count = 0;
  for (const { cat, checks, onDemand } of groupsOf(d)) {
    if (filt && !checks.some((c) => c.status === filt)) continue;
    grid.appendChild(onDemand ? buildOnDemandCard(cat) : buildCard(cat, checks, delay));
    delay += 40;
    count++;
  }
  if (count === 0) renderState(grid, { icon: 'filter', message: 'No checks match this filter' });
}

/** @param {Envelope} d */
function renderSidebar(d) {
  renderCategories(must('category-list'), groupsOf(d), selectedCat, selectCategory);
}

/** Paint the whole report for the current data and filter. */
function render() {
  if (!data || data.warming) return;
  renderGauge(data);
  renderSummary(data);
  renderGrid(data);
  renderSidebar(data);
  filter.update(data, filt);
  syncMenu();
}

/**
 * Show a centred state in place of the summary card and the cards.
 *
 * @param {Parameters<typeof renderState>[1]} spec
 * @returns {HTMLElement}
 */
function showState(spec) {
  hero.hidden = true;
  grid.hidden = true;
  stateHost.hidden = false;
  return renderState(stateHost, spec);
}

function showReport() {
  clearState(stateHost);
  stateHost.hidden = true;
  hero.hidden = false;
  grid.hidden = false;
}

/** The cold, no-cache state: a first scan is running; suppress stale chrome. */
function renderWarming() {
  filter.update(null, filt);
  must('category-list').textContent = '';
  showState({ icon: 'refresh', message: 'First scan in progress…' });
}

/** @param {string} message */
function renderError(message) {
  const root = showState({
    icon: 'info',
    message: 'Cannot reach the health checks',
    action: { label: 'Retry', onClick: doRefresh },
  });
  root.title = message;
}

// -- actions --------------------------------------------------------------------

/** @param {string | null} status */
function pickFilter(status) {
  filt = status;
  render();
}

/**
 * Jump to a category's card. A card the filter hides brings the filter back
 * to All first. On a narrow tile the overlay sidebar closes after the jump.
 *
 * @param {string} cat
 */
function selectCategory(cat) {
  selectedCat = cat;
  const find = () =>
    Array.from(grid.children).find((n) => n instanceof HTMLElement && n.dataset.category === cat);
  if (!find()) filt = null;
  render();
  const card = find();
  if (card && typeof card.scrollIntoView === 'function') {
    card.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }
  if (categories.hasAttribute('data-overlay')) sidebar.close();
}

/**
 * The detail keys of every expandable row of the report, shown or filtered.
 *
 * @returns {string[]}
 */
function detailKeys() {
  if (!data || data.warming) return [];
  /** @type {string[]} */
  const keys = [];
  for (const group of groupsOf(data)) {
    if (group.onDemand) continue;
    for (const ck of group.checks) if (detailText(ck)) keys.push(detailKey(ck));
  }
  return keys;
}

/** Mirror `openDetails` onto the shown rows, then the menu label. */
function syncDetails() {
  for (const node of grid.querySelectorAll('[data-detail]')) {
    if (!(node instanceof HTMLElement)) continue;
    const open = openDetails.has(node.dataset.detail);
    node.classList.toggle(node.classList.contains('dt-box') ? 'show' : 'open', open);
  }
  syncMenu();
}

/** Expand every detail; when all are already open, collapse them. */
function expandAll() {
  const keys = detailKeys();
  const allOpen = keys.length > 0 && keys.every((k) => openDetails.has(k));
  for (const key of keys) {
    if (allOpen) openDetails.delete(key);
    else openDetails.add(key);
  }
  syncDetails();
}

function syncMenu() {
  const report = data && !data.warming ? data : null;
  const keys = detailKeys();
  contributeMenu({
    hasData: report !== null,
    hasDetails: keys.length > 0,
    allOpen: keys.length > 0 && keys.every((k) => openDetails.has(k)),
    hasCli: report !== null && groupsOf(report).some((g) => g.onDemand),
  });
}

// -- envelope application + polling ------------------------------------------

/** @param {Envelope} d */
function applyEnvelope(d) {
  data = d;
  document.title = d.title || 'System Health';
  if (d.warming) {
    renderWarming();
    syncMenu();
    return;
  }
  showReport();
  render();
}

function doRefresh() {
  if (busy) return;
  busy = true;
  must('sw').classList.add('on');

  fetchData()
    .then((d) => {
      applyEnvelope(d);
      resetCd(
        d.warming ? WARMING_REPOLL_S : Math.max(1, Math.round(d.interval_s || DEFAULT_INTERVAL_S))
      );
    })
    .catch((e) => {
      // Keep the last good render on a transient error; only the cold path
      // (no prior data) shows the error state.
      if (!data) renderError(e instanceof Error ? e.message : String(e));
      resetCd(DEFAULT_INTERVAL_S);
    })
    .finally(() => {
      busy = false;
      must('sw').classList.remove('on');
    });
}

/** @returns {Promise<Envelope>} */
function fetchData() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('demo')) return Promise.resolve(demoData(params.get('demo') || ''));
  return fetch('/checks').then((r) => {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  });
}

/** @param {number} seconds */
function resetCd(seconds) {
  cd = seconds;
  scan.setCountdown(cd);
  clearInterval(tmr);
  tmr = setInterval(() => {
    cd--;
    scan.setCountdown(cd);
    if (cd <= 0) doRefresh();
  }, 1000);
}

// -- boot (module is deferred; the skeleton is already parsed) ---------------

// Open on a wide tile; under 720 px the component's narrow rule starts it
// closed, as an overlay.
const sidebar = initPanelSidebar({
  sidebar: categories,
  handle: must('categories-handle'),
  toggle: must('sidebar-toggle'),
  storageKey: 'osprey-health-sidebar',
  defaultOpen: true,
});
const filter = initFilter(must('filter'), pickFilter);
const scan = initScanButton(must('scan-now'), doRefresh);
onMenuAction({
  [MENU_ENTRY.expand]: expandAll,
  [MENU_ENTRY.summary]: () => {
    if (data && !data.warming) copyText(summaryText(data));
  },
  [MENU_ENTRY.cli]: () => {
    if (data && !data.warming) copyText(cliCommands(groupsOf(data)));
  },
});

showState({ icon: 'refresh', message: 'Scanning…' });
syncMenu();
doRefresh();
