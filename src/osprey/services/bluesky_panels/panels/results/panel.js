// @ts-check
/**
 * OSPREY Scan Panels — live results panel (Phase-6 task 2.2, panel-results).
 *
 * Read-only. Lets an operator pick a scan run (from `GET /runs`, or an
 * initial `?run_id=` deep link) and follows it: polls `GET /runs/{id}` for
 * status/`tiled_degraded`/`completion` and `GET /runs/{id}/data` for a
 * bounded rows/columns window, both through the sidecar's same-origin
 * read-proxy — never a write verb.
 *
 * API base: every fetch is prefix-relative, derived from the panel's own
 * mount path (`/panel/<id>/...`) rather than an absolute origin, so the
 * panel works unmodified whether it's opened directly or embedded through
 * the web-terminal reverse proxy.
 *
 * Poll cadence: ~1s while the run is non-terminal (`pending`/`running`) or
 * the data response reports `partial: true`; polling stops once the run
 * reaches a terminal status (`completed`/`stopped`/`error`) and its data is
 * no longer partial.
 *
 * Chart: an inline SVG line chart, one polyline per numeric column,
 * min-max normalized per column and plotted against row order (a generic
 * stand-in for an orbit-response overlay — this panel has no knowledge of
 * which column is a "position" vs a "signal"). Every color comes from the
 * theme-manager computed-style bridges (`chartTheme()`/`chartSeries()`),
 * re-read inside a `subscribe()` callback so the chart re-themes live.
 *
 * Graceful degradation: no run selected, an unreachable API, or an empty
 * table/chart all render a clean empty state rather than throwing — this
 * panel is also served standalone with no bridge behind it (the visual
 * regression harness does exactly that).
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { chartSeries, chartTheme, subscribe } from '/design-system/js/theme-manager.js';

/** @typedef {{
 *   id: string,
 *   status: string,
 *   tiled_degraded?: boolean,
 *   completion?: number,
 *   launched_by?: string,
 *   run_uid?: string,
 *   error?: string,
 *   plan_name?: string,
 * }} RunStatus */

/** @typedef {{
 *   run_uid?: string|null,
 *   columns: string[],
 *   rows: Array<Array<unknown>>,
 *   row_count: number,
 *   truncated: boolean,
 *   partial?: boolean,
 * }} RunData */

/** @typedef {{ok: true, data: RunStatus} | {ok: false, notFound: boolean, message: string}} StatusFetch */
/** @typedef {{ok: true, data: RunData} | {ok: false, notFound: boolean, notStarted: boolean, message: string}} DataFetch */

// ---- API base resolution (prefix-relative fetches only) ----

const PREFIX = (() => {
  const match = window.location.pathname.match(/^\/panel\/[^/]+/);
  return match ? match[0] : '';
})();

/**
 * @param {string} path
 * @returns {string}
 */
function api(path) {
  return `${PREFIX}${path}`;
}

const POLL_INTERVAL_MS = 1000;
const TERMINAL_STATUSES = new Set(['completed', 'stopped', 'error']);

const CHART_WIDTH = 640;
const CHART_HEIGHT = 220;
const CHART_PADDING = 24;
const SVG_NS = 'http://www.w3.org/2000/svg';

// ---- DOM refs ----

const runSelect = /** @type {HTMLSelectElement} */ (document.getElementById('run-select'));
const runStatusBadge = /** @type {HTMLElement} */ (document.getElementById('run-status-badge'));
const runMeta = /** @type {HTMLElement} */ (document.getElementById('run-meta'));
const degradedBanner = /** @type {HTMLElement} */ (document.getElementById('degraded-banner'));
const emptyState = /** @type {HTMLElement} */ (document.getElementById('empty-state'));
const tableCard = /** @type {HTMLElement} */ (document.getElementById('table-card'));
const tableNote = /** @type {HTMLElement} */ (document.getElementById('table-note'));
const resultsTable = /** @type {HTMLTableElement} */ (document.getElementById('results-table'));
const tableHeadRow = /** @type {HTMLTableRowElement} */ (resultsTable.querySelector('thead tr'));
const tableBody = /** @type {HTMLTableSectionElement} */ (resultsTable.querySelector('tbody'));
const chartCard = /** @type {HTMLElement} */ (document.getElementById('chart-card'));
const chartSvg = /** @type {SVGSVGElement} */ (
  /** @type {unknown} */ (document.getElementById('results-chart'))
);
const chartLegend = /** @type {HTMLElement} */ (document.getElementById('chart-legend'));

/** @type {{runId: string|null, timer: ReturnType<typeof setTimeout>|null, lastData: RunData|null}} */
const state = { runId: null, timer: null, lastData: null };

// ---- Small visibility helpers ----

/** @param {HTMLElement} el */
function show(el) {
  el.hidden = false;
}

/** @param {HTMLElement} el */
function hide(el) {
  el.hidden = true;
}

/**
 * @param {string|null} message Pass `null` to hide the empty state entirely.
 * @param {boolean} [isError]
 */
function setEmptyState(message, isError = false) {
  if (message === null) {
    hide(emptyState);
    return;
  }
  emptyState.textContent = message;
  emptyState.classList.toggle('error', isError);
  show(emptyState);
}

// ---- Run list ----

/** @returns {Promise<boolean>} */
async function loadRunList() {
  try {
    const response = await fetch(api('/runs'));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const runs = await response.json();
    populateRunSelect(Array.isArray(runs) ? runs : []);
    return true;
  } catch {
    setEmptyState('Unable to reach the scan panels API for the run list.', true);
    return false;
  }
}

/** @param {RunStatus[]} runs */
function populateRunSelect(runs) {
  const initialRunId = state.runId;
  runSelect.innerHTML = '';

  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '— choose a run —';
  runSelect.appendChild(placeholder);

  const seen = new Set();
  for (const run of runs) {
    const option = document.createElement('option');
    option.value = run.id;
    option.textContent = `${run.id.slice(0, 8)}… — ${run.status}`;
    option.title = run.id;
    runSelect.appendChild(option);
    seen.add(run.id);
  }

  // A `?run_id=` deep link not present in the bounded, newest-first list is
  // still honored — add it as a synthetic option rather than dropping it.
  if (initialRunId && !seen.has(initialRunId)) {
    const option = document.createElement('option');
    option.value = initialRunId;
    option.textContent = `${initialRunId.slice(0, 8)}… (from link)`;
    option.title = initialRunId;
    runSelect.appendChild(option);
  }

  if (initialRunId) {
    runSelect.value = initialRunId;
  }
}

// ---- Fetch helpers (never throw — every failure becomes a discriminated result) ----

/**
 * @param {string} runId
 * @returns {Promise<StatusFetch>}
 */
async function fetchRunStatus(runId) {
  try {
    const response = await fetch(api(`/runs/${encodeURIComponent(runId)}`));
    if (response.status === 404) {
      return { ok: false, notFound: true, message: 'run not found' };
    }
    if (!response.ok) {
      return { ok: false, notFound: false, message: `status HTTP ${response.status}` };
    }
    const data = /** @type {RunStatus} */ (await response.json());
    return { ok: true, data };
  } catch {
    return { ok: false, notFound: false, message: 'network error' };
  }
}

/**
 * @param {string} runId
 * @returns {Promise<DataFetch>}
 */
async function fetchRunData(runId) {
  try {
    const response = await fetch(api(`/runs/${encodeURIComponent(runId)}/data`));
    if (response.status === 404) {
      return { ok: false, notFound: true, notStarted: false, message: 'run not found' };
    }
    if (response.status === 409) {
      return { ok: false, notFound: false, notStarted: true, message: 'run has not started; no data yet' };
    }
    if (!response.ok) {
      return { ok: false, notFound: false, notStarted: false, message: `data HTTP ${response.status}` };
    }
    const data = /** @type {RunData} */ (await response.json());
    return { ok: true, data };
  } catch {
    return { ok: false, notFound: false, notStarted: false, message: 'network error' };
  }
}

// ---- Selecting a run ----

/** @param {string} runId */
function selectRun(runId) {
  stopPolling();
  state.runId = runId;
  state.lastData = null;
  runSelect.value = runId;

  hide(degradedBanner);
  hide(tableCard);
  hide(chartCard);
  hide(runStatusBadge);
  hide(runMeta);

  if (!runId) {
    setEmptyState('Select a run above to view its results.');
    return;
  }
  setEmptyState(null);
  pollOnce();
}

function stopPolling() {
  if (state.timer !== null) {
    clearTimeout(state.timer);
    state.timer = null;
  }
}

/** @param {number} delayMs */
function scheduleNextPoll(delayMs) {
  stopPolling();
  state.timer = setTimeout(pollOnce, delayMs);
}

async function pollOnce() {
  const runId = state.runId;
  if (!runId) return;

  const [statusFetch, dataFetch] = await Promise.all([fetchRunStatus(runId), fetchRunData(runId)]);

  // The selected run changed while these requests were in flight — this
  // response is stale, drop it (the newer selectRun() call already issued
  // its own poll).
  if (state.runId !== runId) return;

  if (!statusFetch.ok) {
    hide(tableCard);
    hide(chartCard);
    hide(runStatusBadge);
    hide(runMeta);
    setEmptyState(`Could not load run status: ${statusFetch.message}`, true);
    if (!statusFetch.notFound) {
      // Transient status error (network hiccup, non-404 HTTP status) — keep
      // polling so a still-running run's live updates don't permanently
      // freeze. Only a definitive not-found hard-stops.
      scheduleNextPoll(POLL_INTERVAL_MS);
    }
    return;
  }

  renderRunMeta(statusFetch.data);

  if (dataFetch.ok) {
    renderTable(dataFetch.data);
    renderChart(dataFetch.data);
    state.lastData = dataFetch.data;
    setEmptyState(
      dataFetch.data.columns.length === 0 ? 'No data columns for this run yet.' : null
    );
  } else if (dataFetch.notStarted) {
    hide(tableCard);
    hide(chartCard);
    setEmptyState('Run has not started — no data yet.');
  } else if (dataFetch.notFound) {
    hide(tableCard);
    hide(chartCard);
    setEmptyState('Run data not found.', true);
  } else {
    // Transient data-fetch error — keep the last good table/chart on
    // screen rather than clearing it out from under the operator.
    setEmptyState(null);
  }

  const terminal = TERMINAL_STATUSES.has(statusFetch.data.status);
  const partial = dataFetch.ok && dataFetch.data.partial === true;
  if (!terminal || partial) {
    scheduleNextPoll(POLL_INTERVAL_MS);
  }
}

// ---- Rendering: run status/meta ----

/**
 * @param {string} status
 * @returns {string}
 */
function badgeClassForStatus(status) {
  if (status === 'completed') return 'ok';
  if (status === 'running' || status === 'pending') return 'info';
  if (status === 'error') return 'err';
  return 'warn'; // stopped
}

/** @param {RunStatus} run */
function renderRunMeta(run) {
  runStatusBadge.textContent = run.status;
  runStatusBadge.className = `badge ${badgeClassForStatus(run.status)}`;
  show(runStatusBadge);

  const parts = [`<strong>status</strong> ${escapeHtml(run.status)}`];
  if (typeof run.completion === 'number') {
    parts.push(`<strong>completion</strong> ${escapeHtml(Math.round(run.completion * 100))}%`);
  }
  if (run.launched_by) {
    parts.push(`<strong>launched by</strong> ${escapeHtml(run.launched_by)}`);
  }
  if (run.run_uid) {
    parts.push(`<strong>run uid</strong> ${escapeHtml(run.run_uid)}`);
  }
  if (run.error) {
    parts.push(`<strong>error</strong> ${escapeHtml(run.error)}`);
  }
  runMeta.innerHTML = parts.join(' &nbsp;&middot;&nbsp; ');
  show(runMeta);

  if (run.tiled_degraded) {
    show(degradedBanner);
  } else {
    hide(degradedBanner);
  }
}

// ---- Rendering: bounded data table ----

/**
 * @param {unknown} value
 * @returns {string}
 */
function formatCell(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(Number(value.toFixed(6))) : String(value);
  }
  return String(value);
}

/** @param {RunData} data */
function renderTable(data) {
  tableHeadRow.innerHTML = '';
  tableBody.innerHTML = '';

  for (const column of data.columns) {
    const th = document.createElement('th');
    th.textContent = column;
    tableHeadRow.appendChild(th);
  }

  for (const row of data.rows) {
    const tr = document.createElement('tr');
    for (const value of row) {
      const td = document.createElement('td');
      td.textContent = formatCell(value);
      tr.appendChild(td);
    }
    tableBody.appendChild(tr);
  }

  // row_count is the bridge's true total; never recomputed from rows.length.
  tableNote.textContent = data.truncated
    ? `showing ${data.rows.length} of ${data.row_count} (truncated)`
    : `${data.row_count} row${data.row_count === 1 ? '' : 's'}`;

  if (data.columns.length === 0) {
    hide(tableCard);
  } else {
    show(tableCard);
  }
}

// ---- Rendering: token-themed inline SVG trace ----

/**
 * Columns whose value is a finite number in every row of this window —
 * the only columns plotted (non-numeric columns still appear in the table).
 * @param {RunData} data
 * @returns {number[]}
 */
function numericColumnIndices(data) {
  if (data.rows.length === 0) return [];
  const indices = [];
  for (let col = 0; col < data.columns.length; col++) {
    const allNumeric = data.rows.every(
      (row) => typeof row[col] === 'number' && Number.isFinite(row[col])
    );
    if (allNumeric) indices.push(col);
  }
  return indices;
}

/** @param {RunData} data */
function renderChart(data) {
  const columnIndices = numericColumnIndices(data);
  if (columnIndices.length === 0) {
    hide(chartCard);
    return;
  }
  show(chartCard);
  drawChart(data, columnIndices);
}

/**
 * @param {RunData} data
 * @param {number[]} columnIndices
 */
function drawChart(data, columnIndices) {
  const theme = chartTheme();
  const seriesColors = chartSeries();

  while (chartSvg.firstChild) chartSvg.removeChild(chartSvg.firstChild);
  chartLegend.innerHTML = '';

  const background = document.createElementNS(SVG_NS, 'rect');
  background.setAttribute('x', '0');
  background.setAttribute('y', '0');
  background.setAttribute('width', String(CHART_WIDTH));
  background.setAttribute('height', String(CHART_HEIGHT));
  background.setAttribute('fill', theme.plot_bgcolor || 'transparent');
  chartSvg.appendChild(background);

  const gridColor = theme.xaxis.gridcolor || theme.yaxis.gridcolor;
  for (let i = 0; i <= 4; i++) {
    const y = CHART_PADDING + (i / 4) * (CHART_HEIGHT - 2 * CHART_PADDING);
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', String(CHART_PADDING));
    line.setAttribute('x2', String(CHART_WIDTH - CHART_PADDING));
    line.setAttribute('y1', String(y));
    line.setAttribute('y2', String(y));
    line.setAttribute('stroke', gridColor || 'currentColor');
    line.setAttribute('stroke-width', '1');
    line.setAttribute('opacity', '0.5');
    chartSvg.appendChild(line);
  }

  const rowCount = data.rows.length;
  const xDenominator = Math.max(1, rowCount - 1);

  columnIndices.forEach((col, seriesIndex) => {
    const values = data.rows.map((row) => /** @type {number} */ (row[col]));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min;

    const points = values.map((value, rowIndex) => {
      const x = CHART_PADDING + (rowIndex / xDenominator) * (CHART_WIDTH - 2 * CHART_PADDING);
      const normalized = span === 0 ? 0.5 : (value - min) / span;
      const y = CHART_PADDING + (1 - normalized) * (CHART_HEIGHT - 2 * CHART_PADDING);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const color = seriesColors[seriesIndex % (seriesColors.length || 1)] || theme.font.color || 'currentColor';

    const polyline = document.createElementNS(SVG_NS, 'polyline');
    polyline.setAttribute('points', points.join(' '));
    polyline.setAttribute('fill', 'none');
    polyline.setAttribute('stroke', color);
    polyline.setAttribute('stroke-width', '2');
    polyline.setAttribute('stroke-linejoin', 'round');
    polyline.setAttribute('stroke-linecap', 'round');
    chartSvg.appendChild(polyline);

    const legendItem = document.createElement('li');
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.backgroundColor = color;
    const label = document.createElement('span');
    label.textContent = data.columns[col];
    legendItem.appendChild(swatch);
    legendItem.appendChild(label);
    chartLegend.appendChild(legendItem);
  });
}

// ---- Wiring ----

runSelect.addEventListener('change', () => {
  selectRun(runSelect.value);
});

/** @returns {string|null} */
function initialRunIdFromUrl() {
  try {
    return new URLSearchParams(window.location.search).get('run_id');
  } catch {
    return null;
  }
}

async function init() {
  // Any token-derived chart color must be re-read from getComputedStyle on
  // every theme change, not cached — subscribe() re-fires on every apply,
  // including the hidden-iframe recovery case theme-manager.js documents.
  subscribe(() => {
    if (state.lastData) renderChart(state.lastData);
  });

  const initial = initialRunIdFromUrl();
  if (initial) state.runId = initial;

  const loaded = await loadRunList();

  if (initial) {
    selectRun(initial);
  } else if (loaded) {
    setEmptyState('Select a run above to view its results.');
  }
  // else: loadRunList() already set the unreachable-API empty state.
}

init();
