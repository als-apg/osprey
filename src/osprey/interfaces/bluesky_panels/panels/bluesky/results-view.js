// @ts-check
/**
 * BLUESKY panel — the Results view.
 *
 * Follows ONE run: polls `GET /runs/{id}` for its record and
 * `GET /runs/{id}/data` for a bounded rows/columns window, both through the
 * sidecar's read proxy, and renders a table plus an inline SVG trace. Read
 * only — nothing in this file issues a write verb.
 *
 * Cadence: ~1 s while the run is non-terminal or its data still reports
 * `partial: true`; polling stops once the run is terminal and its data is
 * settled, so a finished run costs nothing.
 *
 * Two shapes of "gone" are distinguished, because they mean opposite things:
 * `GET /runs/{id}` 404s once the manager has rotated the run out of its
 * history, while `GET /runs/{id}/data` keeps serving it from Tiled long
 * afterwards. A record 404 therefore does NOT clear the results — the data is
 * shown with a note, because the durable record is the one an operator came
 * for.
 *
 * Colors come from the theme-manager computed-style bridges, re-read inside a
 * `subscribe()` callback so the chart re-themes live rather than caching a
 * resolved token value.
 */

import { chartSeries, chartTheme, subscribe } from '/design-system/js/theme-manager.js';

import { describeProgress } from './queue-client.js';

/** @typedef {{
 *   id: string,
 *   status: string,
 *   plan_name?: string|null,
 *   plan_args?: Record<string, unknown>,
 *   run_uid?: string,
 *   error?: string,
 *   progress?: any,
 * }} RunRecord */

/** @typedef {{
 *   run_uid?: string|null,
 *   columns: string[],
 *   rows: Array<Array<unknown>>,
 *   row_count: number,
 *   truncated: boolean,
 *   partial?: boolean,
 * }} RunData */

/** @typedef {{ok: true, data: RunRecord} | {ok: false, notFound: boolean, message: string}} RecordFetch */
/** @typedef {{ok: true, data: RunData} | {ok: false, notFound: boolean, notStarted: boolean, message: string}} DataFetch */

const POLL_INTERVAL_MS = 1000;
const TERMINAL_STATUSES = ['completed', 'stopped', 'error'];

const CHART_WIDTH = 640;
const CHART_HEIGHT = 220;
const CHART_PADDING = 24;
const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Badge tone for a run status. `pending`/`running` read as in-flight, a clean
 * finish as success, a stop as a warning (an operator halted it — not a
 * failure), and anything the manager could not account for as an error.
 *
 * @param {string} status
 * @returns {'ok'|'info'|'warn'|'err'}
 */
export function badgeToneForStatus(status) {
  if (status === 'completed') return 'ok';
  if (status === 'running' || status === 'pending') return 'info';
  if (status === 'error') return 'err';
  return 'warn';
}

/**
 * The run record as label/value pairs for the meta line.
 *
 * Every field is optional on the wire (a pending item has no `run_uid`, a
 * clean run has no `error`, a run the document plane knows nothing about has
 * no `progress`), so each is emitted only when present — an absent field is
 * never rendered as an empty or fabricated value.
 *
 * @param {RunRecord} run
 * @returns {Array<[string, string]>}
 */
export function runMetaParts(run) {
  /** @type {Array<[string, string]>} */
  const parts = [['status', run.status]];
  if (run.plan_name) parts.push(['plan', run.plan_name]);
  if (run.progress) parts.push(['progress', describeProgress(run.progress).label]);
  if (run.run_uid) parts.push(['run uid', run.run_uid]);
  if (run.error) parts.push(['error', run.error]);
  return parts;
}

/**
 * @param {unknown} value
 * @returns {string}
 */
export function formatCell(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(Number(value.toFixed(6))) : String(value);
  }
  return String(value);
}

/**
 * Columns whose value is a finite number in every row of this window — the
 * only columns plotted. Non-numeric columns still appear in the table.
 *
 * @param {RunData} data
 * @returns {number[]}
 */
export function numericColumnIndices(data) {
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

/**
 * Whether following `run` should keep polling.
 *
 * Partial data outranks a terminal status: the run's stop document can land
 * before the last events are recorded, and stopping the poll there would
 * freeze the table one row short of the truth.
 *
 * @param {string} status
 * @param {boolean} partial
 * @returns {boolean}
 */
export function shouldKeepPolling(status, partial) {
  return partial || !TERMINAL_STATUSES.includes(status);
}

/** @typedef {{
 *   statusBadge: HTMLElement,
 *   meta: HTMLElement,
 *   note: HTMLElement,
 *   emptyState: HTMLElement,
 *   tableCard: HTMLElement,
 *   tableNote: HTMLElement,
 *   tableHeadRow: HTMLTableRowElement,
 *   tableBody: HTMLTableSectionElement,
 *   chartCard: HTMLElement,
 *   chartSvg: SVGSVGElement,
 *   chartLegend: HTMLElement,
 * }} ResultsElements */

/**
 * @param {{
 *   api: (path: string) => string,
 *   elements: ResultsElements,
 *   onPollingChange?: (polling: boolean) => void,
 * }} deps `onPollingChange` reports whether this view is currently polling for
 *   a run — i.e. whether data is still arriving. The panel shell turns that
 *   into the Results tab's activity marker when the operator is looking at
 *   another tab. It fires only on a transition, so a caller may re-render
 *   freely from it.
 * @returns {{follow: (runId: string|null) => void, currentRunId: () => string|null, destroy: () => void}}
 */
export function createResultsView({ api, elements, onPollingChange }) {
  /** @type {{runId: string|null, timer: ReturnType<typeof setTimeout>|null, lastData: RunData|null}} */
  const state = { runId: null, timer: null, lastData: null };

  /**
   * Announce a change in whether the view is polling. Derived from the timer
   * alone — the single fact that decides it — so there is no second flag that
   * could survive a `follow()` or an early `stopPolling()` return.
   *
   * @param {boolean} polling
   */
  function notifyPolling(polling) {
    if (onPollingChange) onPollingChange(polling);
  }

  /** @param {HTMLElement} node */
  const show = (node) => {
    node.hidden = false;
  };
  /** @param {HTMLElement} node */
  const hide = (node) => {
    node.hidden = true;
  };

  /**
   * @param {string|null} message Pass `null` to hide the empty state.
   * @param {boolean} [isError]
   */
  function setEmptyState(message, isError = false) {
    if (message === null) {
      hide(elements.emptyState);
      return;
    }
    elements.emptyState.textContent = message;
    elements.emptyState.classList.toggle('error', isError);
    show(elements.emptyState);
  }

  /** @param {string|null} message */
  function setNote(message) {
    if (message === null) {
      hide(elements.note);
      return;
    }
    elements.note.textContent = message;
    show(elements.note);
  }

  /**
   * @param {string} runId
   * @returns {Promise<RecordFetch>}
   */
  async function fetchRecord(runId) {
    try {
      const response = await fetch(api(`/runs/${encodeURIComponent(runId)}`));
      if (response.status === 404) return { ok: false, notFound: true, message: 'run not found' };
      if (!response.ok) {
        return { ok: false, notFound: false, message: `record HTTP ${response.status}` };
      }
      return { ok: true, data: /** @type {RunRecord} */ (await response.json()) };
    } catch {
      return { ok: false, notFound: false, message: 'network error' };
    }
  }

  /**
   * @param {string} runId
   * @returns {Promise<DataFetch>}
   */
  async function fetchData(runId) {
    try {
      const response = await fetch(api(`/runs/${encodeURIComponent(runId)}/data`));
      if (response.status === 404) {
        return { ok: false, notFound: true, notStarted: false, message: 'run not found' };
      }
      if (response.status === 409) {
        return { ok: false, notFound: false, notStarted: true, message: 'no data yet' };
      }
      if (!response.ok) {
        return { ok: false, notFound: false, notStarted: false, message: `data HTTP ${response.status}` };
      }
      return { ok: true, data: /** @type {RunData} */ (await response.json()) };
    } catch {
      return { ok: false, notFound: false, notStarted: false, message: 'network error' };
    }
  }

  /**
   * The one place `state.timer` is written, so "is this view polling?" has a
   * single answer and the transition is reported exactly once. Rescheduling
   * (timer -> timer) is not a transition and stays silent.
   *
   * @param {ReturnType<typeof setTimeout>|null} timer
   */
  function setPollTimer(timer) {
    const wasPolling = state.timer !== null;
    if (state.timer !== null) clearTimeout(state.timer);
    state.timer = timer;
    const polling = timer !== null;
    if (polling !== wasPolling) notifyPolling(polling);
  }

  function stopPolling() {
    setPollTimer(null);
  }

  /** @param {number} delayMs */
  function scheduleNextPoll(delayMs) {
    setPollTimer(setTimeout(pollOnce, delayMs));
  }

  async function pollOnce() {
    const runId = state.runId;
    if (!runId) return;

    const [recordFetch, dataFetch] = await Promise.all([fetchRecord(runId), fetchData(runId)]);
    // The selection changed while these were in flight — drop the stale
    // response; the newer follow() already issued its own poll.
    if (state.runId !== runId) return;

    let status = 'unknown';
    if (recordFetch.ok) {
      status = recordFetch.data.status;
      renderRecord(recordFetch.data);
      setNote(null);
    } else if (recordFetch.notFound) {
      // Rotated out of the manager's history. Its data may still be durable,
      // so keep going rather than blanking the half.
      hide(elements.statusBadge);
      hide(elements.meta);
      setNote('This run is no longer in the queue manager’s history; showing its stored data.');
      status = 'completed';
    } else {
      hide(elements.statusBadge);
      hide(elements.meta);
      setNote(`Could not load the run record: ${recordFetch.message}`);
    }

    let partial = false;
    if (dataFetch.ok) {
      partial = dataFetch.data.partial === true;
      renderTable(dataFetch.data);
      renderChart(dataFetch.data);
      state.lastData = dataFetch.data;
      setEmptyState(dataFetch.data.columns.length === 0 ? 'No data columns yet.' : null);
    } else if (dataFetch.notStarted) {
      hide(elements.tableCard);
      hide(elements.chartCard);
      setEmptyState('This run has not started — no data yet.');
    } else if (dataFetch.notFound && !recordFetch.ok) {
      hide(elements.tableCard);
      hide(elements.chartCard);
      setEmptyState('No record and no data for this run.', true);
      stopPolling();
      return;
    } else if (dataFetch.notFound) {
      hide(elements.tableCard);
      hide(elements.chartCard);
      setEmptyState('No data recorded for this run yet.');
    } else {
      // Transient data error: keep the last good table/chart on screen rather
      // than clearing it out from under the operator.
      setEmptyState(null);
    }

    // `shouldKeepPolling` is the whole decision. In particular a run the
    // manager has rotated away (record 404) whose data still reports
    // `partial: true` keeps polling — the durable store is still filling in,
    // and the record's absence says nothing about that.
    if (shouldKeepPolling(status, partial)) scheduleNextPoll(POLL_INTERVAL_MS);
  }

  /** @param {RunRecord} run */
  function renderRecord(run) {
    elements.statusBadge.textContent = run.status;
    elements.statusBadge.className = `badge ${badgeToneForStatus(run.status)}`;
    show(elements.statusBadge);

    elements.meta.replaceChildren();
    for (const [label, value] of runMetaParts(run)) {
      const key = document.createElement('strong');
      key.textContent = label;
      const text = document.createElement('span');
      text.textContent = ` ${value}`;
      const part = document.createElement('span');
      part.className = 'meta-part';
      part.append(key, text);
      elements.meta.append(part);
    }
    show(elements.meta);
  }

  /** @param {RunData} data */
  function renderTable(data) {
    elements.tableHeadRow.replaceChildren();
    elements.tableBody.replaceChildren();

    for (const column of data.columns) {
      const th = document.createElement('th');
      th.textContent = column;
      elements.tableHeadRow.appendChild(th);
    }

    for (const row of data.rows) {
      const tr = document.createElement('tr');
      for (const value of row) {
        const td = document.createElement('td');
        td.textContent = formatCell(value);
        tr.appendChild(td);
      }
      elements.tableBody.appendChild(tr);
    }

    // row_count is the bridge's true total — the number of rows the run has
    // produced, which can exceed what is stored. Never recomputed from
    // rows.length.
    elements.tableNote.textContent = data.truncated
      ? `showing ${data.rows.length} of ${data.row_count} (truncated)`
      : `${data.row_count} row${data.row_count === 1 ? '' : 's'}`;

    if (data.columns.length === 0) hide(elements.tableCard);
    else show(elements.tableCard);
  }

  /** @param {RunData} data */
  function renderChart(data) {
    const columnIndices = numericColumnIndices(data);
    if (columnIndices.length === 0) {
      hide(elements.chartCard);
      return;
    }
    show(elements.chartCard);
    drawChart(data, columnIndices);
  }

  /**
   * @param {RunData} data
   * @param {number[]} columnIndices
   */
  function drawChart(data, columnIndices) {
    const theme = chartTheme();
    const seriesColors = chartSeries();
    const svg = elements.chartSvg;

    svg.replaceChildren();
    elements.chartLegend.replaceChildren();

    const background = document.createElementNS(SVG_NS, 'rect');
    background.setAttribute('x', '0');
    background.setAttribute('y', '0');
    background.setAttribute('width', String(CHART_WIDTH));
    background.setAttribute('height', String(CHART_HEIGHT));
    background.setAttribute('fill', theme.plot_bgcolor || 'transparent');
    svg.appendChild(background);

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
      svg.appendChild(line);
    }

    const xDenominator = Math.max(1, data.rows.length - 1);

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

      const color =
        seriesColors[seriesIndex % (seriesColors.length || 1)] || theme.font.color || 'currentColor';

      const polyline = document.createElementNS(SVG_NS, 'polyline');
      polyline.setAttribute('points', points.join(' '));
      polyline.setAttribute('fill', 'none');
      polyline.setAttribute('stroke', color);
      polyline.setAttribute('stroke-width', '2');
      polyline.setAttribute('stroke-linejoin', 'round');
      polyline.setAttribute('stroke-linecap', 'round');
      svg.appendChild(polyline);

      const legendItem = document.createElement('li');
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.backgroundColor = color;
      const label = document.createElement('span');
      label.textContent = data.columns[col];
      legendItem.append(swatch, label);
      elements.chartLegend.appendChild(legendItem);
    });
  }

  // Any token-derived chart color must be re-read from getComputedStyle on
  // every theme change, not cached.
  const unsubscribe = subscribe(() => {
    if (state.lastData) renderChart(state.lastData);
  });

  return {
    /** @param {string|null} runId */
    follow(runId) {
      stopPolling();
      state.runId = runId;
      state.lastData = null;
      hide(elements.tableCard);
      hide(elements.chartCard);
      hide(elements.statusBadge);
      hide(elements.meta);
      setNote(null);
      if (!runId) {
        setEmptyState('Select a queued or completed run to see its results.');
        return;
      }
      setEmptyState(null);
      void pollOnce();
    },
    currentRunId() {
      return state.runId;
    },
    destroy() {
      stopPolling();
      if (typeof unsubscribe === 'function') unsubscribe();
    },
  };
}
