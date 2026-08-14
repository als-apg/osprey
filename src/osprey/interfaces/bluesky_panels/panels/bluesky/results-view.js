// @ts-check
/**
 * BLUESKY panel — the Results view.
 *
 * Follows ONE run through three reads per tick, all on the sidecar's read
 * proxy: `GET /runs/{id}` for its record, `GET /runs/{id}/data` for a bounded
 * rows/columns window, and `GET /runs/{id}/figure` for the plan's own view of
 * itself. The first two feed the raw table; the third is handed to
 * `figure-renderer.js` to draw. Read only — nothing in this file issues a
 * write verb.
 *
 * The FIGURE leads and the table follows, collapsed. A run's figure is the
 * plan's own answer to what the run meant; the table is the raw material behind
 * it, and it can be thousands of rows. So the table renders inside a `<details>`
 * that ships closed, and the view never touches its `open` state after that —
 * an operator stepping through runs comparing rows keeps it open for as long as
 * they want it, because nothing here closes it again.
 *
 * That makes the table a PREVIEW, and it is labelled as one. `/data` serves at
 * most `max_rows` (100 by default) while reporting the run's true `row_count`
 * separately, so a truncated window says how much it is withholding and points
 * at the export. The export is the other half: it re-reads `/data` with
 * `max_rows` set to the full count and writes a CSV through `csv-export.js`.
 * If even that comes back short — `live_rows` caps stored rows per run — the
 * file still saves and the note says exactly how short, because a partial
 * export presented as complete is the one outcome worth preventing.
 *
 * Cadence: ~1 s while the run is non-terminal or either the data or the figure
 * still reports `partial: true`; polling stops once the run is terminal and
 * both halves are settled, so a finished run costs nothing.
 *
 * Two shapes of "gone" are distinguished, because they mean opposite things:
 * `GET /runs/{id}` 404s once the manager has rotated the run out of its
 * history, while `/data` and `/figure` keep serving it from Tiled long
 * afterwards. A record 404 therefore does NOT clear the results — the data is
 * shown with a note, because the durable record is the one an operator came
 * for.
 *
 * A figure that carries a `reason` is a DEFAULT view, not an error: the bridge
 * could not run the plan's own `render()` and fell back to plotting the raw
 * columns. It is drawn like any other, and the renderer's note line says why.
 * A transient figure fetch failure keeps the last good figure on screen rather
 * than blanking it, exactly as the table does with its last good rows.
 *
 * Colors live in the renderer, which re-reads the theme bridges on every call;
 * this file re-renders the retained figure from a `subscribe()` callback so a
 * settled run re-themes without waiting for a poll tick that will never come.
 */

import { subscribe } from '/design-system/js/theme-manager.js';

import { exportFilename, saveCsv, toCsv } from './csv-export.js';
import { renderFigure } from './figure-renderer.js';
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

/** The bridge's figure wire shape, owned by the renderer that draws it. */
/** @typedef {import('./figure-renderer.js').Figure} Figure */

/** @typedef {{ok: true, data: RunRecord} | {ok: false, notFound: boolean, message: string}} RecordFetch */
/** @typedef {{ok: true, data: RunData} | {ok: false, notFound: boolean, notStarted: boolean, message: string}} DataFetch */
/** @typedef {{ok: true, data: Figure} | {ok: false, notFound: boolean, message: string}} FigureFetch */

const POLL_INTERVAL_MS = 1000;
const TERMINAL_STATUSES = ['completed', 'stopped', 'error'];

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
 * Whether following `run` should keep polling.
 *
 * Partial data outranks a terminal status: the run's stop document can land
 * before the last events are recorded, and stopping the poll there would
 * freeze the table one row short of the truth. `partial` is the OR of what the
 * two data-bearing reads report — either one still filling in is reason enough
 * to look again.
 *
 * @param {string} status
 * @param {boolean} partial
 * @returns {boolean}
 */
export function shouldKeepPolling(status, partial) {
  return partial || !TERMINAL_STATUSES.includes(status);
}

/**
 * A count with thousands separators, pinned to `en-US` so the panel reads the
 * same everywhere and so tests assert one string rather than the runner's
 * locale.
 *
 * @param {number} value
 * @returns {string}
 */
function count(value) {
  return value.toLocaleString('en-US');
}

/** @typedef {{
 *   statusBadge: HTMLElement,
 *   meta: HTMLElement,
 *   note: HTMLElement,
 *   emptyState: HTMLElement,
 *   tableCard: HTMLElement,
 *   tableDetails: HTMLDetailsElement,
 *   tableSummaryCount: HTMLElement,
 *   tableNote: HTMLElement,
 *   tableHeadRow: HTMLTableRowElement,
 *   tableBody: HTMLTableSectionElement,
 *   exportButton: HTMLButtonElement,
 *   exportNote: HTMLElement,
 *   figureCard: HTMLElement,
 *   figurePanels: HTMLElement,
 *   figureNote: HTMLElement,
 * }} ResultsElements `figurePanels` is handed straight to the renderer, which
 *   empties and refills it; nothing else in this file writes to it.
 *   `figureNote` takes the renderer's provenance line (source, partial,
 *   reason) — it is a separate element from `note`, which carries this view's
 *   own message about the run RECORD, and from `exportNote`, which carries the
 *   outcome of the last save.
 *
 *   `tableCard` is the whole data section (disclosure + export). `tableDetails`
 *   is the disclosure itself and this view READS its `open` state but never
 *   writes it — see the module docstring. */

/**
 * @param {{
 *   api: (path: string) => string,
 *   elements: ResultsElements,
 *   onPollingChange?: (polling: boolean) => void,
 *   saveFile?: (filename: string, text: string) => Promise<{saved: boolean, method: 'picker'|'download'}>,
 * }} deps `onPollingChange` reports whether this view is currently polling for
 *   a run — i.e. whether data is still arriving. The panel shell turns that
 *   into the Results tab's activity marker when the operator is looking at
 *   another tab. It fires only on a transition, so a caller may re-render
 *   freely from it.
 *
 *   `saveFile` defaults to `csv-export.js`'s `saveCsv`. It is a seam rather
 *   than a hard import so the three save outcomes (dialog, cancelled, download
 *   fallback) are reachable under a DOM shim with no File System Access API.
 * @returns {{follow: (runId: string|null) => void, currentRunId: () => string|null, destroy: () => void}}
 */
export function createResultsView({ api, elements, onPollingChange, saveFile = saveCsv }) {
  /** @type {{
   *   runId: string|null,
   *   timer: ReturnType<typeof setTimeout>|null,
   *   lastFigure: Figure|null,
   *   rowCount: number,
   *   planName: string|null,
   *   exporting: boolean,
   * }} `lastFigure` is the last figure that had something to draw. It is what a
   *   theme change re-renders from, and what a transient figure fetch failure
   *   leaves on screen.
   *
   *   `rowCount` and `planName` are what the export needs and the click handler
   *   cannot re-derive: the run's TRUE total (not the rendered window's length)
   *   and the name the file is called after. `exporting` guards against a
   *   second click while a large fetch is in flight. */
  const state = {
    runId: null,
    timer: null,
    lastFigure: null,
    rowCount: 0,
    planName: null,
    exporting: false,
  };

  /** @type {import('./figure-renderer.js').FigureSelection} */
  const figureSelection = { series: {}, section: {}, open: {} };

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
   * The outcome of the last save. Cleared on every new export and on every
   * `follow()`, so a "Saved orm-r1.csv" line can never end up sitting under a
   * different run's figure, where it would read as a claim about that run.
   *
   * @param {string|null} message
   * @param {boolean} [isError]
   */
  function setExportNote(message, isError = false) {
    if (message === null) {
      hide(elements.exportNote);
      return;
    }
    elements.exportNote.textContent = message;
    elements.exportNote.classList.toggle('error', isError);
    show(elements.exportNote);
  }

  // All three reads below check the SHAPE of a 200 before reporting success,
  // and treat an unrecognizable body as a transient failure.
  //
  // This is not defensive habit, it is what keeps the panel alive: the three
  // reads share one tick, so anything that throws between the `await` and
  // `scheduleNextPoll` abandons the tick without rescheduling it. Because
  // `state.timer` is never rewritten, the view then stops polling for good AND
  // leaves the Results tab's activity marker stuck on, with nothing on screen
  // to say so. And an unparseable 200 is not hypothetical: the read proxy's
  // `safe_json` relays a null body on a 200, so any of these can arrive as
  // 200 + `null`.

  /**
   * The run's record. `status` is the field the whole tick turns on, so a body
   * without one is not a record.
   *
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
      const body = await response.json();
      if (!body || typeof body.status !== 'string') {
        return { ok: false, notFound: false, message: 'record response was not a run record' };
      }
      return { ok: true, data: /** @type {RunRecord} */ (body) };
    } catch {
      return { ok: false, notFound: false, message: 'network error' };
    }
  }

  /**
   * A bounded window of the run's rows. `columns` and `rows` are both checked
   * because `renderTable` walks both, so a body carrying one without the other
   * would throw downstream.
   *
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
      const body = await response.json();
      if (!body || !Array.isArray(body.columns) || !Array.isArray(body.rows)) {
        return {
          ok: false,
          notFound: false,
          notStarted: false,
          message: 'data response was not a data window',
        };
      }
      return { ok: true, data: /** @type {RunData} */ (body) };
    } catch {
      return { ok: false, notFound: false, notStarted: false, message: 'network error' };
    }
  }

  /**
   * The plan's own view of the run.
   *
   * There is no "not started" shape here: the bridge answers 200 for any run
   * either its live buffer or Tiled knows, and 404 — byte-identical to
   * `/data`'s — only for a run neither source has heard of. A 502 (the sidecar
   * cannot reach the bridge at all) lands in the transient bucket alongside a
   * network error, which is what keeps the last good figure on screen.
   *
   * A 200 carrying something that is not a figure is treated the same way, per
   * the shape rule above.
   *
   * @param {string} runId
   * @returns {Promise<FigureFetch>}
   */
  async function fetchFigure(runId) {
    try {
      const response = await fetch(api(`/runs/${encodeURIComponent(runId)}/figure`));
      if (response.status === 404) return { ok: false, notFound: true, message: 'run not found' };
      if (!response.ok) {
        return { ok: false, notFound: false, message: `figure HTTP ${response.status}` };
      }
      const body = await response.json();
      if (!body || !Array.isArray(body.panels)) {
        return { ok: false, notFound: false, message: 'figure response was not a figure' };
      }
      return { ok: true, data: /** @type {Figure} */ (body) };
    } catch {
      return { ok: false, notFound: false, message: 'network error' };
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

    const [recordFetch, dataFetch, figureFetch] = await Promise.all([
      fetchRecord(runId),
      fetchData(runId),
      fetchFigure(runId),
    ]);
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

    let dataPartial = false;
    // Neither half of the run exists — the one case where there is nothing
    // left to wait for. Decided here, acted on below, so the figure still gets
    // its turn to clear itself before the poll shuts down.
    let nothingLeft = false;
    if (dataFetch.ok) {
      dataPartial = dataFetch.data.partial === true;
      renderTable(dataFetch.data);
      setEmptyState(dataFetch.data.columns.length === 0 ? 'No data columns yet.' : null);
    } else if (dataFetch.notStarted) {
      hide(elements.tableCard);
      setEmptyState('This run has not started — no data yet.');
    } else if (dataFetch.notFound && !recordFetch.ok) {
      hide(elements.tableCard);
      setEmptyState('No record and no data for this run.', true);
      nothingLeft = true;
    } else if (dataFetch.notFound) {
      hide(elements.tableCard);
      setEmptyState('No data recorded for this run yet.');
    } else {
      // Transient data error: keep the last good table on screen rather than
      // clearing it out from under the operator.
      setEmptyState(null);
    }

    let figurePartial = false;
    if (figureFetch.ok) {
      figurePartial = figureFetch.data.partial === true;
      if (figureFetch.data.panels.length > 0) {
        // Retained only once it has actually been drawn — a figure the
        // renderer choked on is not a figure worth re-rendering on every
        // theme change for the rest of the session.
        if (drawFigure(figureFetch.data)) state.lastFigure = figureFetch.data;
      } else if (state.lastFigure === null) {
        // Nothing to draw and nothing drawn before. An empty figure means the
        // run has no plottable columns yet, or the bridge could not read the
        // store this tick (`source_unavailable`) — in both cases the empty
        // state above is already telling the operator where things stand, so
        // an empty card would only add noise.
        hide(elements.figureCard);
      }
      // An empty figure NEVER replaces a good one: a Tiled read that failed
      // this tick says nothing about the figure it served last tick.
    } else if (figureFetch.notFound) {
      state.lastFigure = null;
      hide(elements.figureCard);
    }
    // Any other figure failure is transient (a 502 while the bridge restarts,
    // a dropped connection): leave the retained figure exactly where it is.

    if (nothingLeft) {
      // Neither source has this run any more, so anything still on screen is a
      // plot of a run that no longer exists. Clearing it is what keeps the
      // "No record and no data" banner from sitting over a drawn figure and
      // contradicting it.
      state.lastFigure = null;
      hide(elements.figureCard);
      stopPolling();
      return;
    }

    // `shouldKeepPolling` is the whole decision. In particular a run the
    // manager has rotated away (record 404) whose data or figure still reports
    // `partial: true` keeps polling — the durable store is still filling in,
    // and the record's absence says nothing about that.
    if (shouldKeepPolling(status, dataPartial || figurePartial)) scheduleNextPoll(POLL_INTERVAL_MS);
  }

  /** @param {RunRecord} run */
  function renderRecord(run) {
    // Kept for the export filename. A run rotated out of the manager's history
    // has no record at all, so this is the last name seen rather than a name
    // the export can look up on demand.
    state.planName = run.plan_name ?? null;
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
    // rows.length. It is what the disclosure summary names (so the operator
    // knows the size of the thing before opening it) and what the export asks
    // the bridge for.
    state.rowCount = data.row_count;
    elements.tableSummaryCount.textContent = `${count(data.row_count)} row${
      data.row_count === 1 ? '' : 's'
    }`;

    // The caveat is printed ONLY when something is actually being withheld.
    // On a settled 12-row run the summary already says "12 rows", and a second
    // line repeating it would be noise that trains the operator to skip the
    // place the truncation warning appears.
    if (data.truncated) {
      elements.tableNote.textContent =
        `preview — showing ${count(data.rows.length)} of ${count(data.row_count)}` +
        ` · export for all rows`;
      show(elements.tableNote);
    } else {
      hide(elements.tableNote);
    }

    elements.exportButton.disabled = state.exporting || data.row_count === 0;

    if (data.columns.length === 0) hide(elements.tableCard);
    else show(elements.tableCard);
  }

  /**
   * Export the run's data as a CSV.
   *
   * Deliberately re-fetches rather than serialising what is on screen: the
   * table holds at most `max_rows` rows (100 by default) and the operator is
   * asking for the RUN, not for the window. `max_rows` is set to the true
   * `row_count`, which is the largest number the bridge could possibly honour.
   *
   * The bridge may still answer with fewer — `live_rows` caps stored rows per
   * run, and a Tiled entry can be rotated. That file is still worth having, so
   * it is saved; what must not happen is saving it silently, so the note names
   * both numbers.
   */
  async function exportCsv() {
    const runId = state.runId;
    if (!runId || state.exporting) return;

    state.exporting = true;
    elements.exportButton.disabled = true;
    setExportNote(null);

    try {
      const response = await fetch(
        api(`/runs/${encodeURIComponent(runId)}/data?max_rows=${Math.max(1, state.rowCount)}`)
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const body = await response.json();
      if (!body || !Array.isArray(body.columns) || !Array.isArray(body.rows)) {
        throw new Error('the response was not a data window');
      }
      // The selection moved while the fetch was in flight. Saving now would
      // drop the previous run's file under the current run's figure.
      if (state.runId !== runId) return;

      const filename = exportFilename(state.planName, runId);
      const result = await saveFile(filename, toCsv(body.columns, body.rows));
      if (!result.saved) {
        // Cancelled at the dialog. Nothing was written, so nothing is claimed.
        setExportNote(null);
        return;
      }

      const written = body.rows.length;
      const total = typeof body.row_count === 'number' ? body.row_count : written;
      const rows =
        written < total
          ? `${count(written)} of ${count(total)} rows — the rest is no longer stored`
          : `${count(written)} row${written === 1 ? '' : 's'}`;
      const where = result.method === 'download' ? ' to your Downloads folder' : '';
      setExportNote(`Saved ${filename}${where} · ${rows}.`);
    } catch (error) {
      setExportNote(`Could not export: ${/** @type {Error} */ (error).message}`, true);
    } finally {
      state.exporting = false;
      // Re-enabled even after a failure: a 502 while the bridge restarts says
      // nothing about whether the next attempt will work, and a control that
      // disarms itself on one bad tick is a control the operator has to reload
      // the panel to get back.
      elements.exportButton.disabled = state.rowCount === 0;
    }
  }

  elements.exportButton.addEventListener('click', exportCsv);

  /**
   * Hand a figure to the renderer and reveal the card.
   *
   * The renderer owns everything below `figurePanels` — it empties and refills
   * that element, writes the provenance line into `figureNote`, and re-reads
   * the theme on every call. This function exists so the poll path and the
   * theme path draw through exactly the same door.
   *
   * A throwing renderer is contained here for the same reason a bad body is
   * contained in `fetchFigure`: this runs inside the shared poll tick, so an
   * escaping error would abandon the tick before it could reschedule itself,
   * freezing the record and table reads too and leaving the tab's activity
   * marker stuck on. A 200 whose `panels` array is well-formed but whose
   * contents are not is the case that gets here.
   *
   * @param {Figure} figure
   * @returns {boolean} Whether the figure actually reached the screen.
   */
  function drawFigure(figure) {
    try {
      renderFigure(
        { panels: elements.figurePanels, note: elements.figureNote },
        figure,
        { selection: figureSelection }
      );
    } catch (error) {
      console.error('osprey bluesky results: the figure renderer threw', error);
      return false;
    }
    show(elements.figureCard);
    return true;
  }

  // A theme change must re-draw from the retained figure, because a settled
  // run has stopped polling: without this, flipping to light mode would leave
  // a finished scan drawn in the dark palette until the operator picked
  // another run. The renderer re-reads the color tokens itself, so there is
  // nothing to invalidate here beyond calling it again.
  const unsubscribe = subscribe(() => {
    if (state.lastFigure) drawFigure(state.lastFigure);
  });

  return {
    /** @param {string|null} runId */
    follow(runId) {
      stopPolling();
      state.runId = runId;
      state.lastFigure = null;
      state.rowCount = 0;
      state.planName = null;
      hide(elements.tableCard);
      hide(elements.figureCard);
      hide(elements.statusBadge);
      hide(elements.meta);
      setNote(null);
      // `tableDetails.open` is deliberately NOT reset here — see the module
      // docstring. The operator's choice to have the table open outlives any
      // one run.
      //
      // The figure's pickers follow the same split. `series`/`section` name
      // devices and panels belonging to ONE run — re-applying "show hcm3" to
      // the next run would silently pick a different magnet, or nothing — so
      // they are cleared. `open` is a disclosure preference exactly like
      // `tableDetails.open`, so it survives.
      figureSelection.series = {};
      figureSelection.section = {};
      setExportNote(null);
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
      elements.exportButton.removeEventListener('click', exportCsv);
      if (typeof unsubscribe === 'function') unsubscribe();
    },
  };
}
