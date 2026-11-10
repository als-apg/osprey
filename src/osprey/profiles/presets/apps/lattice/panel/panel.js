// The lattice panel: what this terminal is looking at (model, energy, analysis) and a
// state loop that polls the backend and draws the shared result. Nothing here computes:
// the backend runs each analysis once for the whole deployment.
// Every address is relative: the page is served under the terminal's /panel/<id>/.

import { diffusionColor, esc, extent, fmt, plot, resonanceSegments } from './chart.js';

const STATE_URL = 'api/state';
const RESULT_URL = 'api/result';
const EXPORT_URL = 'api/export';
const REFRESH_MS = 2000;
const CHOICE_KEY = 'osprey.lattice.choice';

/**
 * @typedef {'ready' | 'queued' | 'running' | 'failed' | 'absent'} Status
 */

/**
 * @typedef {object} ModelInfo
 * @property {string} name
 * @property {string} label
 * @property {string} engine
 * @property {Record<string, unknown>} engine_settings
 * @property {string} deck
 * @property {number[]} energies_gev
 * @property {string[]} analyses
 */

/**
 * One analysis of the selected model, as the state reports it.
 * @typedef {object} Entry
 * @property {Status} status
 * @property {string} [key]
 * @property {Record<string, unknown>} settings
 * @property {number} deadline_s
 * @property {number} [elapsed_s]
 * @property {string} [error]
 * @property {string} [computed_at]
 * @property {Record<string, number | string | null>} [scalars]
 * @property {Record<string, string>} [units]
 */

/**
 * @typedef {object} State
 * @property {string | null} error
 * @property {ModelInfo[]} models
 * @property {{name: string, label: string}[]} analyses
 * @property {{model: string, energy_gev: number | null} | null} selected
 * @property {Record<string, Entry>} results
 */

/**
 * @typedef {object} Table
 * @property {Record<string, (number | null)[]>} columns
 * @property {Record<string, string>} [units]
 */

/**
 * @typedef {Entry & {model: string, analysis: string, tables?: Record<string, Table>}} Result
 */

/**
 * @typedef {object} Choice
 * @property {string | null} model
 * @property {number | null} energy
 * @property {string | null} analysis
 */

/**
 * @typedef {object} Panel
 * @property {HTMLSelectElement} model
 * @property {HTMLSelectElement} energy
 * @property {HTMLElement} analyses
 * @property {HTMLElement} status
 * @property {HTMLElement} error
 * @property {HTMLElement} meta
 * @property {HTMLElement} body
 * @property {Choice} choice
 * @property {Result | null} shown the result the body shows
 * @property {string | null} asked the key a result request is out for
 */

const SERIES = [1, 2, 3, 4].map((n) => `var(--chart-series-${n})`);

/** @type {WeakMap<Element, string>} */
const drawn = new WeakMap();

/**
 * Set an element's markup unless it already shows it, so a poll does not close an
 * open select or restart a pressed button. Every value in `html` is escaped by its caller.
 * @param {Element} element
 * @param {string} html
 */
function setHtml(element, html) {
  if (drawn.get(element) === html) return;
  drawn.set(element, html);
  element.innerHTML = html;
}

/** @returns {Choice} */
function storedChoice() {
  try {
    const stored = JSON.parse(localStorage.getItem(CHOICE_KEY) ?? '{}');
    return {
      model: typeof stored.model === 'string' ? stored.model : null,
      energy: typeof stored.energy === 'number' ? stored.energy : null,
      analysis: typeof stored.analysis === 'string' ? stored.analysis : null,
    };
  } catch {
    return { model: null, energy: null, analysis: null };
  }
}

/** @param {Choice} choice */
function storeChoice(choice) {
  try {
    localStorage.setItem(CHOICE_KEY, JSON.stringify(choice));
  } catch {
    // A terminal without storage starts from the first model every time.
  }
}

/**
 * The panel's elements under `root`.
 * @param {ParentNode} root
 * @returns {Panel}
 */
export function panelOf(root) {
  /**
   * @template {HTMLElement} T
   * @param {string} name
   * @returns {T}
   */
  const field = (name) => {
    const element = root.querySelector(`[data-field="${name}"]`);
    if (!element) throw new Error(`the panel has no ${name} field`);
    return /** @type {T} */ (element);
  };
  return {
    model: field('model'),
    energy: field('energy'),
    analyses: field('analyses'),
    status: field('status'),
    error: field('error'),
    meta: field('meta'),
    body: field('body'),
    choice: storedChoice(),
    shown: null,
    asked: null,
  };
}

/**
 * The query of a request for the panel's choice.
 * @param {Choice} choice
 * @param {Record<string, string>} [extra]
 */
export function query(choice, extra = {}) {
  const params = new URLSearchParams();
  if (choice.model) params.set('model', choice.model);
  if (choice.energy !== null) params.set('energy_gev', String(choice.energy));
  if (choice.analysis) params.set('analysis', choice.analysis);
  for (const [key, value] of Object.entries(extra)) params.set(key, value);
  return params.toString();
}

/**
 * @param {Table} table
 * @param {string} name
 * @returns {(number | null)[]}
 */
function column(table, name) {
  return table.columns[name] ?? [];
}

/** @param {number | null} tune */
function fractional(tune) {
  return tune === null ? null : tune - Math.floor(tune);
}

/**
 * A window on the tune plane around the values, at least `minSpan` wide: a tight
 * footprint still shows the resonance lines it sits between.
 * @param {(number | null)[]} values
 * @param {number} minSpan
 * @returns {[number, number]}
 */
export function tuneWindow(values, minSpan) {
  const [low, high] = extent([values], 0.25);
  const half = Math.max(high - low, minSpan) / 2;
  const centre = (low + high) / 2;
  return [centre - half, centre + half];
}

/**
 * A tune diagram: the points over the resonance lines up to fourth order.
 * @param {(number | null)[]} x
 * @param {(number | null)[]} y
 * @param {string[]} [colors]
 */
function tuneDiagram(x, y, colors) {
  const minSpan = x.length > 1 ? 0.1 : 0.4;
  const xRange = tuneWindow(x, minSpan);
  const yRange = tuneWindow(y, minSpan);
  return plot({
    xLabel: 'νx (fractional)',
    yLabel: 'νy (fractional)',
    xRange,
    yRange,
    segments: resonanceSegments(xRange, yRange, 4),
    series: [{ label: 'tunes', color: SERIES[0], x, y, mode: 'points', colors }],
    height: 340,
  });
}

/**
 * The charts of one analysis's result.
 * @param {string} analysis
 * @param {Result} result
 * @returns {string[]}
 */
export function charts(analysis, result) {
  const tables = result.tables ?? {};
  const scalars = result.scalars ?? {};
  /** @param {string} name */
  const num = (name) => {
    const value = scalars[name];
    return typeof value === 'number' ? value : null;
  };
  if (analysis === 'optics' && tables.twiss) {
    const t = tables.twiss;
    const s = column(t, 's');
    return [
      plot({
        xLabel: 's [m]',
        yLabel: 'β [m]',
        series: [
          { label: 'βx', color: SERIES[0], x: s, y: column(t, 'beta_x') },
          { label: 'βy', color: SERIES[1], x: s, y: column(t, 'beta_y') },
        ],
      }),
      plot({
        xLabel: 's [m]',
        yLabel: 'ηx [m]',
        series: [{ label: 'ηx', color: SERIES[2], x: s, y: column(t, 'eta_x') }],
        height: 200,
      }),
      tuneDiagram([fractional(num('tune_x'))], [fractional(num('tune_y'))]),
    ];
  }
  if (analysis === 'chromaticity' && tables.tune_vs_dp) {
    const t = tables.tune_vs_dp;
    const dp = column(t, 'dp_pct');
    return [
      plot({
        xLabel: 'δ [%]',
        yLabel: 'νx',
        series: [{ label: 'νx', color: SERIES[0], x: dp, y: column(t, 'tune_x'), mode: 'both' }],
        height: 240,
      }),
      plot({
        xLabel: 'δ [%]',
        yLabel: 'νy',
        series: [{ label: 'νy', color: SERIES[1], x: dp, y: column(t, 'tune_y'), mode: 'both' }],
        height: 240,
      }),
    ];
  }
  if (analysis === 'da' && tables.aperture) {
    const t = tables.aperture;
    return [
      plot({
        xLabel: 'x [mm]',
        yLabel: 'y [mm]',
        series: [
          {
            label: 'dynamic aperture',
            color: SERIES[0],
            x: column(t, 'x_mm'),
            y: column(t, 'y_mm'),
            mode: 'both',
          },
        ],
        height: 340,
      }),
    ];
  }
  if (analysis === 'footprint' && tables.tunes) {
    const t = tables.tunes;
    const colors = column(t, 'diffusion').map(diffusionColor);
    return [
      tuneDiagram(column(t, 'tune_x').map(fractional), column(t, 'tune_y').map(fractional), colors),
      plot({
        xLabel: 'x [mm]',
        yLabel: 'y [mm]',
        series: [
          {
            label: 'diffusion (blue stable, red chaotic)',
            color: SERIES[0],
            x: column(t, 'x_mm'),
            y: column(t, 'y_mm'),
            mode: 'points',
            colors,
          },
        ],
        height: 240,
      }),
    ];
  }
  if (analysis === 'lma' && tables.aperture) {
    const t = tables.aperture;
    const s = column(t, 's');
    return [
      plot({
        xLabel: 's [m]',
        yLabel: 'δ [%]',
        series: [
          { label: 'δ+', color: SERIES[0], x: s, y: column(t, 'dp_plus_pct'), mode: 'both' },
          { label: 'δ−', color: SERIES[1], x: s, y: column(t, 'dp_minus_pct'), mode: 'both' },
        ],
      }),
    ];
  }
  return [];
}

/**
 * The scalars of a result as a list of name and value.
 * @param {Entry} entry
 */
function scalarList(entry) {
  const rows = Object.entries(entry.scalars ?? {}).map(([name, value]) => {
    const unit = entry.units?.[name];
    return `<div class="osprey-list-row"><span class="lt-name">${esc(name)}</span><span class="lt-value">${esc(fmt(value))}${unit ? ` ${esc(unit)}` : ''}</span></div>`;
  });
  return rows.join('');
}

/**
 * Draw one analysis's entry: its charts when ready, else where it stands.
 * @param {Panel} panel
 * @param {Result} result
 */
export function showResult(panel, result) {
  panel.shown = result;
  const analysis = result.analysis;
  if (result.status === 'failed') {
    setHtml(
      panel.body,
      `<div class="osprey-state"><p class="osprey-state-message">${esc(analysis)} failed</p>` +
        `<p class="osprey-state-detail lt-reason">${esc(result.error ?? '')}</p>` +
        '<button type="button" class="osprey-btn" data-action="retry">Retry</button></div>'
    );
    return;
  }
  if (result.status !== 'ready') {
    const elapsed = result.elapsed_s === undefined ? '' : ` for ${Math.round(result.elapsed_s)} s`;
    const what =
      result.status === 'queued' ? 'queued behind other analyses' : `computing${elapsed}`;
    setHtml(
      panel.body,
      `<div class="osprey-state"><p class="osprey-state-message">${esc(analysis)}: ${what}</p>` +
        `<p class="osprey-state-detail">Computed once for every terminal; deadline ${fmt(result.deadline_s)} s.</p></div>`
    );
    return;
  }
  const exports = ['csv', 'sdds']
    .map(
      (format) =>
        `<a class="osprey-btn" href="${EXPORT_URL}?${esc(query(panel.choice, { format }))}" download>${format.toUpperCase()}</a>`
    )
    .join('');
  setHtml(
    panel.body,
    `<div class="lt-charts">${charts(analysis, result).join('')}</div>` +
      `<aside class="osprey-card lt-side"><div class="osprey-section-label">Results</div>${scalarList(result)}` +
      `<div class="osprey-section-label">Files</div><div class="lt-files">${exports}</div></aside>`
  );
}

/**
 * The line under the toolbar: what computed the numbers, and with which settings.
 * @param {ModelInfo} model
 * @param {Entry | undefined} entry
 */
export function metaLine(model, entry) {
  /** @param {Record<string, unknown>} settings */
  const list = (settings) =>
    Object.entries(settings)
      .map(([key, value]) => `${key} ${typeof value === 'object' ? JSON.stringify(value) : value}`)
      .join(', ');
  const parts = [`engine ${model.engine}`, `deck ${model.deck}`];
  if (Object.keys(model.engine_settings).length) parts.push(list(model.engine_settings));
  if (entry && Object.keys(entry.settings).length) parts.push(list(entry.settings));
  if (entry?.computed_at)
    parts.push(`computed ${entry.computed_at} in ${fmt(entry.elapsed_s ?? null)} s`);
  return parts.join(' · ');
}

/** @type {Record<Status, [string, string]>} */
const CHIPS = {
  ready: ['osprey-chip--ok', 'ready'],
  running: ['osprey-chip--connecting', 'computing'],
  queued: ['osprey-chip--info', 'queued'],
  failed: ['osprey-chip--err', 'failed'],
  absent: ['osprey-chip--info', 'not computed'],
};

/**
 * Draw the toolbar and the status from the backend's state; settle the choice on what it offers.
 * @param {Panel} panel
 * @param {State} state
 */
export function showState(panel, state) {
  panel.error.hidden = !state.error;
  panel.error.textContent = state.error ?? '';
  const selected = state.models.find((m) => m.name === state.selected?.model);
  if (!selected) {
    panel.model.hidden = panel.energy.hidden = panel.status.hidden = panel.meta.hidden = true;
    setHtml(panel.analyses, '');
    setHtml(panel.body, '');
    panel.shown = null;
    return;
  }
  const { choice } = panel;
  choice.model = selected.name;
  choice.energy = state.selected?.energy_gev ?? null;
  if (!choice.analysis || !selected.analyses.includes(choice.analysis)) {
    choice.analysis = selected.analyses[0] ?? null;
  }

  panel.model.hidden = state.models.length < 2;
  setHtml(
    panel.model,
    state.models
      .map(
        (m) =>
          `<option value="${esc(m.name)}"${m.name === selected.name ? ' selected' : ''}>${esc(m.label)}</option>`
      )
      .join('')
  );
  panel.energy.hidden = selected.energies_gev.length === 0;
  setHtml(
    panel.energy,
    selected.energies_gev
      .map(
        (e) =>
          `<option value="${e}"${e === choice.energy ? ' selected' : ''}>${fmt(e)} GeV</option>`
      )
      .join('')
  );
  const buttons = state.analyses
    .filter((a) => selected.analyses.includes(a.name))
    .map((a) => {
      const status = state.results[a.name]?.status ?? 'absent';
      return `<button type="button" data-analysis="${esc(a.name)}" data-status="${esc(status)}" aria-pressed="${a.name === choice.analysis}">${esc(a.label)}</button>`;
    });
  setHtml(panel.analyses, buttons.join(''));

  const entry = choice.analysis ? state.results[choice.analysis] : undefined;
  panel.status.hidden = !entry;
  if (entry) {
    const [tone, text] = CHIPS[entry.status] ?? CHIPS.absent;
    panel.status.className = `osprey-chip ${tone}`;
    panel.status.textContent = text;
  }
  panel.meta.hidden = false;
  panel.meta.textContent = metaLine(selected, entry);
}

/**
 * Read the state, and the selected analysis's result when the body does not show it yet.
 * @param {Panel} panel
 * @param {{retry?: boolean}} [options]
 */
export async function refresh(panel, options = {}) {
  /** @type {State} */
  let state;
  try {
    const response = await fetch(`${STATE_URL}?${query({ ...panel.choice, analysis: null })}`);
    if (response.status === 400 || response.status === 404) {
      // The stored choice names a model or energy this deployment no longer offers.
      panel.choice = { model: null, energy: null, analysis: panel.choice.analysis };
      return;
    }
    if (!response.ok) throw new Error(`the state read answered ${response.status}`);
    state = await response.json();
  } catch (error) {
    panel.error.hidden = false;
    panel.error.textContent = `The lattice backend is not answering: ${error instanceof Error ? error.message : error}`;
    return;
  }
  showState(panel, state);
  const analysis = panel.choice.analysis;
  const entry = analysis ? state.results[analysis] : undefined;
  if (!analysis || !entry || !state.selected) return;
  const current = { ...entry, model: state.selected.model, analysis };
  const shown = panel.shown;
  const same = shown && shown.key === entry.key && shown.analysis === analysis;
  if (same && shown.status === 'ready' && entry.status === 'ready') return;
  // A ready result's tables, or the first ask that starts an absent one, come from the
  // result route; a running or failed one is drawn from the state alone.
  const fetchResult = options.retry || entry.status === 'ready' || entry.status === 'absent';
  if (!fetchResult) {
    showResult(panel, current);
    return;
  }
  const asked = `${entry.key}:${options.retry ? 'retry' : entry.status}`;
  if (panel.asked === asked) return;
  panel.asked = asked;
  try {
    const extra = options.retry ? { retry: 'true' } : undefined;
    const response = await fetch(`${RESULT_URL}?${query(panel.choice, extra)}`);
    if (!response.ok) throw new Error((await response.json()).detail ?? response.statusText);
    /** @type {Result} */
    const result = await response.json();
    if (panel.choice.analysis === result.analysis) showResult(panel, result);
  } catch (error) {
    panel.error.hidden = false;
    panel.error.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    panel.asked = null;
  }
}

/**
 * Wire the controls and start the state loop.
 * @param {Panel} panel
 */
export function start(panel) {
  const changed = () => {
    storeChoice(panel.choice);
    panel.shown = null;
    setHtml(panel.body, '');
    void refresh(panel);
  };
  panel.model.addEventListener('change', () => {
    panel.choice = { model: panel.model.value, energy: null, analysis: panel.choice.analysis };
    changed();
  });
  panel.energy.addEventListener('change', () => {
    panel.choice.energy = Number(panel.energy.value);
    changed();
  });
  panel.analyses.addEventListener('click', (event) => {
    const button = event.target instanceof Element ? event.target.closest('button') : null;
    if (!button?.dataset.analysis) return;
    panel.choice.analysis = button.dataset.analysis;
    changed();
  });
  panel.body.addEventListener('click', (event) => {
    const button =
      event.target instanceof Element ? event.target.closest('[data-action="retry"]') : null;
    if (button) void refresh(panel, { retry: true });
  });
  void refresh(panel);
  return setInterval(() => {
    if (!document.hidden) void refresh(panel);
  }, REFRESH_MS);
}

const root = typeof document === 'undefined' ? null : document.getElementById('lattice');
if (root) start(panelOf(root));
