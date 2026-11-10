// @ts-check
/**
 * The live run view: iteration progress, best so far, and a two-row Plotly
 * figure — the objective per evaluation with its running best, and each
 * variable normalized to its run bounds so they share one axis.
 *
 * The series are built by pure functions from the job's `detail=full` record
 * (`data`, `variable_names`, `objective_name`, `objective_direction`,
 * `variable_bounds`, `total_evaluations`); only {@link renderLivePlot} touches
 * Plotly (the classic-script global loaded by index.html).
 *
 * @module live-plot
 */

import { chartTheme, chartRelayout } from '/design-system/js/theme-manager.js';

/**
 * @typedef {Record<string, number|string|null>} DataRecord
 * @typedef {{
 *   status?: string,
 *   data?: DataRecord[]|null,
 *   data_version?: number,
 *   objective_name?: string|null,
 *   objective_direction?: string|null,
 *   variable_names?: string[]|null,
 *   variable_bounds?: Record<string, number[]>|null,
 *   total_evaluations?: number|null,
 * }} LiveJob
 */

/**
 * @param {unknown} value
 * @returns {number|null}
 */
function num(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The running best of `values` in `direction`; null until the first finite
 * value. Null when the direction is unknown — an inverted best line is worse
 * than none.
 *
 * @param {(number|null)[]} values
 * @param {string|null|undefined} direction
 * @returns {(number|null)[]|null}
 */
export function runningBest(values, direction) {
  if (direction !== 'MAXIMIZE' && direction !== 'MINIMIZE') return null;
  const better = direction === 'MAXIMIZE' ? Math.max : Math.min;
  /** @type {number|null} */
  let best = null;
  return values.map((v) => {
    if (v !== null) best = best === null ? v : better(best, v);
    return best;
  });
}

/**
 * Progress through the run: evaluations seen, the total the runner resolved
 * (initial samples + iterations; null before it reports one), and the best
 * objective value so far.
 *
 * @param {LiveJob} job
 * @returns {{ done: number, total: number|null, best: number|null, percent: number|null }}
 */
export function progress(job) {
  const data = job.data || [];
  const values = job.objective_name ? data.map((r) => num(r[job.objective_name || ''])) : [];
  const bests = runningBest(values, job.objective_direction);
  const best = bests && bests.length ? bests[bests.length - 1] : null;
  const total = num(job.total_evaluations);
  const done = data.length;
  return {
    done,
    total,
    best,
    percent: total && total > 0 ? Math.min(100, (done / total) * 100) : null,
  };
}

/**
 * Normalize `value` into [0, 1] over `bounds` (`[min, max]`); raw value when
 * the bounds are missing or degenerate.
 *
 * @param {number|null} value
 * @param {number[]|undefined} bounds
 * @returns {number|null}
 */
export function normalize(value, bounds) {
  if (value === null) return null;
  if (!bounds || bounds.length < 2) return value;
  const [lo, hi] = bounds;
  return hi > lo ? (value - lo) / (hi - lo) : value;
}

/**
 * The Plotly traces for a job.
 *
 * @param {LiveJob} job
 * @returns {Record<string, unknown>[]}
 */
export function liveTraces(job) {
  const data = job.data || [];
  const x = data.map((_, i) => i);
  /** @type {Record<string, unknown>[]} */
  const traces = [];
  const objective = job.objective_name;
  if (objective) {
    const y = data.map((r) => num(r[objective]));
    traces.push({ x, y, name: objective, mode: 'lines+markers', xaxis: 'x', yaxis: 'y' });
    const best = runningBest(y, job.objective_direction);
    if (best) {
      traces.push({
        x,
        y: best,
        name: 'Best so far',
        mode: 'lines',
        line: { dash: 'dash', shape: 'hv' },
        xaxis: 'x',
        yaxis: 'y',
      });
    }
  }
  for (const name of job.variable_names || []) {
    const bounds = job.variable_bounds?.[name];
    traces.push({
      x,
      y: data.map((r) => normalize(num(r[name]), bounds)),
      name,
      mode: 'lines+markers',
      marker: { size: 4 },
      xaxis: 'x2',
      yaxis: 'y2',
    });
  }
  return traces;
}

/**
 * @param {LiveJob} job
 * @returns {Record<string, unknown>}
 */
export function liveLayout(job) {
  const normalized = !!job.variable_bounds;
  return {
    ...chartTheme(),
    margin: { l: 56, r: 16, t: 16, b: 40 },
    showlegend: true,
    legend: { orientation: 'h', y: -0.15 },
    grid: { rows: 2, columns: 1, roworder: 'top to bottom' },
    xaxis: { anchor: 'y', matches: 'x2', showticklabels: false },
    yaxis: { title: { text: job.objective_name || 'objective' }, domain: [0.55, 1] },
    xaxis2: { anchor: 'y2', title: { text: 'evaluation' } },
    yaxis2: {
      title: { text: normalized ? 'variables (0–1 of bounds)' : 'variables' },
      domain: [0, 0.45],
      range: normalized ? [-0.05, 1.05] : undefined,
    },
    uirevision: 'live',
  };
}

/**
 * Draw (or update in place) the live figure. A no-op when Plotly did not load,
 * so the rest of the panel still works offline without its bundle.
 *
 * @param {HTMLElement} gd
 * @param {LiveJob} job
 * @returns {boolean} whether a figure was drawn
 */
export function renderLivePlot(gd, job) {
  const plotly = /** @type {any} */ (globalThis).Plotly;
  if (!plotly) return false;
  plotly.react(gd, liveTraces(job), liveLayout(job), { responsive: true, displaylogo: false });
  return true;
}

/**
 * Re-theme an already drawn figure from the design tokens.
 *
 * @param {HTMLElement} gd
 */
export function rethemeLivePlot(gd) {
  const plotly = /** @type {any} */ (globalThis).Plotly;
  if (!plotly || !(/** @type {any} */ (gd)._fullLayout)) return;
  plotly.relayout(gd, chartRelayout(gd));
}
